/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

namespace {
using Scalar = std::complex<double>;
using Verdict = VerificationVerdict;
using Relation = VerificationRelation;
using Method = VerificationMethod;
using Phase = VerificationPhase;

// The oracle deliberately uses scalar dense arithmetic only. It does not call
// Core's gate matrices, DD construction, interpreter, or layout normalization.
struct Matrix {
  size_t dimension;
  std::vector<Scalar> values;
  explicit Matrix(size_t size) : dimension(size), values(size * size) {}
  Scalar& at(size_t row, size_t column) {
    return values[row * dimension + column];
  }
  [[nodiscard]] Scalar at(size_t row, size_t column) const {
    return values[row * dimension + column];
  }
};

} // namespace

static Matrix identity(size_t dimension) {
  Matrix result(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    result.at(i, i) = 1;
  }
  return result;
}

static Matrix multiply(const Matrix& left, const Matrix& right) {
  Matrix result(left.dimension);
  for (size_t row = 0; row < result.dimension; ++row) {
    for (size_t column = 0; column < result.dimension; ++column) {
      for (size_t inner = 0; inner < result.dimension; ++inner) {
        result.at(row, column) += left.at(row, inner) * right.at(inner, column);
      }
    }
  }
  return result;
}

static double maxDifference(const Matrix& left, const Matrix& right,
                            Scalar phase = 1) {
  double maximum = 0;
  for (size_t i = 0; i < left.values.size(); ++i) {
    maximum =
        std::max(maximum, std::abs(left.values[i] - phase * right.values[i]));
  }
  return maximum;
}

static double amplitudeDistance(const Matrix& left, const Matrix& right,
                                size_t inputDimension, Scalar phase = 1) {
  double sum = 0;
  for (size_t i = 0; i < left.values.size(); ++i) {
    sum += std::norm(left.values[i] - phase * right.values[i]);
  }
  return std::sqrt(sum / static_cast<double>(inputDimension));
}

static double processDistance(const Matrix& left, const Matrix& right,
                              size_t inputDimension) {
  Scalar overlap = 0;
  for (size_t i = 0; i < left.values.size(); ++i) {
    overlap += std::conj(left.values[i]) * right.values[i];
  }
  // Avoid subtractive cancellation for exactly projective fixtures. Other
  // cases use the defining normalized trace formula directly.
  if (std::abs(overlap) != 0 &&
      amplitudeDistance(left, right, inputDimension,
                        std::conj(overlap) / std::abs(overlap)) < 1e-13) {
    return 0;
  }
  return std::sqrt(
      std::max(0., 1 - std::norm(overlap) /
                           std::pow(static_cast<double>(inputDimension), 2)));
}

namespace {

enum class GateKind : uint8_t {
  H,
  RX,
  RY,
  RZ,
  X,
  Z,
  CX,
  CPhase,
  Swap,
  GlobalPhase,
};
struct Gate {
  GateKind kind;
  size_t target = 0;
  size_t control = 0;
  double angle = 0;
};
struct Circuit {
  size_t qubits;
  std::vector<Gate> gates;
  std::vector<std::pair<size_t, size_t>> measurements;
  std::string registerName = "c";
};

} // namespace

static Matrix denseGate(size_t qubits, const Gate& gate) {
  const auto dimension = size_t{1} << qubits;
  Matrix result(dimension);
  if (gate.kind == GateKind::GlobalPhase) {
    for (size_t i = 0; i < dimension; ++i) {
      result.at(i, i) = std::polar(1., gate.angle);
    }
    return result;
  }
  if (gate.kind == GateKind::Swap) {
    for (size_t column = 0; column < dimension; ++column) {
      auto row = column;
      if (((column >> gate.target) & 1U) != ((column >> gate.control) & 1U)) {
        row ^= (size_t{1} << gate.target) | (size_t{1} << gate.control);
      }
      result.at(row, column) = 1;
    }
    return result;
  }
  std::array<Scalar, 4> local{};
  const auto cosine = std::cos(gate.angle / 2);
  const auto sine = std::sin(gate.angle / 2);
  switch (gate.kind) {
  case GateKind::H: {
    const auto h = std::numbers::sqrt2 / 2;
    local = {h, h, h, -h};
    break;
  }
  case GateKind::RX:
    local = {cosine, Scalar{0, -sine}, Scalar{0, -sine}, cosine};
    break;
  case GateKind::RY:
    local = {cosine, -sine, sine, cosine};
    break;
  case GateKind::RZ:
    local = {
        std::polar(1., -gate.angle / 2),
        0.,
        0.,
        std::polar(1., gate.angle / 2),
    };
    break;
  case GateKind::X:
  case GateKind::CX:
    local = {0., 1., 1., 0.};
    break;
  case GateKind::Z:
    local = {1., 0., 0., -1.};
    break;
  case GateKind::CPhase:
    local = {1., 0., 0., std::polar(1., gate.angle)};
    break;
  case GateKind::Swap:
  case GateKind::GlobalPhase:
    break;
  }
  const bool controlled =
      gate.kind == GateKind::CX || gate.kind == GateKind::CPhase;
  for (size_t column = 0; column < dimension; ++column) {
    if (controlled && ((column >> gate.control) & 1U) == 0) {
      result.at(column, column) = 1;
      continue;
    }
    for (size_t bit = 0; bit < 2; ++bit) {
      const auto row =
          (column & ~(size_t{1} << gate.target)) | (bit << gate.target);
      result.at(row, column) = local[2 * bit + ((column >> gate.target) & 1U)];
    }
  }
  return result;
}

static Matrix denseCircuit(const Circuit& circuit,
                           const std::vector<size_t>& zeroAncillas = {}) {
  auto result = identity(size_t{1} << circuit.qubits);
  for (const auto& gate : circuit.gates) {
    result = multiply(denseGate(circuit.qubits, gate), result);
  }
  for (size_t column = 0; column < result.dimension; ++column) {
    if (std::ranges::any_of(zeroAncillas, [column](size_t bit) {
          return ((column >> bit) & 1U) != 0;
        })) {
      for (size_t row = 0; row < result.dimension; ++row) {
        result.at(row, column) = 0;
      }
    }
  }
  return result;
}

static Matrix effect(const Matrix& unitary, size_t observedWire, bool outcome) {
  Matrix result(unitary.dimension);
  for (size_t input = 0; input < result.dimension; ++input) {
    for (size_t other = 0; other < result.dimension; ++other) {
      for (size_t output = 0; output < result.dimension; ++output) {
        if (((output >> observedWire) & 1U) == static_cast<size_t>(outcome)) {
          result.at(input, other) +=
              std::conj(unitary.at(output, input)) * unitary.at(output, other);
        }
      }
    }
  }
  return result;
}

static std::string program(const Circuit& circuit) {
  std::ostringstream ir;
  ir << std::scientific << std::setprecision(17);
  ir << "module { func.func @main()";
  if (!circuit.measurements.empty()) {
    ir << " -> !cbit.reg<9>";
  }
  ir << " {\n";
  std::vector<std::string> wires(circuit.qubits);
  for (size_t wire = 0; wire < circuit.qubits; ++wire) {
    wires[wire] = "%q" + std::to_string(wire);
    ir << wires[wire] << " = qco.static " << wire << " : !qco.qubit\n";
  }
  for (size_t index = 0; index < circuit.gates.size(); ++index) {
    const auto& gate = circuit.gates[index];
    const auto name = "%g" + std::to_string(index);
    const auto parameter = "%a" + std::to_string(index);
    const bool parameterized =
        gate.kind == GateKind::RX || gate.kind == GateKind::RY ||
        gate.kind == GateKind::RZ || gate.kind == GateKind::CPhase ||
        gate.kind == GateKind::GlobalPhase;
    if (parameterized) {
      ir << parameter << " = arith.constant " << gate.angle << " : f64\n";
    }
    if (gate.kind == GateKind::GlobalPhase) {
      ir << "qco.gphase(" << parameter << ")\n";
      continue;
    }
    if (gate.kind == GateKind::Swap) {
      ir << name << "a, " << name << "b = qco.swap " << wires[gate.target]
         << ", " << wires[gate.control] << " : !qco.qubit, !qco.qubit\n";
      wires[gate.target] = name + "a";
      wires[gate.control] = name + "b";
      continue;
    }
    if (gate.kind == GateKind::CX || gate.kind == GateKind::CPhase) {
      ir << name << "c, " << name << "t = qco.ctrl(" << wires[gate.control]
         << ") targets(" << name << "arg = " << wires[gate.target] << ") {\n"
         << name << "body = qco.";
      if (gate.kind == GateKind::CX) {
        ir << "x";
      } else {
        ir << "p(" << parameter << ")";
      }
      ir << " " << name << "arg : !qco.qubit\nqco.yield " << name
         << "body : !qco.qubit\n} : ({!qco.qubit}, {!qco.qubit})\n";
      wires[gate.control] = name + "c";
      wires[gate.target] = name + "t";
      continue;
    }
    ir << name << " = qco.";
    switch (gate.kind) {
    case GateKind::H:
      ir << "h";
      break;
    case GateKind::RX:
      ir << "rx";
      break;
    case GateKind::RY:
      ir << "ry";
      break;
    case GateKind::RZ:
      ir << "rz";
      break;
    case GateKind::X:
      ir << "x";
      break;
    case GateKind::Z:
      ir << "z";
      break;
    default:
      break;
    }
    if (parameterized) {
      ir << "(" << parameter << ")";
    }
    ir << " " << wires[gate.target] << " : !qco.qubit\n";
    wires[gate.target] = name;
  }
  if (!circuit.measurements.empty()) {
    ir << "%zero = arith.constant 0 : i9\n"
          "%c = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = \""
       << circuit.registerName
       << "\"} : !cbit.reg<9>\n"
          "cbit.write %zero, %c : i9, !cbit.reg<9>\n";
    for (size_t index = 0; index < circuit.measurements.size(); ++index) {
      const auto [wire, destination] = circuit.measurements[index];
      const auto name = "%m" + std::to_string(index);
      ir << name << "idx = arith.constant " << destination << " : index\n"
         << name << "q, " << name << "b = qco.measure " << wires[wire]
         << " : !qco.qubit\ncbit.store " << name << "b, %c[" << name
         << "idx] : !cbit.reg<9>\n";
      wires[wire] = name + "q";
    }
  }
  for (const auto& wire : wires) {
    ir << "qco.sink " << wire << " : !qco.qubit\n";
  }
  if (circuit.measurements.empty()) {
    ir << "return\n";
  } else {
    ir << "return %c : !cbit.reg<9>\n";
  }
  ir << "} }";
  return ir.str();
}

namespace {

class DenseVerification : public ::testing::Test {
protected:
  MLIRContext context;
  ScopedDiagnosticHandler diagnostics{&context,
                                      [](Diagnostic&) { return success(); }};
  DenseVerification() {
    context.loadDialect<arith::ArithDialect, func::FuncDialect, mqt::MQTDialect,
                        QCODialect, cbit::CBitDialect>();
  }
  VerificationResult check(const Circuit& left, const Circuit& right,
                           const VerificationOptions& options,
                           const VerificationContract& contract = {}) {
    auto a = parseSourceString<ModuleOp>(program(left), &context);
    auto b = parseSourceString<ModuleOp>(program(right), &context);
    EXPECT_TRUE(a && b);
    if (!a || !b) {
      return {.reason = "dense fixture failed parsing"};
    }
    auto result =
        qco::verify(a->lookupSymbol<func::FuncOp>("main"),
                    b->lookupSymbol<func::FuncOp>("main"), contract,
                    DDArgumentBindings(), DDArgumentBindings(), options);
    EXPECT_TRUE(succeeded(result));
    return succeeded(result)
               ? *result
               : VerificationResult{.reason = "invalid dense fixture"};
  }
};

TEST_F(DenseVerification, CoherentCorpusMatchesIndependentMatrixArithmetic) {
  const Circuit left{
      .qubits = 2,
      .gates =
          {
              {.kind = GateKind::H, .target = 0},
              {.kind = GateKind::RX, .target = 1, .control = 0, .angle = .37},
              {.kind = GateKind::CX, .target = 1, .control = 0},
              {.kind = GateKind::RZ, .target = 0, .control = 0, .angle = .19},
          },
  };
  std::vector<Circuit> corpus{left};
  auto wrongAngle = left;
  wrongAngle.gates[1].angle += .071;
  corpus.push_back(wrongAngle);
  auto wrongControl = left;
  wrongControl.gates[2] = {.kind = GateKind::CX, .target = 0, .control = 1};
  corpus.push_back(wrongControl);
  for (const auto gate : {
           Gate{.kind = GateKind::Z, .target = 1},
           Gate{
               .kind = GateKind::GlobalPhase,
               .target = 0,
               .control = 0,
               .angle = .23,
           },
           Gate{.kind = GateKind::Swap, .target = 0, .control = 1},
       }) {
    auto changed = left;
    changed.gates.push_back(gate);
    corpus.push_back(std::move(changed));
  }
  const auto reference = denseCircuit(left);
  for (size_t index = 0; index < corpus.size(); ++index) {
    SCOPED_TRACE(index);
    const auto candidate = denseCircuit(corpus[index]);
    for (const auto relation : {
             Relation::Exact,
             Relation::AmplitudeDistance,
             Relation::ProcessDistance,
         }) {
      const auto expected = relation == Relation::Exact
                                ? maxDifference(reference, candidate)
                            : relation == Relation::AmplitudeDistance
                                ? amplitudeDistance(reference, candidate, 4)
                                : processDistance(reference, candidate, 4);
      for (const auto method : {Method::Construction, Method::Alternating}) {
        if (relation == Relation::Exact && method == Method::Alternating) {
          continue;
        }
        for (const auto ratio : {.9, 1.1}) {
          VerificationOptions options;
          options.method = method;
          options.relation = relation;
          options.phase = relation == Relation::ProcessDistance
                              ? Phase::Ignore
                              : Phase::Preserve;
          options.tolerance = expected < 1e-13 ? 1e-10 : ratio * expected;
          const auto result = check(left, corpus[index], options);
          EXPECT_EQ(result.verdict, expected < options.tolerance
                                        ? Verdict::Equivalent
                                        : Verdict::NotEquivalent)
              << result.reason;
          ASSERT_TRUE(result.distance);
          EXPECT_NEAR(*result.distance, expected, 2e-12);
        }
      }
    }
  }
}

TEST_F(DenseVerification, ZeroAncillaColumnsDetermineNormalizationAndDomain) {
  const Circuit left{
      .qubits = 3,
      .gates =
          {
              {.kind = GateKind::H, .target = 0},
              {.kind = GateKind::RX, .target = 1, .control = 0, .angle = .37},
              {.kind = GateKind::CX, .target = 1, .control = 0},
              {.kind = GateKind::CX, .target = 2, .control = 1},
          },
  };
  auto right = left;
  right.gates.push_back(
      {.kind = GateKind::RZ, .target = 1, .control = 0, .angle = .13});
  VerificationContract contract;
  contract.zeroAncillas = {2};
  const auto a = denseCircuit(left, contract.zeroAncillas);
  const auto b = denseCircuit(right, contract.zeroAncillas);
  for (const auto relation :
       {Relation::AmplitudeDistance, Relation::ProcessDistance}) {
    const auto expected = relation == Relation::AmplitudeDistance
                              ? amplitudeDistance(a, b, 4)
                              : processDistance(a, b, 4);
    VerificationOptions options;
    options.method = Method::Construction;
    options.relation = relation;
    options.phase =
        relation == Relation::ProcessDistance ? Phase::Ignore : Phase::Preserve;
    options.tolerance = 1.1 * expected;
    const auto result = check(left, right, options, contract);
    EXPECT_EQ(result.verdict, Verdict::Equivalent) << result.reason;
    ASSERT_TRUE(result.distance);
    EXPECT_NEAR(*result.distance, expected, 2e-12);
    EXPECT_EQ(result.numFreeInputs, 2);
  }
  auto onInitializedOnly = left;
  onInitializedOnly.gates.insert(onInitializedOnly.gates.begin(),
                                 {.kind = GateKind::Z, .target = 2});
  EXPECT_DOUBLE_EQ(maxDifference(a, denseCircuit(onInitializedOnly, {2})), 0);
  EXPECT_GT(maxDifference(denseCircuit(left), denseCircuit(onInitializedOnly)),
            .5);
  VerificationOptions options;
  options.method = Method::Construction;
  EXPECT_EQ(check(left, onInitializedOnly, options, contract).verdict,
            Verdict::Equivalent);
  EXPECT_EQ(check(left, onInitializedOnly, options).verdict,
            Verdict::NotEquivalent);
}

TEST_F(DenseVerification, SelectedSparseEffectsIncludeAncillaAndGarbage) {
  const Circuit left{
      .qubits = 3,
      .gates =
          {
              {.kind = GateKind::H, .target = 0},
              {.kind = GateKind::RY, .target = 1, .control = 0, .angle = .27},
              {.kind = GateKind::CX, .target = 0, .control = 1},
              {.kind = GateKind::RZ, .target = 0, .control = 0, .angle = .13},
          },
      .measurements = {{0, 7}, {1, 2}},
  };
  auto garbage = left;
  garbage.gates.push_back({.kind = GateKind::CX, .target = 2, .control = 0});
  garbage.gates.push_back(
      {.kind = GateKind::RY, .target = 2, .control = 0, .angle = .43});
  auto faulty = garbage;
  faulty.gates.push_back({.kind = GateKind::X, .target = 0});
  VerificationContract contract;
  contract.zeroAncillas = {2};
  contract.observedOutputs = {"result0:c[7]"};
  const auto reference = denseCircuit(left, {2});
  for (const auto& candidate : {garbage, faulty}) {
    double expected = 0;
    const auto unitary = denseCircuit(candidate, {2});
    for (const bool outcome : {false, true}) {
      expected = std::max(expected, maxDifference(effect(reference, 0, outcome),
                                                  effect(unitary, 0, outcome)));
    }
    VerificationOptions options;
    options.method = Method::Construction;
    options.relation = Relation::TerminalProbabilities;
    options.limits.maxOutcomes = 2;
    const auto result = check(left, candidate, options, contract);
    EXPECT_EQ(result.verdict,
              expected < 1e-12 ? Verdict::Equivalent : Verdict::NotEquivalent)
        << result.reason;
    ASSERT_TRUE(result.distance);
    EXPECT_NEAR(*result.distance, expected, 2e-12);
    EXPECT_EQ(result.outputLabels, contract.observedOutputs);
  }
  auto mislabeled = garbage;
  mislabeled.registerName = "other";
  VerificationOptions options;
  options.method = Method::Construction;
  options.relation = Relation::TerminalProbabilities;
  EXPECT_EQ(check(left, mislabeled, options, contract).verdict,
            Verdict::Unsupported);
  auto moved = garbage;
  moved.measurements[0].second = 8;
  EXPECT_EQ(check(left, moved, options, contract).verdict,
            Verdict::NotEquivalent);
}

TEST_F(DenseVerification, EqualEffectsDoNotClaimEqualReducedQuantumStates) {
  const Circuit left{
      .qubits = 2,
      .gates = {{.kind = GateKind::H, .target = 0}},
      .measurements = {{0, 7}},
  };
  auto right = left;
  right.gates.push_back({.kind = GateKind::CX, .target = 1, .control = 0});
  const auto a = denseCircuit(left, {1});
  const auto b = denseCircuit(right, {1});
  EXPECT_NEAR(maxDifference(effect(a, 0, false), effect(b, 0, false)), 0,
              1e-15);
  const auto reduced = [](const Matrix& unitary) {
    Matrix state(2);
    for (size_t row = 0; row < 2; ++row) {
      for (size_t column = 0; column < 2; ++column) {
        for (size_t garbage = 0; garbage < 2; ++garbage) {
          state.at(row, column) +=
              unitary.at(row | (garbage << 1U), 0) *
              std::conj(unitary.at(column | (garbage << 1U), 0));
        }
      }
    }
    return state;
  };
  EXPECT_NEAR(maxDifference(reduced(a), reduced(b)), .5, 1e-15);
  VerificationContract contract;
  contract.zeroAncillas = {1};
  contract.observedOutputs = {"result0:c[7]"};
  VerificationOptions options;
  options.method = Method::Construction;
  options.relation = Relation::TerminalProbabilities;
  EXPECT_EQ(check(left, right, options, contract).verdict, Verdict::Equivalent);
}

TEST_F(DenseVerification, OverlapPhaseDoesNotRefuteTheMinimaxEntryRelation) {
  const Circuit left{.qubits = 2};
  const Circuit right{
      .qubits = 2,
      .gates =
          {
              {
                  .kind = GateKind::CPhase,
                  .target = 1,
                  .control = 0,
                  .angle = 1e-4,
              },
          },
  };
  const auto a = denseCircuit(left);
  const auto b = denseCircuit(right);
  const auto tolerance = 6e-5;
  EXPECT_LT(maxDifference(a, b, std::polar(1., -5e-5)), tolerance);
  Scalar overlap = 0;
  for (size_t i = 0; i < 4; ++i) {
    overlap += b.at(i, i);
  }
  EXPECT_GT(maxDifference(a, b, std::conj(overlap) / std::abs(overlap)),
            tolerance);
  VerificationOptions options;
  options.method = Method::Construction;
  options.phase = Phase::Ignore;
  options.tolerance = tolerance;
  const auto result = check(left, right, options);
  EXPECT_EQ(result.verdict, Verdict::Inconclusive) << result.reason;
}

TEST_F(DenseVerification,
       PowerCannotAmplifyAcceptedMatrixRoundingIntoNonunitarity) {
  // The literal passes the existing 1e-10 entrywise Gram-matrix verifier.
  // Raising it to a large power amplifies its radial error, not its phase.
  // p=1e10 already invalidates process-distance algebra while its squared
  // residual remains below 2; p=1e12 exceeds that range, and p=1e14 overflows.
  auto identityProgram =
      parseSourceString<ModuleOp>(program(Circuit{.qubits = 1}), &context);
  ASSERT_TRUE(identityProgram);
  for (const auto exponent : {1e10, 1e12, 1e14}) {
    SCOPED_TRACE(exponent);
    std::ostringstream ir;
    ir << std::scientific << std::setprecision(17);
    ir << "module { func.func @main() { %p = arith.constant " << exponent
       << R"mlir( : f64
      %q = qco.static 0 : !qco.qubit
      %out = qco.pow(%p) (%a = %q) {
        %r = qco.unitary dense<[
          [(1.00000000004, 0.0), (0.0, 0.0)],
          [(0.0, 0.0), (1.00000000004, 0.0)]
        ]> : tensor<2x2xcomplex<f64>> %a : !qco.qubit
        qco.yield %r : !qco.qubit
      } : {!qco.qubit} -> {!qco.qubit}
      qco.sink %out : !qco.qubit
      return
    } })mlir";
    auto powered = parseSourceString<ModuleOp>(ir.str(), &context);
    ASSERT_TRUE(powered);
    const auto entry = powered->lookupSymbol<func::FuncOp>("main");
    EXPECT_EQ(resolveExecution(entry).status, DDTraceStatus::Unsupported);
    for (const auto relation : {Relation::Exact, Relation::ProcessDistance}) {
      VerificationOptions options;
      options.relation = relation;
      options.phase = relation == Relation::ProcessDistance ? Phase::Ignore
                                                            : Phase::Preserve;
      options.tolerance = .5;
      for (const auto method :
           {Method::Construction, Method::Alternating, Method::Auto}) {
        options.method = method;
        const auto result = qco::verify(
            identityProgram->lookupSymbol<func::FuncOp>("main"), entry, {},
            DDArgumentBindings(), DDArgumentBindings(), options);
        ASSERT_TRUE(succeeded(result));
        EXPECT_EQ(result->verdict, Verdict::Unsupported) << result->reason;
      }
    }
  }
}

} // namespace
