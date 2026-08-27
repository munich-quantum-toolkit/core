/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"
#include "qasm_programs.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace mqt::test::qasm3_translation {
using namespace mlir;

namespace {

struct QASM3TranslationTestCase {
  std::string name;
  std::string source;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QASM3TranslationTestCase& test);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QASM3TranslationTestCase& test) {
  return os << "QASM3Translation{" << test.name << ", reference="
            << ::mqt::test::displayName(test.referenceBuilder.name) << "}";
}

class QASM3TranslationTest
    : public testing::TestWithParam<QASM3TranslationTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect,
                    math::MathDialect, memref::MemRefDialect,
                    mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                    qtensor::QTensorDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static Value measureToRegister(qc::QCProgramBuilder& b, ValueRange qubits) {
  auto c = b.allocClassicalBitRegister(static_cast<int64_t>(qubits.size()));
  for (auto [i, q] : llvm::enumerate(qubits)) {
    b.measure(q, c, static_cast<int64_t>(i));
  }
  return c;
}

static Value loadBit(qc::QCProgramBuilder& b, Value reg,
                     const int64_t index = 0) {
  return b.loadClassicalBit(reg, index);
}

static SmallVector<Value> allocMultipleQubitRegisters(qc::QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  auto c0 = measureToRegister(b, {q0[0], q0[1]});
  auto c1 = measureToRegister(b, {q1[0], q1[1], q1[2]});
  return {c0, c1};
}

static SmallVector<Value> twoX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.x(q[0]);
  b.x(q[1]);
  return {measureToRegister(b, {q[0], q[1]})};
}

static void legacyU2(qc::QCProgramBuilder& b, Value target) {
  b.gphase(-0.5 * (0.234 + 0.567));
  b.u2(0.234, 0.567, target);
}

static Value legacyU2(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  legacyU2(b, q[0]);
  return measureToRegister(b, {q[0]});
}

static SmallVector<Value> legacySingleControlledU2(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](Value target) { legacyU2(b, target); });
  return {measureToRegister(b, {q[0], q[1]})};
}

static SmallVector<Value> legacyMultipleControlledU2(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(ValueRange{q[0], q[1]}, q[2],
         [&](Value target) { legacyU2(b, target); });
  return {measureToRegister(b, {q[0], q[1], q[2]})};
}

static void legacyU(qc::QCProgramBuilder& b, Value target) {
  b.gphase(-0.5 * (0.2 + 0.3));
  b.u(0.1, 0.2, 0.3, target);
}

static Value legacyU(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  legacyU(b, q[0]);
  return measureToRegister(b, {q[0]});
}

static SmallVector<Value> legacySingleControlledU(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](Value target) { legacyU(b, target); });
  return {measureToRegister(b, {q[0], q[1]})};
}

static SmallVector<Value> legacyMultipleControlledU(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(ValueRange{q[0], q[1]}, q[2],
         [&](Value target) { legacyU(b, target); });
  return {measureToRegister(b, {q[0], q[1], q[2]})};
}

static Value legacyTripleControlledSx(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(ValueRange{q[0], q[1], q[2]}, q[3],
         [&](Value target) { b.sx(target); });
  return measureToRegister(b, q.qubits);
}

using Complex = std::complex<double>;

namespace {

template <size_t Dimension> struct TestMatrix {
  std::array<Complex, Dimension * Dimension> data{};

  template <typename... Elements>
  [[nodiscard]] static TestMatrix fromElements(Elements... elements) {
    static_assert(sizeof...(elements) == Dimension * Dimension);
    return {{Complex{elements}...}};
  }

  [[nodiscard]] static TestMatrix identity() {
    TestMatrix result;
    for (size_t diagonal = 0; diagonal < Dimension; ++diagonal) {
      result(diagonal, diagonal) = 1.0;
    }
    return result;
  }

  [[nodiscard]] Complex& operator()(const size_t row, const size_t column) {
    return data[(row * Dimension) + column];
  }

  [[nodiscard]] Complex operator()(const size_t row,
                                   const size_t column) const {
    return data[(row * Dimension) + column];
  }

  [[nodiscard]] TestMatrix operator*(const TestMatrix& rhs) const {
    TestMatrix result;
    for (size_t row = 0; row < Dimension; ++row) {
      for (size_t column = 0; column < Dimension; ++column) {
        for (size_t inner = 0; inner < Dimension; ++inner) {
          result(row, column) += (*this)(row, inner) * rhs(inner, column);
        }
      }
    }
    return result;
  }

  [[nodiscard]] TestMatrix operator*(const Complex scalar) const {
    auto result = *this;
    result *= scalar;
    return result;
  }

  TestMatrix& operator*=(const Complex scalar) {
    for (auto& element : data) {
      element *= scalar;
    }
    return *this;
  }

  [[nodiscard]] TestMatrix adjoint() const {
    TestMatrix result;
    for (size_t row = 0; row < Dimension; ++row) {
      for (size_t column = 0; column < Dimension; ++column) {
        result(row, column) = std::conj(data[(column * Dimension) + row]);
      }
    }
    return result;
  }

  [[nodiscard]] bool isApprox(const TestMatrix& other,
                              const double tolerance) const {
    return std::ranges::equal(data, other.data,
                              [=](const Complex lhs, const Complex rhs) {
                                return std::abs(lhs - rhs) <= tolerance;
                              });
  }
};

using Matrix2 = TestMatrix<2>;
using Matrix4 = TestMatrix<4>;

} // namespace

[[nodiscard]] static Matrix4 embedInTwoQubit(const Matrix2& gate,
                                             const size_t qubit) {
  EXPECT_LT(qubit, 2U);
  if (qubit == 0) {
    return Matrix4::fromElements(
        gate(0, 0), 0.0, gate(0, 1), 0.0, 0.0, gate(0, 0), 0.0, gate(0, 1),
        gate(1, 0), 0.0, gate(1, 1), 0.0, 0.0, gate(1, 0), 0.0, gate(1, 1));
  }
  return Matrix4::fromElements(gate(0, 0), gate(0, 1), 0.0, 0.0, gate(1, 0),
                               gate(1, 1), 0.0, 0.0, 0.0, 0.0, gate(0, 0),
                               gate(0, 1), 0.0, 0.0, gate(1, 0), gate(1, 1));
}

[[nodiscard]] static Matrix2
openQASM3UMatrix(const double theta, const double phi, const double lambda) {
  using namespace std::complex_literals;
  const auto thetaPhase = std::exp(1i * theta);
  const auto common = 0.5 * (1.0 + thetaPhase);
  const auto difference = Complex{0.0, 0.5} * (1.0 - thetaPhase);
  return Matrix2::fromElements(common, -std::exp(1i * lambda) * difference,
                               std::exp(1i * phi) * difference,
                               std::exp(1i * (phi + lambda)) * common);
}

[[nodiscard]] static Matrix2
conventionalUMatrix(const double theta, const double phi, const double lambda) {
  using namespace std::complex_literals;
  const auto cosine = std::cos(theta / 2.0);
  const auto sine = std::sin(theta / 2.0);
  return Matrix2::fromElements(cosine, -std::exp(1i * lambda) * sine,
                               std::exp(1i * phi) * sine,
                               std::exp(1i * (phi + lambda)) * cosine);
}

[[nodiscard]] static Matrix2
openQASM2UMatrix(const double theta, const double phi, const double lambda) {
  using namespace std::complex_literals;
  return conventionalUMatrix(theta, phi, lambda) *
         std::exp(Complex{0.0, -0.5} * (phi + lambda));
}

[[nodiscard]] static Matrix4 controlledMatrix(const Matrix2& body,
                                              const bool negative = false) {
  if (negative) {
    return Matrix4::fromElements(body(0, 0), body(0, 1), 0.0, 0.0, body(1, 0),
                                 body(1, 1), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0);
  }
  return Matrix4::fromElements(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                               body(0, 0), body(0, 1), 0.0, 0.0, body(1, 0),
                               body(1, 1));
}

[[nodiscard]] static std::optional<double> evaluateScalar(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (const auto floatValue = dyn_cast<FloatAttr>(constant.getValue())) {
      return floatValue.getValueAsDouble();
    }
    if (const auto integerValue = dyn_cast<IntegerAttr>(constant.getValue())) {
      return static_cast<double>(integerValue.getInt());
    }
  }
  if (auto add = value.getDefiningOp<arith::AddFOp>()) {
    const auto lhs = evaluateScalar(add.getLhs());
    const auto rhs = evaluateScalar(add.getRhs());
    if (lhs && rhs) {
      return *lhs + *rhs;
    }
  }
  if (auto multiply = value.getDefiningOp<arith::MulFOp>()) {
    const auto lhs = evaluateScalar(multiply.getLhs());
    const auto rhs = evaluateScalar(multiply.getRhs());
    if (lhs && rhs) {
      return *lhs * *rhs;
    }
  }
  return std::nullopt;
}

[[nodiscard]] static Matrix2 integerPower(Matrix2 base, int64_t exponent) {
  if (exponent < 0) {
    base = base.adjoint();
    exponent = -exponent;
  }
  auto result = Matrix2::identity();
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      result = base * result;
    }
    base = base * base;
    exponent >>= 1;
  }
  return result;
}

[[nodiscard]] static std::optional<Matrix2>
evaluateOneQubitRegion(Region& region);

[[nodiscard]] static std::optional<Matrix2>
evaluateOneQubitOperation(Operation* operation) {
  if (auto gphase = dyn_cast<qc::GPhaseOp>(operation)) {
    const auto theta = evaluateScalar(gphase.getTheta());
    if (theta) {
      return Matrix2::identity() * std::exp(std::complex<double>{0.0, *theta});
    }
  } else if (isa<qc::XOp>(operation)) {
    return Matrix2::fromElements(0.0, 1.0, 1.0, 0.0);
  } else if (auto phase = dyn_cast<qc::POp>(operation)) {
    const auto theta = evaluateScalar(phase.getTheta());
    if (theta) {
      return Matrix2::fromElements(1.0, 0.0, 0.0,
                                   std::exp(std::complex<double>{0.0, *theta}));
    }
  } else if (auto u = dyn_cast<qc::UOp>(operation)) {
    const auto theta = evaluateScalar(u.getTheta());
    const auto phi = evaluateScalar(u.getPhi());
    const auto lambda = evaluateScalar(u.getLambda());
    if (theta && phi && lambda) {
      return conventionalUMatrix(*theta, *phi, *lambda);
    }
  } else if (auto u2 = dyn_cast<qc::U2Op>(operation)) {
    const auto phi = evaluateScalar(u2.getPhi());
    const auto lambda = evaluateScalar(u2.getLambda());
    if (phi && lambda) {
      return conventionalUMatrix(std::numbers::pi / 2.0, *phi, *lambda);
    }
  } else if (auto inverse = dyn_cast<qc::InvOp>(operation)) {
    if (const auto body = evaluateOneQubitRegion(inverse.getRegion())) {
      return body->adjoint();
    }
  } else if (auto power = dyn_cast<qc::PowOp>(operation)) {
    const auto exponent = evaluateScalar(power.getExponent());
    const auto body = evaluateOneQubitRegion(power.getRegion());
    if (exponent && body && std::trunc(*exponent) == *exponent) {
      return integerPower(*body, static_cast<int64_t>(*exponent));
    }
  }
  return std::nullopt;
}

[[nodiscard]] static std::optional<Matrix2>
evaluateOneQubitRegion(Region& region) {
  auto result = Matrix2::identity();
  for (Operation& operation : region.front()) {
    if (!isa<qc::UnitaryOpInterface>(&operation)) {
      continue;
    }
    const auto matrix = evaluateOneQubitOperation(&operation);
    if (!matrix) {
      return std::nullopt;
    }
    result = *matrix * result;
  }
  return result;
}

[[nodiscard]] static std::optional<size_t> topLevelQubitIndex(Value qubit) {
  auto load = qubit.getDefiningOp<memref::LoadOp>();
  if (!load || load.getIndices().size() != 1) {
    return std::nullopt;
  }
  auto index = load.getIndices().front().getDefiningOp<arith::ConstantOp>();
  if (!index) {
    return std::nullopt;
  }
  const auto value = dyn_cast<IntegerAttr>(index.getValue());
  if (!value || value.getInt() < 0) {
    return std::nullopt;
  }
  return static_cast<size_t>(value.getInt());
}

[[nodiscard]] static Matrix2
translatedOneQubitUnitary(const llvm::StringRef source) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_TRUE(moduleOp);
  if (!moduleOp) {
    return Matrix2::identity();
  }
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  const auto matrix = evaluateOneQubitRegion(function.getBody());
  EXPECT_TRUE(matrix);
  return matrix.value_or(Matrix2::identity());
}

[[nodiscard]] static Matrix4
translatedTwoQubitUnitary(const llvm::StringRef source) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_TRUE(moduleOp);
  if (!moduleOp) {
    return Matrix4::identity();
  }
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto result = Matrix4::identity();
  for (Operation& operation : function.getBody().front()) {
    if (auto gphase = dyn_cast<qc::GPhaseOp>(&operation)) {
      const auto theta = evaluateScalar(gphase.getTheta());
      EXPECT_TRUE(theta);
      if (!theta) {
        return Matrix4::identity();
      }
      result *= std::exp(std::complex<double>{0.0, *theta});
      continue;
    }
    if (auto control = dyn_cast<qc::CtrlOp>(&operation)) {
      const auto controlIndex = topLevelQubitIndex(control.getControl(0));
      const auto targetIndex = topLevelQubitIndex(control.getTarget(0));
      const auto body = evaluateOneQubitRegion(control.getRegion());
      EXPECT_EQ(controlIndex, 0U);
      EXPECT_EQ(targetIndex, 1U);
      EXPECT_TRUE(body);
      if (!controlIndex || !targetIndex || !body) {
        return Matrix4::identity();
      }
      result = controlledMatrix(*body) * result;
      continue;
    }
    if (!isa<qc::UnitaryOpInterface>(&operation)) {
      continue;
    }
    auto unitary = dyn_cast<qc::UnitaryOpInterface>(&operation);
    const auto matrix = evaluateOneQubitOperation(&operation);
    const auto targetIndex = topLevelQubitIndex(unitary.getTarget(0));
    EXPECT_TRUE(matrix);
    EXPECT_TRUE(targetIndex);
    if (!matrix || !targetIndex) {
      return Matrix4::identity();
    }
    result = embedInTwoQubit(*matrix, *targetIndex) * result;
  }
  return result;
}

TEST(QASM3TranslationMatrixTest, PreservesOpenQASMGatePhaseConventions) {
  constexpr double theta = 0.37;
  constexpr double phi = -0.29;
  constexpr double lambda = 0.83;
  constexpr double gamma = -0.41;
  const auto qasm3U = openQASM3UMatrix(theta, phi, lambda);
  const auto qasm2U = openQASM2UMatrix(theta, phi, lambda);

  struct OneQubitCase {
    llvm::StringRef source;
    Matrix2 expected;
  };
  const auto oneQubitCases = std::to_array<OneQubitCase>({
      {.source = "OPENQASM 3.1; qubit q; U(0.37, -0.29, 0.83) q;",
       .expected = qasm3U},
      {.source = "OPENQASM 2.0; qreg q[1]; U(0.37, -0.29, 0.83) q[0];",
       .expected = qasm2U},
      {.source = "OPENQASM 3.1; include \"stdgates.inc\"; qubit q; "
                 "u2(-0.29, 0.83) q;",
       .expected = openQASM2UMatrix(std::numbers::pi / 2.0, phi, lambda)},
      {.source = "OPENQASM 3.1; include \"stdgates.inc\"; qubit q; "
                 "u3(0.37, -0.29, 0.83) q;",
       .expected = qasm2U},
      {.source = "OPENQASM 3.1; qubit q; u(0.37, -0.29, 0.83) q;",
       .expected = qasm2U},
      {.source = "OPENQASM 3.1; qubit q; inv @ U(0.37, -0.29, 0.83) q;",
       .expected = qasm3U.adjoint()},
      {.source = "OPENQASM 3.1; qubit q; pow(2) @ U(0.37, -0.29, 0.83) q;",
       .expected = qasm3U * qasm3U},
  });
  for (const auto& test : oneQubitCases) {
    SCOPED_TRACE(test.source.str());
    EXPECT_TRUE(
        translatedOneQubitUnitary(test.source).isApprox(test.expected, 1e-10));
  }

  const auto controlledQASM3 = controlledMatrix(qasm3U);
  const auto controlledQASM2 = controlledMatrix(qasm2U);
  const auto phasedQASM3 = qasm3U * std::exp(std::complex<double>{0.0, gamma});
  struct TwoQubitCase {
    llvm::StringRef source;
    Matrix4 expected;
  };
  const auto twoQubitCases = std::to_array<TwoQubitCase>({
      {.source = "OPENQASM 3.1; qubit[2] q; "
                 "ctrl @ U(0.37, -0.29, 0.83) q[0], q[1];",
       .expected = controlledQASM3},
      {.source = "OPENQASM 3.1; qubit[2] q; "
                 "negctrl @ U(0.37, -0.29, 0.83) q[0], q[1];",
       .expected = controlledMatrix(qasm3U, true)},
      {.source = "OPENQASM 3.1; include \"stdgates.inc\"; qubit[2] q; "
                 "cu(0.37, -0.29, 0.83, -0.41) q[0], q[1];",
       .expected = controlledMatrix(phasedQASM3)},
      {.source = "OPENQASM 3.1; include \"qelib1.inc\"; qubit[2] q; "
                 "cu3(0.37, -0.29, 0.83) q[0], q[1];",
       .expected = controlledQASM2},
  });
  for (const auto& test : twoQubitCases) {
    SCOPED_TRACE(test.source.str());
    EXPECT_TRUE(
        translatedTwoQubitUnitary(test.source).isApprox(test.expected, 1e-10));
  }
}

static SmallVector<Value> singleNegControlledX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.x(q[0]);
  b.cx(q[0], q[1]);
  b.x(q[0]);
  return {measureToRegister(b, {q[0], q[1]})};
}

static SmallVector<Value> tripleControlledX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcx({q[0], q[1], q[2]}, q[3]);
  return {measureToRegister(b, {q[0], q[1], q[2], q[3]})};
}

static SmallVector<Value> mixedControlledX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.x(q[1]);
  b.mcx({q[0], q[1]}, q[2]);
  b.x(q[1]);
  return {measureToRegister(b, {q[0], q[1], q[2]})};
}

static SmallVector<Value> twoMixedControlledX(qc::QCProgramBuilder& b) {
  auto q1 = b.allocQubitRegister(2);
  auto q2 = b.allocQubitRegister(2);
  auto q3 = b.allocQubitRegister(2);
  b.x(q2[0]);
  b.mcx({q1[0], q2[0]}, q3[0]);
  b.x(q2[0]);
  b.x(q2[1]);
  b.mcx({q1[1], q2[1]}, q3[1]);
  b.x(q2[1]);
  auto c1 = measureToRegister(b, {q1[0], q1[1]});
  auto c2 = measureToRegister(b, {q2[0], q2[1]});
  auto c3 = measureToRegister(b, {q3[0], q3[1]});
  return {c1, c2, c3};
}

static Value ifNot(qc::QCProgramBuilder& b) {
  // Only `out` is declared `output` in the QASM source, so the non-output
  // condition bit `c` is not returned.
  auto trueValue = b.boolConstant(true);
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto condition = b.allocClassicalBitRegister(1);
  b.measure(q[0], condition, 0);
  auto cond =
      arith::XOrIOp::create(b, b.loadClassicalBit(condition, 0), trueValue)
          .getResult();
  b.scfIf(cond, [&] { b.x(q[0]); });
  auto out = b.allocClassicalBitRegister(1);
  b.measure(q[0], out, 0);
  return out;
}

static SmallVector<Value> ifWithMeasurement(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(1);
  auto measurement = b.allocClassicalBitRegister(1);
  b.h(q[0]);
  b.measure(q[0], c, 0);
  b.scfIf(
      c, 0, [&] { b.measure(q[0], measurement, 0); },
      [&] { b.measure(q[0], measurement, 0); });
  return {c, measurement};
}

template <typename ThenBuilder, typename ElseBuilder>
static Value buildShortCircuitCondition(qc::QCProgramBuilder& b, Value lhs,
                                        ThenBuilder&& thenBuilder,
                                        ElseBuilder&& elseBuilder) {
  auto ifOp = scf::IfOp::create(b, b.getI1Type(), lhs, true);
  OpBuilder::InsertionGuard guard(b);
  auto& thenBlock = ifOp.getThenRegion().front();
  if (!thenBlock.empty()) {
    thenBlock.back().erase();
  }
  b.setInsertionPointToEnd(&thenBlock);
  scf::YieldOp::create(b, std::forward<ThenBuilder>(thenBuilder)());
  auto& elseBlock = ifOp.getElseRegion().front();
  if (!elseBlock.empty()) {
    elseBlock.back().erase();
  }
  b.setInsertionPointToEnd(&elseBlock);
  scf::YieldOp::create(b, std::forward<ElseBuilder>(elseBuilder)());
  return ifOp.getResult(0);
}

static Value shortCircuitAnd(qc::QCProgramBuilder& b, Value lhs,
                             const function_ref<Value()>& rhs) {
  return buildShortCircuitCondition(b, lhs, rhs,
                                    [&] { return b.boolConstant(false); });
}

static Value shortCircuitOr(qc::QCProgramBuilder& b, Value lhs,
                            const function_ref<Value()>& rhs) {
  return buildShortCircuitCondition(
      b, lhs, [&] { return b.boolConstant(true); }, rhs);
}

static Value powTwoX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.pow(2.0, q, [&](Value qubit) { b.x(qubit); });
  return measureToRegister(b, {q});
}

static Value powZeroX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.pow(0.0, q, [&](Value qubit) { b.x(qubit); });
  return measureToRegister(b, {q});
}

static Value negativePowS(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.pow(2.0, q, [&](Value powQubit) {
    b.inv(powQubit, [&](Value invQubit) { b.s(invQubit); });
  });
  return measureToRegister(b, {q});
}

static SmallVector<Value> controlledInversePowS(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](Value target) {
    b.pow(2.0, target, [&](Value powQubit) {
      b.inv(powQubit, [&](Value invQubit) { b.s(invQubit); });
    });
  });
  return {measureToRegister(b, {q[0], q[1]})};
}

static Value nestedPowX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.pow(6.0, q, [&](Value qubit) { b.x(qubit); });
  return measureToRegister(b, {q});
}

static Value customPowHS(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.pow(2.0, q, [&](Value qubit) {
    b.h(qubit);
    b.s(qubit);
  });
  return measureToRegister(b, {q});
}

static SmallVector<Value> broadcastPowX(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  for (auto qubit : q.qubits) {
    b.pow(2.0, qubit, [&](Value argument) { b.x(argument); });
  }
  return {measureToRegister(b, {q[0], q[1]})};
}

static SmallVector<Value> broadcastRegisterAndQubit(qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q = b.allocQubit();
  b.cx(reg[0], q);
  b.cx(reg[1], q);
  b.cx(reg[2], q);
  auto left = measureToRegister(b, {reg[0], reg[1], reg[2]});
  auto right = measureToRegister(b, {q});
  return {left, right};
}

static SmallVector<Value> broadcastCompoundGate(qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q = b.allocQubit();
  for (auto qubit : reg.qubits) {
    b.x(qubit);
    b.cx(qubit, q);
  }
  auto left = measureToRegister(b, {reg[0], reg[1], reg[2]});
  auto right = measureToRegister(b, {q});
  return {left, right};
}

static Value expressionArithmetic(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.rx((((1.0 + 2.0) * 3.0) / 2.0) - 0.5, q);
  return measureToRegister(b, {q});
}

static Value expressionUnaryMinus(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.rx(-0.5, q);
  b.ry(-(1.0 + 2.0), q);
  b.rz(-(-0.25), q);
  return measureToRegister(b, {q});
}

static Value expressionBuiltinConstants(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.rx(std::numbers::pi / 2.0, q);
  b.ry((2.0 * std::numbers::pi) / 4.0, q);
  b.rz(std::numbers::e, q);
  return measureToRegister(b, {q});
}

static Value expressionMathFunctions(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.rx(std::acos(0.5), q);
  b.rx(std::asin(0.5), q);
  b.rx(std::atan(0.5), q);
  b.rx(std::cos(0.5), q);
  b.rx(std::exp(0.5), q);
  b.rx(std::numbers::ln2, q);
  b.rx(std::fmod(5.5, 2.0), q);
  b.rx(std::pow(2.0, 3.0), q);
  b.rx(std::sin(0.5), q);
  b.rx(std::numbers::sqrt2, q);
  b.rx(std::tan(0.5), q);
  return measureToRegister(b, {q});
}

static Value expressionNestedMathFunctions(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.rx(std::sqrt(std::pow(std::sin(0.5), 2.0) + std::pow(std::cos(0.5), 2.0)),
       q);
  return measureToRegister(b, {q});
}

static Value expressionConstFloat(qc::QCProgramBuilder& b) {
  constexpr double theta = std::numbers::pi / 4.0;
  auto q = b.allocQubit();
  b.h(q);
  b.rx(theta, q);
  b.ry(theta * 2.0, q);
  return measureToRegister(b, {q});
}

static SmallVector<Value> expressionMutableFloat(qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.rx(0.5, q);
  b.ry(0.75, q);
  auto theta =
      arith::ConstantOp::create(b, b.getF64FloatAttr(0.75)).getResult();
  return {theta, measureToRegister(b, {q})};
}

static SmallVector<Value>
expressionConstIntArithmetic(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(8);
  b.h(q[3]);
  b.h(q[5]);
  b.rx(8.0, q[3]);
  return {measureToRegister(b, {q[3], q[5]})};
}

static SmallVector<Value> conditionLiteral(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.scfIf(true, [&] { b.x(q[0]); });
  b.scfIf(false, [&] { b.x(q[1]); });
  return {measureToRegister(b, {q[0], q[1]})};
}

static SmallVector<Value> conditionMeasurement(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto enabled = b.allocClassicalBitRegister(1);
  b.measure(q[0], enabled, 0);
  b.scfIf(loadBit(b, enabled), [&] { b.x(q[1]); });
  auto c = measureToRegister(b, {q[1]});
  return {enabled, c};
}

static SmallVector<Value> conditionAnd(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0Reg = measureToRegister(b, {q[0]});
  auto c1Reg = measureToRegister(b, {q[1]});
  auto c0 = loadBit(b, c0Reg);
  auto condition = shortCircuitAnd(b, c0, [&] { return loadBit(b, c1Reg); });
  b.scfIf(condition, [&] { b.x(q[2]); });
  auto out = measureToRegister(b, {q[2]});
  return {c0Reg, c1Reg, out};
}

static SmallVector<Value> conditionOr(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0Reg = measureToRegister(b, {q[0]});
  auto c1Reg = measureToRegister(b, {q[1]});
  auto c0 = loadBit(b, c0Reg);
  auto condition = shortCircuitOr(b, c0, [&] { return loadBit(b, c1Reg); });
  b.scfIf(condition, [&] { b.x(q[2]); }, [&] { b.h(q[2]); });
  auto out = measureToRegister(b, {q[2]});
  return {c0Reg, c1Reg, out};
}

static SmallVector<Value> conditionNotAndOr(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.h(q[0]);
  b.h(q[1]);
  b.h(q[2]);
  auto c0Reg = measureToRegister(b, {q[0]});
  auto c1Reg = measureToRegister(b, {q[1]});
  auto c2Reg = measureToRegister(b, {q[2]});
  auto c0 = loadBit(b, c0Reg);
  auto both = shortCircuitAnd(b, c0, [&] { return loadBit(b, c1Reg); });
  auto notBoth = arith::XOrIOp::create(b, both, b.boolConstant(true));
  auto condition =
      shortCircuitOr(b, notBoth, [&] { return loadBit(b, c2Reg); });
  b.scfIf(condition, [&] { b.x(q[3]); });
  auto out = measureToRegister(b, {q[3]});
  return {c0Reg, c1Reg, c2Reg, out};
}

static SmallVector<Value> conditionBoolVariable(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c0Reg = measureToRegister(b, {q[0]});
  auto c1Reg = measureToRegister(b, {q[1]});
  auto c0 = loadBit(b, c0Reg);
  auto both = shortCircuitAnd(b, c0, [&] { return loadBit(b, c1Reg); });
  auto neither = arith::XOrIOp::create(b, both, b.boolConstant(true));
  b.scfIf(neither, [&] { b.x(q[2]); });
  auto out = measureToRegister(b, {q[2]});
  return {c0Reg, c1Reg, both, neither, out};
}

static SmallVector<Value> conditionIndexedBit(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.h(q[0]);
  b.h(q[1]);
  auto c = b.allocClassicalBitRegister(2);
  b.measure(q[0], c, 0);
  b.measure(q[1], c, 1);
  b.scfIf(loadBit(b, c, 1), [&] { b.x(q[2]); });
  auto out = measureToRegister(b, {q[2]});
  return {c, out};
}

static LogicalResult convertQCToQCO(ModuleOp moduleOp) {
  PassManager manager(moduleOp.getContext());
  manager.addPass(createQCToQCO());
  return manager.run(moduleOp);
}

TEST_P(QASM3TranslationTest, ProgramEquivalence) {
  const auto name = " (" + GetParam().name + ")";
  const auto& source = GetParam().source;
  const auto referenceBuilder = GetParam().referenceBuilder;
  ::mqt::test::DeferredPrinter printer;

  auto translated = qc::translateQASM3ToQC(source, context.get());
  ASSERT_TRUE(translated);
  printer.record(translated.get(), "Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(translated.get()).succeeded());
  printer.record(translated.get(), "Canonicalized Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  const auto initialization = StringRef(source).contains("OPENQASM 2")
                                  ? cbit::Initialization::Zero
                                  : cbit::Initialization::Undefined;
  auto reference =
      ::mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  reference->walk(
      [&](cbit::AllocOp op) { op.setInitialization(initialization); });
  printer.record(reference.get(), "Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  ASSERT_TRUE(succeeded(convertQCToQCO(translated.get())));
  ASSERT_TRUE(succeeded(convertQCToQCO(reference.get())));
  ASSERT_TRUE(runQCOCleanupPipeline(translated.get()).succeeded());
  ASSERT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  printer.record(translated.get(), "Lowered Translated QCO IR" + name);
  printer.record(reference.get(), "Lowered Reference QCO IR" + name);
  ASSERT_TRUE(verify(*translated).succeeded());
  ASSERT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(translated.get(), reference.get()));
}

TEST(QASM3TranslationErrors, AcceptsFloatingAndRejectsBooleanPowerExponent) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto translated = qc::translateQASM3ToQC(qasm::floatingPowX, &context);
  ASSERT_TRUE(translated);
  SmallVector<qc::PowOp> powers;
  translated->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 1U);
  ASSERT_TRUE(powers.front().getExponentValue().has_value());
  EXPECT_DOUBLE_EQ(*powers.front().getExponentValue(), 0.5);

  EXPECT_FALSE(qc::translateQASM3ToQC(qasm::booleanPowX, &context));
}

TEST(QASM3TranslationErrors, ChecksPowerExponentPrecisionAndNesting) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto translated = qc::translateQASM3ToQC(qasm::exactLargePowX, &context);
  ASSERT_TRUE(translated);
  SmallVector<qc::PowOp> powers;
  translated->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 1U);
  const auto exponent = powers.front().getExponentValue();
  ASSERT_TRUE(exponent.has_value());
  EXPECT_DOUBLE_EQ(*exponent, 9007199254740992.0);

  EXPECT_FALSE(qc::translateQASM3ToQC(qasm::inexactLargePowX, &context));

  translated = qc::translateQASM3ToQC(qasm::overflowingNestedPowX, &context);
  ASSERT_TRUE(translated);
  powers.clear();
  translated->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 2U);
  for (auto power : powers) {
    ASSERT_TRUE(power.getExponentValue().has_value());
    EXPECT_DOUBLE_EQ(*power.getExponentValue(), 4294967296.0);
  }
}

TEST_F(QASM3TranslationTest, RetainsClassicalRegisterName) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.0;
qubit q;
output bit named_result;
named_result = measure q;
)qasm";
  auto translated = qc::translateQASM3ToQC(source, context.get());
  ASSERT_TRUE(translated);

  cbit::AllocOp classicalRegister;
  translated->walk([&](cbit::AllocOp op) { classicalRegister = op; });
  ASSERT_TRUE(classicalRegister);
  const auto name = classicalRegister->getAttrOfType<StringAttr>(
      ::mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_result");
}

TEST_F(QASM3TranslationTest, UsesVersionSpecificBitInitialization) {
  constexpr std::array sources{std::pair{R"qasm(OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
)qasm",
                                         cbit::Initialization::Zero},
                               std::pair{R"qasm(OPENQASM 3.0;
qubit q;
bit[1] c;
c[0] = measure q;
)qasm",
                                         cbit::Initialization::Undefined}};

  for (const auto& [source, expected] : sources) {
    auto translated = qc::translateQASM3ToQC(source, context.get());
    ASSERT_TRUE(translated);
    SmallVector<cbit::AllocOp> registers;
    bool containsPoison = false;
    translated->walk([&](Operation* op) {
      if (auto alloc = dyn_cast<cbit::AllocOp>(op)) {
        registers.push_back(alloc);
      }
      containsPoison |= op->getName().getStringRef() == "ub.poison";
    });
    ASSERT_EQ(registers.size(), 1U);
    EXPECT_EQ(registers.front().getInitialization(), expected);
    EXPECT_FALSE(containsPoison);
  }
}

TEST_F(QASM3TranslationTest, RetainsQubitRegisterName) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.0;
qubit[2] named_qubits;
)qasm";
  auto translated = qc::translateQASM3ToQC(source, context.get());
  ASSERT_TRUE(translated);

  memref::AllocOp qubitRegister;
  translated->walk([&](memref::AllocOp op) {
    if (isa<qc::QubitType>(op.getType().getElementType())) {
      qubitRegister = op;
    }
  });
  ASSERT_TRUE(qubitRegister);
  const auto name = qubitRegister->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

TEST_F(QASM3TranslationTest, DistinguishesScalarAndWidthOneQubitAllocations) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
qubit scalar;
qubit[1] vector;
output bit[2] result;
result[0] = measure scalar;
result[1] = measure vector[0];
)qasm";
  auto translated = qc::translateQASM3ToQC(source, context.get());
  ASSERT_TRUE(translated);

  size_t scalarAllocations = 0;
  size_t registerAllocations = 0;
  translated->walk([&](qc::AllocOp /*op*/) { ++scalarAllocations; });
  translated->walk([&](memref::AllocOp allocation) {
    if (isa<qc::QubitType>(allocation.getType().getElementType())) {
      ++registerAllocations;
      const auto shape = allocation.getType().getShape();
      ASSERT_EQ(shape.size(), 1);
      EXPECT_EQ(shape.front(), 1);
    }
  });
  EXPECT_EQ(scalarAllocations, 1);
  EXPECT_EQ(registerAllocations, 1);
}

TEST_F(QASM3TranslationTest, JoinsMeasurementsFromBothBranches) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.0;
qubit[2] q;
bit condition;
bit measured_on_all_paths;
condition = measure q[0];
if (condition) {
  measured_on_all_paths = measure q[1];
} else {
  measured_on_all_paths = measure q[1];
}
if (measured_on_all_paths) {
  x q[1];
}
)qasm";
  auto translated = qc::translateQASM3ToQC(source, context.get());
  ASSERT_TRUE(translated);
  EXPECT_TRUE(succeeded(verify(*translated)));
}

TEST(QASM3TranslationRegression, ReloadsConditionAfterBranchMeasurement) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr auto source = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit c = measure q;
if (c) {
  c = measure q;
}
if (c) {
  x q;
}
)qasm";
  auto translated = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(translated);
  EXPECT_TRUE(succeeded(verify(*translated)));
}

INSTANTIATE_TEST_SUITE_P(
    QASM3TranslationProgramsTest, QASM3TranslationTest,
    testing::Values(

        QASM3TranslationTestCase{"AllocQubit", qasm::allocQubit,
                                 MQT_NAMED_BUILDER(qc::allocQubit)},
        QASM3TranslationTestCase{"AllocQubitRegister", qasm::allocQubitRegister,
                                 MQT_NAMED_BUILDER(qc::allocQubitRegister)},
        QASM3TranslationTestCase{
            "AllocMultipleQubitRegisters", qasm::allocMultipleQubitRegisters,
            MQT_NAMED_BUILDER(allocMultipleQubitRegisters)},
        QASM3TranslationTestCase{"AllocLargeRegister", qasm::allocLargeRegister,
                                 MQT_NAMED_BUILDER(qc::allocLargeRegister)},
        QASM3TranslationTestCase{
            "SingleMeasurementToSingleBit", qasm::singleMeasurementToSingleBit,
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        QASM3TranslationTestCase{
            "RepeatedMeasurementToSameBit", qasm::repeatedMeasurementToSameBit,
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        QASM3TranslationTestCase{
            "RepeatedMeasurementToDifferentBits",
            qasm::repeatedMeasurementToDifferentBits,
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        QASM3TranslationTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            qasm::multipleClassicalRegistersAndMeasurements,
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)},
        QASM3TranslationTestCase{
            "ResetQubitAfterSingleOp", qasm::resetQubitAfterSingleOp,
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QASM3TranslationTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            qasm::resetMultipleQubitsAfterSingleOp,
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        QASM3TranslationTestCase{
            "RepeatedResetAfterSingleOp", qasm::repeatedResetAfterSingleOp,
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp)},
        QASM3TranslationTestCase{"GlobalPhase", qasm::globalPhase,
                                 MQT_NAMED_BUILDER(qc::globalPhase)},
        QASM3TranslationTestCase{"InverseGlobalPhase", qasm::inverseGlobalPhase,
                                 MQT_NAMED_BUILDER(qc::inverseGlobalPhase)},
        QASM3TranslationTestCase{"Identity", qasm::identity,
                                 MQT_NAMED_BUILDER(qc::identity)},
        QASM3TranslationTestCase{"SingleControlledIdentity",
                                 qasm::singleControlledIdentity,
                                 MQT_NAMED_BUILDER(qc::twoQubitsOneIdentity)},
        QASM3TranslationTestCase{"MultipleControlledIdentity",
                                 qasm::multipleControlledIdentity,
                                 MQT_NAMED_BUILDER(qc::threeQubitsOneIdentity)},
        QASM3TranslationTestCase{"X", qasm::x, MQT_NAMED_BUILDER(qc::x)},
        QASM3TranslationTestCase{"TwoX", qasm::twoX, MQT_NAMED_BUILDER(twoX)},
        QASM3TranslationTestCase{"SingleControlledX", qasm::singleControlledX,
                                 MQT_NAMED_BUILDER(qc::singleControlledX)},
        QASM3TranslationTestCase{"SingleNegControlledX",
                                 qasm::singleNegControlledX,
                                 MQT_NAMED_BUILDER(singleNegControlledX)},
        QASM3TranslationTestCase{"MultipleControlledX",
                                 qasm::multipleControlledX,
                                 MQT_NAMED_BUILDER(qc::multipleControlledX)},
        QASM3TranslationTestCase{"TripleControlledXOpenQASM2",
                                 qasm::tripleControlledXOpenQASM2,
                                 MQT_NAMED_BUILDER(tripleControlledX)},
        QASM3TranslationTestCase{"MixedControlledX", qasm::mixedControlledX,
                                 MQT_NAMED_BUILDER(mixedControlledX)},
        QASM3TranslationTestCase{"TwoMixedControlledX",
                                 qasm::twoMixedControlledX,
                                 MQT_NAMED_BUILDER(twoMixedControlledX)},
        QASM3TranslationTestCase{"InverseX", qasm::inverseX,
                                 MQT_NAMED_BUILDER(qc::inverseX)},
        QASM3TranslationTestCase{
            "InverseMultipleControlledX", qasm::inverseMultipleControlledX,
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledX)},
        QASM3TranslationTestCase{"Y", qasm::y, MQT_NAMED_BUILDER(qc::y)},
        QASM3TranslationTestCase{"SingleControlledY", qasm::singleControlledY,
                                 MQT_NAMED_BUILDER(qc::singleControlledY)},
        QASM3TranslationTestCase{"MultipleControlledY",
                                 qasm::multipleControlledY,
                                 MQT_NAMED_BUILDER(qc::multipleControlledY)},
        QASM3TranslationTestCase{"Z", qasm::z, MQT_NAMED_BUILDER(qc::z)},
        QASM3TranslationTestCase{"SingleControlledZ", qasm::singleControlledZ,
                                 MQT_NAMED_BUILDER(qc::singleControlledZ)},
        QASM3TranslationTestCase{"MultipleControlledZ",
                                 qasm::multipleControlledZ,
                                 MQT_NAMED_BUILDER(qc::multipleControlledZ)},
        QASM3TranslationTestCase{"H", qasm::h, MQT_NAMED_BUILDER(qc::h)},
        QASM3TranslationTestCase{"SingleControlledH", qasm::singleControlledH,
                                 MQT_NAMED_BUILDER(qc::singleControlledH)},
        QASM3TranslationTestCase{"MultipleControlledH",
                                 qasm::multipleControlledH,
                                 MQT_NAMED_BUILDER(qc::multipleControlledH)},
        QASM3TranslationTestCase{"S", qasm::s, MQT_NAMED_BUILDER(qc::s)},
        QASM3TranslationTestCase{"SingleControlledS", qasm::singleControlledS,
                                 MQT_NAMED_BUILDER(qc::singleControlledS)},
        QASM3TranslationTestCase{"MultipleControlledS",
                                 qasm::multipleControlledS,
                                 MQT_NAMED_BUILDER(qc::multipleControlledS)},
        QASM3TranslationTestCase{"Sdg", qasm::sdg, MQT_NAMED_BUILDER(qc::sdg)},
        QASM3TranslationTestCase{"SingleControlledSdg",
                                 qasm::singleControlledSdg,
                                 MQT_NAMED_BUILDER(qc::singleControlledSdg)},
        QASM3TranslationTestCase{"MultipleControlledSdg",
                                 qasm::multipleControlledSdg,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSdg)},
        QASM3TranslationTestCase{"T", qasm::t_, MQT_NAMED_BUILDER(qc::t_)},
        QASM3TranslationTestCase{"SingleControlledT", qasm::singleControlledT,
                                 MQT_NAMED_BUILDER(qc::singleControlledT)},
        QASM3TranslationTestCase{"MultipleControlledT",
                                 qasm::multipleControlledT,
                                 MQT_NAMED_BUILDER(qc::multipleControlledT)},
        QASM3TranslationTestCase{"Tdg", qasm::tdg, MQT_NAMED_BUILDER(qc::tdg)},
        QASM3TranslationTestCase{"SingleControlledTdg",
                                 qasm::singleControlledTdg,
                                 MQT_NAMED_BUILDER(qc::singleControlledTdg)},
        QASM3TranslationTestCase{"MultipleControlledTdg",
                                 qasm::multipleControlledTdg,
                                 MQT_NAMED_BUILDER(qc::multipleControlledTdg)},
        QASM3TranslationTestCase{"SX", qasm::sx, MQT_NAMED_BUILDER(qc::sx)},
        QASM3TranslationTestCase{"SingleControlledSX", qasm::singleControlledSx,
                                 MQT_NAMED_BUILDER(qc::singleControlledSx)},
        QASM3TranslationTestCase{"MultipleControlledSX",
                                 qasm::multipleControlledSx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSx)},
        QASM3TranslationTestCase{
            "LegacyTripleControlledSqrtX",
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[4]; creg c[4]; "
            "c3sqrtx q[0], q[1], q[2], q[3]; measure q -> c;",
            MQT_NAMED_BUILDER(legacyTripleControlledSx)},
        QASM3TranslationTestCase{"SXdg", qasm::sxdg,
                                 MQT_NAMED_BUILDER(qc::sxdg)},
        QASM3TranslationTestCase{"SingleControlledSXdg",
                                 qasm::singleControlledSxdg,
                                 MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        QASM3TranslationTestCase{"MultipleControlledSXdg",
                                 qasm::multipleControlledSxdg,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSxdg)},
        QASM3TranslationTestCase{"RX", qasm::rx, MQT_NAMED_BUILDER(qc::rx)},
        QASM3TranslationTestCase{"SingleControlledRX", qasm::singleControlledRx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRx)},
        QASM3TranslationTestCase{"MultipleControlledRX",
                                 qasm::multipleControlledRx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRx)},
        QASM3TranslationTestCase{"RY", qasm::ry, MQT_NAMED_BUILDER(qc::ry)},
        QASM3TranslationTestCase{"SingleControlledRY", qasm::singleControlledRy,
                                 MQT_NAMED_BUILDER(qc::singleControlledRy)},
        QASM3TranslationTestCase{"MultipleControlledRY",
                                 qasm::multipleControlledRy,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRy)},
        QASM3TranslationTestCase{"RZ", qasm::rz, MQT_NAMED_BUILDER(qc::rz)},
        QASM3TranslationTestCase{"SingleControlledRZ", qasm::singleControlledRz,
                                 MQT_NAMED_BUILDER(qc::singleControlledRz)},
        QASM3TranslationTestCase{"MultipleControlledRZ",
                                 qasm::multipleControlledRz,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRz)},
        QASM3TranslationTestCase{"P", qasm::p, MQT_NAMED_BUILDER(qc::p)},
        QASM3TranslationTestCase{"SingleControlledP", qasm::singleControlledP,
                                 MQT_NAMED_BUILDER(qc::singleControlledP)},
        QASM3TranslationTestCase{"MultipleControlledP",
                                 qasm::multipleControlledP,
                                 MQT_NAMED_BUILDER(qc::multipleControlledP)},
        QASM3TranslationTestCase{"R", qasm::r, MQT_NAMED_BUILDER(qc::r)},
        QASM3TranslationTestCase{"SingleControlledR", qasm::singleControlledR,
                                 MQT_NAMED_BUILDER(qc::singleControlledR)},
        QASM3TranslationTestCase{"MultipleControlledR",
                                 qasm::multipleControlledR,
                                 MQT_NAMED_BUILDER(qc::multipleControlledR)},
        QASM3TranslationTestCase{"U2", qasm::u2, MQT_NAMED_BUILDER(legacyU2)},
        QASM3TranslationTestCase{"SingleControlledU2", qasm::singleControlledU2,
                                 MQT_NAMED_BUILDER(legacySingleControlledU2)},
        QASM3TranslationTestCase{"MultipleControlledU2",
                                 qasm::multipleControlledU2,
                                 MQT_NAMED_BUILDER(legacyMultipleControlledU2)},
        QASM3TranslationTestCase{"U", qasm::u, MQT_NAMED_BUILDER(legacyU)},
        QASM3TranslationTestCase{"SingleControlledU", qasm::singleControlledU,
                                 MQT_NAMED_BUILDER(legacySingleControlledU)},
        QASM3TranslationTestCase{"MultipleControlledU",
                                 qasm::multipleControlledU,
                                 MQT_NAMED_BUILDER(legacyMultipleControlledU)},
        QASM3TranslationTestCase{"SWAP", qasm::swap,
                                 MQT_NAMED_BUILDER(qc::swap)},
        QASM3TranslationTestCase{"SingleControlledSWAP",
                                 qasm::singleControlledSwap,
                                 MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        QASM3TranslationTestCase{"MultipleControlledSWAP",
                                 qasm::multipleControlledSwap,
                                 MQT_NAMED_BUILDER(qc::multipleControlledSwap)},
        QASM3TranslationTestCase{"iSWAP", qasm::iswap,
                                 MQT_NAMED_BUILDER(qc::iswap)},
        QASM3TranslationTestCase{"SingleControllediSWAP",
                                 qasm::singleControlledIswap,
                                 MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        QASM3TranslationTestCase{
            "MultipleControllediSWAP", qasm::multipleControlledIswap,
            MQT_NAMED_BUILDER(qc::multipleControlledIswap)},
        QASM3TranslationTestCase{"InverseISWAP", qasm::inverseIswap,
                                 MQT_NAMED_BUILDER(qc::inverseIswap)},
        QASM3TranslationTestCase{
            "InverseMultiControlledISWAP", qasm::inverseMultipleControlledIswap,
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        QASM3TranslationTestCase{"DCX", qasm::dcx, MQT_NAMED_BUILDER(qc::dcx)},
        QASM3TranslationTestCase{"SingleControlledDCX",
                                 qasm::singleControlledDcx,
                                 MQT_NAMED_BUILDER(qc::singleControlledDcx)},
        QASM3TranslationTestCase{"MultipleControlledDCX",
                                 qasm::multipleControlledDcx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledDcx)},
        QASM3TranslationTestCase{"ECR", qasm::ecr, MQT_NAMED_BUILDER(qc::ecr)},
        QASM3TranslationTestCase{"SingleControlledECR",
                                 qasm::singleControlledEcr,
                                 MQT_NAMED_BUILDER(qc::singleControlledEcr)},
        QASM3TranslationTestCase{"MultipleControlledECR",
                                 qasm::multipleControlledEcr,
                                 MQT_NAMED_BUILDER(qc::multipleControlledEcr)},
        QASM3TranslationTestCase{"RXX", qasm::rxx, MQT_NAMED_BUILDER(qc::rxx)},
        QASM3TranslationTestCase{"SingleControlledRXX",
                                 qasm::singleControlledRxx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRxx)},
        QASM3TranslationTestCase{"MultipleControlledRXX",
                                 qasm::multipleControlledRxx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRxx)},
        QASM3TranslationTestCase{"TripleControlledRXX",
                                 qasm::tripleControlledRxx,
                                 MQT_NAMED_BUILDER(qc::tripleControlledRxx)},
        QASM3TranslationTestCase{"RYY", qasm::ryy, MQT_NAMED_BUILDER(qc::ryy)},
        QASM3TranslationTestCase{"SingleControlledRYY",
                                 qasm::singleControlledRyy,
                                 MQT_NAMED_BUILDER(qc::singleControlledRyy)},
        QASM3TranslationTestCase{"MultipleControlledRYY",
                                 qasm::multipleControlledRyy,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRyy)},
        QASM3TranslationTestCase{"RZX", qasm::rzx, MQT_NAMED_BUILDER(qc::rzx)},
        QASM3TranslationTestCase{"SingleControlledRZX",
                                 qasm::singleControlledRzx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRzx)},
        QASM3TranslationTestCase{"MultipleControlledRZX",
                                 qasm::multipleControlledRzx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRzx)},
        QASM3TranslationTestCase{"RZZ", qasm::rzz, MQT_NAMED_BUILDER(qc::rzz)},
        QASM3TranslationTestCase{"SingleControlledRZZ",
                                 qasm::singleControlledRzz,
                                 MQT_NAMED_BUILDER(qc::singleControlledRzz)},
        QASM3TranslationTestCase{"MultipleControlledRZZ",
                                 qasm::multipleControlledRzz,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRzz)},
        QASM3TranslationTestCase{"XXPlusYY", qasm::xxPlusYY,
                                 MQT_NAMED_BUILDER(qc::xxPlusYY)},
        QASM3TranslationTestCase{
            "SingleControlledXXPlusYY", qasm::singleControlledXxPlusYY,
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        QASM3TranslationTestCase{
            "MultipleControlledXXPlusYY", qasm::multipleControlledXxPlusYY,
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)},
        QASM3TranslationTestCase{"XXMinusYY", qasm::xxMinusYY,
                                 MQT_NAMED_BUILDER(qc::xxMinusYY)},
        QASM3TranslationTestCase{
            "SingleControlledXXMinusYY", qasm::singleControlledXxMinusYY,
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        QASM3TranslationTestCase{
            "MultipleControlledXXMinusYY", qasm::multipleControlledXxMinusYY,
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)},
        QASM3TranslationTestCase{"RCCX", qasm::rccx,
                                 MQT_NAMED_BUILDER(qc::rccx)},
        QASM3TranslationTestCase{"SingleControlledRCCX",
                                 qasm::singleControlledRccx,
                                 MQT_NAMED_BUILDER(qc::singleControlledRccx)},
        QASM3TranslationTestCase{"MultipleControlledRCCX",
                                 qasm::multipleControlledRccx,
                                 MQT_NAMED_BUILDER(qc::multipleControlledRccx)},
        QASM3TranslationTestCase{"Barrier", qasm::barrier,
                                 MQT_NAMED_BUILDER(qc::barrier)},
        QASM3TranslationTestCase{"PowTwoX", qasm::powTwoX,
                                 MQT_NAMED_BUILDER(powTwoX)},
        QASM3TranslationTestCase{"PowZeroX", qasm::powZeroX,
                                 MQT_NAMED_BUILDER(powZeroX)},
        QASM3TranslationTestCase{"NegativePowS", qasm::negativePowS,
                                 MQT_NAMED_BUILDER(negativePowS)},
        QASM3TranslationTestCase{"ControlledInversePowS",
                                 qasm::controlledInversePowS,
                                 MQT_NAMED_BUILDER(controlledInversePowS)},
        QASM3TranslationTestCase{"NestedPowX", qasm::nestedPowX,
                                 MQT_NAMED_BUILDER(nestedPowX)},
        QASM3TranslationTestCase{"CustomPowHS", qasm::customPowHS,
                                 MQT_NAMED_BUILDER(customPowHS)},
        QASM3TranslationTestCase{"BroadcastPowX", qasm::broadcastPowX,
                                 MQT_NAMED_BUILDER(broadcastPowX)},
        QASM3TranslationTestCase{"BarrierTwoQubits", qasm::barrierTwoQubits,
                                 MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
        QASM3TranslationTestCase{"BarrierMultipleQubits",
                                 qasm::barrierMultipleQubits,
                                 MQT_NAMED_BUILDER(qc::barrierMultipleQubits)},
        QASM3TranslationTestCase{"CtrlTwo", qasm::ctrlTwo,
                                 MQT_NAMED_BUILDER(qc::ctrlTwo)},
        QASM3TranslationTestCase{"CtrlTwoMixed", qasm::ctrlTwoMixed,
                                 MQT_NAMED_BUILDER(qc::ctrlTwoMixed)},
        QASM3TranslationTestCase{"SimpleIf", qasm::simpleIf,
                                 MQT_NAMED_BUILDER(qc::simpleIf)},
        QASM3TranslationTestCase{"IfElse", qasm::ifElse,
                                 MQT_NAMED_BUILDER(qc::ifElse)},
        QASM3TranslationTestCase{"IfTwoQubits", qasm::ifTwoQubits,
                                 MQT_NAMED_BUILDER(qc::ifTwoQubits)},
        QASM3TranslationTestCase{"IfWithMeasurement", qasm::ifWithMeasurement,
                                 MQT_NAMED_BUILDER(ifWithMeasurement)},
        QASM3TranslationTestCase{"IfNot", qasm::ifNot,
                                 MQT_NAMED_BUILDER(ifNot)},
        QASM3TranslationTestCase{"BroadcastRegisterAndQubit",
                                 qasm::broadcastRegisterAndQubit,
                                 MQT_NAMED_BUILDER(broadcastRegisterAndQubit)},
        QASM3TranslationTestCase{"BroadcastCompoundGate",
                                 qasm::broadcastCompoundGate,
                                 MQT_NAMED_BUILDER(broadcastCompoundGate)},
        QASM3TranslationTestCase{"ExpressionArithmetic",
                                 qasm::expressionArithmetic,
                                 MQT_NAMED_BUILDER(expressionArithmetic)},
        QASM3TranslationTestCase{"ExpressionUnaryMinus",
                                 qasm::expressionUnaryMinus,
                                 MQT_NAMED_BUILDER(expressionUnaryMinus)},
        QASM3TranslationTestCase{"ExpressionBuiltinConstants",
                                 qasm::expressionBuiltinConstants,
                                 MQT_NAMED_BUILDER(expressionBuiltinConstants)},
        QASM3TranslationTestCase{"ExpressionMathFunctions",
                                 qasm::expressionMathFunctions,
                                 MQT_NAMED_BUILDER(expressionMathFunctions)},
        QASM3TranslationTestCase{
            "ExpressionNestedMathFunctions",
            qasm::expressionNestedMathFunctions,
            MQT_NAMED_BUILDER(expressionNestedMathFunctions)},
        QASM3TranslationTestCase{"ExpressionConstFloat",
                                 qasm::expressionConstFloat,
                                 MQT_NAMED_BUILDER(expressionConstFloat)},
        QASM3TranslationTestCase{"ExpressionMutableFloat",
                                 qasm::expressionMutableFloat,
                                 MQT_NAMED_BUILDER(expressionMutableFloat)},
        QASM3TranslationTestCase{
            "ExpressionConstIntArithmetic", qasm::expressionConstIntArithmetic,
            MQT_NAMED_BUILDER(expressionConstIntArithmetic)},
        QASM3TranslationTestCase{"ConditionLiteral", qasm::conditionLiteral,
                                 MQT_NAMED_BUILDER(conditionLiteral)},
        QASM3TranslationTestCase{"ConditionMeasurement",
                                 qasm::conditionMeasurement,
                                 MQT_NAMED_BUILDER(conditionMeasurement)},
        QASM3TranslationTestCase{"ConditionAnd", qasm::conditionAnd,
                                 MQT_NAMED_BUILDER(conditionAnd)},
        QASM3TranslationTestCase{"ConditionOr", qasm::conditionOr,
                                 MQT_NAMED_BUILDER(conditionOr)},
        QASM3TranslationTestCase{"ConditionNotAndOr", qasm::conditionNotAndOr,
                                 MQT_NAMED_BUILDER(conditionNotAndOr)},
        QASM3TranslationTestCase{"ConditionBoolVariable",
                                 qasm::conditionBoolVariable,
                                 MQT_NAMED_BUILDER(conditionBoolVariable)},
        QASM3TranslationTestCase{"ConditionIndexedBit",
                                 qasm::conditionIndexedBit,
                                 MQT_NAMED_BUILDER(conditionIndexedBit)}));

} // namespace mqt::test::qasm3_translation
