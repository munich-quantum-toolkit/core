/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Edge.hpp"
#include "dd/Package.hpp"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Dialect/CBit/IR/CBitAttributes.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Support/Passes.h"
#include "mqt/Target/OpenQASM/TranslateOpenQASMToQC.h"

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "qasm_programs.h"
#include "qc_programs.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <utility>

namespace mqt::test::openqasm_translation {
using namespace mlir;

namespace {

struct OpenQASMTranslationTestCase {
  std::string name;
  std::string source;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> referenceBuilder;
};

class OpenQASMTranslationTest
    : public testing::TestWithParam<OpenQASMTranslationTestCase> {
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

static Value legacyTripleControlledSx(qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(ValueRange{q[0], q[1], q[2]}, q[3],
         [&](Value target) { b.sx(target); });
  return measureToRegister(b, q.qubits);
}

[[nodiscard]] static qco::Matrix2x2
openQASM3UMatrix(const double theta, const double phi, const double lambda) {
  using namespace std::complex_literals;
  const auto thetaPhase = std::exp(1i * theta);
  const auto common = 0.5 * (1.0 + thetaPhase);
  const auto difference = qco::Complex{0.0, 0.5} * (1.0 - thetaPhase);
  return qco::Matrix2x2::fromElements(
      common, -std::exp(1i * lambda) * difference,
      std::exp(1i * phi) * difference, std::exp(1i * (phi + lambda)) * common);
}

[[nodiscard]] static qco::Matrix2x2
conventionalUMatrix(const double theta, const double phi, const double lambda) {
  using namespace std::complex_literals;
  const auto cosine = std::cos(theta / 2.0);
  const auto sine = std::sin(theta / 2.0);
  return qco::Matrix2x2::fromElements(cosine, -std::exp(1i * lambda) * sine,
                                      std::exp(1i * phi) * sine,
                                      std::exp(1i * (phi + lambda)) * cosine);
}

[[nodiscard]] static qco::Matrix2x2
openQASM2UMatrix(const double theta, const double phi, const double lambda) {
  using namespace std::complex_literals;
  return conventionalUMatrix(theta, phi, lambda) *
         std::exp(qco::Complex{0.0, -0.5} * (phi + lambda));
}

[[nodiscard]] static qco::Matrix4x4
controlledMatrix(const qco::Matrix2x2& body, const bool negative = false) {
  auto result = qco::Matrix4x4::identity();
  // The control is q[0], the least significant bit of the basis index.
  const size_t controlBit = negative ? 0 : 1;
  for (size_t row = 0; row < 2; ++row) {
    for (size_t column = 0; column < 2; ++column) {
      result(2 * row + controlBit, 2 * column + controlBit) = body(row, column);
    }
  }
  return result;
}

template <typename Matrix>
static void expectTranslatedMatrix(const llvm::StringRef source,
                                   const Matrix& expected) {
  SCOPED_TRACE(source.str());
  MLIRContext context;
  auto module = qc::translateOpenQASMToQC(source, &context);
  ASSERT_TRUE(module);
  PassManager manager(&context);
  manager.addPass(createQCToQCO());
  ASSERT_TRUE(succeeded(manager.run(*module)));

  constexpr auto numQubits = std::bit_width(Matrix::K_ROWS) - 1;
  auto package = llvm::cantFail(dd::Package::create(numQubits));
  const auto matrix =
      qco::buildFunctionality(mlir::mqt::getEntryPoint(*module), *package);
  ASSERT_TRUE(succeeded(matrix));
  const auto actual = llvm::cantFail(dd::getMatrix(*matrix, numQubits));
  for (size_t row = 0; row < Matrix::K_ROWS; ++row) {
    for (size_t column = 0; column < Matrix::K_COLS; ++column) {
      EXPECT_NEAR(std::abs(actual[row][column] - expected(row, column)), 0.,
                  1e-10)
          << "at (" << row << ", " << column << ")";
    }
  }
}

TEST(OpenQASMTranslationMatrixTest, PreservesOpenQASMGatePhaseConventions) {
  constexpr double theta = 0.37;
  constexpr double phi = -0.29;
  constexpr double lambda = 0.83;
  constexpr double gamma = -0.41;
  const auto qasm3U = openQASM3UMatrix(theta, phi, lambda);
  const auto qasm2U = openQASM2UMatrix(theta, phi, lambda);
  const auto nativeU = conventionalUMatrix(theta, phi, lambda);

  struct OneQubitCase {
    llvm::StringRef source;
    qco::Matrix2x2 expected;
  };
  const auto oneQubitCases = std::to_array<OneQubitCase>({
      {
          .source = "OPENQASM 3.1; qubit q; U(0.37, -0.29, 0.83) q;",
          .expected = qasm3U,
      },
      {
          .source = "OPENQASM 2.0; qreg q[1]; U(0.37, -0.29, 0.83) q[0];",
          .expected = qasm2U,
      },
      {
          .source = "OPENQASM 3.1; include \"stdgates.inc\"; qubit q; "
                    "u2(-0.29, 0.83) q;",
          .expected = openQASM2UMatrix(std::numbers::pi / 2.0, phi, lambda),
      },
      {
          .source = "OPENQASM 3.1; include \"stdgates.inc\"; qubit q; "
                    "u3(0.37, -0.29, 0.83) q;",
          .expected = qasm2U,
      },
      {
          .source = "OPENQASM 3.1; qubit q; u(0.37, -0.29, 0.83) q;",
          .expected = nativeU,
      },
      {
          .source = "OPENQASM 3.1; qubit q; inv @ U(0.37, -0.29, 0.83) q;",
          .expected = qasm3U.adjoint(),
      },
      {
          .source = "OPENQASM 3.1; qubit q; pow(2) @ U(0.37, -0.29, 0.83) q;",
          .expected = qasm3U * qasm3U,
      },
  });
  for (const auto& test : oneQubitCases) {
    expectTranslatedMatrix(test.source, test.expected);
  }

  const auto controlledQASM3 = controlledMatrix(qasm3U);
  const auto controlledQASM2 = controlledMatrix(qasm2U);
  const auto phasedQASM3 = qasm3U * std::exp(std::complex<double>{0.0, gamma});
  struct TwoQubitCase {
    llvm::StringRef source;
    qco::Matrix4x4 expected;
  };
  const auto twoQubitCases = std::to_array<TwoQubitCase>({
      {
          .source = "OPENQASM 3.1; qubit[2] q; "
                    "ctrl @ u(0.37, -0.29, 0.83) q[0], q[1];",
          .expected = controlledMatrix(nativeU),
      },
      {
          .source = "OPENQASM 3.1; qubit[2] q; "
                    "negctrl @ u(0.37, -0.29, 0.83) q[0], q[1];",
          .expected = controlledMatrix(nativeU, true),
      },
      {
          .source = "OPENQASM 3.1; qubit[2] q; "
                    "ctrl @ U(0.37, -0.29, 0.83) q[0], q[1];",
          .expected = controlledQASM3,
      },
      {
          .source = "OPENQASM 3.1; qubit[2] q; "
                    "negctrl @ U(0.37, -0.29, 0.83) q[0], q[1];",
          .expected = controlledMatrix(qasm3U, true),
      },
      {
          .source = "OPENQASM 3.1; include \"stdgates.inc\"; qubit[2] q; "
                    "cu(0.37, -0.29, 0.83, -0.41) q[0], q[1];",
          .expected = controlledMatrix(phasedQASM3),
      },
      {
          .source = "OPENQASM 3.1; include \"qelib1.inc\"; qubit[2] q; "
                    "cu3(0.37, -0.29, 0.83) q[0], q[1];",
          .expected = controlledQASM2,
      },
  });
  for (const auto& test : twoQubitCases) {
    expectTranslatedMatrix(test.source, test.expected);
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
  auto hs = b.createUnitaryFunction(
      "hs", TypeRange{qc::QubitType::get(b.getContext())},
      [&](ValueRange arguments) {
        b.h(arguments[0]);
        b.s(arguments[0]);
      });
  auto q = b.allocQubit();
  b.pow(2.0, q, [&](Value qubit) { b.call(hs, qubit); });
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
  auto compound =
      b.createUnitaryFunction("compound",
                              TypeRange{
                                  qc::QubitType::get(b.getContext()),
                                  qc::QubitType::get(b.getContext()),
                              },
                              [&](ValueRange arguments) {
                                b.x(arguments[0]);
                                b.cx(arguments[0], arguments[1]);
                              });
  auto reg = b.allocQubitRegister(3);
  auto q = b.allocQubit();
  for (auto qubit : reg.qubits) {
    b.call(compound, {qubit, q});
  }
  auto left = measureToRegister(b, {reg[0], reg[1], reg[2]});
  auto right = measureToRegister(b, {q});
  return {left, right};
}

static Value ctrlTwoCalls(qc::QCProgramBuilder& b) {
  auto compound =
      b.createUnitaryFunction("compound",
                              TypeRange{
                                  qc::QubitType::get(b.getContext()),
                                  qc::QubitType::get(b.getContext()),
                              },
                              [&](ValueRange arguments) {
                                b.x(arguments[0]);
                                b.rxx(0.123, arguments[0], arguments[1]);
                              });
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]},
         [&](ValueRange targets) { b.call(compound, targets); });
  return measureToRegister(b, q.qubits);
}

static Value ctrlTwoMixedCalls(qc::QCProgramBuilder& b) {
  auto compound =
      b.createUnitaryFunction("compound",
                              TypeRange{
                                  qc::QubitType::get(b.getContext()),
                                  qc::QubitType::get(b.getContext()),
                              },
                              [&](ValueRange arguments) {
                                b.cx(arguments[0], arguments[1]);
                                b.rxx(0.123, arguments[0], arguments[1]);
                              });
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]},
         [&](ValueRange targets) { b.call(compound, targets); });
  return measureToRegister(b, q.qubits);
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

TEST_P(OpenQASMTranslationTest, ProgramEquivalence) {
  SCOPED_TRACE(
      llvm::formatv("OpenQASMTranslation: {0}, reference={1}", GetParam().name,
                    ::mqt::test::displayName(GetParam().referenceBuilder.name))
          .str());
  const auto name = " (" + GetParam().name + ")";
  const auto& source = GetParam().source;
  const auto referenceBuilder = GetParam().referenceBuilder;
  ::mqt::test::DeferredPrinter printer;

  auto translated = qc::translateOpenQASMToQC(source, context.get());
  ASSERT_TRUE(translated);
  printer.record(translated.get(), "Translated QC IR" + name);
  EXPECT_TRUE(verify(*translated).succeeded());

  ASSERT_TRUE(succeeded(runPassPipeline(translated.get(), "simplify-quantum")));
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

  ASSERT_TRUE(succeeded(runPassPipeline(reference.get(), "simplify-quantum")));
  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  // Shared gate fixtures have no source names; dedicated tests check them.
  translated->walk([](Operation* op) {
    op->removeAttr(mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  });

  ASSERT_TRUE(succeeded(convertQCToQCO(translated.get())));
  ASSERT_TRUE(succeeded(convertQCToQCO(reference.get())));
  ASSERT_TRUE(succeeded(runPassPipeline(translated.get(), "simplify-quantum")));
  ASSERT_TRUE(runQCOCleanupPipeline(translated.get()).succeeded());
  ASSERT_TRUE(succeeded(runPassPipeline(reference.get(), "simplify-quantum")));
  ASSERT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  printer.record(translated.get(), "Lowered Translated QCO IR" + name);
  printer.record(reference.get(), "Lowered Reference QCO IR" + name);
  ASSERT_TRUE(verify(*translated).succeeded());
  ASSERT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(translated.get(), reference.get()));
}

TEST(OpenQASMTranslationErrors, AcceptsFloatingAndRejectsBooleanPowerExponent) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto translated = qc::translateOpenQASMToQC(qasm::floatingPowX, &context);
  ASSERT_TRUE(translated);
  SmallVector<qc::PowOp> powers;
  translated->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 1U);
  ASSERT_TRUE(powers.front().getExponentValue().has_value());
  EXPECT_DOUBLE_EQ(*powers.front().getExponentValue(), 0.5);

  EXPECT_FALSE(qc::translateOpenQASMToQC(qasm::booleanPowX, &context));
}

TEST(OpenQASMTranslationErrors, ChecksPowerExponentPrecisionAndNesting) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto translated = qc::translateOpenQASMToQC(qasm::exactLargePowX, &context);
  ASSERT_TRUE(translated);
  SmallVector<qc::PowOp> powers;
  translated->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 1U);
  const auto exponent = powers.front().getExponentValue();
  ASSERT_TRUE(exponent.has_value());
  EXPECT_DOUBLE_EQ(*exponent, 9007199254740992.0);

  EXPECT_FALSE(qc::translateOpenQASMToQC(qasm::inexactLargePowX, &context));

  translated = qc::translateOpenQASMToQC(qasm::overflowingNestedPowX, &context);
  ASSERT_TRUE(translated);
  powers.clear();
  translated->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 2U);
  for (auto power : powers) {
    ASSERT_TRUE(power.getExponentValue().has_value());
    EXPECT_DOUBLE_EQ(*power.getExponentValue(), 4294967296.0);
  }
}

TEST_F(OpenQASMTranslationTest, RetainsClassicalRegisterName) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.0;
qubit q;
output bit named_result;
named_result = measure q;
)qasm";
  auto translated = qc::translateOpenQASMToQC(source, context.get());
  ASSERT_TRUE(translated);

  cbit::AllocOp classicalRegister;
  translated->walk([&](cbit::AllocOp op) { classicalRegister = op; });
  ASSERT_TRUE(classicalRegister);
  const auto name = classicalRegister->getAttrOfType<StringAttr>(
      ::mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_result");
}

TEST_F(OpenQASMTranslationTest, UsesVersionSpecificBitInitialization) {
  constexpr std::array sources{
      std::pair{R"qasm(OPENQASM 2.0;
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
                cbit::Initialization::Undefined},
  };

  for (const auto& [source, expected] : sources) {
    auto translated = qc::translateOpenQASMToQC(source, context.get());
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

TEST_F(OpenQASMTranslationTest, RetainsQubitRegisterName) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.0;
qubit[2] named_qubits;
)qasm";
  auto translated = qc::translateOpenQASMToQC(source, context.get());
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

TEST_F(OpenQASMTranslationTest,
       DistinguishesScalarAndWidthOneQubitAllocations) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
qubit scalar;
qubit[1] vector;
output bit[2] result;
result[0] = measure scalar;
result[1] = measure vector[0];
)qasm";
  auto translated = qc::translateOpenQASMToQC(source, context.get());
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

TEST_F(OpenQASMTranslationTest, JoinsMeasurementsFromBothBranches) {
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
  auto translated = qc::translateOpenQASMToQC(source, context.get());
  ASSERT_TRUE(translated);
  EXPECT_TRUE(succeeded(verify(*translated)));
}

TEST(OpenQASMTranslationRegression, ReloadsConditionAfterBranchMeasurement) {
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
  auto translated = qc::translateOpenQASMToQC(source, &context);
  ASSERT_TRUE(translated);
  EXPECT_TRUE(succeeded(verify(*translated)));
}

INSTANTIATE_TEST_SUITE_P(
    OpenQASMTranslationProgramsTest, OpenQASMTranslationTest,
    testing::Values(

        OpenQASMTranslationTestCase{"AllocQubit", qasm::allocQubit,
                                    MQT_NAMED_BUILDER(qc::allocQubit)},
        OpenQASMTranslationTestCase{"AllocQubitRegister",
                                    qasm::allocQubitRegister,
                                    MQT_NAMED_BUILDER(qc::allocQubitRegister)},
        OpenQASMTranslationTestCase{
            "AllocMultipleQubitRegisters", qasm::allocMultipleQubitRegisters,
            MQT_NAMED_BUILDER(allocMultipleQubitRegisters)},
        OpenQASMTranslationTestCase{"AllocLargeRegister",
                                    qasm::allocLargeRegister,
                                    MQT_NAMED_BUILDER(qc::allocLargeRegister)},
        OpenQASMTranslationTestCase{
            "SingleMeasurementToSingleBit", qasm::singleMeasurementToSingleBit,
            MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        OpenQASMTranslationTestCase{
            "RepeatedMeasurementToSameBit", qasm::repeatedMeasurementToSameBit,
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        OpenQASMTranslationTestCase{
            "RepeatedMeasurementToDifferentBits",
            qasm::repeatedMeasurementToDifferentBits,
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        OpenQASMTranslationTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            qasm::multipleClassicalRegistersAndMeasurements,
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)},
        OpenQASMTranslationTestCase{
            "ResetQubitAfterSingleOp", qasm::resetQubitAfterSingleOp,
            MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        OpenQASMTranslationTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            qasm::resetMultipleQubitsAfterSingleOp,
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        OpenQASMTranslationTestCase{
            "RepeatedResetAfterSingleOp", qasm::repeatedResetAfterSingleOp,
            MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp)},
        OpenQASMTranslationTestCase{"GlobalPhase", qasm::globalPhase,
                                    MQT_NAMED_BUILDER(qc::globalPhase)},
        OpenQASMTranslationTestCase{"InverseGlobalPhase",
                                    qasm::inverseGlobalPhase,
                                    MQT_NAMED_BUILDER(qc::inverseGlobalPhase)},
        OpenQASMTranslationTestCase{"Identity", qasm::identity,
                                    MQT_NAMED_BUILDER(qc::identity)},
        OpenQASMTranslationTestCase{
            "SingleControlledIdentity", qasm::singleControlledIdentity,
            MQT_NAMED_BUILDER(qc::twoQubitsOneIdentity)},
        OpenQASMTranslationTestCase{
            "MultipleControlledIdentity", qasm::multipleControlledIdentity,
            MQT_NAMED_BUILDER(qc::threeQubitsOneIdentity)},
        OpenQASMTranslationTestCase{"X", qasm::x, MQT_NAMED_BUILDER(qc::x)},
        OpenQASMTranslationTestCase{"TwoX", qasm::twoX,
                                    MQT_NAMED_BUILDER(twoX)},
        OpenQASMTranslationTestCase{"SingleControlledX",
                                    qasm::singleControlledX,
                                    MQT_NAMED_BUILDER(qc::singleControlledX)},
        OpenQASMTranslationTestCase{"SingleNegControlledX",
                                    qasm::singleNegControlledX,
                                    MQT_NAMED_BUILDER(singleNegControlledX)},
        OpenQASMTranslationTestCase{"MultipleControlledX",
                                    qasm::multipleControlledX,
                                    MQT_NAMED_BUILDER(qc::multipleControlledX)},
        OpenQASMTranslationTestCase{"TripleControlledXOpenQASM2",
                                    qasm::tripleControlledXOpenQASM2,
                                    MQT_NAMED_BUILDER(tripleControlledX)},
        OpenQASMTranslationTestCase{"MixedControlledX", qasm::mixedControlledX,
                                    MQT_NAMED_BUILDER(mixedControlledX)},
        OpenQASMTranslationTestCase{"TwoMixedControlledX",
                                    qasm::twoMixedControlledX,
                                    MQT_NAMED_BUILDER(twoMixedControlledX)},
        OpenQASMTranslationTestCase{"InverseX", qasm::inverseX,
                                    MQT_NAMED_BUILDER(qc::inverseX)},
        OpenQASMTranslationTestCase{
            "InverseMultipleControlledX", qasm::inverseMultipleControlledX,
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledX)},
        OpenQASMTranslationTestCase{"Y", qasm::y, MQT_NAMED_BUILDER(qc::y)},
        OpenQASMTranslationTestCase{"SingleControlledY",
                                    qasm::singleControlledY,
                                    MQT_NAMED_BUILDER(qc::singleControlledY)},
        OpenQASMTranslationTestCase{"MultipleControlledY",
                                    qasm::multipleControlledY,
                                    MQT_NAMED_BUILDER(qc::multipleControlledY)},
        OpenQASMTranslationTestCase{"Z", qasm::z, MQT_NAMED_BUILDER(qc::z)},
        OpenQASMTranslationTestCase{"SingleControlledZ",
                                    qasm::singleControlledZ,
                                    MQT_NAMED_BUILDER(qc::singleControlledZ)},
        OpenQASMTranslationTestCase{"MultipleControlledZ",
                                    qasm::multipleControlledZ,
                                    MQT_NAMED_BUILDER(qc::multipleControlledZ)},
        OpenQASMTranslationTestCase{"H", qasm::h, MQT_NAMED_BUILDER(qc::h)},
        OpenQASMTranslationTestCase{"SingleControlledH",
                                    qasm::singleControlledH,
                                    MQT_NAMED_BUILDER(qc::singleControlledH)},
        OpenQASMTranslationTestCase{"MultipleControlledH",
                                    qasm::multipleControlledH,
                                    MQT_NAMED_BUILDER(qc::multipleControlledH)},
        OpenQASMTranslationTestCase{"S", qasm::s, MQT_NAMED_BUILDER(qc::s)},
        OpenQASMTranslationTestCase{"SingleControlledS",
                                    qasm::singleControlledS,
                                    MQT_NAMED_BUILDER(qc::singleControlledS)},
        OpenQASMTranslationTestCase{"MultipleControlledS",
                                    qasm::multipleControlledS,
                                    MQT_NAMED_BUILDER(qc::multipleControlledS)},
        OpenQASMTranslationTestCase{"Sdg", qasm::sdg,
                                    MQT_NAMED_BUILDER(qc::sdg)},
        OpenQASMTranslationTestCase{"SingleControlledSdg",
                                    qasm::singleControlledSdg,
                                    MQT_NAMED_BUILDER(qc::singleControlledSdg)},
        OpenQASMTranslationTestCase{
            "MultipleControlledSdg", qasm::multipleControlledSdg,
            MQT_NAMED_BUILDER(qc::multipleControlledSdg)},
        OpenQASMTranslationTestCase{"T", qasm::t_, MQT_NAMED_BUILDER(qc::t_)},
        OpenQASMTranslationTestCase{"SingleControlledT",
                                    qasm::singleControlledT,
                                    MQT_NAMED_BUILDER(qc::singleControlledT)},
        OpenQASMTranslationTestCase{"MultipleControlledT",
                                    qasm::multipleControlledT,
                                    MQT_NAMED_BUILDER(qc::multipleControlledT)},
        OpenQASMTranslationTestCase{"Tdg", qasm::tdg,
                                    MQT_NAMED_BUILDER(qc::tdg)},
        OpenQASMTranslationTestCase{"SingleControlledTdg",
                                    qasm::singleControlledTdg,
                                    MQT_NAMED_BUILDER(qc::singleControlledTdg)},
        OpenQASMTranslationTestCase{
            "MultipleControlledTdg", qasm::multipleControlledTdg,
            MQT_NAMED_BUILDER(qc::multipleControlledTdg)},
        OpenQASMTranslationTestCase{"SX", qasm::sx, MQT_NAMED_BUILDER(qc::sx)},
        OpenQASMTranslationTestCase{"SingleControlledSX",
                                    qasm::singleControlledSx,
                                    MQT_NAMED_BUILDER(qc::singleControlledSx)},
        OpenQASMTranslationTestCase{
            "MultipleControlledSX", qasm::multipleControlledSx,
            MQT_NAMED_BUILDER(qc::multipleControlledSx)},
        OpenQASMTranslationTestCase{
            "LegacyTripleControlledSqrtX",
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[4]; creg c[4]; "
            "c3sqrtx q[0], q[1], q[2], q[3]; measure q -> c;",
            MQT_NAMED_BUILDER(legacyTripleControlledSx)},
        OpenQASMTranslationTestCase{"SXdg", qasm::sxdg,
                                    MQT_NAMED_BUILDER(qc::sxdg)},
        OpenQASMTranslationTestCase{
            "SingleControlledSXdg", qasm::singleControlledSxdg,
            MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        OpenQASMTranslationTestCase{
            "MultipleControlledSXdg", qasm::multipleControlledSxdg,
            MQT_NAMED_BUILDER(qc::multipleControlledSxdg)},
        OpenQASMTranslationTestCase{"RX", qasm::rx, MQT_NAMED_BUILDER(qc::rx)},
        OpenQASMTranslationTestCase{"SingleControlledRX",
                                    qasm::singleControlledRx,
                                    MQT_NAMED_BUILDER(qc::singleControlledRx)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRX", qasm::multipleControlledRx,
            MQT_NAMED_BUILDER(qc::multipleControlledRx)},
        OpenQASMTranslationTestCase{"RY", qasm::ry, MQT_NAMED_BUILDER(qc::ry)},
        OpenQASMTranslationTestCase{"SingleControlledRY",
                                    qasm::singleControlledRy,
                                    MQT_NAMED_BUILDER(qc::singleControlledRy)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRY", qasm::multipleControlledRy,
            MQT_NAMED_BUILDER(qc::multipleControlledRy)},
        OpenQASMTranslationTestCase{"RZ", qasm::rz, MQT_NAMED_BUILDER(qc::rz)},
        OpenQASMTranslationTestCase{"SingleControlledRZ",
                                    qasm::singleControlledRz,
                                    MQT_NAMED_BUILDER(qc::singleControlledRz)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRZ", qasm::multipleControlledRz,
            MQT_NAMED_BUILDER(qc::multipleControlledRz)},
        OpenQASMTranslationTestCase{"P", qasm::p, MQT_NAMED_BUILDER(qc::p)},
        OpenQASMTranslationTestCase{"SingleControlledP",
                                    qasm::singleControlledP,
                                    MQT_NAMED_BUILDER(qc::singleControlledP)},
        OpenQASMTranslationTestCase{"MultipleControlledP",
                                    qasm::multipleControlledP,
                                    MQT_NAMED_BUILDER(qc::multipleControlledP)},
        OpenQASMTranslationTestCase{"R", qasm::r, MQT_NAMED_BUILDER(qc::r)},
        OpenQASMTranslationTestCase{"SingleControlledR",
                                    qasm::singleControlledR,
                                    MQT_NAMED_BUILDER(qc::singleControlledR)},
        OpenQASMTranslationTestCase{"MultipleControlledR",
                                    qasm::multipleControlledR,
                                    MQT_NAMED_BUILDER(qc::multipleControlledR)},
        OpenQASMTranslationTestCase{"U2", qasm::u2,
                                    MQT_NAMED_BUILDER(legacyU2)},
        OpenQASMTranslationTestCase{
            "SingleControlledU2", qasm::singleControlledU2,
            MQT_NAMED_BUILDER(legacySingleControlledU2)},
        OpenQASMTranslationTestCase{
            "MultipleControlledU2", qasm::multipleControlledU2,
            MQT_NAMED_BUILDER(legacyMultipleControlledU2)},
        OpenQASMTranslationTestCase{"U", qasm::u, MQT_NAMED_BUILDER(qc::u)},
        OpenQASMTranslationTestCase{"SingleControlledU",
                                    qasm::singleControlledU,
                                    MQT_NAMED_BUILDER(qc::singleControlledU)},
        OpenQASMTranslationTestCase{"MultipleControlledU",
                                    qasm::multipleControlledU,
                                    MQT_NAMED_BUILDER(qc::multipleControlledU)},
        OpenQASMTranslationTestCase{"SWAP", qasm::swap,
                                    MQT_NAMED_BUILDER(qc::swap)},
        OpenQASMTranslationTestCase{
            "SingleControlledSWAP", qasm::singleControlledSwap,
            MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        OpenQASMTranslationTestCase{
            "MultipleControlledSWAP", qasm::multipleControlledSwap,
            MQT_NAMED_BUILDER(qc::multipleControlledSwap)},
        OpenQASMTranslationTestCase{"iSWAP", qasm::iswap,
                                    MQT_NAMED_BUILDER(qc::iswap)},
        OpenQASMTranslationTestCase{
            "SingleControllediSWAP", qasm::singleControlledIswap,
            MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        OpenQASMTranslationTestCase{
            "MultipleControllediSWAP", qasm::multipleControlledIswap,
            MQT_NAMED_BUILDER(qc::multipleControlledIswap)},
        OpenQASMTranslationTestCase{"InverseISWAP", qasm::inverseIswap,
                                    MQT_NAMED_BUILDER(qc::inverseIswap)},
        OpenQASMTranslationTestCase{
            "InverseMultiControlledISWAP", qasm::inverseMultipleControlledIswap,
            MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        OpenQASMTranslationTestCase{"DCX", qasm::dcx,
                                    MQT_NAMED_BUILDER(qc::dcx)},
        OpenQASMTranslationTestCase{"SingleControlledDCX",
                                    qasm::singleControlledDcx,
                                    MQT_NAMED_BUILDER(qc::singleControlledDcx)},
        OpenQASMTranslationTestCase{
            "MultipleControlledDCX", qasm::multipleControlledDcx,
            MQT_NAMED_BUILDER(qc::multipleControlledDcx)},
        OpenQASMTranslationTestCase{"ECR", qasm::ecr,
                                    MQT_NAMED_BUILDER(qc::ecr)},
        OpenQASMTranslationTestCase{"SingleControlledECR",
                                    qasm::singleControlledEcr,
                                    MQT_NAMED_BUILDER(qc::singleControlledEcr)},
        OpenQASMTranslationTestCase{
            "MultipleControlledECR", qasm::multipleControlledEcr,
            MQT_NAMED_BUILDER(qc::multipleControlledEcr)},
        OpenQASMTranslationTestCase{"RXX", qasm::rxx,
                                    MQT_NAMED_BUILDER(qc::rxx)},
        OpenQASMTranslationTestCase{"SingleControlledRXX",
                                    qasm::singleControlledRxx,
                                    MQT_NAMED_BUILDER(qc::singleControlledRxx)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRXX", qasm::multipleControlledRxx,
            MQT_NAMED_BUILDER(qc::multipleControlledRxx)},
        OpenQASMTranslationTestCase{"TripleControlledRXX",
                                    qasm::tripleControlledRxx,
                                    MQT_NAMED_BUILDER(qc::tripleControlledRxx)},
        OpenQASMTranslationTestCase{"RYY", qasm::ryy,
                                    MQT_NAMED_BUILDER(qc::ryy)},
        OpenQASMTranslationTestCase{"SingleControlledRYY",
                                    qasm::singleControlledRyy,
                                    MQT_NAMED_BUILDER(qc::singleControlledRyy)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRYY", qasm::multipleControlledRyy,
            MQT_NAMED_BUILDER(qc::multipleControlledRyy)},
        OpenQASMTranslationTestCase{"RZX", qasm::rzx,
                                    MQT_NAMED_BUILDER(qc::rzx)},
        OpenQASMTranslationTestCase{"SingleControlledRZX",
                                    qasm::singleControlledRzx,
                                    MQT_NAMED_BUILDER(qc::singleControlledRzx)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRZX", qasm::multipleControlledRzx,
            MQT_NAMED_BUILDER(qc::multipleControlledRzx)},
        OpenQASMTranslationTestCase{"RZZ", qasm::rzz,
                                    MQT_NAMED_BUILDER(qc::rzz)},
        OpenQASMTranslationTestCase{"SingleControlledRZZ",
                                    qasm::singleControlledRzz,
                                    MQT_NAMED_BUILDER(qc::singleControlledRzz)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRZZ", qasm::multipleControlledRzz,
            MQT_NAMED_BUILDER(qc::multipleControlledRzz)},
        OpenQASMTranslationTestCase{"XXPlusYY", qasm::xxPlusYY,
                                    MQT_NAMED_BUILDER(qc::xxPlusYY)},
        OpenQASMTranslationTestCase{
            "SingleControlledXXPlusYY", qasm::singleControlledXxPlusYY,
            MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        OpenQASMTranslationTestCase{
            "MultipleControlledXXPlusYY", qasm::multipleControlledXxPlusYY,
            MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)},
        OpenQASMTranslationTestCase{"XXMinusYY", qasm::xxMinusYY,
                                    MQT_NAMED_BUILDER(qc::xxMinusYY)},
        OpenQASMTranslationTestCase{
            "SingleControlledXXMinusYY", qasm::singleControlledXxMinusYY,
            MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        OpenQASMTranslationTestCase{
            "MultipleControlledXXMinusYY", qasm::multipleControlledXxMinusYY,
            MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)},
        OpenQASMTranslationTestCase{"RCCX", qasm::rccx,
                                    MQT_NAMED_BUILDER(qc::rccx)},
        OpenQASMTranslationTestCase{
            "SingleControlledRCCX", qasm::singleControlledRccx,
            MQT_NAMED_BUILDER(qc::singleControlledRccx)},
        OpenQASMTranslationTestCase{
            "MultipleControlledRCCX", qasm::multipleControlledRccx,
            MQT_NAMED_BUILDER(qc::multipleControlledRccx)},
        OpenQASMTranslationTestCase{"Barrier", qasm::barrier,
                                    MQT_NAMED_BUILDER(qc::barrier)},
        OpenQASMTranslationTestCase{"PowTwoX", qasm::powTwoX,
                                    MQT_NAMED_BUILDER(powTwoX)},
        OpenQASMTranslationTestCase{"PowZeroX", qasm::powZeroX,
                                    MQT_NAMED_BUILDER(powZeroX)},
        OpenQASMTranslationTestCase{"NegativePowS", qasm::negativePowS,
                                    MQT_NAMED_BUILDER(negativePowS)},
        OpenQASMTranslationTestCase{"ControlledInversePowS",
                                    qasm::controlledInversePowS,
                                    MQT_NAMED_BUILDER(controlledInversePowS)},
        OpenQASMTranslationTestCase{"NestedPowX", qasm::nestedPowX,
                                    MQT_NAMED_BUILDER(nestedPowX)},
        OpenQASMTranslationTestCase{"CustomPowHS", qasm::customPowHS,
                                    MQT_NAMED_BUILDER(customPowHS)},
        OpenQASMTranslationTestCase{"BroadcastPowX", qasm::broadcastPowX,
                                    MQT_NAMED_BUILDER(broadcastPowX)},
        OpenQASMTranslationTestCase{"BarrierTwoQubits", qasm::barrierTwoQubits,
                                    MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
        OpenQASMTranslationTestCase{
            "BarrierMultipleQubits", qasm::barrierMultipleQubits,
            MQT_NAMED_BUILDER(qc::barrierMultipleQubits)},
        OpenQASMTranslationTestCase{"CtrlTwo", qasm::ctrlTwo,
                                    MQT_NAMED_BUILDER(ctrlTwoCalls)},
        OpenQASMTranslationTestCase{"CtrlTwoMixed", qasm::ctrlTwoMixed,
                                    MQT_NAMED_BUILDER(ctrlTwoMixedCalls)},
        OpenQASMTranslationTestCase{"SimpleIf", qasm::simpleIf,
                                    MQT_NAMED_BUILDER(qc::simpleIf)},
        OpenQASMTranslationTestCase{"IfElse", qasm::ifElse,
                                    MQT_NAMED_BUILDER(qc::ifElse)},
        OpenQASMTranslationTestCase{"IfTwoQubits", qasm::ifTwoQubits,
                                    MQT_NAMED_BUILDER(qc::ifTwoQubits)},
        OpenQASMTranslationTestCase{"IfWithMeasurement",
                                    qasm::ifWithMeasurement,
                                    MQT_NAMED_BUILDER(ifWithMeasurement)},
        OpenQASMTranslationTestCase{"IfNot", qasm::ifNot,
                                    MQT_NAMED_BUILDER(ifNot)},
        OpenQASMTranslationTestCase{
            "BroadcastRegisterAndQubit", qasm::broadcastRegisterAndQubit,
            MQT_NAMED_BUILDER(broadcastRegisterAndQubit)},
        OpenQASMTranslationTestCase{"BroadcastCompoundGate",
                                    qasm::broadcastCompoundGate,
                                    MQT_NAMED_BUILDER(broadcastCompoundGate)},
        OpenQASMTranslationTestCase{"ExpressionArithmetic",
                                    qasm::expressionArithmetic,
                                    MQT_NAMED_BUILDER(expressionArithmetic)},
        OpenQASMTranslationTestCase{"ExpressionUnaryMinus",
                                    qasm::expressionUnaryMinus,
                                    MQT_NAMED_BUILDER(expressionUnaryMinus)},
        OpenQASMTranslationTestCase{
            "ExpressionBuiltinConstants", qasm::expressionBuiltinConstants,
            MQT_NAMED_BUILDER(expressionBuiltinConstants)},
        OpenQASMTranslationTestCase{"ExpressionMathFunctions",
                                    qasm::expressionMathFunctions,
                                    MQT_NAMED_BUILDER(expressionMathFunctions)},
        OpenQASMTranslationTestCase{
            "ExpressionNestedMathFunctions",
            qasm::expressionNestedMathFunctions,
            MQT_NAMED_BUILDER(expressionNestedMathFunctions)},
        OpenQASMTranslationTestCase{"ExpressionConstFloat",
                                    qasm::expressionConstFloat,
                                    MQT_NAMED_BUILDER(expressionConstFloat)},
        OpenQASMTranslationTestCase{"ExpressionMutableFloat",
                                    qasm::expressionMutableFloat,
                                    MQT_NAMED_BUILDER(expressionMutableFloat)},
        OpenQASMTranslationTestCase{
            "ExpressionConstIntArithmetic", qasm::expressionConstIntArithmetic,
            MQT_NAMED_BUILDER(expressionConstIntArithmetic)},
        OpenQASMTranslationTestCase{"ConditionLiteral", qasm::conditionLiteral,
                                    MQT_NAMED_BUILDER(conditionLiteral)},
        OpenQASMTranslationTestCase{"ConditionMeasurement",
                                    qasm::conditionMeasurement,
                                    MQT_NAMED_BUILDER(conditionMeasurement)},
        OpenQASMTranslationTestCase{"ConditionAnd", qasm::conditionAnd,
                                    MQT_NAMED_BUILDER(conditionAnd)},
        OpenQASMTranslationTestCase{"ConditionOr", qasm::conditionOr,
                                    MQT_NAMED_BUILDER(conditionOr)},
        OpenQASMTranslationTestCase{"ConditionNotAndOr",
                                    qasm::conditionNotAndOr,
                                    MQT_NAMED_BUILDER(conditionNotAndOr)},
        OpenQASMTranslationTestCase{"ConditionBoolVariable",
                                    qasm::conditionBoolVariable,
                                    MQT_NAMED_BUILDER(conditionBoolVariable)},
        OpenQASMTranslationTestCase{"ConditionIndexedBit",
                                    qasm::conditionIndexedBit,
                                    MQT_NAMED_BUILDER(conditionIndexedBit)}));

} // namespace mqt::test::openqasm_translation
