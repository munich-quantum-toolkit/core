/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <numbers>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using namespace mlir;
using namespace mlir::qco;

/// A constant for the value of \f$\pi\f$.
constexpr double PI = std::numbers::pi;

/// A constant for the value of \f$\tau\f$.
constexpr auto TAU = 2.0 * std::numbers::pi;

class QCOQuaternionMergeTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;
  OwningOpRef<ModuleOp> module;

  /**
   * @brief Struct to easily construct a rotation gate inline.
   *        opName uses the getOperationName() mnemonic.
   */
  struct RotationGate {
    llvm::StringLiteral opName;
    std::vector<double> angles;
  };

  QCOQuaternionMergeTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();

    builder.initialize();
  }

  /**
   * @brief Counts the amount of operations the current module/circuit
   *        contains.
   */
  template <typename OpTy> int countOps() {
    int count = 0;
    module->walk([&](OpTy) { ++count; });
    return count;
  }

  /**
   * @brief Extract constant floating point value from a mlir::Value
   */
  static std::optional<double> toDouble(mlir::Value v) {
    if (auto constOp = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto floatAttr =
              mlir::dyn_cast<mlir::FloatAttr>(constOp.getValue())) {
        return floatAttr.getValueAsDouble();
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Find the first occurrence of a u-gate in the current module and get
   *        the numeric value of its parameters. This assumes that parameters
   *        are constant and can be extracted.
   */
  std::optional<std::tuple<double, double, double>> getUGateParams() {
    UOp uOp = nullptr;
    module->walk([&](UOp op) {
      uOp = op;
      // stop after finding first UOp
      return mlir::WalkResult::interrupt();
    });

    if (!uOp) {
      return std::nullopt;
    }

    auto theta = toDouble(uOp.getTheta());
    auto phi = toDouble(uOp.getPhi());
    auto lambda = toDouble(uOp.getLambda());

    if (!theta || !phi || !lambda) {
      return std::nullopt;
    }

    return std::make_tuple(*theta, *phi, *lambda);
  }

  /**
   * @brief Gets the first u-gate of a module and tests whether its
   *        angle parameters are equal the the expected ones.
   */
  void expectUGateParams(double expectedTheta, double expectedPhi,
                         double expectedLambda, double tolerance = 1e-8) {
    // TODO: maybe check angle equality modulo 2*PI
    auto params = getUGateParams();
    ASSERT_TRUE(params.has_value());

    auto [theta, phi, lambda] = *params;
    EXPECT_NEAR(theta, expectedTheta, tolerance);
    EXPECT_NEAR(phi, expectedPhi, tolerance);
    EXPECT_NEAR(lambda, expectedLambda, tolerance);
  }

  Value buildRotations(const std::vector<RotationGate>& rotations, Value& q) {
    Value qubit = q;

    for (const auto& gate : rotations) {
      if (gate.opName == RXOp::getOperationName()) {
        qubit = builder.rx(gate.angles[0], qubit);
      } else if (gate.opName == RYOp::getOperationName()) {
        qubit = builder.ry(gate.angles[0], qubit);
      } else if (gate.opName == RZOp::getOperationName()) {
        qubit = builder.rz(gate.angles[0], qubit);
      } else if (gate.opName == UOp::getOperationName()) {
        qubit =
            builder.u(gate.angles[0], gate.angles[1], gate.angles[2], qubit);
      }
    }

    return qubit;
  }

  /**
   * @brief Takes a list of rotation gates (rx, ry, rz and u) and uses the
   * builder api to build a small quantum circuit, where a qubit is feed through
   * all rotations in the list.
   */
  LogicalResult testGateMerge(const std::vector<RotationGate>& rotations) {

    auto q = builder.allocQubitRegister(1);

    buildRotations(rotations, q[0]);

    module = builder.finalize();
    return runMergePass(module.get());
  }

  /**
   * @brief Adds the mergeRotationGates Pass to the current context and runs it.
   */
  static LogicalResult runMergePass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(qco::createMergeRotationGates());
    return pm.run(module);
  }
};

} // namespace

// ##################################################
// # Two Gate Merging Tests
// ##################################################

/**
 * @brief Test: RX->RY should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRXRYGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
}

/**
 * @brief Test: RX->RZ should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRXRZGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

/**
 * @brief Test: RY->RX should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRYRXGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
}

/**
 * @brief Test: RY->RZ should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRYRZGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

/**
 * @brief Test: RZ->RX should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRZRXGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
}

/**
 * @brief Test: RZ->RY should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRZRYGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
}

/**
 * @brief Test: U->U should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeUUGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {UOp::getOperationName(), {4., 5., 6.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
}

/**
 * @brief Test: U->RX should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeURXGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
}

/**
 * @brief Test: U->RY should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeURYGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
}

/**
 * @brief Test: U->RZ should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeURZGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

/**
 * @brief Test: RX->U should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRXUGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {UOp::getOperationName(), {1., 2., .3}}})
                  .succeeded());
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}

/**
 * @brief Test: RY->U should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRYUGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {UOp::getOperationName(), {1., 2., .3}}})
                  .succeeded());
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}

/**
 * @brief Test: RZ->U should merge into a single U gate
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeRZUGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {UOp::getOperationName(), {1., 2., .3}}})
                  .succeeded());
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}

// ##################################################
// # Not Merging Tests
// ##################################################

/**
 * @brief Test: RX->RX should not merge
 */
TEST_F(QCOQuaternionMergeTest, quaternionNoMergeRXRXGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 2);
}

/**
 * @brief Test: RY->RY should not merge
 */
TEST_F(QCOQuaternionMergeTest, quaternionNoMergeRYRYGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 2);
}

/**
 * @brief Test: RZ->RZ should not merge
 */
TEST_F(QCOQuaternionMergeTest, quaternionNoMergeRZRZGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 2);
}

/**
 * @brief Test: single RX should not convert to U
 */
TEST_F(QCOQuaternionMergeTest, quaternionNoMergeSingleRXGate) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}}}).succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 1);
}

/**
 * @brief Test: single RY should not convert to U
 */
TEST_F(QCOQuaternionMergeTest, quaternionNoMergeSingleRYGate) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}}}).succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 1);
}

/**
 * @brief Test: single RZ should not convert to U
 */
TEST_F(QCOQuaternionMergeTest, quaternionNoMergeSingleRZGate) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}}}).succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 1);
}

/**
 * @brief Test: Gates on different qubits should not merge
 */
TEST_F(QCOQuaternionMergeTest, dontMergeGatesFromDifferentQubits) {
  auto q = builder.allocQubitRegister(2);

  const Value qubit1 = q[0];
  const Value qubit2 = q[1];
  builder.rx(1.0, qubit1);
  builder.rx(1.0, qubit2);
  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<RXOp>(), 2);
}

/**
 * @brief Test: Non-consecutive gates should not merge
 */
TEST_F(QCOQuaternionMergeTest, dontMergeNonConsecutiveGates) {
  auto q = builder.allocQubitRegister(1);

  auto q1 = builder.rx(1.0, q[0]);
  auto q2 = builder.h(q1);
  builder.ry(1.0, q2);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<RXOp>(), 1);
  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 1);
}

// ##################################################
// # Greedy Merging Tests
// ##################################################

/**
 * @brief Test: Many gates should greedily merge into one U
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeManyGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RXOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {2.}},
                             {RZOp::getOperationName(), {3.}},
                             {UOp::getOperationName(), {4., 5., 6.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

/**
 * @brief Test: Many gates with one unmergeable in between
 * should merge into two U with the unmergeable in between.
 */
TEST_F(QCOQuaternionMergeTest, quaternionMergeManyWithUnmergeable) {
  auto q = builder.allocQubitRegister(1);
  Value qubit = buildRotations({{UOp::getOperationName(), {1., 2., .3}},
                                {RXOp::getOperationName(), {1.}},
                                {RYOp::getOperationName(), {2.}},
                                {RZOp::getOperationName(), {3.}}},
                               q[0]);
  qubit = builder.h(qubit);
  qubit = buildRotations({{RZOp::getOperationName(), {4.}},
                          {RYOp::getOperationName(), {5.}},
                          {RXOp::getOperationName(), {6.}},
                          {UOp::getOperationName(), {4., 5., 6.}}},
                         qubit);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<UOp>(), 2);
  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

// ##################################################
// # Special Cases Tests
// ##################################################

/**
 * @brief Test: Consecutive gates with another gate in between should merge
 */
TEST_F(QCOQuaternionMergeTest, mergeConsecutiveWithGateInBetween) {
  auto q = builder.allocQubitRegister(2);

  auto q1 = builder.rx(1.0, q[0]);
  builder.h(q[1]);
  builder.ry(1.0, q1);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}

/**
 * @brief Test: Gates with multiple uses should not be merged but pass should
 * still succeed
 */
TEST_F(QCOQuaternionMergeTest, nonLinearCodeHandling) {
  // QCOProgramBuilder does not allow non-linear circuits,
  // so strings are used
  const char* mlirCode = R"(
      module {
        func.func @nonLinearCodeHandling() {
          %0 = qco.alloc : !qco.qubit
          %cst = arith.constant 1.000000e+00 : f64
          %1 = qco.ry(%cst) %0 : !qco.qubit -> !qco.qubit

          // %1 is used by BOTH operations - violates linearity!
          %2 = qco.rz(%cst) %1 : !qco.qubit -> !qco.qubit
          %3 = qco.rz(%cst) %1 : !qco.qubit -> !qco.qubit

          qco.dealloc %2 : !qco.qubit
          qco.dealloc %3 : !qco.qubit
          return
        }
      }
    )";

  module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runMergePass(module.get()).succeeded());

  // Gates should remain unchanged (not merged) due to multiple uses
  EXPECT_EQ(countOps<RZOp>(), 2);
  EXPECT_EQ(countOps<RYOp>(), 1);
  EXPECT_EQ(countOps<UOp>(), 0);
}

/**
 * @brief Test: Gates with multiple uses should not be merged but pass should
 * still succeed
 */
TEST_F(QCOQuaternionMergeTest, multipleUseInIf) {
  const char* mlirCode = R"(
      module {
        func.func @scfIfTest(%cond: i1) {
          %0 = qco.alloc : !qco.qubit
          %cst = arith.constant 1.0 : f64
          %1 = qco.ry(%cst) %0 : !qco.qubit -> !qco.qubit

          // qubit %1 used in both branches - multiple uses
          %2 = scf.if %cond -> !qco.qubit {
            %t = qco.rz(%cst) %1 : !qco.qubit -> !qco.qubit
            scf.yield %t : !qco.qubit
          } else {
            %e = qco.rx(%cst) %1 : !qco.qubit -> !qco.qubit
            scf.yield %e : !qco.qubit
          }

          qco.dealloc %2 : !qco.qubit
          return
        }
      }
    )";

  context.loadDialect<scf::SCFDialect>();
  module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runMergePass(module.get()).succeeded());

  // Gates should remain unchanged (not merged) due to multiple uses
  EXPECT_EQ(countOps<RXOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 1);
  EXPECT_EQ(countOps<UOp>(), 0);
}

/**
 * @brief Test: Gates with no final users should still succeed
 *        but will be removed by dead code removal from
 *        applyPatternsGreedily
 */
TEST_F(QCOQuaternionMergeTest, noUsedGate) {
  const char* mlirCode = R"(
      module {
        func.func @noUsedGate() {
          %0 = qco.alloc : !qco.qubit
          %cst = arith.constant 1.000000e+00 : f64
          %1 = qco.ry(%cst) %0 : !qco.qubit -> !qco.qubit
          %2 = qco.rz(%cst) %1 : !qco.qubit -> !qco.qubit
          return
        }
      }
    )";

  module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runMergePass(module.get()).succeeded());

  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 0);
}

// ##################################################
// # Numerical Correctness
// ##################################################

/**
 * @brief Test: RX(1)->RY(1) should merge into
 *        U(0.495367289218673, 1.27455578230629, -1.07542903757622)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyRXRY) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);

  expectUGateParams(0.495367289218673, 1.27455578230629, -1.07542903757622);
}

/**
 * @brief Test: RX(1)->RZ(1) should merge into
 *        U(1.57079632679490, 1.00000000000000, -0.570796326794897)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyRXRZ) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);

  expectUGateParams(1.57079632679490, 1.00000000000000, -0.570796326794897);
}

/**
 * @brief Test: RY(1)->RX(1) should merge into
 *        U(1.07542903757622, 1.27455578230629, -0.495367289218673)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyRYRX) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);

  expectUGateParams(1.07542903757622, 1.27455578230629, -0.495367289218673);
}

/**
 * @brief Test: RY(1)->RZ(1) should merge into
 *        U(0, 1.00000000000000, 1.00000000000000)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyRYRZ) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);

  expectUGateParams(0, 1.00000000000000, 1.00000000000000);
}

/**
 * @brief Test: RZ(1)->RX(1) should merge into
 *        U(2.57079632679490, 1.00000000000000, -1.57079632679490)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyRZRX) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);

  expectUGateParams(2.57079632679490, 1.00000000000000, -1.57079632679490);
}

/**
 * @brief Test: RZ(1)->RY(1) should merge into
 *        U(1.00000000000000, 1.00000000000000, 0)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyRZRY) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);

  expectUGateParams(1.00000000000000, 1.00000000000000, 0);
}

/**
 * @brief Test: U(1,2,3)->U(4,5,6) should merge into
 *        U(0.154763313125030, 1.00116934013043, -5.77770904175559)
 */
TEST_F(QCOQuaternionMergeTest, numericalAccuracyUU) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., 3.}},
                             {UOp::getOperationName(), {4., 5., 6.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);

  expectUGateParams(0.154763313125030, 1.00116934013043, -5.77770904175559);
}

/**
 * @brief Test: RZ(PI)->RY(PI)->RX(PI) should merge into
 *        U(0, 0, 0) or U(2*PI, 0, 0)
 */
TEST_F(QCOQuaternionMergeTest, numericalRotationIdentity) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {PI}},
                             {RYOp::getOperationName(), {PI}},
                             {RXOp::getOperationName(), {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);

  expectUGateParams(TAU, 0., 0.);
}

/**
 * @brief Test: RY(1)->RZ(1)->RY(-1)->RZ(-1) should merge into
 *        U(0, 0, 0)
 */
TEST_F(QCOQuaternionMergeTest, numericalRotationIdentity2) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1}},
                             {RZOp::getOperationName(), {1}},
                             {RZOp::getOperationName(), {-1}},
                             {RYOp::getOperationName(), {-1}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);

  expectUGateParams(0., 0., 0.);
}

/**
 * @brief Test: RX(0.001)->RY(0.001) should merge into
 *        U(0.785397913397407, 0.00141421344452194, -0.785398413397490)
 */
TEST_F(QCOQuaternionMergeTest, numericalSmallAngles) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {0.001}},
                             {RYOp::getOperationName(), {0.001}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);

  expectUGateParams(0.785397913397407, 0.00141421344452194, -0.785398413397490);
}

/**
 * @brief Test: RX(PI)->RY(PI) should merge into
 *        U(-PI, 0, 0)
 */
TEST_F(QCOQuaternionMergeTest, numericalGimbalLock) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {PI}},
                             {RYOp::getOperationName(), {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);

  expectUGateParams(-PI, 0., 0.);
}
