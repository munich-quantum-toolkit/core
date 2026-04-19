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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::qco;

/// A constant for the value of \f$\pi\f$.
constexpr double PI = std::numbers::pi;

class MergeSingleQubitRotationGatesTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;
  OwningOpRef<ModuleOp> module;

  enum class GateType : std::uint8_t { RX, RY, RZ, P, R, U2, U };
  /**
   * @brief Struct to easily construct a rotation gate inline.
   *        opName uses the getOperationName() mnemonic.
   */
  struct RotationGate {
    GateType type;
    llvm::SmallVector<double, 4> angles;
  };

  MergeSingleQubitRotationGatesTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();

    builder.initialize();
  }

  /**
   * @brief Counts the amount of operations the current module/circuit contains.
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
   * the numeric value of its parameters. This assumes that parameters are
   * constant and can be extracted.
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
   * @brief Gets the first u-gate of a module and tests whether its angle
   * parameters are equal to the expected ones.
   */
  void expectUGateParams(double expectedTheta, double expectedPhi,
                         double expectedLambda, double tolerance = 1e-8) {
    auto params = getUGateParams();
    ASSERT_TRUE(params.has_value());

    auto [theta, phi, lambda] = *params;
    EXPECT_NEAR(theta, expectedTheta, tolerance);
    EXPECT_NEAR(phi, expectedPhi, tolerance);
    EXPECT_NEAR(lambda, expectedLambda, tolerance);
  }

  /**
   * @brief Find the first occurrence of a gphase op in the current module and
   * get the numeric value of its parameter.
   */
  std::optional<double> getGPhaseParam() {
    GPhaseOp gOp = nullptr;
    module->walk([&](GPhaseOp op) {
      gOp = op;
      return mlir::WalkResult::interrupt();
    });

    if (!gOp) {
      return std::nullopt;
    }

    return toDouble(gOp.getParameter(0));
  }

  /**
   * @brief Gets the first gphase op of a module and tests whether its angle
   * parameter is equal to the expected one.
   */
  void expectGPhaseParam(double expected, double tolerance = 1e-8) {
    auto param = getGPhaseParam();
    ASSERT_TRUE(param.has_value());
    EXPECT_NEAR(*param, expected, tolerance);
  }

  Value buildRotations(llvm::ArrayRef<RotationGate> rotations, Value& q) {
    Value qubit = q;

    for (const auto& gate : rotations) {
      switch (gate.type) {
      case GateType::RX:
        assert(gate.angles.size() == 1 && "RXOp requires 1 angle parameter");
        qubit = builder.rx(gate.angles[0], qubit);
        break;
      case GateType::RY:
        assert(gate.angles.size() == 1 && "RYOp requires 1 angle parameter");
        qubit = builder.ry(gate.angles[0], qubit);
        break;
      case GateType::RZ:
        assert(gate.angles.size() == 1 && "RZOp requires 1 angle parameter");
        qubit = builder.rz(gate.angles[0], qubit);
        break;
      case GateType::P:
        assert(gate.angles.size() == 1 && "POp requires 1 angle parameter");
        qubit = builder.p(gate.angles[0], qubit);
        break;
      case GateType::R:
        assert(gate.angles.size() == 2 && "ROp requires 2 angle parameters");
        qubit = builder.r(gate.angles[0], gate.angles[1], qubit);
        break;
      case GateType::U2:
        assert(gate.angles.size() == 2 && "U2Op requires 2 angle parameters");
        qubit = builder.u2(gate.angles[0], gate.angles[1], qubit);
        break;
      case GateType::U:
        assert(gate.angles.size() == 3 && "UOp requires 3 angle parameters");
        qubit =
            builder.u(gate.angles[0], gate.angles[1], gate.angles[2], qubit);
        break;
      }
    }

    return qubit;
  }

  /**
   * @brief Takes a list of rotation gates (rx, ry, rz and u) and uses the
   * builder api to build a small quantum circuit, where a qubit is fed through
   * all rotations in the list.
   */
  LogicalResult testGateMerge(llvm::ArrayRef<RotationGate> rotations) {
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
    pm.addPass(qco::createMergeSingleQubitRotationGates());
    return pm.run(module);
  }
};

} // namespace

// Note: All expected values are computed using SymPy's quaternion algebra:
// https://docs.sympy.org/latest/modules/algebras.html#module-sympy.algebras.Quaternion

// ##################################################
// # Two Gate Merging Tests
// ##################################################

/**
 * @brief Test: RX->RX should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRXRXGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {1.}},
                             {.type = GateType::RX, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: RX->RY should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRXRYGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {1.}},
                             {.type = GateType::RY, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.27455578230629, -1.07542903757622, 0.495367289218673);
  expectGPhaseParam(0.290030874178775);
}

/**
 * @brief Test: RX->RZ should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRXRZGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {1.}},
                             {.type = GateType::RZ, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.00000000000000, -0.570796326794897, 1.57079632679490);
  expectGPhaseParam(-0.500000000000000);
}

/**
 * @brief Test: RY->RX should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRYRXGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RY, .angles = {1.}},
                             {.type = GateType::RX, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.27455578230629, -0.495367289218673, 1.07542903757622);
  expectGPhaseParam(-0.290030874178775);
}

/**
 * @brief Test: RY->RY should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRYRYGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RY, .angles = {1.}},
                             {.type = GateType::RY, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: RY->RZ should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRYRZGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RY, .angles = {1.}},
                             {.type = GateType::RZ, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.00000000000000, 1.00000000000000, 0.);
  expectGPhaseParam(-0.500000000000000);
}

/**
 * @brief Test: RZ->RX should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRZRXGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {1.}},
                             {.type = GateType::RX, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.00000000000000, -1.57079632679490, 2.57079632679490);
  expectGPhaseParam(-0.500000000000000);
}

/**
 * @brief Test: RZ->RY should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRZRYGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {1.}},
                             {.type = GateType::RY, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.00000000000000, 0., 1.00000000000000);
  expectGPhaseParam(-0.500000000000000);
}

/**
 * @brief Test: RZ->RZ should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRZRZGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {1.}},
                             {.type = GateType::RZ, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: U->U should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeUUGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U, .angles = {1., 2., 3.}},
                             {.type = GateType::U, .angles = {4., 5., 6.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(2.03289042623884, 0.663830775701153, 0.849231441867857);
  expectGPhaseParam(7.243468891215494);
}

/**
 * @brief Test: U->RX should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeURXGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U, .angles = {1., 2., 3.}},
                             {.type = GateType::RX, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: U->RY should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeURYGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U, .angles = {1., 2., 3.}},
                             {.type = GateType::RY, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: U->RZ should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeURZGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U, .angles = {1., 2., 3.}},
                             {.type = GateType::RZ, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: RX->U should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRXUGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {1.}},
                             {.type = GateType::U, .angles = {1., 2., 3.}}})
                  .succeeded());
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: RY->U should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRYUGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RY, .angles = {1.}},
                             {.type = GateType::U, .angles = {1., 2., 3.}}})
                  .succeeded());
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: RZ->U should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRZUGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {1.}},
                             {.type = GateType::U, .angles = {1., 2., 3.}}})
                  .succeeded());
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}
/**
 * @brief Test: P->RX should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergePRXGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::P, .angles = {1.}},
                             {.type = GateType::RX, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<POp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.00000000000000, -1.57079632679490, 2.57079632679490);
  expectGPhaseParam(1.11022302462516e-16);
}

/**
 * @brief Test: P->RY should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergePRYGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::P, .angles = {1.}},
                             {.type = GateType::RY, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<POp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: P->U should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergePUGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::P, .angles = {1.}},
                             {.type = GateType::U, .angles = {1., 2., 3.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<POp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: R->RX should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRRXGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::R, .angles = {1., 1.}},
                             {.type = GateType::RX, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<ROp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: P->P should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, noMergePPGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::P, .angles = {1.}},
                             {.type = GateType::P, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<POp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: R->R should merge into a single U gate (same multi-parameter
 * type always uses quaternion merge)
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeRRGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::R, .angles = {1., 2.}},
                             {.type = GateType::R, .angles = {3., 4.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<ROp>(), 0);
  expectUGateParams(2.07770669385131, 1.36334275733332, 2.85969871348886);
  expectGPhaseParam(-2.1115207354110845);
}

/**
 * @brief Test: U2->U should merge into a single U gate
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeU2UGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U2, .angles = {1., 2.}},
                             {.type = GateType::U, .angles = {1., 2., 3.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<U2Op>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: U2->U2 should merge into a single U gate (same multi-parameter
 * type always uses quaternion merge)
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeU2U2Gates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U2, .angles = {1., 2.}},
                             {.type = GateType::U2, .angles = {3., 4.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<U2Op>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(1.85840734641021, 1.42920367320511, 0.429203673205103);
  expectGPhaseParam(4.070796326794897);
}

// ##################################################
// # Not Merging Tests
// ##################################################

/**
 * @brief Test: single RX should not convert to U
 */
TEST_F(MergeSingleQubitRotationGatesTest, noMergeSingleRXGate) {
  ASSERT_TRUE(
      testGateMerge({{.type = GateType::RX, .angles = {1.}}}).succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

/**
 * @brief Test: single RY should not convert to U
 */
TEST_F(MergeSingleQubitRotationGatesTest, noMergeSingleRYGate) {
  ASSERT_TRUE(
      testGateMerge({{.type = GateType::RY, .angles = {1.}}}).succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

/**
 * @brief Test: single RZ should not convert to U
 */
TEST_F(MergeSingleQubitRotationGatesTest, noMergeSingleRZGate) {
  ASSERT_TRUE(
      testGateMerge({{.type = GateType::RZ, .angles = {1.}}}).succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

/**
 * @brief Test: Gates on different qubits should not merge
 */
TEST_F(MergeSingleQubitRotationGatesTest, dontMergeGatesFromDifferentQubits) {
  auto q = builder.allocQubitRegister(2);

  const Value qubit1 = q[0];
  const Value qubit2 = q[1];
  builder.rx(1.0, qubit1);
  builder.ry(1.0, qubit2);
  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<RXOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

/**
 * @brief Test: Non-consecutive gates should not merge
 */
TEST_F(MergeSingleQubitRotationGatesTest, dontMergeNonConsecutiveGates) {
  auto q = builder.allocQubitRegister(1);

  auto q1 = builder.rx(1.0, q[0]);
  auto q2 = builder.h(q1);
  builder.ry(1.0, q2);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<RXOp>(), 1);
  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

// ##################################################
// # Greedy Merging Tests
// ##################################################

/**
 * @brief Test: Many gates should greedily merge into one U
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeManyGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::U, .angles = {1., 2., 3.}},
                             {.type = GateType::RX, .angles = {1.}},
                             {.type = GateType::RY, .angles = {2.}},
                             {.type = GateType::RZ, .angles = {3.}},
                             {.type = GateType::U, .angles = {4., 5., 6.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

/**
 * @brief Test: Many gates with one unmergeable in between should merge into two
 * U with the unmergeable in between.
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeManyWithUnmergeable) {
  auto q = builder.allocQubitRegister(1);
  Value qubit = buildRotations({{.type = GateType::U, .angles = {1., 2., 3.}},
                                {.type = GateType::RX, .angles = {1.}},
                                {.type = GateType::RY, .angles = {2.}},
                                {.type = GateType::RZ, .angles = {3.}}},
                               q[0]);
  qubit = builder.h(qubit);
  qubit = buildRotations({{.type = GateType::RZ, .angles = {4.}},
                          {.type = GateType::RY, .angles = {5.}},
                          {.type = GateType::RX, .angles = {6.}},
                          {.type = GateType::U, .angles = {4., 5., 6.}}},
                         qubit);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<UOp>(), 2);
  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 2);
}

// ##################################################
// # Special Cases Tests
// ##################################################

/**
 * @brief Test: Consecutive gates with another gate in between should merge
 */
TEST_F(MergeSingleQubitRotationGatesTest, mergeConsecutiveWithGateInBetween) {
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
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

// ##################################################
// # Numerical Correctness
// ##################################################

/**
 * @brief Test: RZ(PI)->RY(PI)->RX(PI) should merge into U(0, 0, 0) or U(0, 2 *
 * PI, 0)
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalRotationIdentity) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {PI}},
                             {.type = GateType::RY, .angles = {PI}},
                             {.type = GateType::RX, .angles = {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(0, 0, 0.);
  expectGPhaseParam(0);
}

/**
 * @brief Test: RY(1)->RZ(1)->RZ(-1)->RY(-1) should merge into U(0, 0, 0)
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalRotationIdentity2) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RY, .angles = {1}},
                             {.type = GateType::RZ, .angles = {1}},
                             {.type = GateType::RZ, .angles = {-1}},
                             {.type = GateType::RY, .angles = {-1}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(0., 0., 0.);
  expectGPhaseParam(0.0);
}

/**
 * @brief Test: RX(0.001)->RY(0.001) should merge into U(0.00141421344452194,
 * -0.785398413397490, 0.785397913397407)
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalSmallAngles) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {0.001}},
                             {.type = GateType::RY, .angles = {0.001}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(0.00141421344452194, -0.785398413397490, 0.785397913397407);
  expectGPhaseParam(2.50000041668308e-7);
}

/**
 * @brief Test: RX(PI)->RY(PI) should merge into U(0, -PI, 0.)
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalGimbalLock) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {PI}},
                             {.type = GateType::RY, .angles = {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(0, -PI, 0.);
  expectGPhaseParam(1.57079632679490);
}

/**
 * @brief Test: R(1,1)->R(1,1) (same axis) should merge into U(2.00000000000000,
 * -0.570796326794897, 0.570796326794897)
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalAccuracyRRSameAxis) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::R, .angles = {1., 1.}},
                             {.type = GateType::R, .angles = {1., 1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<ROp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  expectUGateParams(2.00000000000000, -0.570796326794897, 0.570796326794897);
  expectGPhaseParam(0.0);
}

/**
 * @brief Test: U(0, -2.0360075460227076, 0)->U(0, 4.157656961105587, 0) should
 * not produce NaN. These specific numbers would produce NaN if acos parameter
 * would not be clamped to [-1, 1]
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalAcosClampingPreventsNaN) {
  ASSERT_TRUE(testGateMerge(
                  {{.type = GateType::U, .angles = {0, -2.0360075460227076, 0}},
                   {.type = GateType::U, .angles = {0, 4.157656961105587, 0}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);

  auto params = getUGateParams();
  ASSERT_TRUE(params.has_value());

  auto [theta, phi, lambda] = *params;
  EXPECT_FALSE(std::isnan(theta));
  EXPECT_FALSE(std::isnan(phi));
  EXPECT_FALSE(std::isnan(lambda));

  auto gphase = getGPhaseParam();
  ASSERT_TRUE(gphase.has_value());
  EXPECT_FALSE(std::isnan(*gphase));
}
