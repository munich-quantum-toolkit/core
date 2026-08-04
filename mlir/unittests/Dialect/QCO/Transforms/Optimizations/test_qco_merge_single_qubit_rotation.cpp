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
#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
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
    SmallVector<double, 4> angles;
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
   * @brief Extract constant floating point value from a Value
   */
  static std::optional<double> toDouble(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        return floatAttr.getValueAsDouble();
      }
    }
    return std::nullopt;
  }

  /// True if `v`'s use-def cone reaches `target` (e.g. a function argument).
  static bool valueDependsOn(Value v, Value target) {
    DenseSet<Value> visited;
    SmallVector<Value> worklist{v};
    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      if (cur == target) {
        return true;
      }
      if (!visited.insert(cur).second) {
        continue;
      }
      if (Operation* def = cur.getDefiningOp()) {
        worklist.append(def->operand_begin(), def->operand_end());
      }
    }
    return false;
  }

  /// Replace the first `values.size()` function arguments with f64 constants so
  /// dynamic SSA can be constant-folded for numeric checks.
  static void bindLeadingArgs(func::FuncOp funcOp, ArrayRef<double> values) {
    OpBuilder b(funcOp.getContext());
    b.setInsertionPointToStart(&funcOp.getBody().front());
    for (auto [idx, value] : llvm::enumerate(values)) {
      Value c = arith::ConstantOp::create(b, funcOp.getLoc(),
                                          b.getF64FloatAttr(value));
      funcOp.getArgument(idx).replaceAllUsesWith(c);
    }
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
      return WalkResult::interrupt();
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
      return WalkResult::interrupt();
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
    expected = utils::normalizeAngle(expected);
    auto param = getGPhaseParam();
    if (expected == 0.0) {
      EXPECT_FALSE(param.has_value());
      return;
    }
    ASSERT_TRUE(param.has_value());
    EXPECT_NEAR(*param, expected, tolerance);
  }

  Value buildRotations(ArrayRef<RotationGate> rotations, Value& q) {
    auto qubit = q;

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
  LogicalResult testGateMerge(ArrayRef<RotationGate> rotations) {
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

// Note: All expected values are computed using the reference script
// compute_expected_merge_single_qubit_rotation.py in this directory, which uses
// SymPy's quaternion algebra:
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
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
  expectUGateParams(1., -0.570796326794897, 1.57079632679490);
  expectGPhaseParam(-0.5);
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
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
  expectUGateParams(1., 1., 0.);
  expectGPhaseParam(-0.5);
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
  expectUGateParams(1., -1.57079632679490, 2.57079632679490);
  expectGPhaseParam(-0.5);
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
  expectUGateParams(1., 0., 1.);
  expectGPhaseParam(-0.5);
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
  expectUGateParams(1., -1.57079632679490, 2.57079632679490);
  expectGPhaseParam(0.0);
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
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
TEST_F(MergeSingleQubitRotationGatesTest, mergePPGates) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::P, .angles = {1.}},
                             {.type = GateType::P, .angles = {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<POp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
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

  builder.rx(1.0, q[0]);
  builder.ry(1.0, q[1]);
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
  auto reg = builder.allocQubitRegister(1);
  auto q = reg[0];
  q = buildRotations({{.type = GateType::U, .angles = {1., 2., 3.}},
                      {.type = GateType::RX, .angles = {1.}},
                      {.type = GateType::RY, .angles = {2.}},
                      {.type = GateType::RZ, .angles = {3.}}},
                     q);
  q = builder.h(q);
  q = buildRotations({{.type = GateType::RZ, .angles = {4.}},
                      {.type = GateType::RY, .angles = {5.}},
                      {.type = GateType::RX, .angles = {6.}},
                      {.type = GateType::U, .angles = {4., 5., 6.}}},
                     q);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<UOp>(), 2);
  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
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
 * @brief Test: RZ(PI)->RY(PI)->RX(PI) should merge into U(0, 0, 0)
 */
TEST_F(MergeSingleQubitRotationGatesTest, numericalRotationIdentity) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {PI}},
                             {.type = GateType::RY, .angles = {PI}},
                             {.type = GateType::RX, .angles = {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
  expectUGateParams(0., 0., 0.);
  expectGPhaseParam(0.);
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
  expectUGateParams(0., 0., 0.);
  expectGPhaseParam(0.);
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
  expectUGateParams(0., -PI, 0.);
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
  expectUGateParams(2., -0.570796326794897, 0.570796326794897);
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
  EXPECT_EQ(countOps<GPhaseOp>(), 0);

  auto params = getUGateParams();
  ASSERT_TRUE(params.has_value());

  auto [theta, phi, lambda] = *params;
  EXPECT_FALSE(std::isnan(theta));
  EXPECT_FALSE(std::isnan(phi));
  EXPECT_FALSE(std::isnan(lambda));

  EXPECT_FALSE(getGPhaseParam().has_value());
}

/**
 * @brief Pure-Z merges must preserve RZ(a);RZ(b) ≡ U(0, a+b, 0) (up to
 * gphase).
 *
 * Fully static chains use the shared `Val<double>` merge path, so singular
 * atan2 cases and tiny beta drift cannot poison gphase or split the Z angle
 * across phi/lambda.
 */
TEST_F(MergeSingleQubitRotationGatesTest,
       mergePureZRotationsDoesNotEmitNanGPhase) {
  // Angles like 0.3 are enough for cos^2+sin^2 drift to push |beta| just
  // above eps while (x,y)≈0 if Euler extraction ran only in SSA.
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {0.3}},
                             {.type = GateType::RZ, .angles = {0.3}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);

  // RZ(0.3);RZ(0.3) → RZ(0.6) → U(0, 0.6, 0); allow tiny beta from float noise.
  expectUGateParams(/*expectedTheta=*/0., /*expectedPhi=*/0.6,
                    /*expectedLambda=*/0., /*tolerance=*/1e-6);

  auto phase = getGPhaseParam();
  ASSERT_TRUE(phase.has_value());
  EXPECT_TRUE(utils::isValidGlobalPhaseAngle(*phase));
  EXPECT_NEAR(*phase, utils::normalizeAngle(*phase), 1e-8);
}

TEST_F(MergeSingleQubitRotationGatesTest,
       mergeDynamicAngleRotationsUsesSsaPath) {
  // Pure-Z chain with unfoldable angle SSA forces Val<Value> merge (not the
  // host Val<double> path exercised by constant-angle tests above).
  // RZ(a);RZ(b) → U(0, wrap(a+b), 0) with SSA phi tied to the angle args.
  constexpr double angleA = 0.3;
  constexpr double angleB = 0.4;
  auto q = builder.allocQubitRegister(1);
  q[0] = builder.rz(angleA, q[0]);
  q[0] = builder.rz(angleB, q[0]);
  module = builder.finalize();

  auto funcOp = cast<func::FuncOp>(module->getBody()->front());
  const auto f64 = Float64Type::get(&context);
  funcOp.insertArgument(0, f64, {}, funcOp.getLoc());
  funcOp.insertArgument(1, f64, {}, funcOp.getLoc());

  SmallVector<RZOp> rzs;
  module->walk([&](RZOp op) { rzs.push_back(op); });
  ASSERT_EQ(rzs.size(), 2U);
  rzs[0].getThetaMutable().assign(funcOp.getArgument(0));
  rzs[1].getThetaMutable().assign(funcOp.getArgument(1));

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_GE(countOps<GPhaseOp>(), 1);

  UOp uOp = nullptr;
  module->walk([&](UOp op) {
    uOp = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(uOp);

  GPhaseOp gOp = nullptr;
  module->walk([&](GPhaseOp op) {
    gOp = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(gOp);

  // Still SSA in the angles / gphase before binding concrete values.
  EXPECT_FALSE(utils::valueToConstantDouble(uOp.getPhi()).has_value());
  EXPECT_TRUE(valueDependsOn(uOp.getPhi(), funcOp.getArgument(0)));
  EXPECT_TRUE(valueDependsOn(uOp.getPhi(), funcOp.getArgument(1)));
  EXPECT_FALSE(utils::valueToConstantDouble(gOp.getParameter(0)).has_value());
  EXPECT_TRUE(valueDependsOn(gOp.getParameter(0), funcOp.getArgument(0)));
  EXPECT_TRUE(valueDependsOn(gOp.getParameter(0), funcOp.getArgument(1)));

  // Guard against atan2(0,0) constant-folder NaNs on the dynamic path.
  module->walk([](arith::ConstantOp c) {
    if (auto floatAttr = dyn_cast<FloatAttr>(c.getValue())) {
      EXPECT_FALSE(std::isnan(floatAttr.getValueAsDouble()));
    }
  });

  // Bind controlled values and check the folded RZ(a);RZ(b) formulas:
  //   U(0, wrap(a+b), 0), gphase = normalize(-(phi+lambda)/2).
  bindLeadingArgs(funcOp, {angleA, angleB});
  const auto theta = utils::valueToConstantDouble(uOp.getTheta());
  const auto phi = utils::valueToConstantDouble(uOp.getPhi());
  const auto lambda = utils::valueToConstantDouble(uOp.getLambda());
  const auto phase = utils::valueToConstantDouble(gOp.getParameter(0));
  ASSERT_TRUE(theta.has_value());
  ASSERT_TRUE(phi.has_value());
  ASSERT_TRUE(lambda.has_value());
  ASSERT_TRUE(phase.has_value());
  EXPECT_NEAR(*theta, 0.0, 1e-6);
  EXPECT_NEAR(*phi, utils::normalizeAngle(angleA + angleB), 1e-6);
  EXPECT_NEAR(*lambda, 0.0, 1e-6);
  EXPECT_NEAR(*phase, utils::normalizeAngle(-(*phi + *lambda) / 2.0), 1e-6);
  EXPECT_TRUE(utils::isValidGlobalPhaseAngle(*phase));
  EXPECT_FALSE(std::isnan(*theta));
  EXPECT_FALSE(std::isnan(*phi));
  EXPECT_FALSE(std::isnan(*lambda));
  EXPECT_FALSE(std::isnan(*phase));
}

/**
 * @brief Two phase-bearing dynamic gates exercise SSA phase accumulation.
 *
 * `mergeDynamicChain` sums each gate's global-phase contribution. P(a);P(b)
 * accumulates both unfoldable angles into the emitted `gphase` SSA.
 */
TEST_F(MergeSingleQubitRotationGatesTest,
       mergeDynamicPhaseGatesAccumulatesGlobalPhase) {
  constexpr double angleA = 0.3;
  constexpr double angleB = 0.4;
  auto q = builder.allocQubitRegister(1);
  q[0] = builder.p(angleA, q[0]);
  q[0] = builder.p(angleB, q[0]);
  module = builder.finalize();

  auto funcOp = cast<func::FuncOp>(module->getBody()->front());
  const auto f64 = Float64Type::get(&context);
  funcOp.insertArgument(0, f64, {}, funcOp.getLoc());
  funcOp.insertArgument(1, f64, {}, funcOp.getLoc());

  SmallVector<POp> ps;
  module->walk([&](POp op) { ps.push_back(op); });
  ASSERT_EQ(ps.size(), 2U);
  ps[0].getThetaMutable().assign(funcOp.getArgument(0));
  ps[1].getThetaMutable().assign(funcOp.getArgument(1));

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<POp>(), 0);
  EXPECT_GE(countOps<GPhaseOp>(), 1);

  UOp uOp = nullptr;
  module->walk([&](UOp op) {
    uOp = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(uOp);

  GPhaseOp gOp = nullptr;
  module->walk([&](GPhaseOp op) {
    gOp = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(gOp);
  // Accumulated input phases depend on both dynamic P angles.
  EXPECT_TRUE(valueDependsOn(gOp.getParameter(0), funcOp.getArgument(0)));
  EXPECT_TRUE(valueDependsOn(gOp.getParameter(0), funcOp.getArgument(1)));

  // P(a);P(b) → U(0, wrap(a+b), 0) with inputPhase (a+b)/2 cancelling the U
  // intrinsic phase, so gphase folds to ~0 under controlled values.
  bindLeadingArgs(funcOp, {angleA, angleB});
  const auto theta = utils::valueToConstantDouble(uOp.getTheta());
  const auto phi = utils::valueToConstantDouble(uOp.getPhi());
  const auto lambda = utils::valueToConstantDouble(uOp.getLambda());
  const auto phase = utils::valueToConstantDouble(gOp.getParameter(0));
  ASSERT_TRUE(theta.has_value());
  ASSERT_TRUE(phi.has_value());
  ASSERT_TRUE(lambda.has_value());
  ASSERT_TRUE(phase.has_value());
  EXPECT_NEAR(*theta, 0.0, 1e-6);
  EXPECT_NEAR(*phi, utils::normalizeAngle(angleA + angleB), 1e-6);
  EXPECT_NEAR(*lambda, 0.0, 1e-6);
  EXPECT_NEAR(*phase, 0.0, 1e-6);
  EXPECT_TRUE(utils::isValidGlobalPhaseAngle(*phase));
}
