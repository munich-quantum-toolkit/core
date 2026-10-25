/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "VerificationTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <optional>

using namespace mlir;
using namespace mlir::qco;

/// Build an ordinary function whose quantum results remain observable.
template <typename Build>
static OwningOpRef<ModuleOp> buildQuantumFunction(MLIRContext* context,
                                                  Build build) {
  QCOProgramBuilder builder(context);
  builder.initialize();
  auto function = builder.createFunction(
      "test", {}, [&](ValueRange) { return build(builder); });
  function.setPublic();
  auto moduleOp = builder.finalize();
  moduleOp->lookupSymbol<func::FuncOp>("main").erase();
  function.setName("main");
  return moduleOp;
}

namespace {

/// A constant for the value of π.
constexpr double PI = std::numbers::pi;

enum class GateType : uint8_t { RX, RY, RZ, P, R, U2, U };

struct RotationGate {
  GateType type;
  SmallVector<double, 4> angles;
};

struct RotationPair {
  const char* name;
  RotationGate first;
  RotationGate second;
  int phaseCount;
};

class MergeSingleQubitRotationGatesTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;
  OwningOpRef<ModuleOp> module;

  MergeSingleQubitRotationGatesTest() : builder(&context) {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();

    builder.initialize();
  }

  /// Counts the amount of operations the current module/circuit contains.
  template <typename OpTy> int countOps() {
    int count = 0;
    module->walk([&](OpTy) { ++count; });
    return count;
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

  Value buildRotations(ArrayRef<RotationGate> rotations, Value q) {
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

  /// Builds a circuit with the given rotations and checks the merge pass.
  ///
  /// Feeds a qubit through every rotation (rx, ry, rz and u) in the list.
  LogicalResult testGateMerge(ArrayRef<RotationGate> rotations) {
    auto q = builder.staticQubit(0);

    q = buildRotations(rotations, q);
    builder.sink(q);

    module = builder.finalize();
    OwningOpRef<ModuleOp> original = module->clone();
    const auto result = runMergePass(*module);
    if (succeeded(result)) {
      ::mqt::test::expectUnitaryEquivalent(*original, *module, 1);
    }
    return result;
  }

  /// Adds the mergeRotationGates Pass to the current context and runs it.
  static LogicalResult runMergePass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(qco::createMergeSingleQubitRotationGates());
    return pm.run(module);
  }
};

enum class FixedGateType : std::uint8_t {
  X,
  Y,
  Z,
  H,
  S,
  Sdg,
  T,
  Tdg,
  SX,
  SXdg,
  Id,
};

class MergeRotationPairTest : public MergeSingleQubitRotationGatesTest,
                              public testing::WithParamInterface<RotationPair> {
};

class SingleRotationTest : public MergeSingleQubitRotationGatesTest,
                           public testing::WithParamInterface<GateType> {};

class MergeFixedSingleQubitGateTest
    : public MergeSingleQubitRotationGatesTest,
      public testing::WithParamInterface<FixedGateType> {};

} // namespace

TEST_P(MergeRotationPairTest, PreservesMatrixAndMergesToOneGate) {
  const auto& pair = GetParam();
  ASSERT_TRUE(succeeded(testGateMerge({pair.first, pair.second})));
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), pair.phaseCount);
  EXPECT_EQ(countOps<UnitaryOpInterface>(), 1 + pair.phaseCount);
}

INSTANTIATE_TEST_SUITE_P(
    RotationPairs, MergeRotationPairTest,
    testing::ValuesIn(std::array<RotationPair, 24>{{
        {"RXRX", {GateType::RX, {1.}}, {GateType::RX, {1.}}, 0},
        {"RXRY", {GateType::RX, {1.}}, {GateType::RY, {1.}}, 1},
        {"RXRZ", {GateType::RX, {1.}}, {GateType::RZ, {1.}}, 1},
        {"RYRX", {GateType::RY, {1.}}, {GateType::RX, {1.}}, 1},
        {"RYRY", {GateType::RY, {1.}}, {GateType::RY, {1.}}, 0},
        {"RYRZ", {GateType::RY, {1.}}, {GateType::RZ, {1.}}, 1},
        {"RZRX", {GateType::RZ, {1.}}, {GateType::RX, {1.}}, 1},
        {"RZRY", {GateType::RZ, {1.}}, {GateType::RY, {1.}}, 1},
        {"RZRZ", {GateType::RZ, {1.}}, {GateType::RZ, {1.}}, 1},
        {"UU", {GateType::U, {1., 2., 3.}}, {GateType::U, {4., 5., 6.}}, 1},
        {"URX", {GateType::U, {1., 2., 3.}}, {GateType::RX, {1.}}, 1},
        {"URY", {GateType::U, {1., 2., 3.}}, {GateType::RY, {1.}}, 1},
        {"URZ", {GateType::U, {1., 2., 3.}}, {GateType::RZ, {1.}}, 1},
        {"RXU", {GateType::RX, {1.}}, {GateType::U, {1., 2., 3.}}, 1},
        {"RYU", {GateType::RY, {1.}}, {GateType::U, {1., 2., 3.}}, 1},
        {"RZU", {GateType::RZ, {1.}}, {GateType::U, {1., 2., 3.}}, 1},
        {"PRX", {GateType::P, {1.}}, {GateType::RX, {1.}}, 0},
        {"PRY", {GateType::P, {1.}}, {GateType::RY, {1.}}, 0},
        {"PU", {GateType::P, {1.}}, {GateType::U, {1., 2., 3.}}, 0},
        {"RRX", {GateType::R, {1., 1.}}, {GateType::RX, {1.}}, 1},
        {"PP", {GateType::P, {1.}}, {GateType::P, {1.}}, 0},
        {"RR", {GateType::R, {1., 2.}}, {GateType::R, {3., 4.}}, 1},
        {"U2U", {GateType::U2, {1., 2.}}, {GateType::U, {1., 2., 3.}}, 1},
        {"U2U2", {GateType::U2, {1., 2.}}, {GateType::U2, {3., 4.}}, 1},
    }}),
    [](const testing::TestParamInfo<RotationPair>& info) {
      return info.param.name;
    });

TEST_P(SingleRotationTest, LeavesSingleGateUnchanged) {
  const auto gate = GetParam();
  ASSERT_TRUE(succeeded(testGateMerge({{gate, {1.}}})));
  EXPECT_EQ(countOps<UnitaryOpInterface>(), 1);
  EXPECT_EQ(countOps<RXOp>(), gate == GateType::RX ? 1 : 0);
  EXPECT_EQ(countOps<RYOp>(), gate == GateType::RY ? 1 : 0);
  EXPECT_EQ(countOps<RZOp>(), gate == GateType::RZ ? 1 : 0);
}

INSTANTIATE_TEST_SUITE_P(RotationAxes, SingleRotationTest,
                         testing::Values(GateType::RX, GateType::RY,
                                         GateType::RZ));

// Test: Gates on different qubits should not merge
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

TEST_P(MergeFixedSingleQubitGateTest, PreservesMatrix) {
  auto q = builder.staticQubit(0);
  switch (GetParam()) {
  case FixedGateType::X:
    q = builder.x(q);
    break;
  case FixedGateType::Y:
    q = builder.y(q);
    break;
  case FixedGateType::Z:
    q = builder.z(q);
    break;
  case FixedGateType::H:
    q = builder.h(q);
    break;
  case FixedGateType::S:
    q = builder.s(q);
    break;
  case FixedGateType::Sdg:
    q = builder.sdg(q);
    break;
  case FixedGateType::T:
    q = builder.t(q);
    break;
  case FixedGateType::Tdg:
    q = builder.tdg(q);
    break;
  case FixedGateType::SX:
    q = builder.sx(q);
    break;
  case FixedGateType::SXdg:
    q = builder.sxdg(q);
    break;
  case FixedGateType::Id:
    q = builder.id(q);
    break;
  }
  q = builder.rx(0.37, q);
  builder.sink(q);
  module = builder.finalize();

  OwningOpRef<ModuleOp> original = module->clone();
  for (const bool symbolic : {false, true}) {
    SCOPED_TRACE(symbolic);
    module = original->clone();
    auto funcOp = cast<func::FuncOp>(module->getBody()->front());
    if (symbolic) {
      ASSERT_TRUE(succeeded(funcOp.insertArgument(0, Float64Type::get(&context),
                                                  {}, funcOp.getLoc())));
      module->walk(
          [&](RXOp op) { op.getThetaMutable().assign(funcOp.getArgument(0)); });
    }
    ASSERT_TRUE(runMergePass(*module).succeeded());
    EXPECT_EQ(countOps<IdOp>(), 0);
    if (GetParam() == FixedGateType::Id) {
      // Folding the identity leaves a single RX, which does not need merging.
      EXPECT_EQ(countOps<UOp>(), 0);
      EXPECT_EQ(countOps<RXOp>(), 1);
    } else {
      EXPECT_EQ(countOps<UOp>(), 1);
      EXPECT_EQ(countOps<RXOp>(), 0);
    }
    if (symbolic) {
      bindLeadingArgs(funcOp, {0.37});
      PassManager canonicalizer(&context);
      canonicalizer.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
    }
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_TRUE(succeeded(verifyLinearity(*module)));
    ::mqt::test::expectUnitaryEquivalent(*original, *module, 1);
  }
}

INSTANTIATE_TEST_SUITE_P(AllFixedGates, MergeFixedSingleQubitGateTest,
                         testing::Values(FixedGateType::X, FixedGateType::Y,
                                         FixedGateType::Z, FixedGateType::H,
                                         FixedGateType::S, FixedGateType::Sdg,
                                         FixedGateType::T, FixedGateType::Tdg,
                                         FixedGateType::SX, FixedGateType::SXdg,
                                         FixedGateType::Id));

// Test: Gates separated by a barrier should not merge.
TEST_F(MergeSingleQubitRotationGatesTest, dontMergeAcrossBarrier) {
  auto q = builder.allocQubitRegister(1);

  auto q1 = builder.rx(1.0, q[0]);
  auto q2 = builder.barrier({q1})[0];
  builder.ry(1.0, q2);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<RXOp>(), 1);
  EXPECT_EQ(countOps<BarrierOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

// ##################################################
// # Greedy Merging Tests
// ##################################################

// Test: Many gates should greedily merge into one U
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

// Test: Many gates with one unmergeable in between should merge into two
// U with the unmergeable in between.
TEST_F(MergeSingleQubitRotationGatesTest, mergeManyWithUnmergeable) {
  auto reg = builder.allocQubitRegister(1);
  auto q = reg[0];
  q = buildRotations(
      {
          {.type = GateType::U, .angles = {1., 2., 3.}},
          {.type = GateType::RX, .angles = {1.}},
          {.type = GateType::RY, .angles = {2.}},
          {.type = GateType::RZ, .angles = {3.}},
      },
      q);
  q = builder.barrier({q})[0];
  q = buildRotations(
      {
          {.type = GateType::RZ, .angles = {4.}},
          {.type = GateType::RY, .angles = {5.}},
          {.type = GateType::RX, .angles = {6.}},
          {.type = GateType::U, .angles = {4., 5., 6.}},
      },
      q);

  module = builder.finalize();

  ASSERT_TRUE(runMergePass(module.get()).succeeded());
  EXPECT_EQ(countOps<UOp>(), 2);
  EXPECT_EQ(countOps<BarrierOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

// ##################################################
// # Special Cases Tests
// ##################################################

// Test: Consecutive gates with another gate in between should merge
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

// RZ(pi); RY(pi); RX(pi) needs only its pi global phase.
TEST_F(MergeSingleQubitRotationGatesTest, numericalRotationIdentity) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {PI}},
                             {.type = GateType::RY, .angles = {PI}},
                             {.type = GateType::RX, .angles = {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

// An identity chain needs no remaining gate.
TEST_F(MergeSingleQubitRotationGatesTest, numericalRotationIdentity2) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RY, .angles = {1}},
                             {.type = GateType::RZ, .angles = {1}},
                             {.type = GateType::RZ, .angles = {-1}},
                             {.type = GateType::RY, .angles = {-1}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

TEST_F(MergeSingleQubitRotationGatesTest, numericalSmallAngles) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {0.001}},
                             {.type = GateType::RY, .angles = {0.001}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

TEST_F(MergeSingleQubitRotationGatesTest,
       preservesSmallRotationsAndSingularPhasesForConstantAndRuntimeAngles) {
  constexpr std::array<std::array<double, 2>, 8> parameterSets{
      {
          {1e-10, 1e-10},
          {1e-8, 1e-8},
          {1e-7, 1e-7},
          {PI, PI},
          {-PI, PI},
          {PI - 1e-8, 1e-8},
          {-PI + 1e-8, 1e-8},
          {PI, 0.3},
      },
  };
  for (unsigned runtimeArguments : {0U, 1U, 2U}) {
    SCOPED_TRACE(runtimeArguments);
    for (const auto& angles : parameterSets) {
      if (runtimeArguments == 1 && angles[0] != angles[1]) {
        continue;
      }
      SCOPED_TRACE(testing::Message() << angles[0] << ", " << angles[1]);
      QCOProgramBuilder circuitBuilder(&context);
      circuitBuilder.initialize();
      Value qubit = circuitBuilder.staticQubit(0);
      qubit = circuitBuilder.rx(angles[0], qubit);
      qubit = circuitBuilder.ry(angles[1], qubit);
      circuitBuilder.sink(qubit);
      module = circuitBuilder.finalize();
      auto funcOp = module->lookupSymbol<func::FuncOp>("main");
      ASSERT_TRUE(funcOp);
      if (runtimeArguments != 0) {
        for (unsigned i = 0; i < runtimeArguments; ++i) {
          ASSERT_TRUE(succeeded(funcOp.insertArgument(
              i, Float64Type::get(&context), {}, funcOp.getLoc())));
        }
        funcOp.walk([&](RXOp op) {
          op.getThetaMutable().assign(funcOp.getArgument(0));
        });
        funcOp.walk([&](RYOp op) {
          op.getThetaMutable().assign(funcOp.getArgument(runtimeArguments - 1));
        });
      }
      ASSERT_TRUE(succeeded(verify(*module)));
      OwningOpRef<ModuleOp> original = module->clone();
      ASSERT_TRUE(succeeded(runMergePass(*module)));
      ASSERT_TRUE(succeeded(verify(*module)));
      EXPECT_EQ(countOps<UOp>(), 1);
      if (runtimeArguments != 0) {
        auto values = ArrayRef(angles).take_front(runtimeArguments);
        bindLeadingArgs(original->lookupSymbol<func::FuncOp>("main"), values);
        bindLeadingArgs(funcOp, values);
        PassManager pm(&context);
        pm.addPass(createCanonicalizerPass());
        ASSERT_TRUE(succeeded(pm.run(*module)));
      }
      ASSERT_TRUE(succeeded(verify(*original)));
      ASSERT_TRUE(succeeded(verify(*module)));
      ::mqt::test::expectUnitaryEquivalent(*original, *module, 1);
    }
  }
}

TEST_F(MergeSingleQubitRotationGatesTest, largePhasesPreserveControlledMatrix) {
  for (const auto gate : {GateType::P, GateType::U2, GateType::U}) {
    for (const auto angles : {
             std::array{1e12, 1.0},
             std::array{-1e16, 1.0},
             std::array{1e308, 1e308},
         }) {
      for (const bool dynamic : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "gate=" << static_cast<unsigned>(gate)
                     << " phi=" << angles[0] << " lambda=" << angles[1]
                     << " dynamic=" << dynamic);
        module = buildQuantumFunction(&context, [&](auto& b) {
          auto [control, target] =
              b.ctrl(b.staticQubit(0), b.staticQubit(1), [&](Value qubit) {
                if (gate == GateType::P) {
                  qubit = b.p(angles[0], qubit);
                } else if (gate == GateType::U2) {
                  qubit = b.u2(angles[0], angles[1], qubit);
                } else {
                  qubit = b.u(0.37, angles[0], angles[1], qubit);
                }
                return b.h(qubit);
              });
          return SmallVector<Value>{control, target};
        });
        ASSERT_TRUE(module);
        auto funcOp = module->lookupSymbol<func::FuncOp>("main");
        if (dynamic) {
          ASSERT_TRUE(succeeded(funcOp.insertArgument(
              0, Float64Type::get(&context), {}, funcOp.getLoc())));
          module->walk([&](UnitaryOpInterface op) {
            if (isa<POp, U2Op, UOp>(op.getOperation())) {
              Value parameter = op.getParameter(isa<UOp>(op) ? 1U : 0U);
              parameter.replaceAllUsesWith(funcOp.getArgument(0));
            }
          });
        }
        ASSERT_TRUE(succeeded(verify(*module)));
        ASSERT_TRUE(succeeded(verifyLinearity(*module)));
        OwningOpRef<ModuleOp> original = module->clone();
        ASSERT_TRUE(succeeded(runMergePass(*module)));
        ASSERT_TRUE(succeeded(verify(*module)));
        ASSERT_TRUE(succeeded(verifyLinearity(*module)));
        EXPECT_EQ(countOps<HOp>(), 0);
        if (dynamic) {
          bindLeadingArgs(original->lookupSymbol<func::FuncOp>("main"),
                          {angles[0]});
          bindLeadingArgs(funcOp, {angles[0]});
          PassManager pm(&context);
          pm.addPass(createCanonicalizerPass());
          ASSERT_TRUE(succeeded(pm.run(*module)));
        }
        ASSERT_TRUE(succeeded(verify(*module)));
        ASSERT_TRUE(succeeded(verifyLinearity(*module)));
        ::mqt::test::expectUnitaryEquivalent(*original, *module, 2);
      }
    }
  }
}

TEST_F(MergeSingleQubitRotationGatesTest, numericalGimbalLock) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::RX, .angles = {PI}},
                             {.type = GateType::RY, .angles = {PI}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 1);
}

TEST_F(MergeSingleQubitRotationGatesTest, numericalAccuracyRRSameAxis) {
  ASSERT_TRUE(testGateMerge({{.type = GateType::R, .angles = {1., 1.}},
                             {.type = GateType::R, .angles = {1., 1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<ROp>(), 0);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);
}

// Pure-Z composition must keep finite Euler angles despite quaternion norm
// roundoff.
TEST_F(MergeSingleQubitRotationGatesTest,
       numericalNormRoundoffDoesNotProduceNaN) {
  ASSERT_TRUE(testGateMerge(
                  {{.type = GateType::U, .angles = {0, -2.0360075460227076, 0}},
                   {.type = GateType::U, .angles = {0, 4.157656961105587, 0}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<GPhaseOp>(), 0);

  module->walk([](UOp op) {
    for (Value parameter : op.getParameters()) {
      const auto value = mlir::mqt::valueToDouble(parameter);
      ASSERT_TRUE(value);
      EXPECT_TRUE(std::isfinite(*value));
    }
  });
}

// Pure-Z merges must preserve RZ(a);RZ(b) ≡ U(0, a+b, 0) (up to
// gphase).
// Constant chains use numerical Euler synthesis; singularities must not
// introduce nonfinite parameters or phases.
TEST_F(MergeSingleQubitRotationGatesTest,
       mergePureZRotationsDoesNotEmitNanGPhase) {
  // Angles like 0.3 are enough for cos^2+sin^2 drift to push |beta| just
  // above eps while (x,y)≈0 if Euler extraction ran only in SSA.
  ASSERT_TRUE(testGateMerge({{.type = GateType::RZ, .angles = {0.3}},
                             {.type = GateType::RZ, .angles = {0.3}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);

  EXPECT_EQ(countOps<GPhaseOp>(), 1);
  module->walk([](GPhaseOp op) {
    const auto phase = mlir::mqt::valueToDouble(op.getParameter(0));
    ASSERT_TRUE(phase);
    EXPECT_TRUE(mlir::mqt::isValidGlobalPhaseAngle(*phase));
    EXPECT_NEAR(*phase, mlir::mqt::normalizeAngle(*phase), 1e-8);
  });
}

TEST_F(MergeSingleQubitRotationGatesTest,
       mergesSymbolicEulerChainsWithBoundedAngles) {
  for (const bool useX : {false, true}) {
    SCOPED_TRACE(useX);
    module = buildQuantumFunction(&context, [&](auto& b) {
      auto [control, target] =
          b.ctrl(b.staticQubit(0), b.staticQubit(1), [&](Value qubit) {
            qubit = b.rz(0.1, qubit);
            qubit = useX ? b.rx(0.2, qubit) : b.ry(0.2, qubit);
            return b.rz(0.4, qubit);
          });
      return SmallVector<Value>{control, target};
    });
    auto funcOp = module->lookupSymbol<func::FuncOp>("main");
    module->walk([&](UnitaryOpInterface op) {
      if (isa<RXOp, RYOp, RZOp>(op.getOperation())) {
        const auto index = funcOp.getNumArguments();
        ASSERT_TRUE(succeeded(funcOp.insertArgument(
            index, Float64Type::get(&context), {}, funcOp.getLoc())));
        op.getParameter(0).replaceAllUsesWith(funcOp.getArgument(index));
      }
    });
    ASSERT_EQ(funcOp.getNumArguments(), 3U);
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_TRUE(succeeded(verifyLinearity(*module)));
    OwningOpRef<ModuleOp> original = module->clone();
    ASSERT_TRUE(succeeded(runMergePass(*module)));
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_TRUE(succeeded(verifyLinearity(*module)));
    EXPECT_EQ(countOps<UOp>(), 1);
    EXPECT_EQ(countOps<math::Atan2Op>(), 0);
    EXPECT_EQ(countOps<math::SinOp>(), 0);
    EXPECT_EQ(countOps<math::CosOp>(), 0);
    EXPECT_EQ(countOps<arith::SelectOp>(), 0);

    for (const auto angles : {
             std::array{0.0, 0.0, 0.0},
             std::array{PI, PI, PI},
             std::array{2 * PI, -2 * PI, 2 * PI},
             std::array{-3 * PI, 0.37, 4 * PI},
             std::array{37 * PI, 0.37, -29 * PI},
             std::array{1.0e5, 0.37, 1.0e5},
         }) {
      SCOPED_TRACE(testing::PrintToString(angles));
      OwningOpRef<ModuleOp> before = original->clone();
      OwningOpRef<ModuleOp> after = module->clone();
      bindLeadingArgs(before->lookupSymbol<func::FuncOp>("main"), angles);
      bindLeadingArgs(after->lookupSymbol<func::FuncOp>("main"), angles);
      PassManager pm(&context);
      pm.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(pm.run(*after)));
      ASSERT_TRUE(succeeded(verify(*after)));
      ASSERT_TRUE(succeeded(verifyLinearity(*after)));
      ::mqt::test::expectUnitaryEquivalent(*before, *after, 2);
    }
  }
}

TEST_F(MergeSingleQubitRotationGatesTest,
       fusesSymbolicEulerChainsDirectlyInCompatibleBases) {
  for (const auto* basisName :
       {"u", "zyz", "zxz", "zsxx", "xzx", "xyx", "r", "gpi"}) {
    SCOPED_TRACE(basisName);
    const auto basis = *decomposition::parseSingleQubitBasis(basisName);
    FuseSingleQubitUnitaryRunsOptions options;
    options.basis = basisName;
    const bool outerX = basis == decomposition::SingleQubitBasis::XZX ||
                        basis == decomposition::SingleQubitBasis::XYX;
    for (const StringRef middleGate : {"rx", "ry", "rz", "h"}) {
      SCOPED_TRACE(middleGate.str());
      const bool hadamard = middleGate == "h";
      const bool rotateX = outerX && !hadamard;
      for (const unsigned outerMask : {0U, 1U, 2U, 3U}) {
        if ((hadamard && (outerMask == 0 || outerMask == 3)) ||
            (middleGate == (outerX ? "rx" : "rz") && outerMask != 0)) {
          continue;
        }
        SCOPED_TRACE(outerMask);
        module = buildQuantumFunction(&context, [&](auto& b) {
          auto [control, target] =
              b.ctrl(b.staticQubit(0), b.staticQubit(1), [&](Value qubit) {
                if ((outerMask & 1U) != 0) {
                  qubit = rotateX ? b.rx(0.1, qubit) : b.rz(0.1, qubit);
                }
                qubit = middleGate == "rx"   ? b.rx(0.2, qubit)
                        : middleGate == "ry" ? b.ry(0.2, qubit)
                        : hadamard           ? b.h(qubit)
                                             : b.rz(0.2, qubit);
                if ((outerMask & 2U) != 0) {
                  qubit = rotateX ? b.rx(0.4, qubit) : b.rz(0.4, qubit);
                }
                return qubit;
              });
          return SmallVector<Value>{control, target};
        });
        auto funcOp = module->lookupSymbol<func::FuncOp>("main");
        module->walk([&](UnitaryOpInterface op) {
          if (isa<RXOp, RYOp, RZOp>(op.getOperation())) {
            const auto index = funcOp.getNumArguments();
            ASSERT_TRUE(succeeded(funcOp.insertArgument(
                index, Float64Type::get(&context), {}, funcOp.getLoc())));
            op.getParameter(0).replaceAllUsesWith(funcOp.getArgument(index));
          }
        });
        ASSERT_TRUE(succeeded(verify(*module)));
        ASSERT_TRUE(succeeded(verifyLinearity(*module)));
        OwningOpRef<ModuleOp> original = module->clone();
        PassManager fusion(&context);
        fusion.addPass(createFuseSingleQubitUnitaryRuns(options));
        ASSERT_TRUE(succeeded(fusion.run(*module)));
        ASSERT_TRUE(succeeded(verify(*module)));
        ASSERT_TRUE(succeeded(verifyLinearity(*module)));
        EXPECT_EQ(countOps<math::Atan2Op>(), 0);
        EXPECT_EQ(countOps<math::SinOp>(), 0);
        EXPECT_EQ(countOps<math::CosOp>(), 0);
        EXPECT_EQ(countOps<arith::SelectOp>(), 0);
        unsigned gateCount = 0;
        module->walk([&](UnitaryOpInterface op) {
          // Phase normalization may lift a P gate onto the control wire.
          if (op.isSingleQubit() && op->getParentOfType<CtrlOp>()) {
            ++gateCount;
            EXPECT_TRUE(decomposition::isSingleQubitBasisGate(op, basis));
          }
        });
        EXPECT_GT(gateCount, 0U);
        EXPECT_LE(gateCount,
                  basis == decomposition::SingleQubitBasis::U ? 1U
                  : !hadamard && basis == decomposition::SingleQubitBasis::ZSXX
                      ? 5U
                      : 3U);

        for (const auto angles : {
                 std::array{0.0, 0.0, 0.0},
                 std::array{PI, PI, PI},
                 std::array{2 * PI, -2 * PI, 2 * PI},
                 std::array{-2 * PI, 2 * PI, -2 * PI},
                 std::array{-3 * PI, 0.37, 4 * PI},
                 std::array{37 * PI, 0.37, -29 * PI},
                 std::array{1.0e5, 0.37, 1.0e5},
                 std::array{0.3, 1.0e16, 0.7},
                 std::array{1.0e16, 0.37, 0.7},
             }) {
          SCOPED_TRACE(testing::PrintToString(angles));
          OwningOpRef<ModuleOp> before = original->clone();
          OwningOpRef<ModuleOp> after = module->clone();
          auto values = ArrayRef(angles).take_front(funcOp.getNumArguments());
          bindLeadingArgs(before->lookupSymbol<func::FuncOp>("main"), values);
          bindLeadingArgs(after->lookupSymbol<func::FuncOp>("main"), values);
          PassManager canonicalizer(&context);
          canonicalizer.addPass(createCanonicalizerPass());
          ASSERT_TRUE(succeeded(canonicalizer.run(*after)));
          ASSERT_TRUE(succeeded(verify(*after)));
          ASSERT_TRUE(succeeded(verifyLinearity(*after)));
          ::mqt::test::expectUnitaryEquivalent(*before, *after, 2);
        }
      }
    }
  }
}

TEST_F(MergeSingleQubitRotationGatesTest, composesHadamardZChainsDirectly) {
  for (const auto* basisName : {"u", "zyz", "zxz", "zsxx", "xzx", "xyx", "r"}) {
    SCOPED_TRACE(basisName);
    for (const bool phaseGate : {false, true}) {
      SCOPED_TRACE(phaseGate);
      for (const unsigned hadamards : {1U, 2U, 3U}) {
        SCOPED_TRACE(hadamards);
        module = buildQuantumFunction(&context, [&](auto& b) {
          auto [control, target] =
              b.ctrl(b.staticQubit(0), b.staticQubit(1), [&](Value qubit) {
                if ((hadamards & 1U) != 0) {
                  qubit = b.h(qubit);
                }
                qubit = phaseGate ? b.p(0.2, qubit) : b.rz(0.2, qubit);
                return (hadamards & 2U) != 0 ? b.h(qubit) : qubit;
              });
          return SmallVector<Value>{control, target};
        });
        auto funcOp = module->lookupSymbol<func::FuncOp>("main");
        const auto index = funcOp.getNumArguments();
        ASSERT_TRUE(succeeded(funcOp.insertArgument(
            index, Float64Type::get(&context), {}, funcOp.getLoc())));
        module->walk([&](UnitaryOpInterface op) {
          if (isa<POp, RZOp>(op.getOperation())) {
            op.getParameter(0).replaceAllUsesWith(funcOp.getArgument(index));
          }
        });
        OwningOpRef<ModuleOp> original = module->clone();
        FuseSingleQubitUnitaryRunsOptions options;
        options.basis = basisName;
        PassManager pm(&context);
        pm.addPass(createFuseSingleQubitUnitaryRuns(options));
        ASSERT_TRUE(succeeded(pm.run(*module)));
        ASSERT_TRUE(succeeded(verify(*module)));
        ASSERT_TRUE(succeeded(verifyLinearity(*module)));
        // Direct identities avoid the classical cost of quaternion extraction.
        EXPECT_EQ(countOps<math::Atan2Op>(), 0);
        EXPECT_EQ(countOps<math::SinOp>(), 0);
        EXPECT_EQ(countOps<math::CosOp>(), 0);
        for (const double angle : {0., 0.37, PI, -2 * PI, 37 * PI, 1.e5}) {
          SCOPED_TRACE(angle);
          OwningOpRef<ModuleOp> before = original->clone();
          OwningOpRef<ModuleOp> after = module->clone();
          bindLeadingArgs(before->lookupSymbol<func::FuncOp>("main"), {angle});
          bindLeadingArgs(after->lookupSymbol<func::FuncOp>("main"), {angle});
          PassManager canonicalizer(&context);
          canonicalizer.addPass(createCanonicalizerPass());
          ASSERT_TRUE(succeeded(canonicalizer.run(*after)));
          ::mqt::test::expectUnitaryEquivalent(*before, *after, 2);
        }
      }
    }
  }
}

TEST_F(MergeSingleQubitRotationGatesTest,
       normalizesSharedGateExpressionsIndependently) {
  for (const auto* basisName : {"u", "zyz", "zsxx"}) {
    SCOPED_TRACE(basisName);
    module = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%theta: f64) -> !qco.qubit {
          %two = arith.constant 2.0 : f64
          %half = arith.divf %theta, %two : f64
          %middle = arith.constant 0.37 : f64
          %q = qco.static 0 : !qco.qubit
          %q1 = qco.rz(%half) %q : !qco.qubit
          %q2 = qco.rx(%middle) %q1 : !qco.qubit
          %q3 = qco.rz(%theta) %q2 : !qco.qubit
          return %q3 : !qco.qubit
        }
      })mlir",
                                         &context);
    ASSERT_TRUE(module);
    OwningOpRef<ModuleOp> original = module->clone();
    FuseSingleQubitUnitaryRunsOptions options;
    options.basis = basisName;
    PassManager fusion(&context);
    fusion.addPass(createFuseSingleQubitUnitaryRuns(options));
    ASSERT_TRUE(succeeded(fusion.run(*module)));
    for (const double angle : {2 * PI, -2 * PI, 17 * PI, 1.0e5}) {
      SCOPED_TRACE(angle);
      OwningOpRef<ModuleOp> before = original->clone();
      OwningOpRef<ModuleOp> after = module->clone();
      bindLeadingArgs(before->lookupSymbol<func::FuncOp>("main"), {angle});
      bindLeadingArgs(after->lookupSymbol<func::FuncOp>("main"), {angle});
      PassManager canonicalizer(&context);
      canonicalizer.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(canonicalizer.run(*after)));
      ASSERT_TRUE(succeeded(verify(*after)));
      ASSERT_TRUE(succeeded(verifyLinearity(*after)));
      after->walk([&](UnitaryOpInterface op) {
        for (Value parameter : op.getParameters()) {
          const auto value = mlir::mqt::valueToConstantDouble(parameter);
          ASSERT_TRUE(value);
          EXPECT_LE(std::abs(*value), 4 * PI);
        }
      });
      ::mqt::test::expectUnitaryEquivalent(*before, *after, 1);
    }
  }
}

TEST_F(MergeSingleQubitRotationGatesTest,
       mergeDynamicAngleRotationsUsesSsaPath) {
  // Pure-Z chain with runtime angles uses scalar SSA composition.
  // RZ(a);RZ(b) → U(0, wrap(a+b), 0) with SSA phi tied to the angle args.
  constexpr double angleA = 0.3;
  constexpr double angleB = 0.4;
  auto q = builder.allocQubitRegister(1);
  q[0] = builder.rz(angleA, q[0]);
  q[0] = builder.rz(angleB, q[0]);
  module = builder.finalize();

  auto funcOp = cast<func::FuncOp>(module->getBody()->front());
  const auto f64 = Float64Type::get(&context);
  ASSERT_TRUE(succeeded(funcOp.insertArgument(0, f64, {}, funcOp.getLoc())));
  ASSERT_TRUE(succeeded(funcOp.insertArgument(1, f64, {}, funcOp.getLoc())));

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
  EXPECT_FALSE(mlir::mqt::valueToConstantDouble(uOp.getPhi()).has_value());
  EXPECT_TRUE(valueDependsOn(uOp.getPhi(), funcOp.getArgument(0)));
  EXPECT_TRUE(valueDependsOn(uOp.getPhi(), funcOp.getArgument(1)));
  EXPECT_FALSE(
      mlir::mqt::valueToConstantDouble(gOp.getParameter(0)).has_value());
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
  const auto theta = mlir::mqt::valueToConstantDouble(uOp.getTheta());
  const auto phi = mlir::mqt::valueToConstantDouble(uOp.getPhi());
  const auto lambda = mlir::mqt::valueToConstantDouble(uOp.getLambda());
  const auto phase = mlir::mqt::valueToConstantDouble(gOp.getParameter(0));
  ASSERT_TRUE(theta.has_value());
  ASSERT_TRUE(phi.has_value());
  ASSERT_TRUE(lambda.has_value());
  ASSERT_TRUE(phase.has_value());
  EXPECT_NEAR(*theta, 0.0, 1e-6);
  EXPECT_NEAR(*phi, mlir::mqt::normalizeAngle(angleA + angleB), 1e-6);
  EXPECT_NEAR(*lambda, 0.0, 1e-6);
  EXPECT_NEAR(*phase, mlir::mqt::normalizeAngle(-(*phi + *lambda) / 2.0), 1e-6);
  EXPECT_TRUE(mlir::mqt::isValidGlobalPhaseAngle(*phase));
  EXPECT_FALSE(std::isnan(*theta));
  EXPECT_FALSE(std::isnan(*phi));
  EXPECT_FALSE(std::isnan(*lambda));
  EXPECT_FALSE(std::isnan(*phase));
}

TEST_F(MergeSingleQubitRotationGatesTest,
       mergeFixedGateWithDynamicRotationPreservesMatrix) {
  constexpr double angle = 0.37;
  auto q = builder.staticQubit(0);
  q = builder.h(q);
  q = builder.rz(angle, q);
  builder.sink(q);
  module = builder.finalize();

  auto funcOp = cast<func::FuncOp>(module->getBody()->front());
  const auto f64 = Float64Type::get(&context);
  ASSERT_TRUE(succeeded(funcOp.insertArgument(0, f64, {}, funcOp.getLoc())));

  RZOp rzOp = nullptr;
  module->walk([&](RZOp op) { rzOp = op; });
  ASSERT_TRUE(rzOp);
  rzOp.getThetaMutable().assign(funcOp.getArgument(0));

  ASSERT_TRUE(runMergePass(*module).succeeded());
  EXPECT_EQ(countOps<HOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<math::SqrtOp>(), 0);
  EXPECT_EQ(countOps<math::Atan2Op>(), 0);

  UOp uOp = nullptr;
  module->walk([&](UOp op) { uOp = op; });
  ASSERT_TRUE(uOp);
  EXPECT_TRUE(valueDependsOn(uOp.getTheta(), funcOp.getArgument(0)) ||
              valueDependsOn(uOp.getPhi(), funcOp.getArgument(0)) ||
              valueDependsOn(uOp.getLambda(), funcOp.getArgument(0)));

  bindLeadingArgs(funcOp, {angle});
  const auto theta = mlir::mqt::valueToConstantDouble(uOp.getTheta());
  const auto phi = mlir::mqt::valueToConstantDouble(uOp.getPhi());
  const auto lambda = mlir::mqt::valueToConstantDouble(uOp.getLambda());
  ASSERT_TRUE(theta.has_value());
  ASSERT_TRUE(phi.has_value());
  ASSERT_TRUE(lambda.has_value());
  double globalPhase = 0.0;
  module->walk([&](GPhaseOp op) {
    const auto phase = mlir::mqt::valueToConstantDouble(op.getParameter(0));
    ASSERT_TRUE(phase.has_value());
    globalPhase += *phase;
  });
  const auto actual =
      UOp::unitaryMatrix(*theta, *phi, *lambda) * std::polar(1.0, globalPhase);
  const auto halfAngle = angle / 2.0;
  const std::complex<double> upperPhase = std::polar(1.0, -halfAngle);
  const std::complex<double> lowerPhase = std::polar(1.0, halfAngle);
  const double invSqrtTwo = 1.0 / std::numbers::sqrt2;
  const Matrix2x2 expected{
      upperPhase * invSqrtTwo,
      upperPhase * invSqrtTwo,
      lowerPhase * invSqrtTwo,
      -lowerPhase * invSqrtTwo,
  };
  EXPECT_TRUE(actual.isApprox(expected, 1e-8));
}

// Two phase-bearing dynamic gates exercise SSA phase accumulation.
//
// `mergeDynamicChain` sums each gate's global-phase contribution. P(a);P(b)
// accumulates both unfoldable angles into the emitted `gphase` SSA.
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
  ASSERT_TRUE(succeeded(funcOp.insertArgument(0, f64, {}, funcOp.getLoc())));
  ASSERT_TRUE(succeeded(funcOp.insertArgument(1, f64, {}, funcOp.getLoc())));

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
  const auto theta = mlir::mqt::valueToConstantDouble(uOp.getTheta());
  const auto phi = mlir::mqt::valueToConstantDouble(uOp.getPhi());
  const auto lambda = mlir::mqt::valueToConstantDouble(uOp.getLambda());
  const auto phase = mlir::mqt::valueToConstantDouble(gOp.getParameter(0));
  ASSERT_TRUE(theta.has_value());
  ASSERT_TRUE(phi.has_value());
  ASSERT_TRUE(lambda.has_value());
  ASSERT_TRUE(phase.has_value());
  EXPECT_NEAR(*theta, 0.0, 1e-6);
  EXPECT_NEAR(*phi, mlir::mqt::normalizeAngle(angleA + angleB), 1e-6);
  EXPECT_NEAR(*lambda, 0.0, 1e-6);
  EXPECT_NEAR(*phase, 0.0, 1e-6);
  EXPECT_TRUE(mlir::mqt::isValidGlobalPhaseAngle(*phase));
}
