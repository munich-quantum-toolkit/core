/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Support/Passes.h"

#include "ExactUnitaryTest.h"
#include "Support/IRVerification.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <limits>
#include <random>
#include <tuple>

using namespace mlir;
using namespace mlir::qco;

static void cx(QCOProgramBuilder& builder, MutableArrayRef<Value> qubits,
               unsigned control, unsigned target) {
  std::tie(qubits[control], qubits[target]) =
      builder.cx(qubits[control], qubits[target]);
}

static void gadget(QCOProgramBuilder& builder, MutableArrayRef<Value> qubits,
                   ArrayRef<unsigned> controls, unsigned target,
                   function_ref<Value(Value)> phase) {
  for (const auto control : controls) {
    cx(builder, qubits, control, target);
  }
  qubits[target] = phase(qubits[target]);
  for (const auto control : llvm::reverse(controls)) {
    cx(builder, qubits, control, target);
  }
}

static SmallVector<Value> staticQubits(QCOProgramBuilder& builder,
                                       unsigned count) {
  SmallVector<Value> qubits;
  for (unsigned i = 0; i < count; ++i) {
    qubits.push_back(builder.staticQubit(i));
  }
  return qubits;
}

static void threeGadgets(QCOProgramBuilder& builder,
                         MutableArrayRef<Value> qubits) {
  gadget(builder, qubits, {0}, 2, [&](Value q) { return builder.rz(0.2, q); });
  gadget(builder, qubits, {1}, 2, [&](Value q) { return builder.p(0.4, q); });
  gadget(builder, qubits, {0, 1}, 2,
         [&](Value q) { return builder.rz(0.3, q); });
}

template <typename Op> static size_t countOps(ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&](Op) { ++count; });
  return count;
}

namespace {

class CNOTPhaseResynthesisTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder{&context};

  void SetUp() override {
    context.loadDialect<QCODialect, func::FuncDialect, arith::ArithDialect,
                        qtensor::QTensorDialect>();
    builder.initialize();
  }

  static LogicalResult run(ModuleOp moduleOp,
                           const ResynthesizeCNOTPhaseOptions& options = {}) {
    PassManager pm(moduleOp.getContext());
    pm.addPass(createResynthesizeCNOTPhase(options));
    return pm.run(moduleOp);
  }

  static void check(ModuleOp moduleOp, unsigned width) {
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
    OwningOpRef<ModuleOp> original = moduleOp.clone();
    const auto before = countOps<CtrlOp>(moduleOp);
    ASSERT_TRUE(succeeded(run(moduleOp)));
    EXPECT_LE(countOps<CtrlOp>(moduleOp), before);
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
    ::mqt::test::expectFullUnitaryEqual(*original, moduleOp, width);
  }
};

TEST_F(CNOTPhaseResynthesisTest, SharesParityComputations) {
  auto qubits = staticQubits(builder, 3);
  threeGadgets(builder, qubits);
  auto moduleOp = builder.finalize();
  ASSERT_EQ(countOps<CtrlOp>(*moduleOp), 8U);
  check(*moduleOp, 3);
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 4U);
  EXPECT_EQ(countOps<RZOp>(*moduleOp), 2U);
  EXPECT_EQ(countOps<POp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<GPhaseOp>(*moduleOp), 0U);
}

TEST_F(CNOTPhaseResynthesisTest, PureLinearMapAndNonidentityResidual) {
  auto qubits = staticQubits(builder, 4);
  threeGadgets(builder, qubits);
  cx(builder, qubits, 2, 0);
  cx(builder, qubits, 0, 3);
  cx(builder, qubits, 3, 1);
  cx(builder, qubits, 1, 3);
  cx(builder, qubits, 1, 3);
  auto moduleOp = builder.finalize();
  check(*moduleOp, 4);
  EXPECT_LT(countOps<CtrlOp>(*moduleOp), 13U);
}

TEST_F(CNOTPhaseResynthesisTest, CancelsPureCNOTNetwork) {
  auto qubits = staticQubits(builder, 3);
  for (unsigned i = 0; i < 2; ++i) {
    cx(builder, qubits, 0, 1);
    cx(builder, qubits, 0, 2);
  }
  auto moduleOp = builder.finalize();
  check(*moduleOp, 3);
  EXPECT_EQ(countOps<CtrlOp>(*moduleOp), 0U);
}

TEST_F(CNOTPhaseResynthesisTest, KeepsAnUnprofitableBlockUnchanged) {
  auto qubits = staticQubits(builder, 3);
  cx(builder, qubits, 0, 1);
  qubits[1] = builder.rz(0.2, qubits[1]);
  cx(builder, qubits, 1, 2);
  auto moduleOp = builder.finalize();
  OwningOpRef<ModuleOp> original = moduleOp->clone();
  ASSERT_TRUE(succeeded(run(*moduleOp)));
  EXPECT_TRUE(areModulesStructurallyEquivalent(*original, *moduleOp));
}

TEST_F(CNOTPhaseResynthesisTest, MixedGatesAndDisjointOperations) {
  auto qubits = staticQubits(builder, 4);
  cx(builder, qubits, 0, 1);
  qubits[3] = builder.h(qubits[3]);
  cx(builder, qubits, 0, 1);
  // A shared-wire H prevents cancellation across it.
  cx(builder, qubits, 0, 1);
  qubits[1] = builder.h(qubits[1]);
  cx(builder, qubits, 0, 1);
  qubits[0] = builder.x(qubits[0]);
  std::tie(qubits[0], qubits[2]) = builder.swap(qubits[0], qubits[2]);
  std::tie(qubits[0], qubits[3]) = builder.cz(qubits[0], qubits[3]);
  threeGadgets(builder, qubits);
  auto moduleOp = builder.finalize();
  check(*moduleOp, 4);
  EXPECT_EQ(countOps<HOp>(*moduleOp), 2U);
  EXPECT_EQ(countOps<SWAPOp>(*moduleOp), 1U);
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 7U);
}

TEST_F(CNOTPhaseResynthesisTest, PreservesNamedPhasesAndExtremeAngles) {
  auto qubits = staticQubits(builder, 3);
  for (const auto angle :
       {1e-14, 1e16, -1e16, std::numeric_limits<double>::max()}) {
    gadget(builder, qubits, {0}, 2,
           [&](Value q) { return builder.rz(angle, q); });
    gadget(builder, qubits, {0}, 2,
           [&](Value q) { return builder.p(angle, q); });
  }
  gadget(builder, qubits, {0, 1}, 2, [&](Value q) {
    return builder.tdg(builder.sdg(builder.z(builder.s(builder.t(q)))));
  });
  auto moduleOp = builder.finalize();
  check(*moduleOp, 3);
  EXPECT_EQ(countOps<RZOp>(*moduleOp), 4U);
  EXPECT_EQ(countOps<POp>(*moduleOp), 4U);
  EXPECT_EQ(countOps<TOp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<TdgOp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<SOp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<SdgOp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<ZOp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<GPhaseOp>(*moduleOp), 0U);
}

TEST_F(CNOTPhaseResynthesisTest, SymbolicAnglesKeepTheirSSAValues) {
  auto qubits = staticQubits(builder, 3);
  threeGadgets(builder, qubits);
  auto moduleOp = builder.finalize();
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  function.insertArgument(0, Float64Type::get(&context), {}, function.getLoc());
  SmallVector<RZOp> rotations;
  moduleOp->walk([&](RZOp op) { rotations.push_back(op); });
  // Compute an angle inside the block: synthesis must not hoist its use.
  OpBuilder irBuilder(rotations.back());
  auto angle = arith::AddFOp::create(irBuilder, function.getLoc(),
                                     function.getArgument(0),
                                     rotations.back().getTheta());
  rotations.back().getThetaMutable().assign(angle);
  rotations.front().getThetaMutable().assign(function.getArgument(0));
  OwningOpRef<ModuleOp> original = moduleOp->clone();
  ASSERT_TRUE(succeeded(run(*moduleOp)));
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 4U);
  EXPECT_EQ(countOps<arith::AddFOp>(*moduleOp), 1U);
  EXPECT_EQ(countOps<GPhaseOp>(*moduleOp), 0U);
  SmallVector<Value> parameters;
  moduleOp->walk([&](RZOp op) { parameters.push_back(op.getTheta()); });
  EXPECT_TRUE(llvm::is_contained(parameters, function.getArgument(0)));
  EXPECT_TRUE(llvm::is_contained(parameters, angle.getResult()));
  for (const auto value : {-2.7, 0.0, 0.819, 1e16}) {
    OwningOpRef<ModuleOp> expected = original->clone();
    OwningOpRef<ModuleOp> actual = moduleOp->clone();
    for (auto candidate : {*expected, *actual}) {
      auto func = *candidate.getOps<func::FuncOp>().begin();
      OpBuilder constants(&func.getBody().front(),
                          func.getBody().front().begin());
      auto constant = arith::ConstantOp::create(
          constants, func.getLoc(), constants.getF64FloatAttr(value));
      func.getArgument(0).replaceAllUsesWith(constant);
    }
    ::mqt::test::expectFullUnitaryEqual(*expected, *actual, 3);
  }
}

TEST_F(CNOTPhaseResynthesisTest, NestedControlPreservesObservablePhase) {
  auto qubits = staticQubits(builder, 4);
  builder.ctrl(ValueRange{qubits[3]},
               ValueRange{qubits[0], qubits[1], qubits[2]},
               [&](ValueRange inputs) {
                 SmallVector<Value> inner(inputs);
                 threeGadgets(builder, inner);
                 return inner;
               });
  auto moduleOp = builder.finalize();
  check(*moduleOp, 4);
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 5U);
}

TEST_F(CNOTPhaseResynthesisTest, FullWordParitiesAndQubitLimit) {
  auto qubits = staticQubits(builder, 65);
  for (unsigned target = 1; target < 65; ++target) {
    cx(builder, qubits, 0, target);
    cx(builder, qubits, 0, target);
    qubits[target] = builder.rz(0.2, qubits[target]);
  }
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(succeeded(run(*moduleOp)));
  EXPECT_EQ(countOps<CtrlOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<RZOp>(*moduleOp), 64U);
  EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
}

TEST_F(CNOTPhaseResynthesisTest, BoundedBlocks) {
  auto qubits = staticQubits(builder, 3);
  threeGadgets(builder, qubits);
  auto moduleOp = builder.finalize();
  OwningOpRef<ModuleOp> original = moduleOp->clone();
  ResynthesizeCNOTPhaseOptions options;
  options.maxQubits = 2;
  options.maxGates = 3;
  ASSERT_TRUE(succeeded(run(*moduleOp, options)));
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 8U);
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 3);
}

TEST_F(CNOTPhaseResynthesisTest, RejectsInvalidOptions) {
  auto moduleOp = builder.finalize();
  ScopedDiagnosticHandler diagnostics(&context,
                                      [](Diagnostic&) { return success(); });
  for (const auto limit : {0U, 65U}) {
    ResynthesizeCNOTPhaseOptions options;
    options.maxQubits = limit;
    EXPECT_TRUE(failed(run(*moduleOp, options)));
  }
  ResynthesizeCNOTPhaseOptions options;
  options.maxGates = 0;
  EXPECT_TRUE(failed(run(*moduleOp, options)));
}

TEST_F(CNOTPhaseResynthesisTest, EffectsEndBlocks) {
  auto qubits = staticQubits(builder, 2);
  cx(builder, qubits, 0, 1);
  qubits[1] = builder.measure(qubits[1]).first;
  cx(builder, qubits, 0, 1);
  qubits[0] = builder.reset(qubits[0]);
  cx(builder, qubits, 0, 1);
  auto moduleOp = builder.finalize();
  OwningOpRef<ModuleOp> original = moduleOp->clone();
  ASSERT_TRUE(succeeded(run(*moduleOp)));
  EXPECT_TRUE(areModulesStructurallyEquivalent(*original, *moduleOp));
}

TEST_F(CNOTPhaseResynthesisTest, TensorBackedWires) {
  auto tensor = builder.qtensorAlloc(3);
  SmallVector<Value> qubits;
  for (unsigned i = 0; i < 3; ++i) {
    auto [remainder, qubit] = builder.qtensorExtract(tensor, i);
    tensor = remainder;
    qubits.push_back(qubit);
  }
  threeGadgets(builder, qubits);
  // Transfer wires through a tensor, in a different slot order.
  for (unsigned i = 0; i < 3; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, 2U - i);
  }
  for (unsigned i = 0; i < 3; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }
  threeGadgets(builder, qubits);
  for (unsigned i = 0; i < 3; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }
  builder.qtensorDealloc(tensor);
  auto moduleOp = builder.finalize();
  check(*moduleOp, 3);
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 8U);
}

TEST_F(CNOTPhaseResynthesisTest, RejectsNonlinearInput) {
  auto qubits = staticQubits(builder, 2);
  cx(builder, qubits, 0, 1);
  auto moduleOp = builder.finalize();
  CtrlOp gate;
  moduleOp->walk([&](CtrlOp op) { gate = op; });
  OpBuilder irBuilder(gate);
  XOp::create(irBuilder, gate.getLoc(), gate.getInputControl(0));
  ScopedDiagnosticHandler diagnostics(&context,
                                      [](Diagnostic&) { return success(); });
  EXPECT_TRUE(failed(run(*moduleOp)));
}

TEST_F(CNOTPhaseResynthesisTest, RandomParityGadgetsAndResidualMaps) {
  std::mt19937 rng(9128);
  for (unsigned trial = 0; trial < 50; ++trial) {
    SCOPED_TRACE(trial);
    QCOProgramBuilder randomBuilder(&context);
    randomBuilder.initialize();
    auto qubits = staticQubits(randomBuilder, 5);
    for (unsigned i = 0; i < 15; ++i) {
      const auto target = rng() % 5U;
      SmallVector<unsigned> controls;
      for (unsigned control = 0; control < 5; ++control) {
        if (control != target && rng() % 2U != 0) {
          controls.push_back(control);
        }
      }
      const auto theta = static_cast<double>(rng() % 101U) / 17;
      gadget(randomBuilder, qubits, controls, target,
             [&](Value q) { return randomBuilder.rz(theta, q); });
    }
    for (unsigned i = 0; i < 8; ++i) {
      const auto control = rng() % 5U;
      cx(randomBuilder, qubits, control, (control + 1U + rng() % 4U) % 5U);
    }
    auto moduleOp = randomBuilder.finalize();
    check(*moduleOp, 5);
  }
}

TEST_F(CNOTPhaseResynthesisTest, RandomCircuitsPreserveCompleteMatrices) {
  std::mt19937 rng(70217);
  for (unsigned trial = 0; trial < 100; ++trial) {
    SCOPED_TRACE(trial);
    QCOProgramBuilder randomBuilder(&context);
    randomBuilder.initialize();
    const auto width = 2U + trial % 4U;
    auto qubits = staticQubits(randomBuilder, width);
    for (unsigned i = 0; i < 50; ++i) {
      const auto target = rng() % width;
      const auto control = (target + 1U + rng() % (width - 1U)) % width;
      switch (rng() % 7U) {
      case 0:
        qubits[target] = randomBuilder.h(qubits[target]);
        break;
      case 1:
        qubits[target] = randomBuilder.t(qubits[target]);
        break;
      case 2:
        qubits[target] = randomBuilder.p(static_cast<double>(rng() % 97U) / 13,
                                         qubits[target]);
        break;
      case 3:
        qubits[target] = randomBuilder.rz(static_cast<double>(rng() % 97U) / 13,
                                          qubits[target]);
        break;
      default:
        cx(randomBuilder, qubits, control, target);
        break;
      }
    }
    auto moduleOp = randomBuilder.finalize();
    check(*moduleOp, width);
  }
}

TEST_F(CNOTPhaseResynthesisTest, RegisteredTextualPipeline) {
  auto qubits = staticQubits(builder, 3);
  threeGadgets(builder, qubits);
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(succeeded(runPassPipeline(
      *moduleOp, "resynthesize-cnot-phase{max-qubits=3 max-gates=32}")));
  EXPECT_LE(countOps<CtrlOp>(*moduleOp), 4U);
}

} // namespace
