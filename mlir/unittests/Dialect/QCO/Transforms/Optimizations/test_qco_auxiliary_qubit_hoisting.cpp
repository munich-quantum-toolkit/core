/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qco_auxiliary_qubit_hoisting.cpp
/// Tests for the `quantum-auxiliary-qubit-hoisting` pass.

#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "IPOTestFixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <tuple>
#include <utility>

namespace {

using QCOAuxiliaryQubitHoistingTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Auxiliary qubit hoisting.
// ==========================================================================

/// A qubit that a callee allocates and releases internally is turned into
/// an extra argument, so the caller owns the allocation and can reuse it.
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitIntoCaller) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    auto [aux, target] =
        programBuilder.cx(programBuilder.allocQubit(), args[0]);
    programBuilder.sink(aux);
    return SmallVector<Value>{target};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // The auxiliary qubit becomes a trailing argument and is returned in a reset
  // state as a trailing result.
  referenceBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange refArgs) {
        auto [refAux, refTarget] = referenceBuilder.cx(refArgs[1], refArgs[0]);
        refAux = referenceBuilder.reset(refAux);
        return SmallVector<Value>{refTarget, refAux};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/// The auxiliary qubit is tracked across a measurement and a reset on its
/// way to the release point.
///
/// The measurement outcome is handed back to the caller so that the measurement
/// is not dead, and the reset sits between two gates so that it is neither
/// folded into the allocation nor into the release.
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitThroughMeasureAndReset) {
  const auto qubitType = getQubitType();
  const auto bitType = programBuilder.getI1Type();

  programBuilder.initialize({bitType});
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    auto [aux, bit] =
        programBuilder.measure(programBuilder.h(programBuilder.allocQubit()));
    aux = programBuilder.reset(aux);
    auto target = args[0];
    std::tie(aux, target) = programBuilder.cx(aux, target);
    programBuilder.sink(aux);
    return SmallVector<Value>{bit, target};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[1]);
  moduleOp = programBuilder.finalize({results[0]});

  referenceBuilder.initialize({bitType});
  referenceBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange refArgs) {
        auto [refAux, refBit] =
            referenceBuilder.measure(referenceBuilder.h(refArgs[1]));
        refAux = referenceBuilder.reset(refAux);
        auto refTarget = refArgs[0];
        std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
        refAux = referenceBuilder.reset(refAux);
        return SmallVector<Value>{refBit, refTarget, refAux};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[1]);
  referenceBuilder.sink(refResults[2]);
  reference = referenceBuilder.finalize({refResults[0]});

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/// The auxiliary qubit is tracked across a nested call on its way to the
/// release point.
///
/// The nested callee returns more than one qubit and the auxiliary one is not
/// the first, so the walk has to match the operand position rather than simply
/// taking the first result.
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitThroughNestedCall) {
  const auto qubitType = getQubitType();

  const auto buildNestedCallee = [&qubitType](QCOProgramBuilder& b) {
    b.createFunction("g", {qubitType, qubitType}, [&](ValueRange innerArgs) {
      return SmallVector<Value>{b.h(innerArgs[0]), innerArgs[1]};
    });
  };

  programBuilder.initialize();
  buildNestedCallee(programBuilder);

  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    auto aux = programBuilder.allocQubit();
    // The auxiliary qubit is the second operand and the second result.
    auto nested = callFunction(programBuilder, "g", {args[0], aux});
    auto target = nested[0];
    aux = nested[1];
    std::tie(aux, target) = programBuilder.cx(aux, target);
    programBuilder.sink(aux);
    return SmallVector<Value>{target};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  buildNestedCallee(referenceBuilder);

  referenceBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange refArgs) {
        auto refNested =
            callFunction(referenceBuilder, "g", {refArgs[0], refArgs[1]});
        auto [refAux, refTarget] =
            referenceBuilder.cx(refNested[1], refNested[0]);
        refAux = referenceBuilder.reset(refAux);
        return SmallVector<Value>{refTarget, refAux};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/// Every call site of a hoisted callee gets its own allocation.
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitWithMultipleCallSites) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    auto [aux, target] =
        programBuilder.cx(programBuilder.allocQubit(), args[0]);
    programBuilder.sink(aux);
    return SmallVector<Value>{target};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto results0 = callFunction(programBuilder, "f", {q0});
  auto results1 = callFunction(programBuilder, "f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange refArgs) {
        auto [refAux, refTarget] = referenceBuilder.cx(refArgs[1], refArgs[0]);
        refAux = referenceBuilder.reset(refAux);
        return SmallVector<Value>{refTarget, refAux};
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refAux0 = referenceBuilder.allocQubit();
  auto refResults0 = callFunction(referenceBuilder, "f", {refQ0, refAux0});
  referenceBuilder.sink(refResults0[1]);
  auto refAux1 = referenceBuilder.allocQubit();
  auto refResults1 = callFunction(referenceBuilder, "f", {refQ1, refAux1});
  referenceBuilder.sink(refResults1[1]);
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/// A recursive function keeps its auxiliary qubit, because hoisting it would
/// have to thread the qubit through every level of the recursion.
TEST_F(QCOAuxiliaryQubitHoistingTest, noHoistingOutOfRecursiveFunction) {
  auto module = parseModule(R"mlir(
func.func private @recursive(%q: !qco.qubit) -> !qco.qubit {
  %aux = qco.alloc : !qco.qubit
  %h = qco.h %aux : !qco.qubit -> !qco.qubit
  qco.sink %h : !qco.qubit
  %r = func.call @recursive(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
func.func @main(%q: !qco.qubit) -> !qco.qubit {
  %r = func.call @recursive(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createAuxiliaryQubitHoisting()).succeeded());
  EXPECT_EQ(countAllocsIn(module.get(), "recursive"), 1U);
}

/// An auxiliary qubit passed through a recursive callee is still hoisted.
///
/// Every callee returns its quantum arguments in argument order, so the call
/// hands the qubit back however deep the recursion goes.
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitThroughRecursiveCallee) {
  auto module = parseModule(R"mlir(
func.func private @recursive(%q: !qco.qubit) -> !qco.qubit {
  %r = func.call @recursive(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
func.func private @outer(%q: !qco.qubit) -> !qco.qubit {
  %aux = qco.alloc : !qco.qubit
  %r = func.call @recursive(%aux) : (!qco.qubit) -> !qco.qubit
  qco.sink %r : !qco.qubit
  return %q : !qco.qubit
}
func.func @main(%q: !qco.qubit) -> !qco.qubit {
  %r = func.call @outer(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createAuxiliaryQubitHoisting()).succeeded());
  EXPECT_EQ(countAllocsIn(module.get(), "outer"), 0U);
  EXPECT_EQ(countAllocsIn(module.get(), "main"), 1U);
}

/// An allocation nested inside a region is not hoisted, because it is not
/// executed on every path through the function.
TEST_F(QCOAuxiliaryQubitHoistingTest, noHoistingForAllocInsideRegion) {
  // The builder only allocates in function entry blocks.
  auto module = parseModule(R"mlir(
func.func private @f(%q: !qco.qubit, %c: i1) -> !qco.qubit {
  %r = qco.if %c args(%a = %q) -> (!qco.qubit) {
    %aux = qco.alloc : !qco.qubit
    %auxOut, %aOut = qco.ctrl(%aux) targets(%t = %a) {
      %x = qco.x %t : !qco.qubit -> !qco.qubit
      qco.yield %x : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    qco.sink %auxOut : !qco.qubit
    qco.yield %aOut : !qco.qubit
  } else args(%a = %q) {
    qco.yield %a : !qco.qubit
  }
  return %r : !qco.qubit
}
func.func @main(%q: !qco.qubit, %c: i1) -> !qco.qubit {
  %r = func.call @f(%q, %c) : (!qco.qubit, i1) -> !qco.qubit
  return %r : !qco.qubit
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createAuxiliaryQubitHoisting()).succeeded());
  EXPECT_EQ(countAllocsIn(module.get(), "f"), 1U);
}

/// The auxiliary qubit is tracked across a call whose callee also returns
/// a classical value ahead of the qubit.
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitThroughCallWithClassicalResult) {
  const auto qubitType = getQubitType();
  const auto bitType = programBuilder.getI1Type();

  const auto buildCallee = [&](QCOProgramBuilder& b) {
    b.createFunction("g", {qubitType}, [&](ValueRange innerArgs) {
      auto [inner, bit] = b.measure(innerArgs[0]);
      return SmallVector<Value>{bit, inner};
    });
  };

  programBuilder.initialize({bitType});
  buildCallee(programBuilder);

  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    auto nested =
        callFunction(programBuilder, "g", {programBuilder.allocQubit()});
    auto [aux, target] = programBuilder.cx(nested[1], args[0]);
    programBuilder.sink(aux);
    return SmallVector<Value>{nested[0], target};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[1]);
  moduleOp = programBuilder.finalize({results[0]});

  referenceBuilder.initialize({bitType});
  buildCallee(referenceBuilder);

  referenceBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange refArgs) {
        auto refNested = callFunction(referenceBuilder, "g", {refArgs[1]});
        auto [refAux, refTarget] =
            referenceBuilder.cx(refNested[1], refArgs[0]);
        refAux = referenceBuilder.reset(refAux);
        return SmallVector<Value>{refNested[0], refTarget, refAux};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[1]);
  referenceBuilder.sink(refResults[2]);
  reference = referenceBuilder.finalize({refResults[0]});

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/// A call site inside a loop body receives the hoisted qubit in that body, so
/// the new allocation is released in the block that allocates it.
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitIntoLoopBody) {
  auto module = parseModule(R"mlir(
func.func private @f(%q: !qco.qubit) -> !qco.qubit {
  %aux = qco.alloc : !qco.qubit
  %auxOut, %qOut = qco.ctrl(%aux) targets(%t = %q) {
    %x = qco.x %t : !qco.qubit -> !qco.qubit
    qco.yield %x : !qco.qubit
  } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
  qco.sink %auxOut : !qco.qubit
  return %qOut : !qco.qubit
}
func.func @main(%q: !qco.qubit, %n: index) -> !qco.qubit {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %q) -> (!qco.qubit) {
    %b = func.call @f(%a) : (!qco.qubit) -> !qco.qubit
    scf.yield %b : !qco.qubit
  }
  return %r : !qco.qubit
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createAuxiliaryQubitHoisting()).succeeded());

  auto f = module->lookupSymbol<func::FuncOp>("f");
  ASSERT_TRUE(f);
  EXPECT_EQ(f.getNumArguments(), 2U);
  EXPECT_EQ(f.getNumResults(), 2U);
  EXPECT_EQ(countAllocsIn(module.get(), "f"), 0U);

  func::CallOp call;
  module->walk([&](func::CallOp op) { call = op; });
  ASSERT_TRUE(call);
  auto alloc = call.getOperands().back().getDefiningOp<AllocOp>();
  ASSERT_TRUE(alloc);
  EXPECT_EQ(alloc->getBlock(), call->getBlock());
  ASSERT_TRUE(call.getResults().back().hasOneUse());
  auto* release = *call.getResults().back().getUsers().begin();
  EXPECT_TRUE(isa<SinkOp>(release));
  EXPECT_EQ(release->getBlock(), call->getBlock());
}

/// Hoisting reaches the outermost caller regardless of declaration order.
///
/// An allocation hoisted into a caller may be hoistable again. Processing in
/// module order used to strand it wherever the declarations happened to sit.
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistingIsIndependentOfDeclarationOrder) {
  const auto qubitType = getQubitType();

  // Builds `main -> mid -> leaf`, where `leaf` owns an auxiliary qubit.
  const auto build = [&](QCOProgramBuilder& builder) {
    builder.initialize();
    builder.createFunction("leaf", {qubitType}, [&](ValueRange leafArgs) {
      auto [aux, target] = builder.cx(builder.allocQubit(), leafArgs[0]);
      builder.sink(aux);
      return SmallVector<Value>{target};
    });

    builder.createFunction("mid", {qubitType}, [&](ValueRange midArgs) {
      auto midResults = callFunction(builder, "leaf", {midArgs[0]});
      return SmallVector<Value>{midResults[0]};
    });

    auto q = builder.allocQubit();
    auto results = callFunction(builder, "mid", {q});
    builder.sink(results[0]);
    return builder.finalize();
  };

  moduleOp = build(programBuilder);
  reference = build(referenceBuilder);

  // The builder has to declare a callee before the call, so the second module
  // is reordered afterwards. Both now describe the same call graph and differ
  // only in the order the module walk visits the two callees.
  auto refLeaf = reference->lookupSymbol<func::FuncOp>("leaf");
  auto refMid = reference->lookupSymbol<func::FuncOp>("mid");
  ASSERT_TRUE(refLeaf);
  ASSERT_TRUE(refMid);
  refLeaf->moveAfter(refMid.getOperation());

  ASSERT_TRUE(
      runStage(moduleOp.get(), createAuxiliaryQubitHoisting()).succeeded());
  ASSERT_TRUE(
      runStage(reference.get(), createAuxiliaryQubitHoisting()).succeeded());

  // The auxiliary allocation belongs in the entry function either way: one
  // allocation for the qubit passed in and one for the hoisted auxiliary.
  for (auto* module : {&moduleOp, &reference}) {
    EXPECT_EQ(countAllocsIn(module->get(), "leaf"), 0U);
    EXPECT_EQ(countAllocsIn(module->get(), "mid"), 0U);
    EXPECT_EQ(countAllocsIn(module->get(), "main"), 2U);
  }
}

// ==========================================================================

} // namespace
