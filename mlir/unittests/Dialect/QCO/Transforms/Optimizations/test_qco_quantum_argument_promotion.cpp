/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qco_quantum_argument_promotion.cpp
/// Tests for the `quantum-argument-promotion` pass.

#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "IPOTestFixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <tuple>

namespace {

using QCOQuantumArgumentPromotionTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Quantum argument promotion.
// ==========================================================================

/// A tensor argument whose elements are extracted and re-inserted at
/// compile-time constant indices is replaced by scalar qubit arguments.
TEST_F(QCOQuantumArgumentPromotionTest, promoteTensorArgumentToQubitArgument) {
  const auto tensorType = getQubitTensorType(2);

  programBuilder.initialize();
  programBuilder.createFunction("f", {tensorType}, [&](ValueRange args) {
    auto [tensorIn, inner] = programBuilder.qtensorExtract(args[0], 0);
    inner = programBuilder.h(inner);
    return SmallVector<Value>{programBuilder.qtensorInsert(inner, tensorIn, 0)};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = callFunction(programBuilder, "f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.h(refArgs[0])};
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  // The caller extracts the promoted element, calls, and re-inserts it.
  auto [refTensorIn, refExtracted] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto refResults = callFunction(referenceBuilder, "f", {refExtracted});
  auto refInserted =
      referenceBuilder.qtensorInsert(refResults[0], refTensorIn, 0);
  referenceBuilder.qtensorDealloc(refInserted);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// An element that is taken out and put straight back at the same index,
/// without any gate in between, leaves nothing to promote once it is folded.
///
/// The folder collapses such an extract/insert pair back into the original
/// tensor, which leaves the callee as an identity function. That fold is the
/// precondition here, so it is applied explicitly: this pass does not run a
/// folder of its own, and promoting an unfolded pass-through is wasted work
/// rather than a miscompile.
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionForFoldedPassThrough) {
  const auto tensorType = getQubitTensorType(2);

  const auto buildProgram = [&tensorType](QCOProgramBuilder& b) {
    b.initialize();
    b.createFunction("f", {tensorType}, [&](ValueRange args) {
      auto [rest, inner] = b.qtensorExtract(args[0], 0);
      return SmallVector<Value>{b.qtensorInsert(inner, rest, 0)};
    });

    auto q0 = b.allocQubit();
    auto q1 = b.allocQubit();
    auto tensor = b.qtensorFromElements({q0, q1});
    auto results = callFunction(b, "f", {tensor});
    b.qtensorDealloc(results[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runCanonicalizerPass(moduleOp.get()).succeeded());
  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// Only the tensor elements the callee actually touches become scalar
/// arguments; untouched elements never cross the call boundary.
TEST_F(QCOQuantumArgumentPromotionTest, promoteOnlyUsedTensorElements) {
  const auto tensorType = getQubitTensorType(3);

  programBuilder.initialize();
  programBuilder.createFunction("f", {tensorType}, [&](ValueRange args) {
    auto [tensorIn, inner] = programBuilder.qtensorExtract(args[0], 1);
    inner = programBuilder.x(inner);
    return SmallVector<Value>{programBuilder.qtensorInsert(inner, tensorIn, 1)};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1, q2});
  auto results = callFunction(programBuilder, "f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.x(refArgs[0])};
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refQ2 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1, refQ2});
  auto [refTensorIn, refExtracted] =
      referenceBuilder.qtensorExtract(refTensor, 1);
  auto refResults = callFunction(referenceBuilder, "f", {refExtracted});
  auto refInserted =
      referenceBuilder.qtensorInsert(refResults[0], refTensorIn, 1);
  referenceBuilder.qtensorDealloc(refInserted);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// A qubit that is moved to a different slot is promoted with the
/// extraction and insertion indices kept apart.
TEST_F(QCOQuantumArgumentPromotionTest, promoteTensorElementIntoDifferentSlot) {
  const auto tensorType = getQubitTensorType(2);

  programBuilder.initialize();
  programBuilder.createFunction("f", {tensorType}, [&](ValueRange args) {
    auto [tensorIn, inner] = programBuilder.qtensorExtract(args[0], 0);
    inner = programBuilder.h(inner);
    return SmallVector<Value>{programBuilder.qtensorInsert(inner, tensorIn, 1)};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = callFunction(programBuilder, "f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.h(refArgs[0])};
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto [refTensorIn, refExtracted] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto refResults = callFunction(referenceBuilder, "f", {refExtracted});
  auto refInserted =
      referenceBuilder.qtensorInsert(refResults[0], refTensorIn, 1);
  referenceBuilder.qtensorDealloc(refInserted);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// An element that is extracted but never re-inserted cannot be promoted,
/// because the promoted callee would have nothing to hand back for that slot.
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionWithoutMatchingInsert) {
  // The builder requires borrowed registers to get every slot back, so the
  // escaping element needs hand-written IR.
  auto module = parseModule(R"mlir(
func.func private @callee(%t: tensor<2x!qco.qubit>)
    -> (!qco.qubit, tensor<2x!qco.qubit>) {
  %c0 = arith.constant 0 : index
  %rest, %q = qtensor.extract %t[%c0] : tensor<2x!qco.qubit>
  %h = qco.h %q : !qco.qubit -> !qco.qubit
  return %h, %rest : !qco.qubit, tensor<2x!qco.qubit>
}
func.func @main(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
  %escaped, %r = func.call @callee(%t)
      : (tensor<2x!qco.qubit>) -> (!qco.qubit, tensor<2x!qco.qubit>)
  qco.sink %escaped : !qco.qubit
  return %r : tensor<2x!qco.qubit>
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createQuantumArgumentPromotion()).succeeded());

  auto callee = module->lookupSymbol<func::FuncOp>("callee");
  ASSERT_TRUE(callee);
  EXPECT_TRUE(isa<RankedTensorType>(callee.getArgumentTypes()[0]))
      << "the tensor argument must survive, slot 0 never comes back";
}

/// An element whose path from extraction to re-insertion runs through a
/// call is not promoted.
///
/// The walk only recognises the operations it knows to thread a qubit. A call
/// is not one of them, and guessing that its first result carries the qubit on
/// would let a slot be promoted that no longer holds the extracted qubit, so
/// the callee is left alone.
TEST_F(QCOQuantumArgumentPromotionTest,
       noPromotionWhenCallSitsOnExtractedPath) {
  const auto tensorType = getQubitTensorType(2);

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    const auto qubitType = getQubitType();
    b.initialize();

    // The helper takes the extracted qubit as its second operand.
    b.createFunction(
        "helper", {qubitType, qubitType}, [&](ValueRange helperArgs) {
          return SmallVector<Value>{helperArgs[0], b.h(helperArgs[1])};
        });

    b.createFunction("f", {qubitType, tensorType}, [&](ValueRange args) {
      auto [rest, inner] = b.qtensorExtract(args[1], 0);
      auto helped = callFunction(b, "helper", {args[0], inner});
      return SmallVector<Value>{helped[0], b.qtensorInsert(helped[1], rest, 0)};
    });

    auto q0 = b.allocQubit();
    auto q1 = b.allocQubit();
    auto spare = b.allocQubit();
    auto tensor = b.qtensorFromElements({q0, q1});
    auto results = callFunction(b, "f", {spare, tensor});
    b.sink(results[0]);
    b.qtensorDealloc(results[1]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// A tensor argument that never has an element taken out of it has
/// nothing to promote.
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionWithoutElementAccess) {
  const auto tensorType = getQubitTensorType(2);

  programBuilder.initialize();
  programBuilder.createFunction("f", {tensorType}, [&](ValueRange args) {
    return SmallVector<Value>{args[0]};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = callFunction(programBuilder, "f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction("f", {tensorType}, [&](ValueRange refArgs) {
    return SmallVector<Value>{refArgs[0]};
  });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto refResults = callFunction(referenceBuilder, "f", {refTensor});
  referenceBuilder.qtensorDealloc(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// A callee that touches several tensor elements gets one scalar argument
/// and one scalar result per element.
TEST_F(QCOQuantumArgumentPromotionTest, promoteMultipleTensorElements) {
  const auto tensorType = getQubitTensorType(2);

  programBuilder.initialize();
  programBuilder.createFunction("f", {tensorType}, [&](ValueRange args) {
    auto [afterFirst, first] = programBuilder.qtensorExtract(args[0], 0);
    auto firstTensor =
        programBuilder.qtensorInsert(programBuilder.h(first), afterFirst, 0);
    auto [afterSecond, second] = programBuilder.qtensorExtract(firstTensor, 1);
    return SmallVector<Value>{
        programBuilder.qtensorInsert(programBuilder.x(second), afterSecond, 1),
    };
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = callFunction(programBuilder, "f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  const auto qubitType = getQubitType();
  referenceBuilder.createFunction("f", {qubitType, qubitType},
                                  [&](ValueRange refArgs) {
                                    return SmallVector<Value>{
                                        referenceBuilder.h(refArgs[0]),
                                        referenceBuilder.x(refArgs[1]),
                                    };
                                  });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  // The caller takes every promoted element out before the call and puts them
  // all back afterwards.
  auto [refAfterFirst, refFirst] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto [refAfterSecond, refSecond] =
      referenceBuilder.qtensorExtract(refAfterFirst, 1);
  auto refResults = callFunction(referenceBuilder, "f", {refFirst, refSecond});
  auto refFirstBack =
      referenceBuilder.qtensorInsert(refResults[0], refAfterSecond, 0);
  referenceBuilder.qtensorDealloc(
      referenceBuilder.qtensorInsert(refResults[1], refFirstBack, 1));
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// The promoted qubit takes the place of the tensor's result wherever that
/// result sits, here behind the outcome of measuring the element, which the
/// caller keeps reading.
TEST_F(QCOQuantumArgumentPromotionTest,
       promoteTensorReturnedAfterOrdinaryResult) {
  const auto tensorType = getQubitTensorType(2);
  const auto bitType = programBuilder.getI1Type();

  programBuilder.initialize({bitType});
  programBuilder.createFunction("f", {tensorType}, [&](ValueRange args) {
    auto [rest, inner] = programBuilder.qtensorExtract(args[0], 0);
    Value bit;
    std::tie(inner, bit) = programBuilder.measure(inner);
    return SmallVector<Value>{
        bit,
        programBuilder.qtensorInsert(inner, rest, 0),
    };
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = callFunction(programBuilder, "f", {tensor});
  programBuilder.qtensorDealloc(results[1]);
  moduleOp = programBuilder.finalize({results[0]});

  referenceBuilder.initialize({bitType});
  const auto qubitType = getQubitType();
  referenceBuilder.createFunction("f", {qubitType}, [&](ValueRange refArgs) {
    Value refBit;
    auto refInner = refArgs[0];
    std::tie(refInner, refBit) = referenceBuilder.measure(refInner);
    return SmallVector<Value>{refBit, refInner};
  });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto [refRest, refExtracted] = referenceBuilder.qtensorExtract(refTensor, 0);
  auto refResults = callFunction(referenceBuilder, "f", {refExtracted});
  referenceBuilder.qtensorDealloc(
      referenceBuilder.qtensorInsert(refResults[1], refRest, 0));
  reference = referenceBuilder.finalize({refResults[0]});

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// A tensor behind another quantum argument is promoted in place on both
/// sides of the signature, so each quantum result still continues the
/// argument at the same position.
TEST_F(QCOQuantumArgumentPromotionTest, promoteTensorAfterScalarQubit) {
  const auto tensorType = getQubitTensorType(2);

  programBuilder.initialize();
  const auto qubitType = getQubitType();
  programBuilder.createFunction(
      "f", {qubitType, tensorType}, [&](ValueRange args) {
        auto [afterFirst, first] = programBuilder.qtensorExtract(args[1], 0);
        auto [afterSecond, second] =
            programBuilder.qtensorExtract(afterFirst, 1);
        auto [control, target] = programBuilder.cx(args[0], first);
        auto firstBack = programBuilder.qtensorInsert(target, afterSecond, 0);
        auto secondBack = programBuilder.qtensorInsert(programBuilder.h(second),
                                                       firstBack, 1);
        return SmallVector<Value>{control, secondBack};
      });

  auto scalar = programBuilder.allocQubit();
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = callFunction(programBuilder, "f", {scalar, tensor});
  programBuilder.sink(results[0]);
  programBuilder.qtensorDealloc(results[1]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  const auto refQubitType = getQubitType();
  referenceBuilder.createFunction(
      "f", {refQubitType, refQubitType, refQubitType}, [&](ValueRange refArgs) {
        auto [refControl, refTarget] =
            referenceBuilder.cx(refArgs[0], refArgs[1]);
        return SmallVector<Value>{
            refControl,
            refTarget,
            referenceBuilder.h(refArgs[2]),
        };
      });

  auto refScalar = referenceBuilder.allocQubit();
  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto [refAfterFirst, refFirst] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto [refAfterSecond, refSecond] =
      referenceBuilder.qtensorExtract(refAfterFirst, 1);
  auto refResults =
      callFunction(referenceBuilder, "f", {refScalar, refFirst, refSecond});
  auto refFirstBack =
      referenceBuilder.qtensorInsert(refResults[1], refAfterSecond, 0);
  auto refTensorBack =
      referenceBuilder.qtensorInsert(refResults[2], refFirstBack, 1);
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.qtensorDealloc(refTensorBack);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/// A slot the callee writes before reading again must not be promoted.
///
/// Extractions move in front of the call and insertions behind it, so such a
/// read would be served from the caller's original tensor. Here the callee
/// computes `x(h(slot 0))`, which promotion would turn into `x(slot 1)`.
TEST_F(QCOQuantumArgumentPromotionTest,
       noPromotionWhenSlotIsWrittenBeforeItIsRead) {
  auto module = parseModule(R"mlir(
func.func private @callee(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t1, %q0 = qtensor.extract %t[%c0] : tensor<2x!qco.qubit>
  %q0h = qco.h %q0 : !qco.qubit -> !qco.qubit
  %t2 = qtensor.insert %q0h into %t1[%c1] : tensor<2x!qco.qubit>
  %t3, %q1 = qtensor.extract %t2[%c1] : tensor<2x!qco.qubit>
  %q1x = qco.x %q1 : !qco.qubit -> !qco.qubit
  %t4 = qtensor.insert %q1x into %t3[%c0] : tensor<2x!qco.qubit>
  return %t4 : tensor<2x!qco.qubit>
}
func.func @main(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
  %r = func.call @callee(%t) : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
  return %r : tensor<2x!qco.qubit>
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createQuantumArgumentPromotion()).succeeded());

  auto callee = module->lookupSymbol<func::FuncOp>("callee");
  ASSERT_TRUE(callee);
  EXPECT_TRUE(isa<RankedTensorType>(callee.getArgumentTypes()[0]))
      << "the tensor argument must survive, the accesses depend on each other";
}

/// An insertion that does not belong to a promoted slot blocks promotion.
///
/// It survives the rewrite still using the tensor argument that is erased right
/// afterwards, which used to abort on MLIR's `use_empty()` assertion.
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionForUnmatchedInsertOnChain) {
  auto module = parseModule(R"mlir(
func.func private @callee(%t: tensor<2x!qco.qubit>, %extra: !qco.qubit) -> tensor<2x!qco.qubit> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t1, %q0 = qtensor.extract %t[%c0] : tensor<2x!qco.qubit>
  %q0h = qco.h %q0 : !qco.qubit -> !qco.qubit
  %t2 = qtensor.insert %q0h into %t1[%c0] : tensor<2x!qco.qubit>
  %t3 = qtensor.insert %extra into %t2[%c1] : tensor<2x!qco.qubit>
  return %t3 : tensor<2x!qco.qubit>
}
func.func @main(%t: tensor<2x!qco.qubit>, %e: !qco.qubit) -> tensor<2x!qco.qubit> {
  %r = func.call @callee(%t, %e) : (tensor<2x!qco.qubit>, !qco.qubit) -> tensor<2x!qco.qubit>
  return %r : tensor<2x!qco.qubit>
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createQuantumArgumentPromotion()).succeeded());

  auto callee = module->lookupSymbol<func::FuncOp>("callee");
  ASSERT_TRUE(callee);
  EXPECT_TRUE(isa<RankedTensorType>(callee.getArgumentTypes()[0]))
      << "the tensor argument must survive, one insertion is unmatched";
}

// ==========================================================================

} // namespace
