/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/CallQubitMapping.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <iterator>
#include <memory>

using namespace mlir;
using namespace mlir::qco;

namespace {
class CallQubitMappingFixture : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;

  [[nodiscard]] OwningOpRef<ModuleOp> parseModule(StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  template <typename OpT> [[nodiscard]] static OpT findOp(Operation* root) {
    OpT found;
    root->walk([&](OpT op) {
      if (!found) {
        found = op;
      }
    });
    return found;
  }
};
} // namespace

TEST_F(CallQubitMappingFixture, FollowsNestedReordering) {
  auto module = parseModule(R"mlir(
func.func private @swap(%flag: i1, %a: !qco.qubit, %b: !qco.qubit)
    -> (i1, !qco.qubit, !qco.qubit) {
  return %flag, %b, %a : i1, !qco.qubit, !qco.qubit
}
func.func private @outer(%flag: i1, %a: !qco.qubit, %b: !qco.qubit)
    -> (i1, !qco.qubit, !qco.qubit) {
  %r:3 = func.call @swap(%flag, %a, %b)
      : (i1, !qco.qubit, !qco.qubit)
      -> (i1, !qco.qubit, !qco.qubit)
  return %r#0, %r#1, %r#2 : i1, !qco.qubit, !qco.qubit
}
func.func @main() {
  %flag = arith.constant true
  %a = qco.alloc : !qco.qubit
  %b = qco.alloc : !qco.qubit
  %r:3 = func.call @outer(%flag, %a, %b)
      : (i1, !qco.qubit, !qco.qubit)
      -> (i1, !qco.qubit, !qco.qubit)
  qco.sink %r#1 : !qco.qubit
  qco.sink %r#2 : !qco.qubit
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  auto call = findOp<func::CallOp>(main);

  // Both callees hand the qubits back swapped, so the outer call restores the
  // original order only if the mapping follows the nested call too.
  CallQubitMapping mapping;
  auto mapped = mapping.getResultForOperand(call, call.getOperand(1));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(2));
  mapped = mapping.getResultForOperand(call, call.getOperand(2));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(1));
}

TEST_F(CallQubitMappingFixture, InvalidateObservesAChangedCallee) {
  auto module = parseModule(R"mlir(
func.func private @swap(%a: !qco.qubit, %b: !qco.qubit)
    -> (!qco.qubit, !qco.qubit) {
  return %b, %a : !qco.qubit, !qco.qubit
}
func.func @main() {
  %a = qco.alloc : !qco.qubit
  %b = qco.alloc : !qco.qubit
  %r:2 = func.call @swap(%a, %b)
      : (!qco.qubit, !qco.qubit) -> (!qco.qubit, !qco.qubit)
  qco.sink %r#0 : !qco.qubit
  qco.sink %r#1 : !qco.qubit
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  auto call = findOp<func::CallOp>(main);

  CallQubitMapping mapping;
  auto mapped = mapping.getResultForOperand(call, call.getOperand(0));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(1));

  // Rewriting the callee to return its arguments in order invalidates what the
  // mapping cached for it.
  auto swap = module->lookupSymbol<func::FuncOp>("swap");
  auto returnOp = cast<func::ReturnOp>(swap.getBody().front().getTerminator());
  returnOp->setOperands(swap.getArguments());

  mapped = mapping.getResultForOperand(call, call.getOperand(0));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(1)) << "expected the stale cached mapping";

  mapping.invalidate();
  mapped = mapping.getResultForOperand(call, call.getOperand(0));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(0));
}

TEST_F(CallQubitMappingFixture, DistinguishesKeptAndCreatedQubits) {
  auto module = parseModule(R"mlir(
func.func private @replace(%old: !qco.qubit) -> !qco.qubit {
  qco.sink %old : !qco.qubit
  %new = qco.alloc : !qco.qubit
  return %new : !qco.qubit
}
func.func @main() {
  %old = qco.alloc : !qco.qubit
  %new = func.call @replace(%old) : (!qco.qubit) -> !qco.qubit
  qco.sink %new : !qco.qubit
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  auto call = findOp<func::CallOp>(main);
  Value old = findOp<qco::AllocOp>(main).getResult();

  // The callee sinks the qubit it is given and allocates the one it returns,
  // so the operand continues into no result at all.
  CallQubitMapping mapping;
  auto mapped = mapping.getResultForOperand(call, old);
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_FALSE(*mapped);
}

TEST_F(CallQubitMappingFixture, FailsClosed) {
  auto module = parseModule(R"mlir(
func.func private @external(!qco.qubit) -> !qco.qubit
func.func private @recursive(%q: !qco.qubit) -> !qco.qubit {
  %r = func.call @recursive(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
func.func private @unknown(%q: !qco.qubit) -> !qco.qubit {
  %r = builtin.unrealized_conversion_cast %q : !qco.qubit to !qco.qubit
  return %r : !qco.qubit
}
func.func @main() {
  %a = qco.alloc : !qco.qubit
  %x = func.call @external(%a) : (!qco.qubit) -> !qco.qubit
  qco.sink %x : !qco.qubit
  %b = qco.alloc : !qco.qubit
  %y = func.call @recursive(%b) : (!qco.qubit) -> !qco.qubit
  qco.sink %y : !qco.qubit
  %c = qco.alloc : !qco.qubit
  %z = func.call @unknown(%c) : (!qco.qubit) -> !qco.qubit
  qco.sink %z : !qco.qubit
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  func::CallOp external;
  func::CallOp recursive;
  func::CallOp unknown;
  main.walk([&](func::CallOp call) {
    if (call.getCallee() == "external") {
      external = call;
    } else if (call.getCallee() == "recursive") {
      recursive = call;
    } else {
      unknown = call;
    }
  });
  ASSERT_TRUE(external);
  ASSERT_TRUE(recursive);
  ASSERT_TRUE(unknown);

  // A declaration has no body to follow, recursion would not terminate, and an
  // operation the wire iterator does not know is not safe to interpret.
  CallQubitMapping mapping;
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(external, external.getOperand(0))));
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(recursive, recursive.getOperand(0))));
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(unknown, unknown.getOperand(0))));
}

TEST_F(CallQubitMappingFixture, GenericCallIsAWireBoundary) {
  auto module = parseModule(R"mlir(
func.func private @thread(%q: !qco.qubit) -> !qco.qubit {
  return %q : !qco.qubit
}
func.func @main() {
  %q = qco.alloc : !qco.qubit
  %r = func.call @thread(%q) : (!qco.qubit) -> !qco.qubit
  qco.sink %r : !qco.qubit
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  auto call = findOp<func::CallOp>(main);
  Value qubit = findOp<qco::AllocOp>(main).getResult();

  // The iterator itself stops at the call. Stepping over it is the mapping's
  // job, which is why the two are used together.
  WireIterator iterator(qubit);
  ++iterator;
  ASSERT_NE(iterator, std::default_sentinel);
  EXPECT_EQ(iterator.operation(), call.getOperation());
  EXPECT_FALSE(WireIterator::isTail(call.getOperation()));
  ++iterator;
  EXPECT_EQ(iterator, std::default_sentinel);

  CallQubitMapping mapping;
  auto mapped = mapping.getResultForOperand(call, qubit);
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(0));
}
