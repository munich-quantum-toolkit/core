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
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/Utils/CallTensorMapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <memory>

using namespace mlir;
using namespace mlir::qtensor;
using namespace mlir::qco;

namespace {
class CallTensorMappingFixture : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect, QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;

  [[nodiscard]] OwningOpRef<ModuleOp> parseModule(StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  [[nodiscard]] static func::CallOp findCall(Operation* root,
                                             StringRef callee) {
    func::CallOp found;
    root->walk([&](func::CallOp call) {
      if (call.getCallee() == callee) {
        found = call;
      }
    });
    return found;
  }
};
} // namespace

TEST_F(CallTensorMappingFixture, FollowsNestedReordering) {
  auto module = parseModule(R"mlir(
func.func private @swap(
    %flag: i1, %a: tensor<2x!qco.qubit>, %b: tensor<2x!qco.qubit>)
    -> (i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>) {
  return %flag, %b, %a
      : i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>
}
func.func private @outer(
    %flag: i1, %a: tensor<2x!qco.qubit>, %b: tensor<2x!qco.qubit>)
    -> (i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>) {
  %r:3 = func.call @swap(%flag, %a, %b)
      : (i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>)
      -> (i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>)
  return %r#0, %r#1, %r#2
      : i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>
}
func.func @main() {
  %flag = arith.constant true
  %c2 = arith.constant 2 : index
  %a = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
  %b = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
  %r:3 = func.call @outer(%flag, %a, %b)
      : (i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>)
      -> (i1, tensor<2x!qco.qubit>, tensor<2x!qco.qubit>)
  qtensor.dealloc %r#1 : tensor<2x!qco.qubit>
  qtensor.dealloc %r#2 : tensor<2x!qco.qubit>
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  auto call = findCall(main, "outer");
  ASSERT_TRUE(call);

  CallTensorMapping mapping;
  auto mapped = mapping.getResultForOperand(call, call.getOperand(1));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(2));
  mapped = mapping.getResultForOperand(call, call.getOperand(2));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(1));
}

TEST_F(CallTensorMappingFixture, ReportsAKeptTensor) {
  auto module = parseModule(R"mlir(
func.func private @consume(%t: tensor<2x!qco.qubit>) {
  qtensor.dealloc %t : tensor<2x!qco.qubit>
  return
}
func.func @main() {
  %c2 = arith.constant 2 : index
  %t = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
  func.call @consume(%t) : (tensor<2x!qco.qubit>) -> ()
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto call = findCall(module->lookupSymbol<func::FuncOp>("main"), "consume");
  ASSERT_TRUE(call);

  CallTensorMapping mapping;
  auto mapped = mapping.getResultForOperand(call, call.getOperand(0));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_FALSE(*mapped);
}

TEST_F(CallTensorMappingFixture, FailsClosed) {
  auto module = parseModule(R"mlir(
func.func private @external(tensor<2x!qco.qubit>)
    -> tensor<2x!qco.qubit>
func.func private @recursive(%t: tensor<2x!qco.qubit>)
    -> tensor<2x!qco.qubit> {
  %r = func.call @recursive(%t)
      : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
  return %r : tensor<2x!qco.qubit>
}
func.func @main() {
  %c2 = arith.constant 2 : index
  %a = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
  %x = func.call @external(%a)
      : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
  qtensor.dealloc %x : tensor<2x!qco.qubit>
  %b = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
  %y = func.call @recursive(%b)
      : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
  qtensor.dealloc %y : tensor<2x!qco.qubit>
  return
}
)mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  auto external = findCall(main, "external");
  auto recursive = findCall(main, "recursive");
  ASSERT_TRUE(external);
  ASSERT_TRUE(recursive);

  // A declaration has no body to follow, and recursion would not terminate.
  CallTensorMapping mapping;
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(external, external.getOperand(0))));
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(recursive, recursive.getOperand(0))));
}
