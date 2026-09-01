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
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>

using namespace mlir;
using namespace mlir::qco;

/**
 * @brief Build a program that constructs a GHZ state using a loop.
 * @param context The MLIR context to build the module.
 * @param n The number of qubits of the GHZ state.
 * @return A module with an entry point function containing the GHZ logic.
 */
static OwningOpRef<ModuleOp> getGHZ(MLIRContext* context, int64_t n) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  Value tensor = builder.qtensorAlloc(n);
  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  q0 = builder.h(q0);
  tensor = builder.qtensorInsert(q0, tensor, 0);

  auto out = builder.scfFor(
      1, n, 1, {tensor}, [&builder](Value iv, ValueRange iterArgs) {
        Value loopTensor = iterArgs[0];
        Value ctrl;
        Value targ;

        std::tie(loopTensor, ctrl) = builder.qtensorExtract(loopTensor, 0);
        std::tie(loopTensor, targ) = builder.qtensorExtract(loopTensor, iv);

        std::tie(ctrl, targ) = builder.cx(ctrl, targ);

        loopTensor = builder.qtensorInsert(ctrl, loopTensor, 0);
        loopTensor = builder.qtensorInsert(targ, loopTensor, iv);

        return SmallVector{loopTensor};
      });

  tensor = out[0];

  builder.qtensorDealloc(tensor);

  return builder.finalize();
}

namespace {

class QuantumLoopUnrollTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static LogicalResult runPass(OwningOpRef<ModuleOp>& program,
                               const QuantumLoopUnrollOptions options) {
    PassManager pm(program->getContext());
    pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll(options));
    return pm.run(*program);
  }

  std::unique_ptr<MLIRContext> context;
};

}; // namespace

TEST_F(QuantumLoopUnrollTest, InvalidUnrollFactor) {
  auto m = getGHZ(context.get(), 2);

  const auto res = runPass(m, QuantumLoopUnrollOptions{.unrollFactor = -2});
  ASSERT_TRUE(res.failed());
}

TEST_F(QuantumLoopUnrollTest, ExcessiveExplicitFactorFailureIsAtomic) {
  auto module = getGHZ(context.get(), 2);
  OwningOpRef<ModuleOp> original(module->clone());

  EXPECT_TRUE(
      failed(runPass(module, QuantumLoopUnrollOptions{.unrollFactor = 4097})));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QuantumLoopUnrollTest,
       ExcessiveExplicitFactorRejectsIdentityLoopWithoutMutation) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    %out = scf.for %iv = %lb to %ub step %step
        iter_args(%arg = %q) -> (!qco.qubit) {
      scf.yield %arg : !qco.qubit
    }
    qco.sink %out : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  EXPECT_TRUE(
      failed(runPass(module, QuantumLoopUnrollOptions{.unrollFactor = 4097})));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QuantumLoopUnrollTest, ExcessiveStaticTripCountFailureIsAtomic) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 4097 : index
    %step = arith.constant 1 : index
    %out = scf.for %iv = %lb to %ub step %step
        iter_args(%arg = %q) -> (!qco.qubit) {
      %next = qco.x %arg : !qco.qubit -> !qco.qubit
      scf.yield %next : !qco.qubit
    }
    qco.sink %out : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  EXPECT_TRUE(failed(runPass(module, QuantumLoopUnrollOptions{})));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QuantumLoopUnrollTest, NestedExpansionBudgetFailureIsAtomic) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 400 : index
    %step = arith.constant 1 : index
    %out = scf.for %outer = %lb to %ub step %step
        iter_args(%outer_arg = %q) -> (!qco.qubit) {
      %inner_out = scf.for %inner = %lb to %ub step %step
          iter_args(%inner_arg = %outer_arg) -> (!qco.qubit) {
        %next = qco.x %inner_arg : !qco.qubit -> !qco.qubit
        scf.yield %next : !qco.qubit
      }
      scf.yield %inner_out : !qco.qubit
    }
    qco.sink %out : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  EXPECT_TRUE(failed(runPass(module, QuantumLoopUnrollOptions{})));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QuantumLoopUnrollTest, NoOp) {
  auto m = getGHZ(context.get(), 2);
  auto mClone = m->clone();

  const auto res = runPass(m, QuantumLoopUnrollOptions{.unrollFactor = 0});
  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(mlir::OperationEquivalence::isEquivalentTo(
      m->getOperation(), mClone.getOperation(),
      mlir::OperationEquivalence::Flags::None));
}

TEST_F(QuantumLoopUnrollTest, PreservesYieldOnlyPermutation) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qco.static 0 : !qco.qubit
    %q1 = qco.static 1 : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %out0, %out1 = scf.for %iv = %lb to %ub step %step
        iter_args(%left = %q0, %right = %q1)
        -> (!qco.qubit, !qco.qubit) {
      scf.yield %right, %left : !qco.qubit, !qco.qubit
    }
    qco.sink %out0 : !qco.qubit
    qco.sink %out1 : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runPass(module, QuantumLoopUnrollOptions{})));
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<SinkOp> sinks;
  module->walk([&](SinkOp sink) { sinks.push_back(sink); });
  ASSERT_EQ(sinks.size(), 2);
  auto first = sinks[0].getQubit().getDefiningOp<StaticOp>();
  auto second = sinks[1].getQubit().getDefiningOp<StaticOp>();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.getIndex(), 1);
  EXPECT_EQ(second.getIndex(), 0);
}

TEST_F(QuantumLoopUnrollTest, DynamicTripCountFailureIsAtomic) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%upper: index) attributes {mqt.entry_point} {
    %q0 = qco.static 0 : !qco.qubit
    %q1 = qco.static 1 : !qco.qubit
    %lb = arith.constant 0 : index
    %static_ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    %out0 = scf.for %iv = %lb to %static_ub step %step
        iter_args(%q = %q0) -> (!qco.qubit) {
      %next = qco.x %q : !qco.qubit -> !qco.qubit
      scf.yield %next : !qco.qubit
    }
    %out1 = scf.for %iv = %lb to %upper step %step
        iter_args(%q = %q1) -> (!qco.qubit) {
      %next = qco.h %q : !qco.qubit -> !qco.qubit
      scf.yield %next : !qco.qubit
    }
    qco.sink %out0 : !qco.qubit
    qco.sink %out1 : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OwningOpRef<ModuleOp> original(module->clone());

  EXPECT_TRUE(failed(runPass(module, QuantumLoopUnrollOptions{})));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QuantumLoopUnrollTest, UnrollFull) {
  auto m = getGHZ(context.get(), 3);
  auto entry = *(m->getOps<func::FuncOp>().begin());

  EXPECT_EQ(range_size(entry.getOps<scf::ForOp>()), 1);
  EXPECT_EQ(range_size(entry.getOps<qtensor::ExtractOp>()), 1);
  EXPECT_EQ(range_size(entry.getOps<qtensor::InsertOp>()), 1);

  const auto res = runPass(m, QuantumLoopUnrollOptions{});
  ASSERT_TRUE(res.succeeded());

  // After the pass, there are no more loops and all extracts and inserts are
  // placed inside the function.

  EXPECT_EQ(range_size(entry.getOps<scf::ForOp>()), 0);
  EXPECT_EQ(range_size(entry.getOps<qtensor::ExtractOp>()), 5);
  EXPECT_EQ(range_size(entry.getOps<qtensor::InsertOp>()), 5);
}

TEST_F(QuantumLoopUnrollTest, UnrollsFunctionWithSiblingSymbolReference) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func private @helper()
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    %out = scf.for %iv = %lb to %ub step %step
        iter_args(%arg = %q) -> (!qco.qubit) {
      func.call @helper() : () -> ()
      %next = qco.x %arg : !qco.qubit -> !qco.qubit
      scf.yield %next : !qco.qubit
    }
    qco.sink %out : !qco.qubit
    return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  ASSERT_TRUE(succeeded(runPass(module, QuantumLoopUnrollOptions{})));
  ASSERT_TRUE(succeeded(verify(*module)));

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  EXPECT_TRUE(main.getOps<scf::ForOp>().empty());
  EXPECT_EQ(llvm::range_size(main.getOps<func::CallOp>()), 2U);
}

TEST_F(QuantumLoopUnrollTest, UnrollFullWithOuterDependentBounds) {
  auto m = QCOProgramBuilder::build(context.get(), [](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(2);
    auto upper = arith::ConstantIndexOp::create(b, 2).getResult();
    auto step = arith::ConstantIndexOp::create(b, 1).getResult();
    tensor =
        b.scfFor(0, 2, 1, {tensor}, [&](Value outer, ValueRange outerArgs) {
          auto lower = arith::AddIOp::create(b, outer, step).getResult();
          return b.scfFor(
              lower, upper, step, outerArgs, [&](Value, ValueRange innerArgs) {
                auto tensor = innerArgs.front();
                Value qubit;
                std::tie(tensor, qubit) = b.qtensorExtract(tensor, 0);
                tensor = b.qtensorInsert(b.h(qubit), tensor, 0);
                return SmallVector{tensor};
              });
        })[0];
    b.qtensorDealloc(tensor);
    return b.intConstant(0);
  });
  ASSERT_TRUE(m);

  EXPECT_TRUE(succeeded(runPass(m, QuantumLoopUnrollOptions{})));
  auto entry = *m->getOps<func::FuncOp>().begin();
  EXPECT_TRUE(entry.getOps<scf::ForOp>().empty());
  EXPECT_EQ(range_size(entry.getOps<HOp>()), 1);
}

TEST_F(QuantumLoopUnrollTest, UnrollPartial) {
  auto m = getGHZ(context.get(), 9);
  auto entry = *(m->getOps<func::FuncOp>().begin());

  EXPECT_EQ(range_size(entry.getOps<scf::ForOp>()), 1);
  EXPECT_EQ(range_size(entry.getOps<qtensor::ExtractOp>()), 1);
  EXPECT_EQ(range_size(entry.getOps<qtensor::InsertOp>()), 1);

  const auto res = runPass(m, QuantumLoopUnrollOptions{.unrollFactor = 2});
  ASSERT_TRUE(res.succeeded());

  // The extraction and insertion of q0 (and the subsequent application) of the
  // hadamard stays inside the function body.
  EXPECT_EQ(range_size(entry.getOps<qtensor::ExtractOp>()), 1);
  EXPECT_EQ(range_size(entry.getOps<qtensor::InsertOp>()), 1);

  // After the pass, there are is still a loop, however with step size = 2.
  // Where previously, the loop consists of 2 extracts and 2 inserts (q0, qi),
  // after the pass it consists of 4 extracts and 4 inserts.

  EXPECT_EQ(range_size(entry.getOps<scf::ForOp>()), 1);

  Region& body = (*(entry.getOps<scf::ForOp>().begin())).getRegion();
  EXPECT_EQ(range_size(body.getOps<qtensor::ExtractOp>()), 4);
  EXPECT_EQ(range_size(body.getOps<qtensor::InsertOp>()), 4);
}
