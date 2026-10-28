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
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Utils/TensorIterator.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>

using namespace mlir;
using namespace mlir::qtensor;
using namespace mlir::qco;

namespace {

class TensorIteratorTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect, QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parseModule(StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }
};
} // namespace

TEST_F(TensorIteratorTest, Traversal) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  constexpr int64_t n = 3;
  auto tensor0 = builder.qtensorAlloc(n);
  auto [tensor1, q00] = builder.qtensorExtract(tensor0, 0);
  auto q01 = builder.h(q00);
  auto tensor2 = builder.qtensorInsert(q01, tensor1, 0);
  auto [tensor3, q02] = builder.qtensorExtract(tensor2, 0);
  auto [tensor4, q10] = builder.qtensorExtract(tensor3, 1);
  auto [q03, q11] = builder.cx(q02, q10);
  auto tensor5 = builder.qtensorInsert(q03, tensor4, 0);
  auto tensor6 = builder.qtensorInsert(q11, tensor5, 1);
  auto tensor7 = builder.scfFor(
      1, n, 1, {tensor6}, [&builder](Value iv, ValueRange iterArgs) {
        auto [loopTensor, q] = builder.qtensorExtract(iterArgs[0], iv);
        q = builder.h(q);
        loopTensor = builder.qtensorInsert(q, loopTensor, 0);
        return SmallVector{loopTensor};
      })[0];

  Value tensorThen0;
  Value tensorThen1;
  Value tensorThen2;

  Value tensorElse0;
  Value tensorElse1;
  Value tensorElse2;

  auto tensor8 = builder.qcoIf(
      false, tensor7,
      [&](ValueRange args) -> SmallVector<Value> {
        Value q;
        tensorThen0 = args[0];
        std::tie(tensorThen1, q) = builder.qtensorExtract(tensorThen0, 0);
        q = builder.h(q);
        tensorThen2 = builder.qtensorInsert(q, tensorThen1, 0);
        return SmallVector{tensorThen2};
      },
      [&](ValueRange args) -> SmallVector<Value> {
        Value q;
        tensorElse0 = args[0];
        std::tie(tensorElse1, q) = builder.qtensorExtract(tensorElse0, 0);
        q = builder.t(q);
        tensorElse2 = builder.qtensorInsert(q, tensorElse1, 0);
        return SmallVector{tensorElse2};
      })[0];
  const auto identity = [](ValueRange args) { return llvm::to_vector(args); };
  const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies{
      identity,
  };
  auto tensor9 = builder.qcoIndexSwitch(0, tensor8, SmallVector<int64_t>{0},
                                        caseBodies, identity)[0];
  builder.qtensorDealloc(tensor9);
  [[maybe_unused]] auto m = builder.finalize();

  TensorIterator it(cast<TypedValue<RankedTensorType>>(tensor0));

  ASSERT_EQ(it.operation(), tensor0.getDefiningOp()); // qtensor.alloc
  ASSERT_EQ(it.tensor(), tensor0);

  ++it;
  ASSERT_EQ(it.operation(), tensor1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor1);

  ++it;
  ASSERT_EQ(it.operation(), tensor2.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor2);

  ++it;
  ASSERT_EQ(it.operation(), tensor3.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor3);

  ++it;
  ASSERT_EQ(it.operation(), tensor4.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor4);

  ++it;
  ASSERT_EQ(it.operation(), tensor5.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor5);

  ++it;
  ASSERT_EQ(it.operation(), tensor6.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor6);

  ++it;
  ASSERT_EQ(it.operation(), tensor7.getDefiningOp()); // scf.for
  ASSERT_EQ(it.tensor(), tensor7);

  ++it;
  ASSERT_EQ(it.operation(), tensor8.getDefiningOp()); // qco.if
  ASSERT_EQ(it.tensor(), tensor8);

  ++it;
  ASSERT_EQ(it.operation(), tensor9.getDefiningOp()); // qco.index_switch
  ASSERT_EQ(it.tensor(), tensor9);

  ++it;
  ASSERT_EQ(it.operation(), *(tensor9.user_begin())); // qtensor.dealloc
  ASSERT_EQ(it.tensor(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_EQ(it.operation(), *(tensor9.user_begin())); // qtensor.dealloc
  ASSERT_EQ(it.tensor(), nullptr);

  --it;
  ASSERT_EQ(it.operation(), tensor9.getDefiningOp()); // qco.index_switch
  ASSERT_EQ(it.tensor(), tensor9);

  --it;
  ASSERT_EQ(it.operation(), tensor8.getDefiningOp()); // qco.if
  ASSERT_EQ(it.tensor(), tensor8);

  --it;
  ASSERT_EQ(it.operation(), tensor7.getDefiningOp()); // scf.for
  ASSERT_EQ(it.tensor(), tensor7);

  --it;
  ASSERT_EQ(it.operation(), tensor6.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor6);

  --it;
  ASSERT_EQ(it.operation(), tensor5.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor5);

  --it;
  ASSERT_EQ(it.operation(), tensor4.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor4);

  --it;
  ASSERT_EQ(it.operation(), tensor3.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor3);

  --it;
  ASSERT_EQ(it.operation(), tensor2.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor2);

  --it;
  ASSERT_EQ(it.operation(), tensor1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor1);

  --it;
  ASSERT_EQ(it.operation(), tensor0.getDefiningOp()); // qtensor.alloc
  ASSERT_EQ(it.tensor(), tensor0);

  --it;
  ASSERT_EQ(it.operation(), tensor0.getDefiningOp()); // qtensor.alloc
  ASSERT_EQ(it.tensor(), tensor0);

  //
  // Test recursive use with block-argument.
  //

  TensorIterator recIt(cast<TypedValue<RankedTensorType>>(tensorElse0));

  ASSERT_EQ(recIt.operation(), nullptr);
  ASSERT_EQ(recIt.tensor(), tensorElse0);

  ++recIt;
  ASSERT_EQ(recIt.operation(), tensorElse1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(recIt.tensor(), tensorElse1);

  ++recIt;
  ASSERT_EQ(recIt.operation(), tensorElse2.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(recIt.tensor(), tensorElse2);

  ++recIt;
  ASSERT_EQ(recIt.operation(), *(tensorElse2.user_begin())); // qco.yield
  ASSERT_EQ(recIt.tensor(), nullptr);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  --recIt;
  ASSERT_EQ(recIt.operation(), *(tensorElse2.user_begin())); // qco.yield
  ASSERT_EQ(recIt.tensor(), nullptr);

  --recIt;
  ASSERT_EQ(recIt.operation(), tensorElse2.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(recIt.tensor(), tensorElse2);

  --recIt;
  ASSERT_EQ(recIt.operation(), tensorElse1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(recIt.tensor(), tensorElse1);

  --recIt;
  ASSERT_EQ(recIt.operation(), nullptr);
  ASSERT_EQ(recIt.tensor(), tensorElse0);

  --recIt;
  ASSERT_EQ(recIt.operation(), nullptr);
  ASSERT_EQ(recIt.tensor(), tensorElse0);
}

TEST_F(TensorIteratorTest, CallResultStartsALifeChain) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
func.func private @relabel(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
  return %t : tensor<2x!qco.qubit>
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %in = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
  %out = func.call @relabel(%in) : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
  %rest, %q = qtensor.extract %out[%c0] : tensor<2x!qco.qubit>
  %h = qco.h %q : !qco.qubit
  %back = qtensor.insert %h into %rest[%c0] : tensor<2x!qco.qubit>
  qtensor.dealloc %back : tensor<2x!qco.qubit>
  return
}
)mlir",
                                              context.get());
  ASSERT_TRUE(moduleOp);

  func::CallOp call;
  ExtractOp extract;
  moduleOp->walk([&](Operation* op) {
    if (auto c = dyn_cast<func::CallOp>(op)) {
      call = c;
    }
    if (auto e = dyn_cast<ExtractOp>(op)) {
      extract = e;
    }
  });
  ASSERT_TRUE(call);
  ASSERT_TRUE(extract);

  auto result = cast<TypedValue<RankedTensorType>>(call.getResult(0));
  TensorIterator it(extract.getOutTensor());
  ASSERT_EQ(it.operation(), extract.getOperation());

  --it;
  EXPECT_EQ(it.operation(), call.getOperation());
  EXPECT_EQ(it.tensor(), result);

  // The call produced this tensor, so the chain starts here: stepping back
  // again must not walk into the tensor that was passed in.
  --it;
  EXPECT_EQ(it.operation(), call.getOperation());
  EXPECT_EQ(it.tensor(), result);
}

TEST_F(TensorIteratorTest, TraversesMixedResultConditionals) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%condition: i1, %selector: index) -> i64 {
        %c1 = arith.constant 1 : index
        %tensor0 = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
        %if_state, %tensor1 = qco.if %condition
            args(%arg0 = %tensor0) -> (i64, tensor<1x!qco.qubit>) {
          %then = arith.constant 1 : i64
          qco.yield %then, %arg0 : i64, tensor<1x!qco.qubit>
        } else args(%arg0 = %tensor0) {
          %else = arith.constant 2 : i64
          qco.yield %else, %arg0 : i64, tensor<1x!qco.qubit>
        }
        %switch_state, %tensor2 = qco.index_switch %selector
            -> (i64, tensor<1x!qco.qubit>)
        case 0 args(%arg0 = %tensor1) {
          qco.yield %if_state, %arg0 : i64, tensor<1x!qco.qubit>
        }
        default args(%arg0 = %tensor1) {
          %default = arith.constant 3 : i64
          qco.yield %default, %arg0 : i64, tensor<1x!qco.qubit>
        }
        qtensor.dealloc %tensor2 : tensor<1x!qco.qubit>
        return %switch_state : i64
      }
    }
  )mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  qtensor::AllocOp alloc;
  moduleOp->walk([&](qtensor::AllocOp candidate) { alloc = candidate; });
  ASSERT_TRUE(alloc);

  TensorIterator iterator(alloc.getResult());
  EXPECT_EQ(iterator.operation(), alloc.getOperation());
  ++iterator;
  ASSERT_TRUE(isa<qco::IfOp>(iterator.operation()));
  auto ifOp = cast<qco::IfOp>(iterator.operation());
  EXPECT_EQ(iterator.tensor(), ifOp.getLinearResults().front());
  ++iterator;
  ASSERT_TRUE(isa<qco::IndexSwitchOp>(iterator.operation()));
  auto switchOp = cast<qco::IndexSwitchOp>(iterator.operation());
  EXPECT_EQ(iterator.tensor(), switchOp.getLinearResults().front());
  ++iterator;
  EXPECT_TRUE(isa<qtensor::DeallocOp>(iterator.operation()));
  EXPECT_EQ(iterator.tensor(), nullptr);

  --iterator;
  EXPECT_EQ(iterator.operation(), switchOp.getOperation());
  EXPECT_EQ(iterator.tensor(), switchOp.getLinearResults().front());
  --iterator;
  EXPECT_EQ(iterator.operation(), ifOp.getOperation());
  EXPECT_EQ(iterator.tensor(), ifOp.getLinearResults().front());
  --iterator;
  EXPECT_EQ(iterator.operation(), alloc.getOperation());
  EXPECT_EQ(iterator.tensor(), alloc.getResult());
}

TEST_F(TensorIteratorTest, TraversesWhileCarriedTensors) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    func.func private @touch(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
      return %t : tensor<2x!qco.qubit>
    }
    func.func @main() {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %zero = arith.constant 0.0 : f64
      %false = arith.constant false
      %t0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
      %q = qco.alloc : !qco.qubit
      %t1 = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
      %out:4 = scf.while (%c = %zero, %a = %t0, %b = %q, %d = %t1)
          : (f64, tensor<2x!qco.qubit>, !qco.qubit, tensor<3x!qco.qubit>)
          -> (tensor<2x!qco.qubit>, !qco.qubit, i1, tensor<3x!qco.qubit>) {
        %called = func.call @touch(%a) : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
        scf.condition(%false) %called, %b, %false, %d
            : tensor<2x!qco.qubit>, !qco.qubit, i1, tensor<3x!qco.qubit>
      } do {
      ^bb0(%a: tensor<2x!qco.qubit>, %b: !qco.qubit, %c: i1, %d: tensor<3x!qco.qubit>):
        scf.yield %zero, %a, %b, %d
            : f64, tensor<2x!qco.qubit>, !qco.qubit, tensor<3x!qco.qubit>
      }
      qtensor.dealloc %out#0 : tensor<2x!qco.qubit>
      qco.sink %out#1 : !qco.qubit
      qtensor.dealloc %out#3 : tensor<3x!qco.qubit>
      return
    }
  )mlir",
                                              context.get());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  scf::WhileOp loop;
  moduleOp->walk([&](scf::WhileOp op) { loop = op; });
  ASSERT_TRUE(loop);
  qco::WireIterator wire(loop.getInits()[2]);
  ++wire;
  EXPECT_EQ(wire.qubit(), loop.getResult(1));
  --wire;
  EXPECT_EQ(wire.qubit(), loop.getInits()[2]);
  for (auto [input, output] : {std::pair{1U, 0U}, std::pair{3U, 3U}}) {
    auto tensor = cast<TypedValue<RankedTensorType>>(loop.getInits()[input]);
    TensorIterator iterator(tensor);
    ++iterator;
    EXPECT_EQ(iterator.operation(), loop.getOperation());
    EXPECT_EQ(iterator.tensor(), loop.getResult(output));
    ++iterator;
    EXPECT_TRUE(isa<qtensor::DeallocOp>(iterator.operation()));
    EXPECT_EQ(iterator.tensor(), nullptr);
    ++iterator;
    EXPECT_EQ(iterator, std::default_sentinel);
    --iterator;
    --iterator;
    EXPECT_EQ(iterator.tensor(), loop.getResult(output));
    --iterator;
    EXPECT_EQ(iterator.tensor(), tensor);
    EXPECT_EQ(iterator.operation(), tensor.getDefiningOp());
  }
}
