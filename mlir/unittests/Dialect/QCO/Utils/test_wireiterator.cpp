/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "gtest/gtest.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
class WireIteratorFixture : public testing::Test {
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

struct Chain {
  SmallVector<Value> values;
  SmallVector<Operation*> ops;
};
} // namespace

static Chain getChain(Value q, std::ptrdiff_t n = 1) {
  Chain chain;
  for (WireIterator it(q); it != std::default_sentinel; std::advance(it, n)) {
    chain.values.emplace_back(it.qubit());
    chain.ops.emplace_back(it.operation());
  }
  return chain;
}

TEST_F(WireIteratorFixture, TraversalRespectsStraightLineSemantics) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q03, c0] = builder.measure(q02);
  const auto [q12, c1] = builder.measure(q11);
  builder.sink(q03);
  builder.sink(q12);
  [[maybe_unused]] auto module = builder.finalize(c0);

  const auto fwChain0 = getChain(q00);
  const auto fwChain1 = getChain(q10);

  ASSERT_EQ(fwChain0.values, (SmallVector<Value>{q00, q01, q02, q03, q03}));
  ASSERT_EQ(fwChain0.ops, (SmallVector{q00.getDefiningOp(), q01.getDefiningOp(),
                                       q02.getDefiningOp(), q03.getDefiningOp(),
                                       *(q03.user_begin())}));

  ASSERT_EQ(fwChain1.values, (SmallVector<Value>{q10, q11, q12, q12}));
  ASSERT_EQ(fwChain1.ops,
            (SmallVector{q10.getDefiningOp(), q11.getDefiningOp(),
                         q12.getDefiningOp(), *(q12.user_begin())}));

  const auto bwChain0 = getChain(q03, -1);
  const auto bwChain1 = getChain(q12, -1);

  ASSERT_EQ(bwChain0.values, (SmallVector<Value>{q03, q02, q01, q00}));
  ASSERT_EQ(bwChain0.ops,
            (SmallVector{q03.getDefiningOp(), q02.getDefiningOp(),
                         q01.getDefiningOp(), q00.getDefiningOp()}));

  ASSERT_EQ(bwChain1.values, (SmallVector<Value>{q12, q11, q10}));
  ASSERT_EQ(bwChain1.ops, (SmallVector{q12.getDefiningOp(), q11.getDefiningOp(),
                                       q10.getDefiningOp()}));
}

TEST_F(WireIteratorFixture, TraversalVisitsSourcesAndSinks) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q0 = builder.staticQubit(0);
  builder.sink(q0);
  [[maybe_unused]] auto module = builder.finalize();

  WireIterator it(q0);
  ASSERT_EQ(it.qubit(), q0);
  ASSERT_EQ(it.operation(), q0.getDefiningOp());
  ASSERT_TRUE(isa<StaticOp>(it.operation()));
  ++it;
  ASSERT_EQ(it.qubit(), q0);
  ASSERT_EQ(it.operation(), *(q0.value.user_begin()));
  ASSERT_TRUE(isa<SinkOp>(it.operation()));
  ++it;
  ASSERT_EQ(it, std::default_sentinel);
  --it;
  ASSERT_EQ(it.qubit(), q0);
  ASSERT_EQ(it.operation(), *(q0.value.user_begin()));
  ASSERT_TRUE(isa<SinkOp>(it.operation()));
  --it;
  ASSERT_EQ(it.qubit(), q0);
  ASSERT_EQ(it.operation(), q0.getDefiningOp());
  ASSERT_TRUE(isa<StaticOp>(it.operation()));
  --it;
  ASSERT_EQ(it, std::default_sentinel);
}

TEST_F(WireIteratorFixture, TraversalRespectsNestedBoundaries) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value inLoop = nullptr;

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.scfFor(1, 4, 1, {q0}, [&](Value, ValueRange args) {
    inLoop = builder.x(args[0]);
    return SmallVector<Value>{inLoop};
  })[0];
  builder.sink(q1);

  [[maybe_unused]] auto module = builder.finalize();

  WireIterator it(inLoop);
  ASSERT_EQ(it.qubit(), inLoop);
  ASSERT_EQ(it.operation(), inLoop.getDefiningOp());
  ++it;
  ASSERT_EQ(it.qubit(), inLoop);
  ASSERT_EQ(it.operation(), *(inLoop.user_begin()));
  ASSERT_TRUE(isa<scf::YieldOp>(it.operation()));
  ++it;
  ASSERT_EQ(it, std::default_sentinel);
  --it;
  ASSERT_EQ(it.qubit(), inLoop);
  ASSERT_EQ(it.operation(), *(inLoop.user_begin()));
  ASSERT_TRUE(isa<scf::YieldOp>(it.operation()));
  --it;
  ASSERT_EQ(it.qubit(), inLoop);
  ASSERT_EQ(it.operation(), inLoop.getDefiningOp());
  --it;
  ASSERT_TRUE(isa<BlockArgument>(it.qubit()));
  ASSERT_EQ(it.operation(), nullptr);
  --it;
  ASSERT_EQ(it, std::default_sentinel);
}

TEST_F(WireIteratorFixture, FailOnSentinelAccess) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q0 = builder.staticQubit(0);
  builder.sink(q0);
  [[maybe_unused]] auto module = builder.finalize();

  WireIterator it(q0);
  --it;
  ASSERT_EQ(it, std::default_sentinel);
  ASSERT_DEATH(it.qubit(), "Trying to access qubit of sentinel!");
  ASSERT_DEATH(it.operation(), "Trying to access operation of sentinel!");
}

TEST_F(WireIteratorFixture, TraversalRespectsStructuredSemantics) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  constexpr auto identity = [](ValueRange args) {
    return SmallVector<Value>(args);
  };

  const auto [tensor0, qubits] = builder.allocQubitRegister(2);
  const auto q00 = qubits[0];
  const auto q10 = qubits[1];
  const auto q01 = builder.h(q00);

  const auto forOut =
      builder.scfFor(1, 4, 1, {q01, q10},
                     [&](Value, ValueRange args) { return identity(args); });
  const auto q02 = forOut[0];
  const auto q11 = forOut[1];

  const auto ifOut = builder.qcoIf(true, {q02, q11}, identity, identity);
  const auto q03 = ifOut[0];
  const auto q12 = ifOut[1];

  const auto whileOut = builder.scfWhile(
      {q03, q12},
      [&](ValueRange args) {
        const auto b = builder.boolConstant(true);
        builder.scfCondition(b, args);
        return SmallVector<Value>(args);
      },
      identity);
  const auto q04 = whileOut[0];
  const auto q13 = whileOut[1];

  const auto switchOut = builder.qcoIndexSwitch(
      0, {q04, q13}, SmallVector<int64_t>{0}, {identity}, identity);
  const auto q05 = switchOut[0];
  const auto q14 = switchOut[1];

  const auto q15 = builder.x(q14);
  const auto tensor1 = builder.qtensorInsert(q05, tensor0, 0);
  const auto tensor2 = builder.qtensorInsert(q15, tensor1, 1);
  builder.qtensorDealloc(tensor2);

  [[maybe_unused]] auto module = builder.finalize();

  const auto fwChain0 = getChain(q00);
  const auto fwChain1 = getChain(q10);

  ASSERT_EQ(fwChain0.values,
            (SmallVector<Value>{q00, q01, q02, q03, q04, q05, q05}));
  ASSERT_EQ(fwChain0.ops, (SmallVector{q00.getDefiningOp(), q01.getDefiningOp(),
                                       q02.getDefiningOp(), q03.getDefiningOp(),
                                       q04.getDefiningOp(), q05.getDefiningOp(),
                                       *(q05.user_begin())}));

  ASSERT_EQ(fwChain1.values,
            (SmallVector<Value>{q10, q11, q12, q13, q14, q15, q15}));
  ASSERT_EQ(fwChain1.ops, (SmallVector{q10.getDefiningOp(), q11.getDefiningOp(),
                                       q12.getDefiningOp(), q13.getDefiningOp(),
                                       q14.getDefiningOp(), q15.getDefiningOp(),
                                       *(q15.user_begin())}));

  const auto bwChain0 = getChain(q05, -1);
  const auto bwChain1 = getChain(q15, -1);

  ASSERT_EQ(bwChain0.values,
            (SmallVector<Value>{q05, q04, q03, q02, q01, q00}));
  ASSERT_EQ(bwChain0.ops,
            (SmallVector{q05.getDefiningOp(), q04.getDefiningOp(),
                         q03.getDefiningOp(), q02.getDefiningOp(),
                         q01.getDefiningOp(), q00.getDefiningOp()}));

  ASSERT_EQ(bwChain1.values,
            (SmallVector<Value>{q15, q14, q13, q12, q11, q10}));
  ASSERT_EQ(bwChain1.ops,
            (SmallVector{q15.getDefiningOp(), q14.getDefiningOp(),
                         q13.getDefiningOp(), q12.getDefiningOp(),
                         q11.getDefiningOp(), q10.getDefiningOp()}));
}

TEST_F(WireIteratorFixture, TraversalTerminatesAtFunctionReturn) {
  Value source;
  Value output;
  auto module =
      qco::QCOProgramBuilder::build(context.get(), [&](auto& builder) -> Value {
        source = builder.allocQubit();
        output = builder.h(source);
        return output;
      });
  ASSERT_TRUE(module);

  qco::WireIterator it(source);
  ASSERT_EQ(it.operation(), source.getDefiningOp());
  ASSERT_EQ(it.qubit(), source);

  ++it;
  ASSERT_EQ(it.operation(), output.getDefiningOp());
  ASSERT_EQ(it.qubit(), output);

  ++it;
  ASSERT_TRUE(isa<func::ReturnOp>(it.operation()));
  ASSERT_EQ(it.qubit(), output);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_TRUE(isa<func::ReturnOp>(it.operation()));
  ASSERT_EQ(it.qubit(), output);

  --it;
  ASSERT_EQ(it.operation(), output.getDefiningOp());
  ASSERT_EQ(it.qubit(), output);
}

TEST_F(WireIteratorFixture, TraversalTerminatesAtUnknownCarrier) {
  OpBuilder builder(context.get());
  const auto location = builder.getUnknownLoc();
  auto module = ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());
  auto function = func::FuncOp::create(builder, location, "main",
                                       builder.getFunctionType({}, {}));
  Block* body = function.addEntryBlock();
  builder.setInsertionPointToStart(body);

  auto source = qco::AllocOp::create(builder, location).getResult();
  auto carrier = UnrealizedConversionCastOp::create(
      builder, location, TypeRange{source.getType()}, ValueRange{source});
  auto carried = carrier.getResult(0);
  qco::SinkOp::create(builder, location, carried);
  func::ReturnOp::create(builder, location);

  qco::WireIterator forward(source);
  ++forward;
  EXPECT_EQ(forward.operation(), carrier.getOperation());
  ++forward;
  EXPECT_EQ(forward, std::default_sentinel);

  qco::WireIterator backward(carried);
  --backward;
  EXPECT_EQ(backward, std::default_sentinel);
}

TEST_F(WireIteratorFixture, CallMappingFollowsNestedReordering) {
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
  SmallVector<Value> allocs;
  main.walk([&](qco::AllocOp op) { allocs.emplace_back(op.getResult()); });
  ASSERT_EQ(allocs.size(), 2U);

  qco::CallQubitMapping mapping;
  auto mapped = mapping.getResultForOperand(call, call.getOperand(1));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(2));
  mapped = mapping.getResultForOperand(call, call.getOperand(2));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(1));

  qco::WireIterator iterator(allocs[0]);
  ++iterator;
  EXPECT_EQ(iterator.qubit(), call.getResult(2));
  --iterator;
  EXPECT_EQ(iterator.qubit(), allocs[0]);

  auto swap = module->lookupSymbol<func::FuncOp>("swap");
  auto returnOp = cast<func::ReturnOp>(swap.getBody().front().getTerminator());
  returnOp->setOperands(swap.getArguments());
  mapping.invalidate();
  mapped = mapping.getResultForOperand(call, call.getOperand(1));
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_EQ(*mapped, call.getResult(1));
}

TEST_F(WireIteratorFixture, CallMappingDistinguishesKeptAndCreatedQubits) {
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

  qco::CallQubitMapping mapping;
  auto mapped = mapping.getResultForOperand(call, old);
  ASSERT_TRUE(succeeded(mapped));
  EXPECT_FALSE(*mapped);

  qco::WireIterator consumed(old);
  ++consumed;
  ASSERT_EQ(consumed.operation(), call);
  ++consumed;
  EXPECT_EQ(consumed, std::default_sentinel);

  qco::WireIterator created(call.getResult(0));
  --created;
  EXPECT_EQ(created, std::default_sentinel);
}

TEST_F(WireIteratorFixture, CallMappingFailsClosed) {
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

  qco::CallQubitMapping mapping;
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(external, external.getOperand(0))));
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(recursive, recursive.getOperand(0))));
  EXPECT_TRUE(
      failed(mapping.getResultForOperand(unknown, unknown.getOperand(0))));
}
