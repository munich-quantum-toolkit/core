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
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;

namespace {
class WireIteratorTest : public testing::TestWithParam<bool> {
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
};
} // namespace

TEST_P(WireIteratorTest, Traversal) {
  const bool isDynamic = GetParam();

  // Build circuit.
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = isDynamic ? builder.allocQubit() : builder.staticQubit(0);
  const auto q10 = isDynamic ? builder.allocQubit() : builder.staticQubit(1);
  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q03, c0] = builder.measure(q02);
  const auto q04 = builder.reset(q03);

  Value iterQ00;
  Value iterQ01;
  Value iterQ02;
  Value iterQ10;
  Value iterQ11;

  const auto loopOut =
      builder.scfFor(1, 4, 1, {q04, q11}, [&](Value, ValueRange iterArgs) {
        iterQ00 = iterArgs[0];
        iterQ10 = iterArgs[1];
        iterQ01 = builder.h(iterQ00);
        std::tie(iterQ02, iterQ11) = builder.cx(iterQ01, iterQ10);
        return SmallVector{iterQ02, iterQ11};
      });
  const auto q05 = loopOut[0];
  const auto q12 = loopOut[1];
  const auto ifOut = builder.qcoIf(
      true, {q05, q12},
      [&](ValueRange args) { return SmallVector{args[0], args[1]}; },
      [&](ValueRange args) { return SmallVector{args[0], args[1]}; });
  const auto q06 = ifOut[0];
  const auto q13 = ifOut[1];
  const auto identity = [](ValueRange args) { return llvm::to_vector(args); };
  const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies{
      identity};
  const auto switchOut = builder.qcoIndexSwitch(
      0, {q06, q13}, SmallVector<int64_t>{0}, caseBodies, identity);
  const auto q07 = switchOut[0];
  const auto q14 = switchOut[1];
  builder.sink(q07);
  builder.sink(q14);
  [[maybe_unused]] auto module = builder.finalize();

  // Setup WireIterator.
  qco::WireIterator it(q00);

  //
  // Test: Forward Iteration
  //

  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc
  ASSERT_EQ(it.qubit(), q00);

  ++it;
  ASSERT_EQ(it.operation(), q01.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q01);

  ++it;
  ASSERT_EQ(it.operation(), q02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(it.qubit(), q02);

  ++it;
  ASSERT_EQ(it.operation(), q03.getDefiningOp()); // qco.measure
  ASSERT_EQ(it.qubit(), q03);

  ++it;
  ASSERT_EQ(it.operation(), q04.getDefiningOp()); // qco.reset
  ASSERT_EQ(it.qubit(), q04);

  ++it;
  ASSERT_EQ(it.operation(), q05.getDefiningOp()); // scf.for
  ASSERT_EQ(it.qubit(), q05);

  ++it;
  ASSERT_EQ(it.operation(), q06.getDefiningOp()); // qco.if
  ASSERT_EQ(it.qubit(), q06);

  ++it;
  ASSERT_EQ(it.operation(), q07.getDefiningOp()); // qco.index_switch
  ASSERT_EQ(it.qubit(), q07);

  ++it;
  ASSERT_EQ(it.operation(), *(q07.getUsers().begin())); // qco.sink
  ASSERT_EQ(it.qubit(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  //
  // Test: Backward Iteration
  //

  --it;
  ASSERT_EQ(it.operation(), *(q07.getUsers().begin())); // qco.sink
  ASSERT_EQ(it.qubit(), nullptr);

  --it;
  ASSERT_EQ(it.operation(), q07.getDefiningOp()); // qco.index_switch
  ASSERT_EQ(it.qubit(), q07);

  --it;
  ASSERT_EQ(it.operation(), q06.getDefiningOp()); // qco.if
  ASSERT_EQ(it.qubit(), q06);

  --it;
  ASSERT_EQ(it.operation(), q05.getDefiningOp()); // scf.for
  ASSERT_EQ(it.qubit(), q05);

  --it;
  ASSERT_EQ(it.operation(), q04.getDefiningOp()); // qco.reset
  ASSERT_EQ(it.qubit(), q04);

  --it;
  ASSERT_EQ(it.operation(), q03.getDefiningOp()); // qco.measure
  ASSERT_EQ(it.qubit(), q03);

  --it;
  ASSERT_EQ(it.operation(), q02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(it.qubit(), q02);

  --it;
  ASSERT_EQ(it.operation(), q01.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q01);

  --it;
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc or qco.static
  ASSERT_EQ(it.qubit(), q00);

  --it;
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc or qco.static
  ASSERT_EQ(it.qubit(), q00);

  //
  // Test: Recursive use with block-argument.
  //

  qco::WireIterator recIt(iterQ00);
  ASSERT_EQ(recIt.operation(), nullptr); // Blockargument
  ASSERT_EQ(recIt.qubit(), iterQ00);

  ++recIt;
  ASSERT_EQ(recIt.operation(), iterQ01.getDefiningOp()); // qco.h
  ASSERT_EQ(recIt.qubit(), iterQ01);

  ++recIt;
  ASSERT_EQ(recIt.operation(), iterQ02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(recIt.qubit(), iterQ02);

  ++recIt;
  ASSERT_EQ(recIt.operation(), *(iterQ02.getUsers().begin())); // scf.yield
  ASSERT_EQ(recIt.qubit(), nullptr);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  --recIt;
  ASSERT_EQ(recIt.operation(), *(iterQ02.getUsers().begin())); // scf.yield
  ASSERT_EQ(recIt.qubit(), nullptr);

  --recIt;
  ASSERT_EQ(recIt.operation(), iterQ02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(recIt.qubit(), iterQ02);

  --recIt;
  ASSERT_EQ(recIt.operation(), iterQ01.getDefiningOp()); // qco.h
  ASSERT_EQ(recIt.qubit(), iterQ01);

  --recIt;
  ASSERT_EQ(recIt.operation(), nullptr); // Blockargument
  ASSERT_EQ(recIt.qubit(), iterQ00);
}

TEST_P(WireIteratorTest, FunctionReturnTerminatesTraversal) {
  const bool isDynamic = GetParam();
  Value source;
  Value output;
  auto module =
      qco::QCOProgramBuilder::build(context.get(), [&](auto& builder) -> Value {
        source = isDynamic ? builder.allocQubit() : builder.staticQubit(0);
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
  ASSERT_EQ(it.qubit(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_TRUE(isa<func::ReturnOp>(it.operation()));
  ASSERT_EQ(it.qubit(), nullptr);

  --it;
  ASSERT_EQ(it.operation(), output.getDefiningOp());
  ASSERT_EQ(it.qubit(), output);
}

INSTANTIATE_TEST_SUITE_P(DynamicAndStatic, WireIteratorTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Dynamic" : "Static";
                         });

/**
 * @brief A wire continues through a call that threads the qubit, even when the
 * callee takes and returns classical values around it.
 */
TEST_F(WireIteratorTest, TraversalThroughThreadingCall) {
  qco::QCOProgramBuilder builder(context.get());

  const auto qubitType = builder.getQubitType();
  const auto floatType = builder.getF64Type();
  const auto bitType = builder.getI1Type();

  // `main` hands the measurement outcome back, so it has to be typed for it.
  builder.initialize({bitType});

  // The qubit is operand 0 but result 1, so pairing by raw index would pick the
  // classical result instead.
  const auto args =
      builder.startFunction("g", {qubitType, floatType}, {bitType, qubitType});
  auto [inner, innerBit] = builder.measure(args[0]);
  builder.endFunction({innerBit, inner});

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.h(q0);
  const auto results = builder.call("g", {q1, builder.floatConstant(0.5)});
  const auto q2 = results[1];
  builder.sink(q2);
  [[maybe_unused]] auto module = builder.finalize({results[0]});

  qco::WireIterator it(q0);
  ASSERT_EQ(it.qubit(), q0); // qco.alloc
  ASSERT_TRUE(it.atWireStart());

  ++it;
  ASSERT_EQ(it.operation(), q1.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q1);

  ++it;
  ASSERT_EQ(it.operation(), q2.getDefiningOp()); // func.call
  ASSERT_EQ(it.qubit(), q2);

  // And back again.
  --it;
  ASSERT_EQ(it.operation(), q1.getDefiningOp());
  ASSERT_EQ(it.qubit(), q1);

  --it;
  ASSERT_EQ(it.operation(), q0.getDefiningOp());
  ASSERT_EQ(it.qubit(), q0);
  ASSERT_TRUE(it.atWireStart());
}

/**
 * @brief A wire ends at a call whose callee keeps the qubit.
 */
TEST_F(WireIteratorTest, TraversalIntoConsumingCall) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto qubitType = builder.getQubitType();
  const auto args = builder.startFunction("consume", {qubitType}, {});
  builder.sink(args[0]);
  builder.endFunction({});

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.h(q0);
  builder.call("consume", {q1});
  [[maybe_unused]] auto module = builder.finalize();

  qco::WireIterator it(q0);
  ++it;
  ASSERT_EQ(it.qubit(), q1); // qco.h

  ++it;
  // The call is the last operation on the wire.
  ASSERT_TRUE(isa<func::CallOp>(it.operation()));

  ++it;
  ASSERT_EQ(it, std::default_sentinel);
}

/**
 * @brief A wire starts at a call whose callee creates the qubit, so backward
 * traversal stops there instead of spinning.
 */
TEST_F(WireIteratorTest, TraversalFromProducingCall) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto qubitType = builder.getQubitType();
  builder.startFunction("produce", {}, {qubitType});
  builder.endFunction({builder.allocQubit()});

  const auto results = builder.call("produce", {});
  const auto q0 = results[0];
  const auto q1 = builder.h(q0);
  builder.sink(q1);
  [[maybe_unused]] auto module = builder.finalize();

  qco::WireIterator it(q1);
  ASSERT_EQ(it.operation(), q1.getDefiningOp()); // qco.h
  ASSERT_FALSE(it.atWireStart());

  --it;
  ASSERT_EQ(it.operation(), q0.getDefiningOp()); // func.call
  ASSERT_EQ(it.qubit(), q0);
  ASSERT_TRUE(it.atWireStart());

  // Going further back must not move the iterator.
  --it;
  ASSERT_EQ(it.operation(), q0.getDefiningOp());
  ASSERT_EQ(it.qubit(), q0);
}

/**
 * @brief A callee may hand its qubits back in a different order than it takes
 * them; the wire follows the qubit, not its position.
 */
TEST_F(WireIteratorTest, TraversalThroughReorderingCall) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto qubitType = builder.getQubitType();
  // `@relabel` returns its arguments swapped, so argument 0 leaves through
  // result 1. Pairing by position would follow the wrong wire here.
  const auto args = builder.startFunction("relabel", {qubitType, qubitType},
                                          {qubitType, qubitType});
  builder.endFunction({args[1], args[0]});

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto results = builder.call("relabel", {q0, q1});
  builder.sink(results[0]);
  builder.sink(results[1]);
  [[maybe_unused]] auto module = builder.finalize();

  qco::CallQubitMapping mapping;

  qco::WireIterator it(q0, &mapping);
  ++it;
  ASSERT_TRUE(isa<func::CallOp>(it.operation()));
  ASSERT_EQ(it.qubit(), results[1]); // not results[0]

  --it;
  ASSERT_EQ(it.qubit(), q0);

  // The other argument leaves through the first result.
  qco::WireIterator other(q1, &mapping);
  ++other;
  ASSERT_EQ(other.qubit(), results[0]);
}

/**
 * @brief Threading a recursive callee terminates instead of descending into
 * itself forever.
 */
TEST_F(WireIteratorTest, TraversalThroughRecursiveCall) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto qubitType = builder.getQubitType();
  const auto args = builder.startFunction("rec", {qubitType}, {qubitType});
  const auto inner = builder.h(args[0]);
  builder.endFunction({builder.call("rec", {inner})[0]});

  const auto q0 = builder.allocQubit();
  const auto results = builder.call("rec", {q0});
  builder.sink(results[0]);
  [[maybe_unused]] auto module = builder.finalize();

  qco::CallQubitMapping mapping;
  qco::WireIterator it(q0, &mapping);
  ++it;
  ASSERT_TRUE(isa<func::CallOp>(it.operation()));
  ASSERT_EQ(it.qubit(), results[0]);

  ++it;
  ASSERT_TRUE(isa<qco::SinkOp>(it.operation()));
}
