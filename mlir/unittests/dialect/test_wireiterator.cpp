/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/IR/WireIterator.h"

#include <gtest/gtest.h>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <string>

using namespace mlir;
using namespace mqt::ir::opt;

namespace {
/** @returns a module containing the circuit from the "Tackling the Qubit
 * Mapping Problem for NISQ-Era Quantum Devices" paper by Li et al.
 */
OwningOpRef<ModuleOp> getModule(MLIRContext& ctx) {
  const char* ir = R"mlir(
module {
  %0 = mqtopt.allocQubit
  %1 = mqtopt.allocQubit
  %out_qubits = mqtopt.h() %0 : !mqtopt.Qubit
  %out_qubits_0 = mqtopt.h() %1 : !mqtopt.Qubit
  %out_qubits_1 = mqtopt.z() %out_qubits : !mqtopt.Qubit
  %out_qubits_2, %pos_ctrl_out_qubits = mqtopt.x() %out_qubits_0 ctrl %out_qubits_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %out_qubits_3 = mqtopt.h() %out_qubits_2 : !mqtopt.Qubit
  %out_qubits_4, %pos_ctrl_out_qubits_5 = mqtopt.x() %pos_ctrl_out_qubits ctrl %out_qubits_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %false = arith.constant false
  %2:2 = scf.if %false -> (!mqtopt.Qubit, !mqtopt.Qubit) {
    %out_qubits_7 = mqtopt.y() %out_qubits_4 : !mqtopt.Qubit
    scf.yield %out_qubits_7, %pos_ctrl_out_qubits_5 : !mqtopt.Qubit, !mqtopt.Qubit
  } else {
    scf.yield %out_qubits_4, %pos_ctrl_out_qubits_5 : !mqtopt.Qubit, !mqtopt.Qubit
  }
  %idx0 = index.constant 0
  %idx8 = index.constant 8
  %idx1 = index.constant 1
  %3:2 = scf.for %arg0 = %idx0 to %idx8 step %idx1 iter_args(%arg1 = %2#0, %arg2 = %2#1) -> (!mqtopt.Qubit, !mqtopt.Qubit) {
    %out_qubits_7 = mqtopt.h() %arg1 : !mqtopt.Qubit
    %out_qubits_8 = mqtopt.h() %arg2 : !mqtopt.Qubit
    scf.yield %out_qubits_7, %out_qubits_8 : !mqtopt.Qubit, !mqtopt.Qubit
  }
  mqtopt.deallocQubit %3#0
  mqtopt.deallocQubit %3#1

  %4 = mqtopt.qubit 42
  %5 = mqtopt.reset %4
  %out_qubits_6 = mqtopt.h() %5 : !mqtopt.Qubit
}
)mlir";
  return parseSourceString<ModuleOp>(ir, &ctx);
}

std::string toString(Operation* op) {
  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  os << *op;
  os.flush();
  return opStr;
}

void checkOperationEqual(Operation* op, const std::string& expected) {
  ASSERT_EQ(expected, toString(op));
}

void checkOperationStartsWith(Operation* op, const std::string& prefix) {
  ASSERT_TRUE(toString(op).starts_with(prefix));
}
} // namespace

class WireIteratorTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<MQTOptDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<index::IndexDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

TEST_F(WireIteratorTest, TestForward) {

  ///
  /// Test the forward iteration.
  ///

  auto module = getModule(*context);
  auto alloc = *(module->getOps<AllocQubitOp>().begin());
  auto q = alloc.getQubit();
  WireIterator it(q, q.getParentRegion());

  checkOperationEqual(*it, "%0 = mqtopt.allocQubit");

  ++it;
  checkOperationEqual(*it, "%out_qubits = mqtopt.h() %0 : !mqtopt.Qubit");

  ++it;
  checkOperationEqual(*it,
                      "%out_qubits_1 = mqtopt.z() %out_qubits : !mqtopt.Qubit");

  ++it;
  checkOperationEqual(
      *it, "%out_qubits_2, %pos_ctrl_out_qubits = mqtopt.x() %out_qubits_0 "
           "ctrl %out_qubits_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit");

  ++it;
  checkOperationEqual(
      *it,
      "%out_qubits_4, %pos_ctrl_out_qubits_5 = mqtopt.x() %pos_ctrl_out_qubits "
      "ctrl %out_qubits_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit");

  ++it;
  checkOperationStartsWith(
      *it, "%2:2 = scf.if %false -> (!mqtopt.Qubit, !mqtopt.Qubit)");

  ++it;
  checkOperationStartsWith(*it,
                           "%3:2 = scf.for %arg0 = %idx0 to %idx8 step %idx1");

  ++it;
  checkOperationEqual(*it, "mqtopt.deallocQubit %3#0");

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);
}

TEST_F(WireIteratorTest, TestBackward) {

  ///
  /// Test the backward iteration.
  ///

  auto module = getModule(*context);
  auto allocs = module->getOps<AllocQubitOp>();
  const auto allocRng = llvm::make_range(allocs.begin(), allocs.end());
  const auto allocVec = llvm::to_vector(allocRng);
  auto alloc = allocVec[1];
  auto q = alloc.getQubit();
  WireIterator it(q, q.getParentRegion());
  const WireIterator begin(it);

  ASSERT_EQ(it, begin);

  for (; it != std::default_sentinel; ++it) {
    llvm::dbgs() << **it << '\n'; /// Keep for debugging purposes.
  }

  ASSERT_EQ(it, std::default_sentinel);

  --it;
  checkOperationEqual(*it, "mqtopt.deallocQubit %3#1");

  --it;
  checkOperationStartsWith(*it,
                           "%3:2 = scf.for %arg0 = %idx0 to %idx8 step %idx1");

  --it;
  checkOperationStartsWith(
      *it, "%2:2 = scf.if %false -> (!mqtopt.Qubit, !mqtopt.Qubit)");

  --it;
  checkOperationEqual(
      *it,
      "%out_qubits_4, %pos_ctrl_out_qubits_5 = mqtopt.x() %pos_ctrl_out_qubits "
      "ctrl %out_qubits_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit");

  --it;
  checkOperationEqual(
      *it, "%out_qubits_3 = mqtopt.h() %out_qubits_2 : !mqtopt.Qubit");

  --it;
  checkOperationEqual(
      *it, "%out_qubits_2, %pos_ctrl_out_qubits = mqtopt.x() %out_qubits_0 "
           "ctrl %out_qubits_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit");

  --it;
  checkOperationEqual(*it, "%out_qubits_0 = mqtopt.h() %1 : !mqtopt.Qubit");

  --it;
  checkOperationEqual(*it, "%1 = mqtopt.allocQubit");

  ASSERT_EQ(it, begin);

  --it;
  checkOperationEqual(*it, "%1 = mqtopt.allocQubit");

  ASSERT_EQ(it, begin);
}

TEST_F(WireIteratorTest, TestForwardAndBackward) {

  ///
  /// Test the forward as well as the backward iteration.
  ///

  auto module = getModule(*context);
  auto alloc = *(module->getOps<AllocQubitOp>().begin());
  auto q = alloc.getQubit();
  WireIterator it(q, q.getParentRegion());
  const WireIterator begin(it);

  checkOperationEqual(*it, "%0 = mqtopt.allocQubit");

  ++it;
  checkOperationEqual(*it, "%out_qubits = mqtopt.h() %0 : !mqtopt.Qubit");

  ++it;
  checkOperationEqual(*it,
                      "%out_qubits_1 = mqtopt.z() %out_qubits : !mqtopt.Qubit");

  ++it;
  checkOperationEqual(
      *it, "%out_qubits_2, %pos_ctrl_out_qubits = mqtopt.x() %out_qubits_0 "
           "ctrl %out_qubits_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit");

  --it;
  checkOperationEqual(*it,
                      "%out_qubits_1 = mqtopt.z() %out_qubits : !mqtopt.Qubit");

  --it;
  checkOperationEqual(*it, "%out_qubits = mqtopt.h() %0 : !mqtopt.Qubit");

  --it;
  checkOperationEqual(*it, "%0 = mqtopt.allocQubit");

  ASSERT_EQ(it, begin);

  for (; it != std::default_sentinel; ++it) {
    llvm::dbgs() << **it << '\n'; /// Keep for debugging purposes.
  }

  ASSERT_EQ(it, std::default_sentinel);

  it = std::prev(it); // Back to last non-sentinel item.

  for (; it != begin; --it) {
    llvm::dbgs() << **it << '\n'; /// Keep for debugging purposes.
  }

  ASSERT_EQ(it, begin);
}

TEST_F(WireIteratorTest, TestRecursiveUse) {

  ///
  /// Test the recursive use of the iterator.
  ///

  auto module = getModule(*context);
  auto alloc = *(module->getOps<AllocQubitOp>().begin());
  auto q = alloc.getQubit();
  WireIterator it(q, q.getParentRegion());

  /// Advance until 'scf.for'.
  for (; it != std::default_sentinel; ++it) {
    if (isa<scf::ForOp>(*it)) {
      break;
    }
    llvm::dbgs() << **it << '\n'; /// Keep for debugging purposes.
  }

  auto loop = cast<scf::ForOp>(*it);
  for (auto [iter, init] :
       llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
    if (init == it.qubit()) {
      WireIterator rec(iter, &loop.getRegion());
      const WireIterator recBegin(rec);
      rec--;

      ASSERT_EQ(rec, recBegin); // Test blockargument handling.

      rec++;
      checkOperationEqual(*rec,
                          "%out_qubits_7 = mqtopt.h() %arg1 : !mqtopt.Qubit");

      rec++;
      checkOperationEqual(*rec, "scf.yield %out_qubits_7, %out_qubits_8 : "
                                "!mqtopt.Qubit, !mqtopt.Qubit");
    }
  }
}

TEST_F(WireIteratorTest, TestStaticQubit) {

  ///
  /// Test the iteration with a static qubit.
  ///

  auto module = getModule(*context);
  auto qubit = *(module->getOps<QubitOp>().begin());
  auto q = qubit.getQubit();
  WireIterator it(q, q.getParentRegion());
  const WireIterator begin(it);

  checkOperationEqual(*it, "%4 = mqtopt.qubit 42");

  ++it;
  checkOperationEqual(*it, "%5 = mqtopt.reset %4");

  ++it;
  checkOperationEqual(*it, "%out_qubits_6 = mqtopt.h() %5 : !mqtopt.Qubit");

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  checkOperationEqual(*it, "%out_qubits_6 = mqtopt.h() %5 : !mqtopt.Qubit");
  ASSERT_EQ(it.qubit(), (*it)->getResult(0)); // q = %out_qubits_6

  --it;
  checkOperationEqual(*it, "%out_qubits_6 = mqtopt.h() %5 : !mqtopt.Qubit");
  ASSERT_EQ(it.qubit(), (*it)->getOperand(0)); // q = %5

  --it;
  checkOperationEqual(*it, "%5 = mqtopt.reset %4");

  --it;
  checkOperationEqual(*it, "%4 = mqtopt.qubit 42");

  ASSERT_EQ(it, begin);
}
