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
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <gtest/gtest.h>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

using namespace mlir;

TEST(WireIteratorTest, MixedUse) {
  // Setup context.
  DialectRegistry registry;
  registry.insert<qco::QCODialect, arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect>();

  auto context = std::make_unique<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  // Build circuit.
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q03, c0] = builder.measure(q02);
  const auto q04 = builder.reset(q03);
  builder.dealloc(q04);

  // Setup WireIterator.
  auto module = builder.finalize();
  auto entry = *(module->getOps<func::FuncOp>().begin());
  auto alloc = *(entry.getBody().getOps<qco::AllocOp>().begin());
  auto qubit = alloc.getResult();
  qco::WireIterator it(qubit);
  qco::WireIterator begin(it);

  //
  // Test: Forward Iteration
  //
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc
  ASSERT_EQ(it.qubit(), q00);

  ++it;
  ASSERT_EQ(it.operation(), q01.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q00);

  ++it;
  ASSERT_EQ(it.operation(), q02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(it.qubit(), q01);

  ++it;
  ASSERT_EQ(it.operation(), q03.getDefiningOp()); // qco.measure
  ASSERT_EQ(it.qubit(), q02);

  ++it;
  ASSERT_EQ(it.operation(), q04.getDefiningOp()); // qco.reset
  ASSERT_EQ(it.qubit(), q03);

  ++it;
  ASSERT_EQ(it.operation(), *(q04.getUsers().begin())); // qco.dealloc
  ASSERT_EQ(it.qubit(), q04);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_EQ(it.operation(), *(q04.getUsers().begin())); // qco.dealloc
  ASSERT_EQ(it.qubit(), q04);

  --it;
  ASSERT_EQ(it.operation(), q04.getDefiningOp()); // qco.reset
  ASSERT_EQ(it.qubit(), q03);

  --it;
  ASSERT_EQ(it.operation(), q03.getDefiningOp()); // qco.measure
  ASSERT_EQ(it.qubit(), q02);

  --it;
  ASSERT_EQ(it.operation(), q02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(it.qubit(), q01);

  --it;
  ASSERT_EQ(it.operation(), q01.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q00);

  --it;
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc
  ASSERT_EQ(it.qubit(), q00);
  ASSERT_EQ(begin, it);

  --it;
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc
  ASSERT_EQ(it.qubit(), q00);
  ASSERT_EQ(begin, it);

  for (; it != std::default_sentinel; ++it) {
    llvm::dbgs() << **it << '\n'; /// Keep for debugging purposes.
  }
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_EQ(it.operation(), *(q04.getUsers().begin())); // qco.dealloc
  ASSERT_EQ(it.qubit(), q04);

  for (; it != begin; --it) {
    llvm::dbgs() << **it << '\n'; /// Keep for debugging purposes.
  }
  ASSERT_EQ(begin, it);
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc
  ASSERT_EQ(it.qubit(), q00);
}
