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
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/Qubits.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <memory>
#include <tuple>

using namespace mlir;

namespace {
class DriversTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;
};
} // namespace

TEST_F(DriversTest, ProgramWalk) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto q20 = builder.allocQubit();
  const auto q30 = builder.allocQubit();

  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q21, q31] = builder.cx(q20, q30);

  const auto [q03, c0] = builder.measure(q02);
  const auto [q12, c1] = builder.measure(q11);
  const auto [q22, c2] = builder.measure(q21);
  const auto [q32, c3] = builder.measure(q31);

  builder.sink(q03);
  builder.sink(q12);
  builder.sink(q22);
  builder.sink(q32);

  auto mod = builder.finalize();
  auto func = *(mod->getOps<func::FuncOp>().begin());

  Value ex0 = nullptr;
  Value ex1 = nullptr;
  Value ex2 = nullptr;
  Value ex3 = nullptr;

  // Walk until the first measurement operation is encountered and stop.
  // Since WalkOrder::PreOrder is used here, the state of the qubits is not yet
  // updated with the SSA values of the measurement op.
  // Consequently, the program qubits point at the outputs of the controlled-Xs.
  std::ignore = qco::walkProgram(func.getBody(),
                                 [&](Operation* op, const qco::Qubits& qubits) {
                                   if (op == q03.getDefiningOp()) {
                                     ex0 = qubits.getProgramQubit(0);
                                     ex1 = qubits.getProgramQubit(1);
                                     ex2 = qubits.getProgramQubit(2);
                                     ex3 = qubits.getProgramQubit(3);
                                     return WalkResult::interrupt();
                                   }
                                   return WalkResult::advance();
                                 });

  ASSERT_EQ(ex0, q02);
  ASSERT_EQ(ex1, q11);
  ASSERT_EQ(ex2, q21);
  ASSERT_EQ(ex3, q31);
}

TEST_F(DriversTest, ProgramGraphWalk) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto q20 = builder.allocQubit();
  const auto q30 = builder.allocQubit();

  const auto q01 = builder.h(q00);

  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q21, q31] = builder.cx(q20, q30);

  const auto q03 = builder.z(q02);
  const auto q22 = builder.h(q21);

  const auto [q12, q23] = builder.cx(q11, q22);

  const auto [q04, q13] = builder.cx(q03, q12);
  const auto q14 = builder.h(q13);

  builder.measure(q04);
  builder.measure(q14);
  builder.measure(q23);
  builder.measure(q31);

  auto mod = builder.finalize();
  auto func = *(mod->getOps<func::FuncOp>().begin());

  // Collect wires.
  SmallVector<qco::WireIterator> wires;
  for (qco::AllocOp op : func.getOps<qco::AllocOp>()) {
    wires.emplace_back(op.getResult());
  }

  // Unit-test supporting datastructure.
  SmallVector<DenseSet<Operation*>> readyPerLayer;

  // Forward pass.
  auto res = qco::walkProgramGraph<qco::WireDirection::Forward>(
      wires, [&](const qco::ReadyRange& ready, qco::ReleasedOps& released) {
        DenseSet<Operation*> layer;
        for (const auto& [op, progs] : ready) {
          layer.insert(op);
          released.emplace_back(op);
        }
        readyPerLayer.emplace_back(layer);
        return WalkResult::advance();
      });

  ASSERT_TRUE(res.succeeded());
  ASSERT_GE(readyPerLayer.size(), 3);
  ASSERT_TRUE(readyPerLayer[0].contains(q02.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[0].contains(q21.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[1].contains(q12.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q04.getDefiningOp()));

  // Backward pass.
  readyPerLayer.clear();
  res = qco::walkProgramGraph<qco::WireDirection::Backward>(
      wires, [&](const qco::ReadyRange& ready, qco::ReleasedOps& released) {
        DenseSet<Operation*> layer;
        for (const auto& [op, progs] : ready) {
          layer.insert(op);
          released.emplace_back(op);
        }
        readyPerLayer.emplace_back(layer);
        return WalkResult::advance();
      });

  ASSERT_TRUE(res.succeeded());
  ASSERT_GE(readyPerLayer.size(), 3);
  ASSERT_TRUE(readyPerLayer[0].contains(q04.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[1].contains(q12.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q02.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q21.getDefiningOp()));

  // Forward, but instead of releasing all, we use ::skip().
  readyPerLayer.clear();
  res = qco::walkProgramGraph<qco::WireDirection::Forward>(
      wires, [&](const qco::ReadyRange& ready, qco::ReleasedOps&) {
        DenseSet<Operation*> layer;
        for (const auto& [op, progs] : ready) {
          layer.insert(op);
        }
        readyPerLayer.emplace_back(layer);

        return WalkResult::skip();
      });

  ASSERT_TRUE(res.succeeded());
  ASSERT_GE(readyPerLayer.size(), 3);
  ASSERT_TRUE(readyPerLayer[0].contains(q02.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[0].contains(q21.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[1].contains(q12.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q04.getDefiningOp()));

  // Backward, but stop after first layer.
  readyPerLayer.clear();
  res = qco::walkProgramGraph<qco::WireDirection::Backward>(
      wires, [&](const qco::ReadyRange& ready, qco::ReleasedOps& released) {
        DenseSet<Operation*> layer;
        for (const auto& [op, progs] : ready) {
          layer.insert(op);
          released.emplace_back(op);
        }
        readyPerLayer.emplace_back(layer);
        return WalkResult::interrupt();
      });

  ASSERT_TRUE(res.failed());
  ASSERT_EQ(readyPerLayer.size(), 1);
  ASSERT_TRUE(readyPerLayer[0].contains(q04.getDefiningOp()));
}
