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
#include <mlir/Dialect/SCF/IR/SCF.h>
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
    registry.insert<qco::QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();

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

  Value q0 = builder.allocQubit();
  Value q1 = builder.allocQubit();
  Value q2 = builder.allocQubit();
  Value q3 = builder.allocQubit();

  q0 = builder.h(q0);
  std::tie(q0, q1) = builder.cx(q0, q1);
  std::tie(q2, q3) = builder.cx(q2, q3);

  const auto forOut = builder.scfFor(
      1, 3, 1, {q0, q1, q2, q3}, [&builder](Value, ValueRange iterArgs) {
        return SmallVector{iterArgs[0], iterArgs[1], iterArgs[2], iterArgs[3]};
      });

  q0 = forOut[0];
  q1 = forOut[1];
  q2 = forOut[2];
  q3 = forOut[3];

  Value c0;
  Value c1;
  Value c2;
  Value c3;

  std::tie(q0, c0) = builder.measure(q0);
  Operation* firstMeasure = q0.getDefiningOp();

  std::tie(q1, c1) = builder.measure(q1);
  std::tie(q2, c2) = builder.measure(q2);
  std::tie(q3, c3) = builder.measure(q3);

  builder.sink(q0);
  builder.sink(q1);
  builder.sink(q2);
  builder.sink(q3);

  auto m = builder.finalize();
  auto func = qco::getEntryPoint(m.get());

  Value ex0 = nullptr;
  Value ex1 = nullptr;
  Value ex2 = nullptr;
  Value ex3 = nullptr;

  // Walk until the first measurement operation is encountered and stop.
  // Since WalkOrder::PreOrder is used here, the state of the qubits is not yet
  // updated with the SSA values of the measurement op.
  // Consequently, the program qubits point at the outputs of the controlled-Xs.
  qco::Qubits qubits;
  std::ignore = qco::walkProgram(func.getBody(), qubits,
                                 [&](Operation* op, const qco::Qubits& qubits) {
                                   if (op == firstMeasure) {
                                     ex0 = qubits.getProgramQubit(0);
                                     ex1 = qubits.getProgramQubit(1);
                                     ex2 = qubits.getProgramQubit(2);
                                     ex3 = qubits.getProgramQubit(3);
                                     return WalkResult::interrupt();
                                   }
                                   return WalkResult::advance();
                                 });

  ASSERT_EQ(ex0, forOut[0]);
  ASSERT_EQ(ex1, forOut[1]);
  ASSERT_EQ(ex2, forOut[2]);
  ASSERT_EQ(ex3, forOut[3]);
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
