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
#include <mlir/IR/ValueRange.h>
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

TEST_F(DriversTest, ProgramGraphWalkTooFewWires) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto [q01, q11] = builder.cx(q00, q10);

  [[maybe_unused]] auto mod = builder.finalize();

  // Collect just one wire.
  SmallVector<qco::WireIterator> wires;
  wires.emplace_back(q00);

  auto res = qco::walkProgramGraph<qco::WireDirection::Forward>(
      wires, [&](const qco::ReadyRange&, qco::ReleasedOps&) {
        return WalkResult::skip();
      });
  ASSERT_TRUE(res.failed());
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

  Value iterQ0;
  Value iterQ1;
  ValueRange blockArgs;
  const auto forResults = builder.scfFor(
      0, 3, 1, {q04, q14, q23, q31}, [&](Value, ValueRange args) {
        blockArgs = args;
        std::tie(iterQ0, iterQ1) = builder.cx(args[0], args[1]);
        return SmallVector<Value>{iterQ0, iterQ1, args[2], args[3]};
      });

  const auto q05 = builder.qcoIf(
      false, forResults[0],
      [&](ValueRange args) { return SmallVector<Value>{builder.h(args[0])}; },
      [&](ValueRange args) {
        return SmallVector<Value>{builder.id(args[0])};
      })[0];

  const auto identity = [](ValueRange args) {
    return llvm::to_vector(args);
  };
  const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies{
      identity};
  const auto q06 = builder.qcoIndexSwitch(
      0, q05, SmallVector<int64_t>{0}, caseBodies, identity)[0];

  builder.measure(q06);
  builder.measure(forResults[1]);
  builder.measure(forResults[2]);
  builder.measure(forResults[3]);

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
  ASSERT_GE(readyPerLayer.size(), 4);
  ASSERT_TRUE(readyPerLayer[0].contains(q02.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[0].contains(q21.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[1].contains(q12.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q04.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[3].contains(forResults[0].getDefiningOp()));

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
  ASSERT_GE(readyPerLayer.size(), 4);
  ASSERT_TRUE(readyPerLayer[0].contains(forResults[0].getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[1].contains(q04.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q12.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[3].contains(q02.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[3].contains(q21.getDefiningOp()));

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
  ASSERT_GE(readyPerLayer.size(), 4);
  ASSERT_TRUE(readyPerLayer[0].contains(q02.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[0].contains(q21.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[1].contains(q12.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[2].contains(q04.getDefiningOp()));
  ASSERT_TRUE(readyPerLayer[3].contains(forResults[0].getDefiningOp()));

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
  ASSERT_TRUE(readyPerLayer[0].contains(forResults[0].getDefiningOp()));

  // Forward, but start at block arguments.
  wires.clear();
  for (Value arg : blockArgs) {
    wires.emplace_back(arg);
  }

  readyPerLayer.clear();
  res = qco::walkProgramGraph<qco::WireDirection::Forward>(
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
  ASSERT_GE(readyPerLayer.size(), 1);
  ASSERT_TRUE(readyPerLayer[0].contains(iterQ0.getDefiningOp()));
}
