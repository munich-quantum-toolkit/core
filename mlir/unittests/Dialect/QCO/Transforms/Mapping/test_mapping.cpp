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
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Algorithms.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/Qubits.h"

#include <gtest/gtest.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

using DeviceSpec = std::pair<size_t, Edges>;

/**
 * @returns llvm::success() if all two-qubit gates inside @p region
 * fulfill the given coupling constraints. llvm::failure(), otherwise.
 */
static LogicalResult isExecutable(Region& region, const Edges& coupling) {
  return walkProgram(region, [&](Operation* curr, const Qubits& qubits) {
    if (auto op = dyn_cast<UnitaryOpInterface>(curr)) {
      if (isa<BarrierOp>(op)) {
        return WalkResult::advance();
      }
      if (op.getNumQubits() > 1) {
        const auto q0 = cast<TypedValue<QubitType>>(op.getInputQubit(0));
        const auto q1 = cast<TypedValue<QubitType>>(op.getInputQubit(1));
        const auto i0 = qubits.getIndex(q0);
        const auto i1 = qubits.getIndex(q1);

        if (!coupling.contains(std::make_pair(i0, i1))) {
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });
}

/**
 * @returns a 9x9 square-grid device.
 */
static DeviceSpec getNineQubitSquareGrid() {
  const static Edges COUPLING{{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                              {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                              {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                              {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}};
  return std::make_pair(9, COUPLING);
}

namespace {

class MappingPassTest : public testing::Test,
                        public testing::WithParamInterface<DeviceSpec> {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static void runPass(OwningOpRef<ModuleOp>& program, const DeviceSpec& device,
                      const qco::MappingPassOptions& options) {
    PassManager pm(program->getContext());
    pm.addPass(qco::createMappingPass(device.first, device.second, options));
    auto res = pm.run(*program);
    ASSERT_TRUE(res.succeeded());
  }

  std::unique_ptr<MLIRContext> context;
};

}; // namespace

TEST_P(MappingPassTest, GHZ) {
  const auto& device = GetParam();

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value q0 = builder.allocQubit();
  Value q1 = builder.allocQubit();
  Value q2 = builder.allocQubit();

  q0 = builder.h(q0);
  std::tie(q0, q1) = builder.cx(q0, q1);
  std::tie(q0, q2) = builder.cx(q0, q2);

  builder.sink(q0);
  builder.sink(q1);
  builder.sink(q2);

  auto program = builder.finalize();

  runPass(program, device, qco::MappingPassOptions{});
  auto entry = *(program->getOps<func::FuncOp>().begin());
  EXPECT_TRUE(isExecutable(entry.getFunctionBody(), device.second).succeeded());
}

TEST_P(MappingPassTest, Sabre) {
  const auto& device = GetParam();

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value q0 = builder.allocQubit();
  Value q1 = builder.allocQubit();
  Value q2 = builder.allocQubit();
  Value q3 = builder.allocQubit();
  Value q4 = builder.allocQubit();
  Value q5 = builder.allocQubit();

  q0 = builder.h(q0);
  q1 = builder.h(q1);
  q4 = builder.h(q4);

  q0 = builder.z(q0);
  std::tie(q1, q2) = builder.cx(q1, q2);
  std::tie(q4, q5) = builder.cx(q4, q5);

  std::tie(q0, q1) = builder.cx(q0, q1);

  q0 = builder.h(q0);
  q1 = builder.y(q1);
  std::tie(q0, q1) = builder.cx(q0, q1);

  std::tie(q2, q3) = builder.cx(q2, q3);

  q2 = builder.h(q2);
  q3 = builder.h(q3);

  std::tie(q1, q2) = builder.cx(q1, q2);
  std::tie(q3, q5) = builder.cx(q3, q5);

  q3 = builder.z(q3);

  std::tie(q3, q4) = builder.cx(q3, q4);

  std::tie(q3, q0) = builder.cx(q3, q0);

  ValueRange out = builder.barrier({q0, q1, q2, q3, q4, q5});
  q0 = out[0];
  q1 = out[1];
  q2 = out[2];
  q3 = out[3];
  q4 = out[4];
  q5 = out[5];

  Value c0;
  Value c1;
  Value c2;
  Value c3;
  Value c4;
  Value c5;

  std::tie(q0, c0) = builder.measure(q0);
  std::tie(q1, c1) = builder.measure(q1);
  std::tie(q2, c2) = builder.measure(q2);
  std::tie(q3, c3) = builder.measure(q3);
  std::tie(q4, c4) = builder.measure(q4);
  std::tie(q5, c5) = builder.measure(q5);

  builder.sink(q0);
  builder.sink(q1);
  builder.sink(q2);
  builder.sink(q3);
  builder.sink(q4);
  builder.sink(q5);

  auto program = builder.finalize();
  runPass(program, device, qco::MappingPassOptions{});
  auto entry = *(program->getOps<func::FuncOp>().begin());
  EXPECT_TRUE(isExecutable(entry.getFunctionBody(), device.second).succeeded());
}

INSTANTIATE_TEST_SUITE_P(NineQubitSquareGrid, MappingPassTest,
                         testing::Values(getNineQubitSquareGrid()));
