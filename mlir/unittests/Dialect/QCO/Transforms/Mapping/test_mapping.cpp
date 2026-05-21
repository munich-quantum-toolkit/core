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
#include "mlir/Dialect/QCO/Utils/Utils.h"

#include <gtest/gtest.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
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

      assert(op.getNumQubits() <= 2 &&
             "isExecutable: expected two-qubit gate decomposition");

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
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static LogicalResult runPass(ModuleOp m, const DeviceSpec& device,
                               const MappingPassOptions& options) {
    PassManager pm(m->getContext());
    pm.addPass(createMappingPass(device.first, device.second, options));
    return pm.run(m);
  }

  std::unique_ptr<MLIRContext> context;
};

}; // namespace

TEST_P(MappingPassTest, NoEntryPoint) {
  const auto& device = GetParam();

  OwningOpRef m = ModuleOp::create(UnknownLoc::get(context.get()));

  auto res = runPass(m.get(), device, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, NoQubitAllocations) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value q0 = builder.allocQubit();
  q0 = builder.h(q0);
  builder.sink(q0);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, NoTwoTensors) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor0 = builder.qtensorAlloc(1);
  Value tensor1 = builder.qtensorAlloc(1);

  Value q0;
  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  Value q1;
  std::tie(tensor1, q1) = builder.qtensorExtract(tensor1, 0);

  q0 = builder.h(q0);
  q1 = builder.h(q1);

  std::tie(q0, q1) = builder.cx(q0, q1);

  tensor0 = builder.qtensorInsert(q0, tensor0, 0);
  tensor1 = builder.qtensorInsert(q1, tensor1, 0);

  builder.qtensorDealloc(tensor0);
  builder.qtensorDealloc(tensor1);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, NoExtractAfterInsert) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor0 = builder.qtensorAlloc(1);

  Value q0;
  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  q0 = builder.h(q0);
  tensor0 = builder.qtensorInsert(q0, tensor0, 0);

  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  q0 = builder.x(q0);
  tensor0 = builder.qtensorInsert(q0, tensor0, 0);

  builder.qtensorDealloc(tensor0);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, TooManyQubitsForArch) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  int64_t nqubits = static_cast<int64_t>(device.first) + 1;
  Value tensor = builder.qtensorAlloc(nqubits);
  SmallVector<Value> qubits(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    Value qi;
    std::tie(tensor, qi) = builder.qtensorExtract(tensor, i);
    qi = builder.h(qi);
    qubits[i] = qi;
  }

  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, GHZ) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(3);

  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);

  Value q1;
  std::tie(tensor, q1) = builder.qtensorExtract(tensor, 1);

  Value q2;
  std::tie(tensor, q2) = builder.qtensorExtract(tensor, 2);

  q0 = builder.h(q0);
  std::tie(q0, q1) = builder.cx(q0, q1);
  std::tie(q0, q2) = builder.cx(q0, q2);

  tensor = builder.qtensorInsert(q0, tensor, 0);
  tensor = builder.qtensorInsert(q1, tensor, 1);
  tensor = builder.qtensorInsert(q2, tensor, 2);
  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device, MappingPassOptions{});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry.getFunctionBody(), device.second).succeeded());
}

TEST_P(MappingPassTest, Sabre) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(6);

  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);

  Value q1;
  std::tie(tensor, q1) = builder.qtensorExtract(tensor, 1);

  Value q2;
  std::tie(tensor, q2) = builder.qtensorExtract(tensor, 2);

  Value q3;
  std::tie(tensor, q3) = builder.qtensorExtract(tensor, 3);

  Value q4;
  std::tie(tensor, q4) = builder.qtensorExtract(tensor, 4);

  Value q5;
  std::tie(tensor, q5) = builder.qtensorExtract(tensor, 5);

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

  tensor = builder.qtensorInsert(q0, tensor, 0);
  tensor = builder.qtensorInsert(q1, tensor, 1);
  tensor = builder.qtensorInsert(q2, tensor, 2);
  tensor = builder.qtensorInsert(q3, tensor, 3);
  tensor = builder.qtensorInsert(q4, tensor, 4);
  tensor = builder.qtensorInsert(q5, tensor, 5);
  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device, MappingPassOptions{});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry.getFunctionBody(), device.second).succeeded());
}

INSTANTIATE_TEST_SUITE_P(NineQubitSquareGrid, MappingPassTest,
                         testing::Values(getNineQubitSquareGrid()));
