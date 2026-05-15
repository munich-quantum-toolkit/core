/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Algorithms.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"

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

#include <cassert>
#include <memory>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {

using DeviceSpec = std::pair<size_t, Edges>;

/**
 * @returns llvm::success() if all two-qubit gates inside @p region fulfill the
 * given coupling constraints. llvm::failure(), otherwise.
 */
LogicalResult isExecutable(Region& region, const Edges& coupling) {
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

class MappingPassTest : public testing::Test,
                        public testing::WithParamInterface<DeviceSpec> {
public:
  static DeviceSpec getNineQubitSquareGrid() {
    const static Edges COUPLING{{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                                {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                                {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                                {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}};
    return std::make_pair(9, COUPLING);
  }

protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static void runPass(OwningOpRef<ModuleOp>& program, const DeviceSpec& device,
                      const qco::MappingPassOptions& options) {
    PassManager pm(program->getContext());
    pm.addPass(createQCToQCO());
    pm.addPass(qco::createMappingPass(device.first, device.second, options));
    auto res = pm.run(*program);
    ASSERT_TRUE(res.succeeded());
  }

  std::unique_ptr<MLIRContext> context;
};

}; // namespace

TEST_P(MappingPassTest, GHZ) {
  const auto& device = GetParam();

  qc::QCProgramBuilder builder(context.get());
  builder.initialize();

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();

  builder.h(q0);
  builder.cx(q0, q1);
  builder.cx(q0, q2);

  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);

  auto program = builder.finalize();

  const qco::MappingPassOptions options{.nlookahead = 5,
                                        .alpha = 1,
                                        .lambda = 0.85,
                                        .niterations = 2,
                                        .ntrials = 4,
                                        .seed = 1337};

  runPass(program, device, options);
  auto entry = *(program->getOps<func::FuncOp>().begin());
  EXPECT_TRUE(isExecutable(entry.getFunctionBody(), device.second).succeeded());
}

TEST_P(MappingPassTest, Sabre) {
  const auto& device = GetParam();

  qc::QCProgramBuilder builder(context.get());
  builder.initialize();

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();
  const auto q3 = builder.allocQubit();
  const auto q4 = builder.allocQubit();
  const auto q5 = builder.allocQubit();

  builder.h(q0);
  builder.h(q1);
  builder.h(q4);

  builder.z(q0);
  builder.cx(q1, q2);
  builder.cx(q4, q5);

  builder.cx(q0, q1);

  builder.h(q0);
  builder.y(q1);
  builder.cx(q0, q1);

  builder.cx(q2, q3);

  builder.h(q2);
  builder.h(q3);

  builder.cx(q1, q2);
  builder.cx(q3, q5);

  builder.z(q3);

  builder.cx(q3, q4);

  builder.cx(q3, q0);

  builder.barrier({q0, q1, q2, q3, q4, q5});
  builder.measure(q0);
  builder.measure(q1);
  builder.measure(q2);
  builder.measure(q3);
  builder.measure(q4);
  builder.measure(q5);

  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);
  builder.dealloc(q3);
  builder.dealloc(q4);
  builder.dealloc(q5);

  auto program = builder.finalize();
  const qco::MappingPassOptions options{.nlookahead = 1,
                                        .alpha = 1,
                                        .lambda = 0.85,
                                        .niterations = 2,
                                        .ntrials = 1,
                                        .seed = 42};
  runPass(program, device, options);
  auto entry = *(program->getOps<func::FuncOp>().begin());
  EXPECT_TRUE(isExecutable(entry.getFunctionBody(), device.second).succeeded());
}

INSTANTIATE_TEST_SUITE_P(
    NineQubitSquareGrid, MappingPassTest,
    testing::Values(MappingPassTest::getNineQubitSquareGrid()));
