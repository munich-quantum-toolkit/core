/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Architecture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::qco;

namespace {
struct ArchitectureParam {
  std::string name;
  Architecture (*factory)();
};

class MappingPassTestBase : public testing::Test {
public:
  /**
   * @brief Walks the IR and validates if each two-qubit op is executable on the
   * given architecture.
   * @returns true iff. all two-qubit gates are executable on the architecture.
   */
  static bool isExecutable(OwningOpRef<ModuleOp>& moduleOp,
                           const Architecture& arch) {
    auto entry = *(moduleOp->getOps<func::FuncOp>().begin());
    DenseMap<Value, std::size_t> mappings;
    for_each(entry.getOps<qc::StaticOp>(), [&](qc::StaticOp op) {
      mappings.try_emplace(op.getQubit(), op.getIndex());
    });

    bool executable = true;
    std::ignore = moduleOp->walk([&](qc::UnitaryOpInterface op) {
      if (isa<qc::BarrierOp>(op)) {
        return WalkResult::advance();
      }
      if (op.getNumQubits() > 1) {
        assert(op.getNumQubits() == 2 &&
               "Expected only 2-qubit gates after decomposition");
        assert(mappings.contains(op.getQubit(0)) && "Qubit 0 not in mapping");
        assert(mappings.contains(op.getQubit(1)) && "Qubit 1 not in mapping");
        const auto i0 = mappings[op.getQubit(0)];
        const auto i1 = mappings[op.getQubit(1)];
        if (!arch.areAdjacent(i0, i1)) {
          executable = false;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    return executable;
  }

  static Architecture getRigettiNovera() {
    // TODO: At some point this should be provided via QDMI.
    const static Architecture::CouplingSet COUPLING{
        {0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1}, {1, 2}, {2, 1},
        {2, 5}, {5, 2}, {3, 6}, {6, 3}, {3, 4}, {4, 3}, {4, 7}, {7, 4},
        {4, 5}, {5, 4}, {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}};
    return Architecture("RigettiNovera", 9, COUPLING);
  }

protected:
  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static void runHeuristicMapping(OwningOpRef<ModuleOp>& moduleOp) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(createQCToQCO());
    pm.addPass(qco::createMappingPass(qco::MappingPassOptions{.nlookahead = 5,
                                                              .alpha = 1,
                                                              .lambda = 0.85,
                                                              .niterations = 2,
                                                              .ntrials = 16,
                                                              .seed = 1337}));
    pm.addPass(createQCOToQC());
    auto res = pm.run(*moduleOp);
    ASSERT_TRUE(succeeded(res));
  }

  static void runQCOMapping(OwningOpRef<ModuleOp>& moduleOp) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(qco::createMappingPass(qco::MappingPassOptions{.nlookahead = 5,
                                                              .alpha = 1,
                                                              .lambda = 0.85,
                                                              .niterations = 2,
                                                              .ntrials = 8,
                                                              .seed = 1337}));
    auto res = pm.run(*moduleOp);
    ASSERT_TRUE(succeeded(res));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
  }

  static void expectSameStaticQubitType(Value input, Value output) {
    const auto qIn = dyn_cast<qco::QubitType>(input.getType());
    const auto qOut = dyn_cast<qco::QubitType>(output.getType());
    ASSERT_TRUE(qIn);
    ASSERT_TRUE(qOut);
    EXPECT_EQ(qIn, qOut);
    (void)qOut;
  }

  std::unique_ptr<MLIRContext> context;
};

class MappingPassTest : public MappingPassTestBase,
                        public testing::WithParamInterface<ArchitectureParam> {
};
}; // namespace

TEST_F(MappingPassTestBase, SynchronizesParameterizedGateAndMeasureTypes) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  auto q0 = builder.allocQubit();
  auto q1 = builder.allocQubit();

  q0 = builder.rx(0.25, q0);
  std::tie(q0, q1) = builder.rxx(0.5, q0, q1);
  std::tie(q0, q1) = builder.xx_plus_yy(0.75, 1.25, q0, q1);
  const auto [measuredQubit, measuredBit] = builder.measure(q0);
  (void)measuredBit;
  q0 = measuredQubit;

  builder.sink(q0);
  builder.sink(q1);

  auto moduleOp = builder.finalize();
  runQCOMapping(moduleOp);

  auto entry = *moduleOp->getOps<func::FuncOp>().begin();
  qco::RXOp rxOp;
  qco::RXXOp rxxOp;
  qco::XXPlusYYOp xxPlusYYOp;
  qco::MeasureOp measureOp;
  entry.walk([&](Operation* op) {
    if (auto rx = dyn_cast<qco::RXOp>(op)) {
      rxOp = rx;
    } else if (auto rxx = dyn_cast<qco::RXXOp>(op)) {
      rxxOp = rxx;
    } else if (auto xxPlusYY = dyn_cast<qco::XXPlusYYOp>(op)) {
      xxPlusYYOp = xxPlusYY;
    } else if (auto measure = dyn_cast<qco::MeasureOp>(op)) {
      measureOp = measure;
    }
  });

  ASSERT_TRUE(rxOp);
  ASSERT_TRUE(rxxOp);
  ASSERT_TRUE(xxPlusYYOp);
  ASSERT_TRUE(measureOp);

  expectSameStaticQubitType(rxOp.getQubitIn(), rxOp.getQubitOut());
  expectSameStaticQubitType(rxxOp.getQubit0In(), rxxOp.getQubit0Out());
  expectSameStaticQubitType(rxxOp.getQubit1In(), rxxOp.getQubit1Out());
  expectSameStaticQubitType(xxPlusYYOp.getQubit0In(),
                            xxPlusYYOp.getQubit0Out());
  expectSameStaticQubitType(xxPlusYYOp.getQubit1In(),
                            xxPlusYYOp.getQubit1Out());
  expectSameStaticQubitType(measureOp.getQubitIn(), measureOp.getQubitOut());
}

TEST_F(MappingPassTestBase, SynchronizesModifierRegionArguments) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  auto q0 = builder.allocQubit();
  auto q1 = builder.allocQubit();

  auto [controlsOut, targetsOut] = builder.ctrl(
      {q0}, {q1}, [&](ValueRange targets) -> llvm::SmallVector<Value> {
        return {builder.rx(0.5, targets[0])};
      });
  q0 = controlsOut.front();
  q1 = targetsOut.front();

  auto invOut =
      builder.inv({q1}, [&](ValueRange qubits) -> llvm::SmallVector<Value> {
        return {builder.rx(0.25, qubits[0])};
      });
  q1 = invOut.front();

  builder.sink(q0);
  builder.sink(q1);

  auto moduleOp = builder.finalize();
  runQCOMapping(moduleOp);

  auto entry = *moduleOp->getOps<func::FuncOp>().begin();
  qco::CtrlOp ctrlOp;
  qco::InvOp invOp;
  qco::RXOp ctrlBodyRxOp;
  qco::RXOp invBodyRxOp;
  entry.walk([&](Operation* op) {
    if (auto ctrl = dyn_cast<qco::CtrlOp>(op)) {
      ctrlOp = ctrl;
    } else if (auto inv = dyn_cast<qco::InvOp>(op)) {
      invOp = inv;
    } else if (auto rx = dyn_cast<qco::RXOp>(op)) {
      if (rx->getParentOfType<qco::CtrlOp>()) {
        ctrlBodyRxOp = rx;
      } else if (rx->getParentOfType<qco::InvOp>()) {
        invBodyRxOp = rx;
      }
    }
  });

  ASSERT_TRUE(ctrlOp);
  ASSERT_TRUE(invOp);
  ASSERT_TRUE(ctrlBodyRxOp);
  ASSERT_TRUE(invBodyRxOp);

  ASSERT_EQ(ctrlOp.getBody()->getNumArguments(), ctrlOp.getNumTargets());
  expectSameStaticQubitType(ctrlOp.getTargetsIn()[0],
                            ctrlOp.getBody()->getArgument(0));
  expectSameStaticQubitType(ctrlBodyRxOp.getQubitIn(),
                            ctrlBodyRxOp.getQubitOut());

  ASSERT_EQ(invOp.getBody()->getNumArguments(), invOp.getNumTargets());
  expectSameStaticQubitType(invOp.getQubitsIn()[0],
                            invOp.getBody()->getArgument(0));
  expectSameStaticQubitType(invBodyRxOp.getQubitIn(),
                            invBodyRxOp.getQubitOut());
}

TEST_P(MappingPassTest, GHZ) {
  auto arch = GetParam().factory();

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

  auto moduleOp = builder.finalize();
  runHeuristicMapping(moduleOp);
  EXPECT_TRUE(isExecutable(moduleOp, arch));
}

TEST_P(MappingPassTest, Sabre) {
  auto arch = GetParam().factory();

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

  auto moduleOp = builder.finalize();
  runHeuristicMapping(moduleOp);
  EXPECT_TRUE(isExecutable(moduleOp, arch));
}

INSTANTIATE_TEST_SUITE_P(
    Architectures, MappingPassTest,
    testing::Values(ArchitectureParam{"RigettiNovera",
                                      &MappingPassTest::getRigettiNovera}),
    [](const testing::TestParamInfo<ArchitectureParam>& info) {
      return info.param.name;
    });
