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
#include "mlir/Dialect/QCO/IR/QCODialect.h"
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

class MappingPassTest : public testing::Test,
                        public testing::WithParamInterface<ArchitectureParam> {
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

  std::unique_ptr<MLIRContext> context;
};
}; // namespace

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
