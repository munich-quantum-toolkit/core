/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// Standalone ``fuse-two-qubit-unitary-runs`` pass tests.

#include "native_synthesis_pass_test_fixture.h"
#include "native_synthesis_test_helpers.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/QCO/IR/QCOOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

namespace {

static void runFuseTwoQubitUnitaryRuns(OwningOpRef<ModuleOp>& moduleOp,
                                       const std::string& nativeGates) {
  PassManager pm(moduleOp->getContext());
  FuseTwoQubitUnitaryRunsOptions opts;
  opts.nativeGates = nativeGates;
  pm.addPass(createFuseTwoQubitUnitaryRuns(opts));
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
}

template <typename... OpTs>
static std::size_t
countOpsOfTypeInModule(const OwningOpRef<ModuleOp>& moduleOp) {
  std::size_t count = 0;
  moduleOp.get()->walk([&](Operation* op) {
    if (llvm::isa<OpTs...>(op)) {
      ++count;
    }
  });
  return count;
}

} // namespace

class FuseTwoQubitUnitaryRunsTest : public NativeSynthesisPassTest {};

TEST_F(FuseTwoQubitUnitaryRunsTest, InvalidMenuFailsPass) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionCxCx);
  runQcToQco(moduleOp);
  PassManager pm(moduleOp->getContext());
  FuseTwoQubitUnitaryRunsOptions opts;
  opts.nativeGates = "not-a-real-menu";
  pm.addPass(createFuseTwoQubitUnitaryRuns(opts));
  EXPECT_TRUE(failed(pm.run(*moduleOp)));
}

TEST_F(FuseTwoQubitUnitaryRunsTest, EmptyMenuIsNoOp) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionCxCx);
  runQcToQco(moduleOp);
  const auto before = countOpsOfTypeInModule<qco::CtrlOp>(moduleOp);
  runFuseTwoQubitUnitaryRuns(moduleOp, "");
  EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(moduleOp), before);
}

TEST_F(FuseTwoQubitUnitaryRunsTest, CancelsAdjacentCxPair) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionCxCx);
  runQcToQco(moduleOp);
  runFuseTwoQubitUnitaryRuns(moduleOp, "u,cx");
  EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(moduleOp), 0U);
}

TEST_F(FuseTwoQubitUnitaryRunsTest, PreservesSingleCx) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::nativeSynthFusionHadamardCxHadamard);
  runQcToQco(moduleOp);
  runFuseTwoQubitUnitaryRuns(moduleOp, "u,cx");
  EXPECT_EQ(countOpsOfTypeInModule<qco::CtrlOp>(moduleOp), 1U);
}

TEST_F(FuseTwoQubitUnitaryRunsTest, FusesCxThroughInterleavedOneQOps) {
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(
        context.get(), mlir::qc::nativeSynthFusionHCxInterleavedTCx);
  };
  auto expected = buildFn();
  runQcToQco(expected);
  const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
  ASSERT_TRUE(expectedUnitary.has_value());

  auto moduleOp = buildFn();
  runQcToQco(moduleOp);
  runFuseTwoQubitUnitaryRuns(moduleOp, "u,cx");
  const auto fusedUnitary = computeTwoQubitUnitaryFromModule(moduleOp);
  ASSERT_TRUE(fusedUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*expectedUnitary, *fusedUnitary));
}
