/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/QC/Builder/QCProgramBuilder.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::qco;
using namespace mqt::test;

using ProgramFn = void (*)(mlir::qc::QCProgramBuilder&);
using NativePredicate = bool (*)(OwningOpRef<ModuleOp>&);

template <typename... Allowed1QOps>
static bool onlyTheseOps(OwningOpRef<ModuleOp>& moduleOp, bool allowCx,
                         bool allowCz) {
  bool ok = true;
  std::ignore = moduleOp->walk([&](UnitaryOpInterface op) {
    Operation* raw = op.getOperation();
    if (llvm::isa_and_present<CtrlOp>(raw->getParentOp())) {
      return WalkResult::advance();
    }
    if (llvm::isa<BarrierOp, GPhaseOp>(raw)) {
      return WalkResult::advance();
    }
    if (auto ctrl = llvm::dyn_cast<CtrlOp>(raw)) {
      if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
        ok = false;
        return WalkResult::interrupt();
      }
      Operation* body = ctrl.getBodyUnitary(0).getOperation();
      const bool isCx = llvm::isa<XOp>(body);
      const bool isCz = llvm::isa<ZOp>(body);
      if ((isCx && allowCx) || (isCz && allowCz)) {
        return WalkResult::advance();
      }
      ok = false;
      return WalkResult::interrupt();
    }
    if (!llvm::isa<Allowed1QOps...>(raw)) {
      ok = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ok;
}

static bool onlyIbmBasicCxOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<XOp, SXOp, RZOp, POp>(m, true, false);
}
static bool onlyIbmBasicCzOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<XOp, SXOp, RZOp, POp>(m, false, true);
}
static bool onlyGenericU3CxOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<UOp>(m, true, false);
}
static bool onlyIqmDefaultOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<ROp>(m, false, true);
}
static bool onlyIbmFractionalOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<XOp, SXOp, RZOp, POp, RXOp, RZZOp>(m, false, true);
}
static bool onlyAxisPairRxRzCxOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<RXOp, RZOp, POp>(m, true, false);
}
static bool onlyAxisPairRxRyCxOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<RXOp, RYOp>(m, true, false);
}
static bool onlyAxisPairRyRzCzOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<RYOp, RZOp, POp>(m, false, true);
}
static bool onlyGenericU3CxOrCzOps(OwningOpRef<ModuleOp>& m) {
  return onlyTheseOps<UOp>(m, true, true);
}

static std::optional<Value> unitaryQubitOperand(UnitaryOpInterface op,
                                                std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getOperand(index);
  if (!llvm::isa<QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

static std::optional<Value> unitaryQubitResult(UnitaryOpInterface op,
                                               std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getResult(index);
  if (!llvm::isa<QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

static bool extractSingleQubitMatrix(UnitaryOpInterface op, Matrix2x2& out) {
  if (op.getUnitaryMatrix2x2(out)) {
    return true;
  }
  DynamicMatrix dynamic;
  if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 2 ||
      dynamic.cols() != 2) {
    return false;
  }
  out = Matrix2x2::fromElements(dynamic(0, 0), dynamic(0, 1), dynamic(1, 0),
                                dynamic(1, 1));
  return true;
}

static bool extractTwoQubitMatrix(UnitaryOpInterface op, Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      out = twoQubitControlledX01();
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      out = twoQubitControlledZ();
      return true;
    }
    return false;
  }
  return op.getUnitaryMatrix4x4(out);
}

static std::optional<DynamicMatrix>
computeUnitaryFromQcoModule(const OwningOpRef<ModuleOp>& moduleOp) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }

  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;
  std::size_t numQubits = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto staticOp = llvm::dyn_cast<StaticOp>(&rawOp)) {
          const auto index = static_cast<std::size_t>(staticOp.getIndex());
          qubitIds.try_emplace(staticOp.getQubit(), index);
          numQubits = std::max(numQubits, index + 1);
        } else if (auto alloc = llvm::dyn_cast<AllocOp>(&rawOp)) {
          qubitIds.try_emplace(alloc.getResult(), nextQubitId++);
          numQubits = std::max(numQubits, nextQubitId);
        }
      }
    }
  }

  if (numQubits == 0) {
    return std::nullopt;
  }

  DynamicMatrix unitary =
      DynamicMatrix::identity(static_cast<std::int64_t>(1ULL << numQubits));

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    const auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<BarrierOp, GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          const auto qIn = unitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          const auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = oneQ.embedInNqubit(numQubits, *qid) * unitary;
          const auto qOut = unitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = unitaryQubitOperand(op, 0);
          const auto q1In = unitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          const auto q0id = getQubitId(*q0In);
          const auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary = twoQ.embedInNqubit(numQubits, *q0id, *q1id) * unitary;
          const auto q0Out = unitaryQubitResult(op, 0);
          const auto q1Out = unitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }

        return std::nullopt;
      }
    }
  }

  return unitary;
}

namespace {

struct ProfileCase {
  const char* name;
  ProgramFn program;
  const char* nativeGates;
  NativePredicate isNative;
  bool checkEquivalence;
};

struct FusionCase {
  const char* name;
  ProgramFn program;
  const char* nativeGates;
  std::optional<std::size_t> exactCtrlCount;
  std::optional<std::size_t> minCtrlCount;
  bool checkTwoQUnitary;
};

class FuseTwoQubitUnitaryRunsPassTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, QCODialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static void runFusePipeline(OwningOpRef<ModuleOp>& moduleOp,
                              StringRef nativeGates) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(createQCToQCO());
    pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
        .nativeGates = nativeGates.str(),
    }));
    ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  }

  static void runQcToQco(OwningOpRef<ModuleOp>& moduleOp) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(createQCToQCO());
    ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  }

  static void runTwoQFuse(OwningOpRef<ModuleOp>& moduleOp,
                          StringRef nativeGates) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
        .nativeGates = nativeGates.str(),
    }));
    ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  }

  static void expectQcoModulesEquivalent(const OwningOpRef<ModuleOp>& lhs,
                                         const OwningOpRef<ModuleOp>& rhs) {
    const auto lhsUnitary = computeUnitaryFromQcoModule(lhs);
    ASSERT_TRUE(lhsUnitary.has_value());
    const auto rhsUnitary = computeUnitaryFromQcoModule(rhs);
    ASSERT_TRUE(rhsUnitary.has_value());
    EXPECT_TRUE(isEquivalentUpToGlobalPhase(*lhsUnitary, *rhsUnitary));
  }

  void expectNativeAfterSynthesis(ProgramFn program, StringRef nativeGates,
                                  NativePredicate isNative) {
    auto moduleOp = mlir::qc::QCProgramBuilder::build(context.get(), program);
    runFusePipeline(moduleOp, nativeGates);
    EXPECT_TRUE(isNative(moduleOp));
  }

  void expectEquivalentAndNativeAfterSynthesis(ProgramFn program,
                                               StringRef nativeGates,
                                               NativePredicate isNative) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), program);
    runQcToQco(expected);
    auto synthesized =
        mlir::qc::QCProgramBuilder::build(context.get(), program);
    runFusePipeline(synthesized, nativeGates);
    EXPECT_TRUE(isNative(synthesized));
    expectQcoModulesEquivalent(expected, synthesized);
  }

  void expectSynthesisFailure(ProgramFn program, StringRef nativeGates) {
    auto moduleOp = mlir::qc::QCProgramBuilder::build(context.get(), program);
    PassManager pm(moduleOp->getContext());
    pm.addPass(createQCToQCO());
    pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
        .nativeGates = nativeGates.str(),
    }));
    EXPECT_TRUE(failed(pm.run(*moduleOp)));
  }

  void expectTwoQFusePreservesUnitary(ProgramFn program,
                                      StringRef nativeGates) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), program);
    ASSERT_TRUE(expected);
    runQcToQco(expected);
    auto fused = mlir::qc::QCProgramBuilder::build(context.get(), program);
    ASSERT_TRUE(fused);
    runQcToQco(fused);
    runTwoQFuse(fused, nativeGates);
    ASSERT_TRUE(succeeded(verify(*fused)));
    expectQcoModulesEquivalent(expected, fused);
  }

  static std::size_t countCtrlOps(const OwningOpRef<ModuleOp>& moduleOp) {
    std::size_t count = 0;
    moduleOp.get()->walk([&](CtrlOp) { ++count; });
    return count;
  }

  std::unique_ptr<MLIRContext> context;
};

class FuseTwoQubitProfileTest
    : public FuseTwoQubitUnitaryRunsPassTest,
      public testing::WithParamInterface<ProfileCase> {};

class FuseTwoQubitFusionTest : public FuseTwoQubitUnitaryRunsPassTest,
                               public testing::WithParamInterface<FusionCase> {
};

} // namespace

static void fusionCxCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q0, q1);
}

static void fusionHCxInterleavedTCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.t(q1);
  b.s(q0);
  b.cx(q0, q1);
}

static void fusionThreeLineCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q1, q2);
  b.cx(q0, q1);
}

static void fusionCxRSharedOtherPair(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.cx(q0, q1);
  b.rz(0.17, q1);
  b.cx(q1, q2);
}

static void fusionCxBarrierCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.barrier({q0, q1});
  b.cx(q0, q1);
}

static void fusionSwapCxPattern(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q1, q0);
  b.cx(q0, q1);
}

static void fusionHRzzSRzz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.rzz(-0.29, q0, q1);
  b.s(q1);
  b.rzz(0.17, q0, q1);
}

static void broadOneQThenCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.y(q1);
  b.h(q0);
  b.sx(q1);
  b.rx(0.13, q0);
  b.ry(-0.47, q1);
  b.rz(0.29, q0);
  b.cz(q0, q1);
}

static void zeroAngleThenCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.rx(0.0, q0);
  b.ry(0.0, q1);
  b.rz(0.0, q0);
  b.p(0.0, q1);
  b.cz(q0, q1);
}

static void ibmFractionalGateFamilies(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.rx(0.13, q1);
  b.cx(q0, q1);
  b.cz(q1, q0);
  b.swap(q0, q1);
  b.rzz(-0.33, q0, q1);
  b.rzx(0.41, q0, q1);
}

static void hstycxTwoQ(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.s(q0);
  b.t(q0);
  b.y(q0);
  b.cx(q0, q1);
}

static void cxYOnQ1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.y(q1);
}

static void hCxTOnQ1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q1);
  b.cx(q0, q1);
  b.t(q1);
}

static void xYSXCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.y(q0);
  b.sx(q0);
  b.cz(q0, q1);
}

static void hYCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q0);
  b.cx(q0, q1);
}

static void zCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.z(q0);
  b.cx(q0, q1);
}

static void xHCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.h(q0);
  b.cz(q0, q1);
}

static void hq0Yq1CxSq0(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q1);
  b.cx(q0, q1);
  b.s(q0);
}

static void hCxSq1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.s(q1);
}

static void threeQGhz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.cx(q1, q2);
}

static void determinismSwap(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.swap(q0, q1);
  b.dealloc(q0);
  b.dealloc(q1);
}

TEST_P(FuseTwoQubitProfileTest, SynthesizesToNativeMenu) {
  const ProfileCase& c = GetParam();
  if (c.checkEquivalence) {
    expectEquivalentAndNativeAfterSynthesis(c.program, c.nativeGates,
                                            c.isNative);
  } else {
    expectNativeAfterSynthesis(c.program, c.nativeGates, c.isNative);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Menus, FuseTwoQubitProfileTest,
    testing::Values(
        ProfileCase{"SwapIbmBasic", mlir::qc::swap, "x,sx,rz,cx",
                    onlyIbmBasicCxOps, false},
        ProfileCase{"SwapGeneric", mlir::qc::swap, "u,cx", onlyGenericU3CxOps,
                    false},
        ProfileCase{"SwapIqm", mlir::qc::swap, "r,cz", onlyIqmDefaultOps,
                    false},
        ProfileCase{"HstycxIbm", hstycxTwoQ, "x,sx,rz,cx", onlyIbmBasicCxOps,
                    false},
        ProfileCase{"CxYIqm", cxYOnQ1, "r,cz", onlyIqmDefaultOps, false},
        ProfileCase{"BroadOneQIqm", broadOneQThenCz, "r,cz", onlyIqmDefaultOps,
                    false},
        ProfileCase{"ZeroAngleRyRzCz", zeroAngleThenCz, "ry,rz,cz",
                    onlyAxisPairRyRzCzOps, false},
        ProfileCase{"HCxTIbmCz", hCxTOnQ1, "x,sx,rz,cz", onlyIbmBasicCzOps,
                    false},
        ProfileCase{"XYSXCzIqm", xYSXCz, "r,cz", onlyIqmDefaultOps, false},
        ProfileCase{"IbmFractional", ibmFractionalGateFamilies,
                    "x,sx,rz,rx,rzz,cz", onlyIbmFractionalOps, false},
        ProfileCase{"HYCxRxRz", hYCx, "rx,rz,cx", onlyAxisPairRxRzCxOps, false},
        ProfileCase{"ZCxRxRy", zCx, "rx,ry,cx", onlyAxisPairRxRyCxOps, false},
        ProfileCase{"Hq0Yq1CxSq0", hq0Yq1CxSq0, "u,cx", onlyGenericU3CxOps,
                    true},
        ProfileCase{"XHCzRyRz", xHCz, "ry,rz,cz", onlyAxisPairRyRzCzOps, true},
        ProfileCase{"HCxSq1MultiEnt", hCxSq1, "u,cx,cz", onlyGenericU3CxOrCzOps,
                    true}),
    [](const testing::TestParamInfo<ProfileCase>& info) {
      return info.param.name;
    });

TEST_F(FuseTwoQubitUnitaryRunsPassTest, FailsForUnsupportedNativeGateMenu) {
  expectSynthesisFailure(mlir::qc::h, "not-a-gate");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(mlir::qc::singleControlledX, "cx,cz");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, FailsForMultiControlledGateStructure) {
  expectSynthesisFailure(mlir::qc::multipleControlledX, "x,sx,rz,cx");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       CandidateSelectionIsDeterministicAcrossRuns) {
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(context.get(), determinismSwap);
  };
  auto firstModule = buildFn();
  runFusePipeline(firstModule, "u,cx");
  auto secondModule = buildFn();
  runFusePipeline(secondModule, "u,cx");

  std::string first;
  std::string second;
  llvm::raw_string_ostream osFirst(first);
  llvm::raw_string_ostream osSecond(second);
  firstModule->print(osFirst);
  secondModule->print(osSecond);
  EXPECT_EQ(first, second);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, ThreeQubitGhzEquivalentOnCoreProfiles) {
  const std::array<ProfileCase, 3> profiles{{
      {.name = "GhzIbm",
       .program = threeQGhz,
       .nativeGates = "x,sx,rz,cx",
       .isNative = onlyIbmBasicCxOps,
       .checkEquivalence = false},
      {.name = "GhzGeneric",
       .program = threeQGhz,
       .nativeGates = "u,cx",
       .isNative = onlyGenericU3CxOps,
       .checkEquivalence = false},
      {.name = "GhzIqm",
       .program = threeQGhz,
       .nativeGates = "r,cz",
       .isNative = onlyIqmDefaultOps,
       .checkEquivalence = false},
  }};
  for (const ProfileCase& profile : profiles) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runQcToQco(expected);
    auto synthesized =
        mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runFusePipeline(synthesized, profile.nativeGates);
    EXPECT_TRUE(profile.isNative(synthesized));
    expectQcoModulesEquivalent(expected, synthesized);
  }
}

TEST_P(FuseTwoQubitFusionTest, WindowFusionBehavior) {
  const FusionCase& c = GetParam();
  if (c.checkTwoQUnitary) {
    expectTwoQFusePreservesUnitary(c.program, c.nativeGates);
  }
  auto module = mlir::qc::QCProgramBuilder::build(context.get(), c.program);
  ASSERT_TRUE(module);
  runQcToQco(module);
  runTwoQFuse(module, c.nativeGates);
  if (c.exactCtrlCount) {
    EXPECT_EQ(countCtrlOps(module), *c.exactCtrlCount);
  }
  if (c.minCtrlCount) {
    EXPECT_GE(countCtrlOps(module), *c.minCtrlCount);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Windows, FuseTwoQubitFusionTest,
    testing::Values(FusionCase{"AdjacentCxCancel", fusionCxCx, "u,cx", 0,
                               std::nullopt, true},
                    FusionCase{"InterleavedOneQ", fusionHCxInterleavedTCx,
                               "u,cx", std::nullopt, std::nullopt, true},
                    FusionCase{"DifferentPairBoundary", fusionThreeLineCx,
                               "u,cx", std::nullopt, 1, false},
                    FusionCase{"SharedWireOneQ", fusionCxRSharedOtherPair,
                               "u,cx", std::nullopt, 2, false},
                    FusionCase{"BarrierBoundary", fusionCxBarrierCx, "u,cx", 2,
                               std::nullopt, false},
                    FusionCase{"SwappedWireOrder", fusionSwapCxPattern, "u,cx",
                               std::nullopt, std::nullopt, true},
                    FusionCase{"RzzBlock", fusionHRzzSRzz, "x,sx,rz,rx,rzz,cz",
                               std::nullopt, std::nullopt, true}),
    [](const testing::TestParamInfo<FusionCase>& info) {
      return info.param.name;
    });

TEST_F(FuseTwoQubitUnitaryRunsPassTest, InvalidNativeGatesFailsPass) {
  auto module = mlir::qc::QCProgramBuilder::build(context.get(), fusionCxCx);
  ASSERT_TRUE(module);
  runQcToQco(module);
  PassManager pm(module->getContext());
  pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
      .nativeGates = "not-a-gate",
  }));
  EXPECT_TRUE(failed(pm.run(*module)));
}
