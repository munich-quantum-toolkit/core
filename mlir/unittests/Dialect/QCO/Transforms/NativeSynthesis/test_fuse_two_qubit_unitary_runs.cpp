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
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
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
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
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

namespace mlir::qco::fuse_two_qubit_test {

using mqt::test::cxGate01;
using mqt::test::czGate;
using mqt::test::expandToTwoQubits;
using mqt::test::fixTwoQubitMatrixQubitOrder;
using mqt::test::isEquivalentUpToGlobalPhase;
using mqt::test::QubitId;

[[nodiscard]] static std::optional<Value>
getUnitaryQubitOperand(UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getOperand(index);
  if (!llvm::isa<QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static std::optional<Value>
getUnitaryQubitResult(UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getResult(index);
  if (!llvm::isa<QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static bool extractSingleQubitMatrix(UnitaryOpInterface op,
                                                   Matrix2x2& out) {
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

[[nodiscard]] static bool extractTwoQubitMatrix(UnitaryOpInterface op,
                                                Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      out = cxGate01();
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      out = czGate();
      return true;
    }
    return false;
  }
  return op.getUnitaryMatrix4x4(out);
}

[[nodiscard]] static std::optional<DynamicMatrix>
computeUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp) {
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
    auto it = qubitIds.find(qubit);
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
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = embedSingleQubitInNqubit(oneQ, numQubits, *qid) * unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary =
              embedTwoQubitInNqubit(twoQ, numQubits, *q0id, *q1id) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
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

[[nodiscard]] static std::optional<Matrix4x4>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }
  Matrix4x4 unitary = Matrix4x4::identity();
  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = llvm::dyn_cast<AllocOp>(&rawOp)) {
          if (nextQubitId >= 2) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), nextQubitId++);
        }
      }
    }
  }

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
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
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary =
              expandToTwoQubits(oneQ, static_cast<QubitId>(*qid)) * unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          const llvm::SmallVector<QubitId, 2> ids{static_cast<QubitId>(*q0id),
                                                  static_cast<QubitId>(*q1id)};
          unitary = fixTwoQubitMatrixQubitOrder(twoQ, ids) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }
      }
    }
  }

  if (nextQubitId != 2) {
    return std::nullopt;
  }
  return unitary;
}

static void expectQcoModulesEquivalent(const OwningOpRef<ModuleOp>& lhs,
                                       const OwningOpRef<ModuleOp>& rhs) {
  const auto lhsUnitary = computeUnitaryFromModule(lhs);
  ASSERT_TRUE(lhsUnitary.has_value());
  const auto rhsUnitary = computeUnitaryFromModule(rhs);
  ASSERT_TRUE(rhsUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*lhsUnitary, *rhsUnitary));
}

static void
expectTwoQubitQcoModulesEquivalent(const OwningOpRef<ModuleOp>& lhs,
                                   const OwningOpRef<ModuleOp>& rhs) {
  const auto lhsUnitary = computeTwoQubitUnitaryFromModule(lhs);
  ASSERT_TRUE(lhsUnitary.has_value());
  const auto rhsUnitary = computeTwoQubitUnitaryFromModule(rhs);
  ASSERT_TRUE(rhsUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*lhsUnitary, *rhsUnitary));
}

/// One row of the standard multi-profile equivalence sweeps in tests.
// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_F` at global scope
struct FuseTwoQubitProfileSweepCase {
  const char* nativeGates;
  bool (*isNative)(OwningOpRef<ModuleOp>&);
};

/// Shared gtest fixture for ``fuse-two-qubit-unitary-runs`` pass tests.
// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_F` at global scope
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

  template <typename... Allowed1QOps>
  static bool onlyTheseOps(OwningOpRef<ModuleOp>& moduleOp, const bool allowCx,
                           const bool allowCz) {
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

  static bool onlyIbmBasicCxOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<XOp, SXOp, RZOp, POp>(moduleOp, /*allowCx=*/true,
                                              /*allowCz=*/false);
  }

  static bool onlyIbmBasicCzOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<XOp, SXOp, RZOp, POp>(moduleOp, /*allowCx=*/false,
                                              /*allowCz=*/true);
  }

  static bool onlyGenericU3CxOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<UOp>(moduleOp, /*allowCx=*/true, /*allowCz=*/false);
  }

  static bool onlyGenericU3CzOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<UOp>(moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool onlyIqmDefaultOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<ROp>(moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool onlyIbmFractionalOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<XOp, SXOp, RZOp, POp, RXOp, RZZOp>(
        moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool onlyAxisPairRxRzCxOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<RXOp, RZOp, POp>(moduleOp, /*allowCx=*/true,
                                         /*allowCz=*/false);
  }

  static bool onlyAxisPairRxRyCxOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<RXOp, RYOp>(moduleOp, /*allowCx=*/true,
                                    /*allowCz=*/false);
  }

  static bool onlyAxisPairRyRzCzOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<RYOp, RZOp, POp>(moduleOp, /*allowCx=*/false,
                                         /*allowCz=*/true);
  }

  static bool onlyUOrAxisPairRxRzCxOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<UOp, RXOp, RZOp, POp>(moduleOp, /*allowCx=*/true,
                                              /*allowCz=*/false);
  }

  static bool onlyGenericU3CxOrCzOps(OwningOpRef<ModuleOp>& moduleOp) {
    return onlyTheseOps<UOp>(moduleOp, /*allowCx=*/true, /*allowCz=*/true);
  }

  static std::array<FuseTwoQubitProfileSweepCase, 3> coreEquivalenceProfiles() {
    return {{{.nativeGates = "x,sx,rz,cx", .isNative = &onlyIbmBasicCxOps},
             {.nativeGates = "u,cx", .isNative = &onlyGenericU3CxOps},
             {.nativeGates = "r,cz", .isNative = &onlyIqmDefaultOps}}};
  }

  static void
  runFuseTwoQubitUnitaryRunsPipeline(OwningOpRef<ModuleOp>& moduleOp,
                                     const std::string& nativeGates) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(createQCToQCO());
    pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
        .nativeGates = nativeGates,
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

  static std::string moduleToString(const OwningOpRef<ModuleOp>& moduleOp) {
    std::string text;
    llvm::raw_string_ostream os(text);
    moduleOp.get()->print(os);
    return text;
  }

  template <typename BuildFn, typename PredicateFn>
  void expectNativeAfterSynthesis(BuildFn buildFn,
                                  const std::string& nativeGates,
                                  PredicateFn isNative) {
    auto moduleOp = buildFn();
    runFuseTwoQubitUnitaryRunsPipeline(moduleOp, nativeGates);
    EXPECT_TRUE(isNative(moduleOp));
  }

  template <typename BuildFn>
  void expectSynthesisFailure(BuildFn buildFn, const std::string& nativeGates) {
    auto moduleOp = buildFn();
    PassManager pm(moduleOp->getContext());
    pm.addPass(createQCToQCO());
    pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
        .nativeGates = nativeGates,
    }));
    EXPECT_TRUE(failed(pm.run(*moduleOp)));
  }

  template <typename BuildFn, typename PredicateFn>
  void expectEquivalentAndNativeAfterSynthesis(BuildFn buildFn,
                                               const std::string& nativeGates,
                                               PredicateFn isNative) {
    auto expectedModule = buildFn();
    runQcToQco(expectedModule);

    auto synthesizedModule = buildFn();
    runFuseTwoQubitUnitaryRunsPipeline(synthesizedModule, nativeGates);
    EXPECT_TRUE(isNative(synthesizedModule));
    expectQcoModulesEquivalent(expectedModule, synthesizedModule);
  }

  template <typename ProgramT>
  static void expectTwoQFusePreservesUnitary(MLIRContext* ctx, ProgramT program,
                                             StringRef nativeGates) {
    auto build = [&](MLIRContext* context) {
      return mlir::qc::QCProgramBuilder::build(context, program);
    };
    auto expected = build(ctx);
    ASSERT_TRUE(expected);
    runQcToQco(expected);

    auto fused = build(ctx);
    ASSERT_TRUE(fused);
    runQcToQco(fused);
    runTwoQFuse(fused, nativeGates);
    ASSERT_TRUE(succeeded(verify(*fused)));
    expectTwoQubitQcoModulesEquivalent(expected, fused);
  }

  static std::size_t countCtrlOps(const OwningOpRef<ModuleOp>& moduleOp) {
    std::size_t count = 0;
    moduleOp.get()->walk([&](CtrlOp) { ++count; });
    return count;
  }

  std::unique_ptr<MLIRContext> context;
};

} // namespace mlir::qco::fuse_two_qubit_test

using namespace mlir::qco::fuse_two_qubit_test;

struct NativeSynthMenuRow {
  const char* name;
  const char* nativeGates;
  bool (*isNative)(OwningOpRef<ModuleOp>&);
};

// --- Inline circuit builders ---

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

// --- Pass profile coverage ---

// NOLINTNEXTLINE(misc-use-internal-linkage)
class FuseTwoQubitSwapProfileTest
    : public FuseTwoQubitUnitaryRunsPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using FuseTwoQubitUnitaryRunsPassTest::onlyGenericU3CxOps;
  using FuseTwoQubitUnitaryRunsPassTest::onlyIbmBasicCxOps;
  using FuseTwoQubitUnitaryRunsPassTest::onlyIqmDefaultOps;
};

TEST_P(FuseTwoQubitSwapProfileTest, DecomposesSwapToProfile) {
  const NativeSynthMenuRow& param = GetParam();
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::swap);
      },
      param.nativeGates, param.isNative);
}

INSTANTIATE_TEST_SUITE_P(
    SwapMenuMatrix, FuseTwoQubitSwapProfileTest,
    testing::Values(
        NativeSynthMenuRow{"IbmBasicCx", "x,sx,rz,cx",
                           &FuseTwoQubitSwapProfileTest::onlyIbmBasicCxOps},
        NativeSynthMenuRow{"GenericU3Cx", "u,cx",
                           &FuseTwoQubitSwapProfileTest::onlyGenericU3CxOps},
        NativeSynthMenuRow{"IqmDefault", "r,cz",
                           &FuseTwoQubitSwapProfileTest::onlyIqmDefaultOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesHstycxToIbmBasicCx) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hstycxTwoQ);
      },
      "x,sx,rz,cx", &FuseTwoQubitUnitaryRunsPassTest::onlyIbmBasicCxOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesCxYOnQ1ToIqmDefault) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), cxYOnQ1); },
      "r,cz", &FuseTwoQubitUnitaryRunsPassTest::onlyIqmDefaultOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, BroadOneQCanonicalizationOnIqmDefault) {
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), broadOneQThenCz);
  runFuseTwoQubitUnitaryRunsPipeline(moduleOp, "r,cz");
  EXPECT_TRUE(onlyIqmDefaultOps(moduleOp));
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, ZeroAngleCanonicalizationOnRyRzCz) {
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), zeroAngleThenCz);
  runFuseTwoQubitUnitaryRunsPipeline(moduleOp, "ry,rz,cz");
  EXPECT_TRUE(onlyAxisPairRyRzCzOps(moduleOp));
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesCxToCzForIbmBasicCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hCxTOnQ1);
      },
      "x,sx,rz,cz", &FuseTwoQubitUnitaryRunsPassTest::onlyIbmBasicCzOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesToIqmDefaultProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), xYSXCz); },
      "r,cz", &FuseTwoQubitUnitaryRunsPassTest::onlyIqmDefaultOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesToIbmFractionalProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 ibmFractionalGateFamilies);
      },
      "x,sx,rz,rx,rzz,cz",
      &FuseTwoQubitUnitaryRunsPassTest::onlyIbmFractionalOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesToAxisPairRxRzCxProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), hYCx); },
      "rx,rz,cx", &FuseTwoQubitUnitaryRunsPassTest::onlyAxisPairRxRzCxOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DecomposesRzToAxisPairRxRyCxProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), zCx); },
      "rx,ry,cx", &FuseTwoQubitUnitaryRunsPassTest::onlyAxisPairRxRyCxOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       GenericProfileMatchesGenericU3CxBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hq0Yq1CxSq0);
      },
      "u,cx", &FuseTwoQubitUnitaryRunsPassTest::onlyGenericU3CxOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       GenericProfileMatchesAxisPairRyRzCzBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), xHCz); },
      "ry,rz,cz", &FuseTwoQubitUnitaryRunsPassTest::onlyAxisPairRyRzCzOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       CustomProfileAcceptsMultipleEntanglersMenu) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), hCxSq1); },
      "u,cx,cz", &FuseTwoQubitUnitaryRunsPassTest::onlyGenericU3CxOrCzOps);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, FailsForUnsupportedNativeGateMenu) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::h);
      },
      "not-a-gate");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::singleControlledX);
      },
      "cx,cz");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, FailsForMultiControlledGateStructure) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::multipleControlledX);
      },
      "x,sx,rz,cx");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       CandidateSelectionIsDeterministicAcrossRuns) {
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(context.get(), determinismSwap);
  };
  auto firstModule = buildFn();
  runFuseTwoQubitUnitaryRunsPipeline(firstModule, "u,cx");
  auto secondModule = buildFn();
  runFuseTwoQubitUnitaryRunsPipeline(secondModule, "u,cx");
  EXPECT_EQ(moduleToString(firstModule), moduleToString(secondModule));
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, ThreeQubitGhzEquivalentOnCoreProfiles) {
  for (const auto& profileCase : coreEquivalenceProfiles()) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runQcToQco(expected);

    auto synthesized =
        mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runFuseTwoQubitUnitaryRunsPipeline(synthesized, profileCase.nativeGates);
    EXPECT_TRUE(profileCase.isNative(synthesized));
    expectQcoModulesEquivalent(expected, synthesized);
  }
}

// --- Two-qubit window fusion (pass internals) ---

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

TEST_F(FuseTwoQubitUnitaryRunsPassTest, AdjacentCxCancel) {
  expectTwoQFusePreservesUnitary(context.get(), fusionCxCx, "u,cx");

  auto module = mlir::qc::QCProgramBuilder::build(context.get(), fusionCxCx);
  ASSERT_TRUE(module);
  runQcToQco(module);
  runTwoQFuse(module, "u,cx");
  EXPECT_EQ(countCtrlOps(module), 0U);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, FusesCxThroughInterleavedOneQOps) {
  expectTwoQFusePreservesUnitary(context.get(), fusionHCxInterleavedTCx,
                                 "u,cx");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, StopsAtDifferentPairBoundary) {
  auto module =
      mlir::qc::QCProgramBuilder::build(context.get(), fusionThreeLineCx);
  ASSERT_TRUE(module);
  runQcToQco(module);
  runTwoQFuse(module, "u,cx");
  EXPECT_GE(countCtrlOps(module), 1U);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, DoesNotFuseAcrossBarrier) {
  auto module =
      mlir::qc::QCProgramBuilder::build(context.get(), fusionCxBarrierCx);
  ASSERT_TRUE(module);
  runQcToQco(module);
  runTwoQFuse(module, "u,cx");
  EXPECT_EQ(countCtrlOps(module), 2U);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, HandlesSwappedWireOrder) {
  expectTwoQFusePreservesUnitary(context.get(), fusionSwapCxPattern, "u,cx");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, HandlesRzzBlock) {
  expectTwoQFusePreservesUnitary(context.get(), fusionHRzzSRzz,
                                 "x,sx,rz,rx,rzz,cz");
}
