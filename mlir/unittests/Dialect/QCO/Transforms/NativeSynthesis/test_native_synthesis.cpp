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
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/QC/Builder/QCProgramBuilder.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
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
using namespace mlir::qco::decomposition;

namespace mlir::qco::native_synth_test {

using mqt::test::cxGate01;
using mqt::test::czGate;
using mqt::test::isEquivalentUpToGlobalPhase;

[[nodiscard]] static std::optional<mlir::Value>
getUnitaryQubitOperand(mlir::qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  mlir::Value v = op->getOperand(index);
  if (!llvm::isa<mlir::qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static std::optional<mlir::Value>
getUnitaryQubitResult(mlir::qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  mlir::Value v = op->getResult(index);
  if (!llvm::isa<mlir::qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static bool
extractSingleQubitMatrix(mlir::qco::UnitaryOpInterface op,
                         mlir::qco::Matrix2x2& out) {
  if (op.getUnitaryMatrix2x2(out)) {
    return true;
  }
  mlir::qco::DynamicMatrix dynamic;
  if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 2 ||
      dynamic.cols() != 2) {
    return false;
  }
  out = mlir::qco::Matrix2x2::fromElements(dynamic(0, 0), dynamic(0, 1),
                                           dynamic(1, 0), dynamic(1, 1));
  return true;
}

[[nodiscard]] static bool
extractTwoQubitMatrix(mlir::qco::UnitaryOpInterface op,
                      mlir::qco::Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<mlir::qco::CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<mlir::qco::XOp>(body)) {
      out = cxGate01();
      return true;
    }
    if (llvm::isa<mlir::qco::ZOp>(body)) {
      out = czGate();
      return true;
    }
    return false;
  }
  return op.getUnitaryMatrix4x4(out);
}

[[nodiscard]] static std::optional<mlir::qco::DynamicMatrix>
computeUnitaryFromModule(const mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
  mlir::ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }

  llvm::DenseMap<mlir::Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;
  std::size_t numQubits = 0;

  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto staticOp = llvm::dyn_cast<mlir::qco::StaticOp>(&rawOp)) {
          const auto index = static_cast<std::size_t>(staticOp.getIndex());
          qubitIds.try_emplace(staticOp.getQubit(), index);
          numQubits = std::max(numQubits, index + 1);
        } else if (auto alloc = llvm::dyn_cast<mlir::qco::AllocOp>(&rawOp)) {
          qubitIds.try_emplace(alloc.getResult(), nextQubitId++);
          numQubits = std::max(numQubits, nextQubitId);
        }
      }
    }
  }

  if (numQubits == 0) {
    return std::nullopt;
  }

  mlir::qco::DynamicMatrix unitary = mlir::qco::DynamicMatrix::identity(
      static_cast<std::int64_t>(1ULL << numQubits));

  auto getQubitId = [&](mlir::Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<mlir::qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<mlir::qco::BarrierOp, mlir::qco::GPhaseOp>(
                op.getOperation())) {
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
          mlir::qco::Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = mlir::qco::embedSingleQubitInNqubit(oneQ, numQubits, *qid) *
                    unitary;
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
          mlir::qco::Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary =
              mlir::qco::embedTwoQubitInNqubit(twoQ, numQubits, *q0id, *q1id) *
              unitary;
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

static void
expectQcoModulesEquivalent(const mlir::OwningOpRef<mlir::ModuleOp>& lhs,
                           const mlir::OwningOpRef<mlir::ModuleOp>& rhs) {
  const auto lhsUnitary = computeUnitaryFromModule(lhs);
  ASSERT_TRUE(lhsUnitary.has_value());
  const auto rhsUnitary = computeUnitaryFromModule(rhs);
  ASSERT_TRUE(rhsUnitary.has_value());
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(*lhsUnitary, *rhsUnitary));
}

/// One row of the standard multi-profile equivalence sweeps in tests.
// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_F` at global scope
struct NativeSynthesisProfileSweepCase {
  const char* nativeGates;
  bool (*isNative)(mlir::OwningOpRef<mlir::ModuleOp>&);
};

/// Shared gtest fixture for native-gate synthesis pass tests.
// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_F` at global scope
class NativeSynthesisPassTest : public testing::Test {
protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  template <typename... Allowed1QOps>
  static bool onlyTheseOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp,
                           const bool allowCx, const bool allowCz) {
    bool ok = true;
    std::ignore = moduleOp->walk([&](mlir::qco::UnitaryOpInterface op) {
      mlir::Operation* raw = op.getOperation();
      if (llvm::isa_and_present<mlir::qco::CtrlOp>(raw->getParentOp())) {
        return mlir::WalkResult::advance();
      }
      if (llvm::isa<mlir::qco::BarrierOp, mlir::qco::GPhaseOp>(raw)) {
        return mlir::WalkResult::advance();
      }
      if (auto ctrl = llvm::dyn_cast<mlir::qco::CtrlOp>(raw)) {
        if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
          ok = false;
          return mlir::WalkResult::interrupt();
        }
        mlir::Operation* body = ctrl.getBodyUnitary(0).getOperation();
        const bool isCx = llvm::isa<mlir::qco::XOp>(body);
        const bool isCz = llvm::isa<mlir::qco::ZOp>(body);
        if ((isCx && allowCx) || (isCz && allowCz)) {
          return mlir::WalkResult::advance();
        }
        ok = false;
        return mlir::WalkResult::interrupt();
      }

      if (!llvm::isa<Allowed1QOps...>(raw)) {
        ok = false;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return ok;
  }

  static bool onlyIbmBasicCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::XOp, mlir::qco::SXOp, mlir::qco::RZOp,
                        mlir::qco::POp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/false);
  }

  static bool onlyIbmBasicCzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::XOp, mlir::qco::SXOp, mlir::qco::RZOp,
                        mlir::qco::POp>(moduleOp, /*allowCx=*/false,
                                        /*allowCz=*/true);
  }

  static bool onlyGenericU3CxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/false);
  }

  static bool onlyGenericU3CzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp>(moduleOp, /*allowCx=*/false,
                                        /*allowCz=*/true);
  }

  static bool onlyIqmDefaultOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::ROp>(moduleOp, /*allowCx=*/false,
                                        /*allowCz=*/true);
  }

  static bool
  onlyIbmFractionalOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::XOp, mlir::qco::SXOp, mlir::qco::RZOp,
                        mlir::qco::POp, mlir::qco::RXOp, mlir::qco::RZZOp>(
        moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool
  onlyAxisPairRxRzCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::RXOp, mlir::qco::RZOp, mlir::qco::POp>(
        moduleOp, /*allowCx=*/true, /*allowCz=*/false);
  }

  static bool
  onlyAxisPairRxRyCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::RXOp, mlir::qco::RYOp>(
        moduleOp, /*allowCx=*/true, /*allowCz=*/false);
  }

  static bool
  onlyAxisPairRyRzCzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::RYOp, mlir::qco::RZOp, mlir::qco::POp>(
        moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool
  onlyUOrAxisPairRxRzCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp, mlir::qco::RXOp, mlir::qco::RZOp,
                        mlir::qco::POp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/false);
  }

  static bool
  onlyGenericU3CxOrCzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/true);
  }

  static std::array<NativeSynthesisProfileSweepCase, 3>
  coreEquivalenceProfiles() {
    return {{{.nativeGates = "x,sx,rz,cx", .isNative = &onlyIbmBasicCxOps},
             {.nativeGates = "u,cx", .isNative = &onlyGenericU3CxOps},
             {.nativeGates = "r,cz", .isNative = &onlyIqmDefaultOps}}};
  }

  static void runNativeSynthesis(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp,
                                 const std::string& nativeGates) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    pm.addPass(mlir::qco::createFuseTwoQubitUnitaryRuns(
        mlir::qco::FuseTwoQubitUnitaryRunsOptions{
            .nativeGates = nativeGates,
        }));
    ASSERT_TRUE(mlir::succeeded(pm.run(*moduleOp)));
  }

  static void runQcToQco(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    ASSERT_TRUE(mlir::succeeded(pm.run(*moduleOp)));
  }

  static std::string
  moduleToString(const mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
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
    runNativeSynthesis(moduleOp, nativeGates);
    EXPECT_TRUE(isNative(moduleOp));
  }

  template <typename BuildFn>
  void expectSynthesisFailure(BuildFn buildFn, const std::string& nativeGates) {
    auto moduleOp = buildFn();
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    pm.addPass(mlir::qco::createFuseTwoQubitUnitaryRuns(
        mlir::qco::FuseTwoQubitUnitaryRunsOptions{
            .nativeGates = nativeGates,
        }));
    EXPECT_TRUE(mlir::failed(pm.run(*moduleOp)));
  }

  template <typename BuildFn, typename PredicateFn>
  void expectEquivalentAndNativeAfterSynthesis(BuildFn buildFn,
                                               const std::string& nativeGates,
                                               PredicateFn isNative) {
    auto expectedModule = buildFn();
    runQcToQco(expectedModule);

    auto synthesizedModule = buildFn();
    runNativeSynthesis(synthesizedModule, nativeGates);
    EXPECT_TRUE(isNative(synthesizedModule));
    expectQcoModulesEquivalent(expectedModule, synthesizedModule);
  }

  std::unique_ptr<mlir::MLIRContext> context;
};

} // namespace mlir::qco::native_synth_test

using namespace mlir::qco::native_synth_test;

struct NativeSynthMenuRow {
  const char* name;
  const char* nativeGates;
  bool (*isNative)(OwningOpRef<ModuleOp>&);
};

// --- Inline circuit builders ---

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

// --- NativeSpec / NativePolicy ---

TEST(NativeSpecTest, ResolveIbmBasicCx) {
  const auto spec = decomposition::parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->allowedGates.contains(decomposition::NativeGateKind::Cx));
  EXPECT_TRUE(spec->allowedGates.contains(decomposition::NativeGateKind::X));
  EXPECT_FALSE(spec->allowRzz);
}

TEST(NativeSpecTest, ResolveRejectsUnknownToken) {
  EXPECT_FALSE(
      decomposition::parseNativeSpec("x,sx,rz,not-a-gate").has_value());
}

TEST(NativeSpecTest, PhaseAliasPMatchesRzInIbmStyleMenu) {
  const auto pMenu = decomposition::parseNativeSpec("x,sx,p,cx");
  const auto rzMenu = decomposition::parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(pMenu);
  ASSERT_TRUE(rzMenu);
  EXPECT_EQ(pMenu->allowedGates, rzMenu->allowedGates);
}

TEST(NativePolicyTest, UsesCxAndCzFromResolvedSpec) {
  const auto cxOnly = decomposition::parseNativeSpec("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(llvm::is_contained(cxOnly->entanglerBases,
                                 decomposition::EntanglerBasis::Cx));
  EXPECT_FALSE(llvm::is_contained(cxOnly->entanglerBases,
                                  decomposition::EntanglerBasis::Cz));

  const auto both = decomposition::parseNativeSpec("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases,
                                 decomposition::EntanglerBasis::Cx));
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases,
                                 decomposition::EntanglerBasis::Cz));
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
class NativePolicyAllowsOpTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder{&context};

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    builder.initialize();
  }
};

TEST_F(NativePolicyAllowsOpTest, RejectsSingleQubitOpNotInMenu) {
  const auto spec = decomposition::parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  Value q = builder.staticQubit(0);
  q = builder.x(q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  XOp xop;
  mod->walk([&](XOp op) {
    xop = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(xop);
}

// --- Pass profile coverage ---

// NOLINTNEXTLINE(misc-use-internal-linkage)
class NativeSynthesisSwapProfileTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyGenericU3CxOps;
  using NativeSynthesisPassTest::onlyIbmBasicCxOps;
  using NativeSynthesisPassTest::onlyIqmDefaultOps;
};

TEST_P(NativeSynthesisSwapProfileTest, DecomposesSwapToProfile) {
  const NativeSynthMenuRow& param = GetParam();
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::swap);
      },
      param.nativeGates, param.isNative);
}

INSTANTIATE_TEST_SUITE_P(
    SwapMenuMatrix, NativeSynthesisSwapProfileTest,
    testing::Values(
        NativeSynthMenuRow{"IbmBasicCx", "x,sx,rz,cx",
                           &NativeSynthesisSwapProfileTest::onlyIbmBasicCxOps},
        NativeSynthMenuRow{"GenericU3Cx", "u,cx",
                           &NativeSynthesisSwapProfileTest::onlyGenericU3CxOps},
        NativeSynthMenuRow{"IqmDefault", "r,cz",
                           &NativeSynthesisSwapProfileTest::onlyIqmDefaultOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

TEST_F(NativeSynthesisPassTest, DecomposesHstycxToIbmBasicCx) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hstycxTwoQ);
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesCxYOnQ1ToIqmDefault) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), cxYOnQ1); },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, BroadOneQCanonicalizationOnIqmDefault) {
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), broadOneQThenCz);
  runNativeSynthesis(moduleOp, "r,cz");
  EXPECT_TRUE(onlyIqmDefaultOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, ZeroAngleCanonicalizationOnRyRzCz) {
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), zeroAngleThenCz);
  runNativeSynthesis(moduleOp, "ry,rz,cz");
  EXPECT_TRUE(onlyAxisPairRyRzCzOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, DecomposesCxToCzForIbmBasicCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hCxTOnQ1);
      },
      "x,sx,rz,cz", &NativeSynthesisPassTest::onlyIbmBasicCzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIqmDefaultProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), xYSXCz); },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIbmFractionalProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 ibmFractionalGateFamilies);
      },
      "x,sx,rz,rx,rzz,cz", &NativeSynthesisPassTest::onlyIbmFractionalOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToAxisPairRxRzCxProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), hYCx); },
      "rx,rz,cx", &NativeSynthesisPassTest::onlyAxisPairRxRzCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesRzToAxisPairRxRyCxProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), zCx); },
      "rx,ry,cx", &NativeSynthesisPassTest::onlyAxisPairRxRyCxOps);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesGenericU3CxBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hq0Yq1CxSq0);
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesAxisPairRyRzCzBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), xHCz); },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps);
}

TEST_F(NativeSynthesisPassTest, CustomProfileAcceptsMultipleEntanglersMenu) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), hCxSq1); },
      "u,cx,cz", &NativeSynthesisPassTest::onlyGenericU3CxOrCzOps);
}

TEST_F(NativeSynthesisPassTest, FailsForUnsupportedNativeGateMenu) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::h);
      },
      "not-a-gate");
}

TEST_F(NativeSynthesisPassTest, FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::singleControlledX);
      },
      "cx,cz");
}

TEST_F(NativeSynthesisPassTest, FailsForMultiControlledGateStructure) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::multipleControlledX);
      },
      "x,sx,rz,cx");
}

TEST_F(NativeSynthesisPassTest, CandidateSelectionIsDeterministicAcrossRuns) {
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(context.get(), determinismSwap);
  };
  auto firstModule = buildFn();
  runNativeSynthesis(firstModule, "u,cx");
  auto secondModule = buildFn();
  runNativeSynthesis(secondModule, "u,cx");
  EXPECT_EQ(moduleToString(firstModule), moduleToString(secondModule));
}

TEST_F(NativeSynthesisPassTest, ThreeQubitGhzEquivalentOnCoreProfiles) {
  for (const auto& profileCase : coreEquivalenceProfiles()) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runQcToQco(expected);

    auto synthesized =
        mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runNativeSynthesis(synthesized, profileCase.nativeGates);
    EXPECT_TRUE(profileCase.isNative(synthesized));
    expectQcoModulesEquivalent(expected, synthesized);
  }
}
