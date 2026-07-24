/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
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

using ProgramFn = Value (*)(mlir::qc::QCProgramBuilder&);

// --- Native-gateset membership check ------------------------------------- //

/// Returns true when every single- or two-qubit operation in @p moduleOp is
/// native to the gateset parsed from @p nativeGates. Operations nested inside a
/// controlled shell are validated through the shell itself, and gates acting on
/// more than two qubits are out of scope for this pass and thus ignored.
static bool allOpsNative(OwningOpRef<ModuleOp>& moduleOp,
                         StringRef nativeGates) {
  const auto spec = decomposition::NativeGateset::parse(nativeGates);
  if (!spec) {
    return false;
  }
  bool ok = true;
  std::ignore = moduleOp->walk([&](UnitaryOpInterface op) {
    Operation* raw = op.getOperation();
    if (isa_and_present<CtrlOp>(raw->getParentOp()) || op.getNumQubits() > 2) {
      return WalkResult::advance();
    }
    if (!spec->allowsOp(raw)) {
      ok = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ok;
}

// --- DD-based equivalence ------------------------------------------------ //

[[nodiscard]] static func::FuncOp mainFunc(ModuleOp module) {
  return *module.getBody()->getOps<func::FuncOp>().begin();
}

[[nodiscard]] static size_t countStaticQubits(func::FuncOp func) {
  size_t numQubits = 0;
  for (StaticOp staticOp : func.getOps<StaticOp>()) {
    numQubits =
        std::max(numQubits, static_cast<size_t>(staticOp.getIndex()) + 1);
  }
  return numQubits;
}

[[nodiscard]] static DynamicMatrix matrixFromDD(const dd::CMat& matrix) {
  const auto dim = static_cast<int64_t>(matrix.size());
  DynamicMatrix out(dim);
  for (int64_t row = 0; row < dim; ++row) {
    for (int64_t col = 0; col < dim; ++col) {
      out(row, col) =
          matrix[static_cast<size_t>(row)][static_cast<size_t>(col)];
    }
  }
  return out;
}

// --- Expressive circuits -------------------------------------------------- //
//
// A handful of circuits, which are crossed with the gateset table below.

/// A bare SWAP (three-entangler class), the canonical two-qubit decomposition.
static Value swapTwoQ(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.swap(q0, q1);
  return b.intConstant(0);
}

/// Rich single-qubit variety on both wires, followed by a two-qubit entangler.
static Value broadOneQThenCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.x(q0);
  b.y(q1);
  b.h(q0);
  b.sx(q1);
  b.rx(0.13, q0);
  b.ry(-0.47, q1);
  b.rz(0.29, q0);
  b.cz(q0, q1);
  return b.intConstant(0);
}

/// Long single-qubit run on one wire, then an entangler (Euler-run fusion).
static Value hstycxTwoQ(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.h(q0);
  b.s(q0);
  b.t(q0);
  b.y(q0);
  b.cx(q0, q1);
  return b.intConstant(0);
}

/// Zero-angle rotations that must canonicalize away before the entangler.
static Value zeroAngleThenCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.rx(0.0, q0);
  b.ry(0.0, q1);
  b.rz(0.0, q0);
  b.p(0.0, q1);
  b.cz(q0, q1);
  return b.intConstant(0);
}

/// Single-qubit gates surrounding an entangler on both sides.
static Value hCxSq1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.h(q0);
  b.cx(q0, q1);
  b.s(q1);
  return b.intConstant(0);
}

/// Three-qubit program with chained entanglers on overlapping pairs.
static Value threeQGhz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  const auto q2 = b.staticQubit(2);
  b.h(q0);
  b.cx(q0, q1);
  b.cx(q1, q2);
  return b.intConstant(0);
}

/// Single-qubit gates wrapped in an inverse modifier (no entangler).
static Value inverseTwoX(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  b.inv(q0, [&](Value qubit) {
    b.x(qubit);
    b.x(qubit);
  });
  return b.intConstant(0);
}

/// A controlled two-gate body that must be synthesized as a two-qubit unitary.
static Value controlledXH(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.ctrl(q0, q1, [&](Value target) {
    b.x(target);
    b.h(target);
  });
  return b.intConstant(0);
}

// --- Fusion-window circuits ---------------------------------------------- //
//
// These probe window geometry (where fusion starts/stops), so they run on a
// single fixed gateset rather than the full table.

static Value fusionCxCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  b.cx(q0, q1);
  return b.intConstant(0);
}

static Value fusionHCxInterleavedTCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.h(q0);
  b.cx(q0, q1);
  b.t(q1);
  b.s(q0);
  b.cx(q0, q1);
  return b.intConstant(0);
}

static Value fusionThreeLineCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  const auto q2 = b.staticQubit(2);
  b.cx(q0, q1);
  b.cx(q1, q2);
  b.cx(q0, q1);
  return b.intConstant(0);
}

static Value fusionCxRSharedOtherPair(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  const auto q2 = b.staticQubit(2);
  b.cx(q0, q1);
  b.rz(0.17, q1);
  b.cx(q1, q2);
  return b.intConstant(0);
}

static Value fusionCxBarrierCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  b.barrier({q0, q1});
  b.cx(q0, q1);
  return b.intConstant(0);
}

/// A single-wire barrier between two entanglers: a non-walkable single-qubit
/// shell terminates the run scan on wire A (and breaks the run-start chain of
/// the second entangler), so neither entangler fuses.
static Value fusionCxSingleWireBarrierCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  b.barrier({q0});
  b.cx(q0, q1);
  return b.intConstant(0);
}

/// An entangler followed by a three-qubit gate sharing both run wires: the run
/// scan must stop at the wider gate (it is neither a single- nor a two-qubit
/// run member) and leave it untouched, since gates on more than two qubits are
/// out of scope for this pass rather than a failure.
static Value fusionCxThenMultiControlledX(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  const auto q2 = b.staticQubit(2);
  b.cx(q0, q1);
  b.mcx({q0, q1}, q2);
  return b.intConstant(0);
}

static Value fusionSwapCxPattern(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  b.cx(q1, q0);
  b.cx(q0, q1);
  return b.intConstant(0);
}

static Value fusionOffMenuGateInWindow(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  b.h(q0);
  b.cx(q0, q1);
  return b.intConstant(0);
}

static Value fusionDualWireOneQBetweenCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  b.rz(0.11, q0);
  b.ry(0.22, q1);
  b.cx(q0, q1);
  return b.intConstant(0);
}

static Value determinismSwap(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.swap(q0, q1);
  b.dealloc(q0);
  b.dealloc(q1);
  return b.intConstant(0);
}

namespace {

/// A named circuit builder, used as a test parameter.
struct NamedProgram {
  const char* name;
  ProgramFn program;
};

/// Native gatesets spanning every supported single-qubit basis and both
/// entangler families (plus a multi-entangler menu). Because the pass
/// re-synthesizes each two-qubit window into the target basis, every circuit is
/// valid input for every gateset.
constexpr std::array<const char*, 9> GATESETS = {
    // CX entangler family
    "x,sx,rz,cx", // ZSXX
    "u,cx",       // U
    "rx,rz,cx",   // XZX
    "rx,ry,cx",   // XYX
    // CZ entangler family
    "r,cz",       // R
    "ry,rz,cz",   // ZYZ
    "x,sx,rz,cz", // ZSXX
    "u,cz",       // U
    // Multiple entanglers (cx preferred)
    "u,cx,cz",
};

/// Gateset used for the fusion-window suite, which asserts on structure rather
/// than on native-basis coverage.
constexpr StringRef FUSION_GATESET = "u,cx";

/// Structural expectations for a fusion-window circuit under @ref
/// FUSION_GATESET.
struct FusionCase {
  const char* name;
  ProgramFn program;
  std::optional<size_t> exactCtrlCount;
  std::optional<size_t> minCtrlCount;
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
    const auto lhsFunc = mainFunc(*lhs);
    const auto rhsFunc = mainFunc(*rhs);
    const auto numQubits = countStaticQubits(lhsFunc);
    ASSERT_EQ(numQubits, countStaticQubits(rhsFunc));
    ASSERT_GT(numQubits, 0U);

    auto dd = std::make_unique<dd::Package>(numQubits);
    const auto lhsUnitary = buildFunctionality(lhsFunc, *dd);
    ASSERT_TRUE(succeeded(lhsUnitary));
    const auto rhsUnitary = buildFunctionality(rhsFunc, *dd);
    ASSERT_TRUE(succeeded(rhsUnitary));

    const auto lhsMatrix = matrixFromDD(lhsUnitary->getMatrix(numQubits));
    const auto rhsMatrix = matrixFromDD(rhsUnitary->getMatrix(numQubits));
    dd->decRef(*lhsUnitary);
    dd->decRef(*rhsUnitary);
    EXPECT_TRUE(lhsMatrix.isApprox(rhsMatrix));
  }

  void expectEquivalentAndNativeAfterSynthesis(ProgramFn program,
                                               StringRef nativeGates) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), program);
    runQcToQco(expected);
    auto synthesized =
        mlir::qc::QCProgramBuilder::build(context.get(), program);
    runFusePipeline(synthesized, nativeGates);
    EXPECT_TRUE(allOpsNative(synthesized, nativeGates));
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

  void expectSynthesisFailure(
      SmallVector<Value> (*program)(mlir::qc::QCProgramBuilder&),
      StringRef nativeGates) {
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

  static size_t countCtrlOps(const OwningOpRef<ModuleOp>& moduleOp) {
    size_t count = 0;
    moduleOp.get()->walk([&](CtrlOp) { ++count; });
    return count;
  }

  /// Counts unitaries acting on more than two qubits, i.e. gates left untouched
  /// for the dedicated multi-controlled synthesis pass.
  static size_t countWideGates(const OwningOpRef<ModuleOp>& moduleOp) {
    size_t count = 0;
    moduleOp.get()->walk([&](UnitaryOpInterface op) {
      if (op.getNumQubits() > 2) {
        ++count;
      }
    });
    return count;
  }

  std::unique_ptr<MLIRContext> context;
};

using SynthesisParam = std::tuple<NamedProgram, const char*>;

class FuseTwoQubitSynthesisTest
    : public FuseTwoQubitUnitaryRunsPassTest,
      public testing::WithParamInterface<SynthesisParam> {};

class FuseTwoQubitFusionTest : public FuseTwoQubitUnitaryRunsPassTest,
                               public testing::WithParamInterface<FusionCase> {
};

} // namespace

// --- Synthesis: every expressive circuit against every gateset ----------- //

TEST_P(FuseTwoQubitSynthesisTest, IsNativeAndEquivalent) {
  const auto& [circuit, gateset] = GetParam();
  expectEquivalentAndNativeAfterSynthesis(circuit.program, gateset);
}

INSTANTIATE_TEST_SUITE_P(
    Circuits, FuseTwoQubitSynthesisTest,
    testing::Combine(
        testing::Values(NamedProgram{"Swap", swapTwoQ},
                        NamedProgram{"BroadOneQThenCz", broadOneQThenCz},
                        NamedProgram{"HstyThenCx", hstycxTwoQ},
                        NamedProgram{"ZeroAngleThenCz", zeroAngleThenCz},
                        NamedProgram{"SurroundedCx", hCxSq1},
                        NamedProgram{"ThreeQubitGhz", threeQGhz},
                        NamedProgram{"InverseBody", inverseTwoX},
                        NamedProgram{"ControlledBody", controlledXH},
                        NamedProgram{"SingleWireBarrier",
                                     fusionCxSingleWireBarrierCx}),
        testing::ValuesIn(GATESETS)),
    [](const testing::TestParamInfo<SynthesisParam>& info) {
      std::string gateset = std::get<1>(info.param);
      std::ranges::replace(gateset, ',', '_');
      return std::string(std::get<0>(info.param).name) + "__" + gateset;
    });

// --- Fusion windows: structural behavior on a fixed gateset -------------- //

TEST_P(FuseTwoQubitFusionTest, WindowFusionBehavior) {
  const FusionCase& c = GetParam();
  if (c.checkTwoQUnitary) {
    expectTwoQFusePreservesUnitary(c.program, FUSION_GATESET);
  }
  auto module = mlir::qc::QCProgramBuilder::build(context.get(), c.program);
  ASSERT_TRUE(module);
  runQcToQco(module);
  runTwoQFuse(module, FUSION_GATESET);
  if (c.exactCtrlCount) {
    EXPECT_EQ(countCtrlOps(module), *c.exactCtrlCount);
  }
  if (c.minCtrlCount) {
    EXPECT_GE(countCtrlOps(module), *c.minCtrlCount);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Windows, FuseTwoQubitFusionTest,
    testing::Values(
        FusionCase{"AdjacentCxCancel", fusionCxCx, 0, std::nullopt, true},
        FusionCase{"InterleavedOneQ", fusionHCxInterleavedTCx, std::nullopt,
                   std::nullopt, true},
        FusionCase{"DifferentPairBoundary", fusionThreeLineCx, std::nullopt, 1,
                   false},
        FusionCase{"SharedWireOneQ", fusionCxRSharedOtherPair, std::nullopt, 2,
                   false},
        FusionCase{"BarrierBoundary", fusionCxBarrierCx, 2, std::nullopt,
                   false},
        FusionCase{"SingleWireBarrierBoundary", fusionCxSingleWireBarrierCx, 2,
                   std::nullopt, true},
        FusionCase{"SwappedWireOrder", fusionSwapCxPattern, std::nullopt,
                   std::nullopt, true},
        FusionCase{"OffMenuGateInWindow", fusionOffMenuGateInWindow,
                   std::nullopt, std::nullopt, true},
        FusionCase{"DualWireOneQBetweenCx", fusionDualWireOneQBetweenCx,
                   std::nullopt, std::nullopt, true}),
    [](const testing::TestParamInfo<FusionCase>& info) {
      return info.param.name;
    });

// --- Pass edge cases ----------------------------------------------------- //

TEST_F(FuseTwoQubitUnitaryRunsPassTest, EmptyNativeGatesSkipsPass) {
  auto module = mlir::qc::QCProgramBuilder::build(context.get(), fusionCxCx);
  ASSERT_TRUE(module);
  runQcToQco(module);
  std::string before;
  llvm::raw_string_ostream osBefore(before);
  module->print(osBefore);

  PassManager pm(module->getContext());
  pm.addPass(createFuseTwoQubitUnitaryRuns(FuseTwoQubitUnitaryRunsOptions{
      .nativeGates = "",
  }));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  std::string after;
  llvm::raw_string_ostream osAfter(after);
  module->print(osAfter);
  EXPECT_EQ(before, after);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, FailsForInvalidNativeGateMenu) {
  expectSynthesisFailure(mlir::qc::h, "not-a-gate");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(mlir::qc::singleControlledX, "cx,cz");
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest, LeavesMultiControlledGateUntouched) {
  // A multi-controlled gate is out of scope for this pass; it is left untouched
  // (for a dedicated multi-controlled synthesis pass) and does not fail the
  // run.
  auto module = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::multipleControlledX);
  ASSERT_TRUE(module);
  runFusePipeline(module, "x,sx,rz,cx");
  EXPECT_TRUE(allOpsNative(module, "x,sx,rz,cx"));
  EXPECT_EQ(countWideGates(module), 1U);
}

TEST_F(FuseTwoQubitUnitaryRunsPassTest,
       LowersTwoQubitRunButLeavesWiderGateBoundary) {
  // The `cx` is off-menu for this cz-family gateset, so it is Weyl-synthesized
  // even though the run scan stops at the three-qubit boundary; the wider gate
  // is left untouched, so the pass succeeds with the two-qubit run lowered.
  auto module = mlir::qc::QCProgramBuilder::build(context.get(),
                                                  fusionCxThenMultiControlledX);
  ASSERT_TRUE(module);
  runFusePipeline(module, "u,cz");
  EXPECT_TRUE(allOpsNative(module, "u,cz"));
  EXPECT_EQ(countWideGates(module), 1U);
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
