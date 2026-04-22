/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "native_synthesis_test_helpers.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/WalkResult.h>

#include <array>
#include <memory>
#include <string>
#include <tuple>

/// One row of the standard multi-profile equivalence sweeps in tests.
struct NativeSynthesisProfileSweepCase {
  const char* nativeGates;
  bool (*isNative)(mlir::OwningOpRef<mlir::ModuleOp>&);
};

class NativeSynthesisPassTest : public testing::Test {
protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
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
      if (mlir::isa_and_present<mlir::qco::CtrlOp>(raw->getParentOp())) {
        return mlir::WalkResult::advance();
      }
      if (mlir::isa<mlir::qco::BarrierOp, mlir::qco::GPhaseOp>(raw)) {
        return mlir::WalkResult::advance();
      }
      if (auto ctrl = mlir::dyn_cast<mlir::qco::CtrlOp>(raw)) {
        if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
          ok = false;
          return mlir::WalkResult::interrupt();
        }
        mlir::Operation* body = ctrl.getBodyUnitary().getOperation();
        const bool isCx = mlir::isa<mlir::qco::XOp>(body);
        const bool isCz = mlir::isa<mlir::qco::ZOp>(body);
        if ((isCx && allowCx) || (isCz && allowCz)) {
          return mlir::WalkResult::advance();
        }
        ok = false;
        return mlir::WalkResult::interrupt();
      }

      if (!mlir::isa<Allowed1QOps...>(raw)) {
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

  /// The nine built-in reference profiles (IBM basic, U3, fractional, IQM,
  /// axis pairs including ``rx,rz,cx``). Used by 2q / multi-qubit equivalence
  /// sweeps.
  static std::array<NativeSynthesisProfileSweepCase, 9>
  allNineEquivalenceProfiles() {
    return {{{.nativeGates = "x,sx,rz,cx", .isNative = &onlyIbmBasicCxOps},
             {.nativeGates = "x,sx,rz,cz", .isNative = &onlyIbmBasicCzOps},
             {.nativeGates = "u,cx", .isNative = &onlyGenericU3CxOps},
             {.nativeGates = "u,cz", .isNative = &onlyGenericU3CzOps},
             {.nativeGates = "x,sx,rz,rx,rzz,cz",
              .isNative = &onlyIbmFractionalOps},
             {.nativeGates = "r,cz", .isNative = &onlyIqmDefaultOps},
             {.nativeGates = "rx,ry,cx", .isNative = &onlyAxisPairRxRyCxOps},
             {.nativeGates = "ry,rz,cz", .isNative = &onlyAxisPairRyRzCzOps},
             {.nativeGates = "rx,rz,cx", .isNative = &onlyAxisPairRxRzCxOps}}};
  }

  /// CX-friendly profiles excluding IQM-default (CZ-only entangler), for
  /// circuits that use a ``cx`` two-qubit primitive in the source.
  static std::array<NativeSynthesisProfileSweepCase, 5>
  fiveCxEntanglerEquivalenceProfiles() {
    return {{{.nativeGates = "x,sx,rz,cx", .isNative = &onlyIbmBasicCxOps},
             {.nativeGates = "u,cx", .isNative = &onlyGenericU3CxOps},
             {.nativeGates = "x,sx,rz,rx,rzz,cz",
              .isNative = &onlyIbmFractionalOps},
             {.nativeGates = "rx,ry,cx", .isNative = &onlyAxisPairRxRyCxOps},
             {.nativeGates = "rx,rz,cx", .isNative = &onlyAxisPairRxRzCxOps}}};
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  buildBroadOneQCanonicalizationCircuit() const {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();

    builder.id(q0);
    builder.x(q0);
    builder.y(q1);
    builder.z(q0);
    builder.h(q1);
    builder.s(q0);
    builder.sdg(q1);
    builder.t(q0);
    builder.tdg(q1);
    builder.sx(q0);
    builder.sxdg(q1);
    builder.rx(0.13, q0);
    builder.ry(-0.47, q1);
    builder.rz(0.29, q0);
    builder.p(-0.38, q1);
    builder.r(0.61, -0.22, q0);
    builder.cz(q0, q1);

    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  buildZeroAngleCanonicalizationCircuit() const {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();

    builder.rx(0.0, q0);
    builder.ry(0.0, q1);
    builder.rz(0.0, q0);
    builder.p(0.0, q1);
    builder.r(0.0, 0.0, q0);
    builder.cz(q0, q1);

    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  }

  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
  buildIbmFractionalAllGateFamiliesCircuit() const {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();

    builder.id(q0);
    builder.x(q0);
    builder.y(q1);
    builder.z(q0);
    builder.h(q1);
    builder.s(q0);
    builder.sdg(q1);
    builder.t(q0);
    builder.tdg(q1);
    builder.sx(q0);
    builder.sxdg(q1);
    builder.rx(0.13, q0);
    builder.ry(-0.47, q1);
    builder.rz(0.29, q0);
    builder.p(-0.38, q1);
    builder.r(0.61, -0.22, q0);

    builder.cx(q0, q1);
    builder.cz(q1, q0);

    builder.swap(q0, q1);
    builder.iswap(q0, q1);
    builder.dcx(q0, q1);
    builder.ecr(q0, q1);
    builder.rxx(0.17, q0, q1);
    builder.ryy(-0.21, q0, q1);
    builder.rzx(0.41, q0, q1);
    builder.rzz(-0.33, q0, q1);
    builder.xx_plus_yy(0.52, -0.14, q0, q1);
    builder.xx_minus_yy(-0.37, 0.26, q0, q1);

    builder.dealloc(q0);
    builder.dealloc(q1);
    return builder.finalize();
  }

  static void runNativeSynthesis(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp,
                                 const std::string& nativeGates,
                                 const double scoreWeightTwoQ = 1.0,
                                 const double scoreWeightOneQ = 0.1,
                                 const double scoreWeightDepth = 0.01) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    pm.addPass(mlir::qco::createNativeGateSynthesisPass(
        mlir::qco::NativeGateSynthesisOptions{
            .nativeGates = nativeGates,
            .scoreWeightTwoQ = scoreWeightTwoQ,
            .scoreWeightOneQ = scoreWeightOneQ,
            .scoreWeightDepth = scoreWeightDepth,
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
                                  PredicateFn isNative,
                                  const double scoreWeightTwoQ = 1.0,
                                  const double scoreWeightOneQ = 0.1,
                                  const double scoreWeightDepth = 0.01) {
    auto moduleOp = buildFn();
    runNativeSynthesis(moduleOp, nativeGates, scoreWeightTwoQ, scoreWeightOneQ,
                       scoreWeightDepth);
    EXPECT_TRUE(isNative(moduleOp));
  }

  template <typename BuildFn>
  void expectSynthesisFailure(BuildFn buildFn, const std::string& nativeGates,
                              const double scoreWeightTwoQ = 1.0,
                              const double scoreWeightOneQ = 0.1,
                              const double scoreWeightDepth = 0.01) {
    auto moduleOp = buildFn();
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    pm.addPass(mlir::qco::createNativeGateSynthesisPass(
        mlir::qco::NativeGateSynthesisOptions{
            .nativeGates = nativeGates,
            .scoreWeightTwoQ = scoreWeightTwoQ,
            .scoreWeightOneQ = scoreWeightOneQ,
            .scoreWeightDepth = scoreWeightDepth,
        }));
    EXPECT_TRUE(mlir::failed(pm.run(*moduleOp)));
  }

  template <typename BuildFn, typename PredicateFn, typename UnitaryFn>
  void expectEquivalentAndNativeAfterSynthesis(
      BuildFn buildFn, const std::string& nativeGates, PredicateFn isNative,
      UnitaryFn computeUnitary, const double scoreWeightTwoQ = 1.0,
      const double scoreWeightOneQ = 0.1,
      const double scoreWeightDepth = 0.01) {
    auto expectedModule = buildFn();
    runQcToQco(expectedModule);
    const auto expectedUnitary = computeUnitary(expectedModule);
    ASSERT_TRUE(expectedUnitary.has_value());

    auto synthesizedModule = buildFn();
    runNativeSynthesis(synthesizedModule, nativeGates, scoreWeightTwoQ,
                       scoreWeightOneQ, scoreWeightDepth);
    EXPECT_TRUE(isNative(synthesizedModule));
    const auto synthesizedUnitary = computeUnitary(synthesizedModule);
    ASSERT_TRUE(synthesizedUnitary.has_value());
    EXPECT_TRUE(mlir::qco::native_synth_test::isEquivalentUpToGlobalPhase(
        *expectedUnitary, *synthesizedUnitary));
  }

  std::unique_ptr<mlir::MLIRContext> context;
};
