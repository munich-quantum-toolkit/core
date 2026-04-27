/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// Multi-qubit equivalence sweeps (3q circuit families, 5q stress) for the
// native-gate synthesis pass.

#include "native_synthesis_pass_test_fixture.h"
#include "native_synthesis_test_helpers.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/QC/Builder/QCProgramBuilder.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <array>
#include <string>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

static OwningOpRef<ModuleOp> buildThreeQGhzCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQGhz);
}

static OwningOpRef<ModuleOp> buildThreeQToffoliCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQToffoli);
}

static OwningOpRef<ModuleOp> buildThreeQQftCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQQft);
}

static OwningOpRef<ModuleOp>
buildThreeQCliffordTMixCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQCliffordTMix);
}

namespace {

struct ThreeQubitCircuitCase {
  const char* name;
  OwningOpRef<ModuleOp> (*build)(MLIRContext*);
};

const std::array<ThreeQubitCircuitCase, 4> THREE_QUBIT_CIRCUIT_CASES{{
    {.name = "ghz-3", .build = &buildThreeQGhzCircuit},
    {.name = "toffoli-3", .build = &buildThreeQToffoliCircuit},
    {.name = "qft-3", .build = &buildThreeQQftCircuit},
    {.name = "clifford-t-3", .build = &buildThreeQCliffordTMixCircuit},
}};

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest fixture at global scope
class NativeSynthesisPassMultiQubitTest : public NativeSynthesisPassTest {
protected:
  template <typename BuildFn>
  void verifyEquivalentAcrossProfiles(BuildFn buildFn,
                                      const char* circuitName = nullptr) {
    const auto profiles = allNineEquivalenceProfiles();
    for (const auto& profileCase : profiles) {
      const std::string prefix =
          circuitName != nullptr ? std::string("circuit=") + circuitName + " "
                                 : "";
      auto expected = buildFn();
      runQcToQco(expected);
      const auto expectedUnitary = computeNQubitUnitaryFromModule(expected);
      ASSERT_TRUE(expectedUnitary.has_value())
          << prefix << "native-gates=" << profileCase.nativeGates;

      auto synthesized = buildFn();
      runNativeSynthesis(synthesized, profileCase.nativeGates);
      EXPECT_TRUE(profileCase.isNative(synthesized))
          << prefix << "native-gates=" << profileCase.nativeGates;

      const auto synthesizedUnitary =
          computeNQubitUnitaryFromModule(synthesized);
      ASSERT_TRUE(synthesizedUnitary.has_value())
          << prefix << "native-gates=" << profileCase.nativeGates;
      EXPECT_TRUE(
          isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary))
          << prefix << "native-gates=" << profileCase.nativeGates;
    }
  }
};

TEST_F(NativeSynthesisPassMultiQubitTest,
       ThreeQubitCircuitsEquivalentAcrossProfiles) {
  for (const auto& circuitCase : THREE_QUBIT_CIRCUIT_CASES) {
    verifyEquivalentAcrossProfiles(
        [&] { return circuitCase.build(context.get()); }, circuitCase.name);
  }
}

static OwningOpRef<ModuleOp> buildFiveQubitStressCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQFiveQStressFourLayers);
}

TEST_F(NativeSynthesisPassMultiQubitTest,
       FiveQubitStressCircuitEquivalentAcrossProfiles) {
  verifyEquivalentAcrossProfiles(
      [&] { return buildFiveQubitStressCircuit(context.get()); });
}
