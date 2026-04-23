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

#include <array>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

namespace {

OwningOpRef<ModuleOp> buildThreeQGhzCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQGhz);
}

OwningOpRef<ModuleOp> buildThreeQToffoliCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQToffoli);
}

OwningOpRef<ModuleOp> buildThreeQQftCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQQft);
}

OwningOpRef<ModuleOp> buildThreeQCliffordTMixCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQThreeQCliffordTMix);
}

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

TEST_F(NativeSynthesisPassTest, ThreeQubitCircuitsEquivalentAcrossProfiles) {
  const auto profiles = NativeSynthesisPassTest::allNineEquivalenceProfiles();

  for (const auto& circuitCase : THREE_QUBIT_CIRCUIT_CASES) {
    for (const auto& profileCase : profiles) {
      auto expected = circuitCase.build(context.get());
      runQcToQco(expected);
      const auto expectedUnitary = computeNQubitUnitaryFromModule(expected);
      ASSERT_TRUE(expectedUnitary.has_value())
          << "circuit=" << circuitCase.name
          << " native-gates=" << profileCase.nativeGates;

      auto synthesized = circuitCase.build(context.get());
      runNativeSynthesis(synthesized, profileCase.nativeGates);
      EXPECT_TRUE(profileCase.isNative(synthesized))
          << "circuit=" << circuitCase.name
          << " native-gates=" << profileCase.nativeGates;

      const auto synthesizedUnitary =
          computeNQubitUnitaryFromModule(synthesized);
      ASSERT_TRUE(synthesizedUnitary.has_value())
          << "circuit=" << circuitCase.name
          << " native-gates=" << profileCase.nativeGates;
      EXPECT_TRUE(
          isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary))
          << "circuit=" << circuitCase.name
          << " native-gates=" << profileCase.nativeGates;
    }
  }
}

namespace {

OwningOpRef<ModuleOp> buildFiveQubitStressCircuit(MLIRContext* context) {
  return mlir::qc::QCProgramBuilder::build(
      context, mlir::qc::nativeSynthMultiQFiveQStressFourLayers);
}

} // namespace

TEST_F(NativeSynthesisPassTest,
       FiveQubitStressCircuitEquivalentAcrossProfiles) {
  const auto profiles = NativeSynthesisPassTest::allNineEquivalenceProfiles();

  for (const auto& profileCase : profiles) {
    auto expected = buildFiveQubitStressCircuit(context.get());
    runQcToQco(expected);
    const auto expectedUnitary = computeNQubitUnitaryFromModule(expected);
    ASSERT_TRUE(expectedUnitary.has_value())
        << "native-gates=" << profileCase.nativeGates;

    auto synthesized = buildFiveQubitStressCircuit(context.get());
    runNativeSynthesis(synthesized, profileCase.nativeGates);
    EXPECT_TRUE(profileCase.isNative(synthesized))
        << "native-gates=" << profileCase.nativeGates;

    const auto synthesizedUnitary = computeNQubitUnitaryFromModule(synthesized);
    ASSERT_TRUE(synthesizedUnitary.has_value())
        << "native-gates=" << profileCase.nativeGates;
    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary))
        << "native-gates=" << profileCase.nativeGates;
  }
}
