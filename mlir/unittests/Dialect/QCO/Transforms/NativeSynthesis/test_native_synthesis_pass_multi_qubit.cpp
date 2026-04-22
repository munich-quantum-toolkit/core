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
#include <numbers>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

namespace {

/// Controlled-phase decomposition: CP(θ) on (ctrl, tgt) expressed with only
/// single-qubit `p` and `cx`, which are supported by every targeted profile.
void emitControlledPhase(mlir::qc::QCProgramBuilder& builder, double theta,
                         Value ctrl, Value tgt) {
  builder.p(theta / 2.0, ctrl);
  builder.cx(ctrl, tgt);
  builder.p(-theta / 2.0, tgt);
  builder.cx(ctrl, tgt);
  builder.p(theta / 2.0, tgt);
}

/// Standard Clifford+T decomposition of CCX on (c1, c2, t).
void emitToffoli(mlir::qc::QCProgramBuilder& builder, Value c1, Value c2,
                 Value t) {
  builder.h(t);
  builder.cx(c2, t);
  builder.tdg(t);
  builder.cx(c1, t);
  builder.t(t);
  builder.cx(c2, t);
  builder.tdg(t);
  builder.cx(c1, t);
  builder.t(c2);
  builder.t(t);
  builder.h(t);
  builder.cx(c1, c2);
  builder.t(c1);
  builder.tdg(c2);
  builder.cx(c1, c2);
}

/// 3-qubit GHZ preparation: H on q0 then CX ladder.
OwningOpRef<ModuleOp> buildThreeQGhzCircuit(MLIRContext* context) {
  mlir::qc::QCProgramBuilder builder(context);
  builder.initialize();
  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();
  builder.h(q0);
  builder.cx(q0, q1);
  builder.cx(q1, q2);
  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);
  return builder.finalize();
}

/// 3-qubit Toffoli via Clifford+T decomposition (15 gates).
OwningOpRef<ModuleOp> buildThreeQToffoliCircuit(MLIRContext* context) {
  mlir::qc::QCProgramBuilder builder(context);
  builder.initialize();
  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();
  emitToffoli(builder, q0, q1, q2);
  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);
  return builder.finalize();
}

/// 3-qubit QFT; final wire reorder done with CXs (no native SWAP in several
/// menus).
OwningOpRef<ModuleOp> buildThreeQQftCircuit(MLIRContext* context) {
  using std::numbers::pi;
  mlir::qc::QCProgramBuilder builder(context);
  builder.initialize();
  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();

  builder.h(q2);
  emitControlledPhase(builder, pi / 2.0, q1, q2);
  builder.h(q1);
  emitControlledPhase(builder, pi / 4.0, q0, q2);
  emitControlledPhase(builder, pi / 2.0, q0, q1);
  builder.h(q0);

  // SWAP(q0, q2) via three CXs.
  builder.cx(q0, q2);
  builder.cx(q2, q0);
  builder.cx(q0, q2);

  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);
  return builder.finalize();
}

/// Deterministic Clifford+T mix on 3 qubits spanning every single-qubit family
/// accepted by `extractSingleQubitMatrix` and both CX/CZ entanglers.
OwningOpRef<ModuleOp> buildThreeQCliffordTMixCircuit(MLIRContext* context) {
  mlir::qc::QCProgramBuilder builder(context);
  builder.initialize();
  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();

  builder.h(q0);
  builder.t(q1);
  builder.x(q2);
  builder.cx(q0, q1);
  builder.rz(0.37, q2);
  builder.cz(q1, q2);
  builder.sdg(q0);
  builder.ry(-0.42, q1);
  builder.cx(q2, q0);
  builder.y(q1);
  builder.tdg(q2);
  builder.cx(q0, q1);
  builder.p(0.21, q2);
  builder.h(q2);
  builder.cz(q0, q2);
  builder.rx(-0.13, q1);
  builder.s(q0);

  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);
  return builder.finalize();
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

/// 5-qubit stress circuit matching the structural
/// `LargeMultiQubitCircuitStaysWithinMinimalMenu` test. Designed to exercise
/// many overlapping 2q blocks and deep 1q chains, using only gates that are
/// supported by every targeted profile's synthesis path.
OwningOpRef<ModuleOp> buildFiveQubitStressCircuit(MLIRContext* context) {
  mlir::qc::QCProgramBuilder builder(context);
  builder.initialize();

  const auto q0 = builder.allocQubit();
  const auto q1 = builder.allocQubit();
  const auto q2 = builder.allocQubit();
  const auto q3 = builder.allocQubit();
  const auto q4 = builder.allocQubit();

  builder.h(q0);
  builder.s(q1);
  builder.t(q2);
  builder.y(q3);
  builder.h(q4);

  builder.cx(q0, q1);
  builder.cz(q1, q2);
  builder.swap(q2, q3);
  builder.cx(q3, q4);

  for (int layer = 0; layer < 4; ++layer) {
    builder.h(q0);
    builder.s(q0);
    builder.t(q0);

    builder.y(q1);
    builder.h(q2);
    builder.s(q3);
    builder.t(q4);

    builder.cx(q0, q2);
    builder.cz(q1, q3);
    builder.cx(q2, q4);

    if ((layer % 2) == 0) {
      builder.swap(q0, q1);
      builder.swap(q3, q4);
    } else {
      builder.cx(q4, q0);
      builder.cz(q2, q1);
    }
  }

  builder.p(0.25, q0);
  builder.p(-0.5, q2);
  builder.p(0.75, q4);

  builder.dealloc(q0);
  builder.dealloc(q1);
  builder.dealloc(q2);
  builder.dealloc(q3);
  builder.dealloc(q4);
  return builder.finalize();
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
