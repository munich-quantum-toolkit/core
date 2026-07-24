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

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qc {
class QCProgramBuilder;

/// Creates an empty QC Program.
Value emptyQC(QCProgramBuilder& b);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
Value allocQubit(QCProgramBuilder& b);

/// Allocates a single qubit without measuring it.
Value allocQubitNoMeasure(QCProgramBuilder& b);

/// Allocates a qubit register of size `1`.
Value alloc1QubitRegister(QCProgramBuilder& b);

/// Allocates a qubit register of size `2`.
Value allocQubitRegister(QCProgramBuilder& b);

/// Allocates a qubit register of size `3`.
Value alloc3QubitRegister(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
Value allocMultipleQubitRegisters(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3` and applies operations.
Value allocMultipleQubitRegistersWithOps(QCProgramBuilder& b);

/// Allocates a large qubit register.
Value allocLargeRegister(QCProgramBuilder& b);

/// Allocates two inline qubits.
Value staticQubits(QCProgramBuilder& b);

/// Allocates two inline qubits without measuring them.
Value staticQubitsNoMeasure(QCProgramBuilder& b);

/// Allocates two static qubits and applies operations.
Value staticQubitsWithOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
Value staticQubitsWithParametricOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
Value staticQubitsWithTwoTargetOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
Value staticQubitsWithCtrl(QCProgramBuilder& b);

/// Allocates a static qubit and applies an inverse modifier.
Value staticQubitsWithInv(QCProgramBuilder& b);

/// Allocates duplicate static qubits and applies operations on both.
Value staticQubitsWithDuplicates(QCProgramBuilder& b);

/// Same as `staticQubitsWithDuplicates`, but with canonical static qubit
/// retrievals.
Value staticQubitsCanonical(QCProgramBuilder& b);

/// Allocates and explicitly deallocates a single qubit.
Value allocDeallocPair(QCProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
Value mixedStaticThenDynamicQubit(QCProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: dynamic register then
/// static.
Value mixedDynamicRegisterThenStaticQubit(QCProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
Value singleMeasurementToSingleBit(QCProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
Value repeatedMeasurementToSameBit(QCProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
SmallVector<Value> repeatedMeasurementToDifferentBits(QCProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
SmallVector<Value>
multipleClassicalRegistersAndMeasurements(QCProgramBuilder& b);

/// Measures one bit of a two-bit register.
Value partialMeasurementToRegister(QCProgramBuilder& b);

/// Measures qubits into a classical register at a dynamic bit index.
Value dynamicallyIndexedMeasurement(QCProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
Value measurementWithoutRegisters(QCProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
Value resetQubitWithoutOp(QCProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
Value resetMultipleQubitsWithoutOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
Value repeatedResetWithoutOp(QCProgramBuilder& b);

/// Resets a single qubit after a single operation.
SmallVector<Value> resetQubitAfterSingleOp(QCProgramBuilder& b);

/// Resets multiple qubits after a single operation.
SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
SmallVector<Value> repeatedResetAfterSingleOp(QCProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
Value globalPhase(QCProgramBuilder& b);

/// Creates a circuit with just a global phase and a single measured qubit.
Value globalPhaseAndMeasure(QCProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
Value singleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
Value multipleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled global phase gate.
Value nestedControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled global phase gate.
Value trivialControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
Value inverseGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
Value inverseMultipleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping a global-phase gate (scales θ).
Value powGphaseScaled(QCProgramBuilder& b);

/// Creates the reference for powGphaseScaled: gphase(3*0.123).
Value powGphaseScaledRef(QCProgramBuilder& b);

/// Creates a circuit with pow(-3.0) wrapping gphase (negative exponent).
Value negPowGphase(QCProgramBuilder& b);

/// Reference for negPowGphase: gphase(-3.0 * 0.123).
Value negPowGphaseRef(QCProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
Value identity(QCProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
Value singleControlledIdentity(QCProgramBuilder& b);

/// Creates an identity gate on a single qubit in a two-qubit register.
Value twoQubitsOneIdentity(QCProgramBuilder& b);

/// Creates an identity gate on a single qubit in a three-qubit register.
Value threeQubitsOneIdentity(QCProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
Value multipleControlledIdentity(QCProgramBuilder& b);

/// Creates an barrier gate on a single qubit in a two-qubit register.
Value twoQubitsOneBarrier(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
Value nestedControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
Value trivialControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
Value inverseIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
Value inverseMultipleControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping id (should pass through).
Value powId(QCProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
Value x(QCProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
Value singleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
Value multipleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
Value nestedControlledX(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
Value trivialControlledX(QCProgramBuilder& b);

/// Creates a circuit with repeated controlled X gates.
Value repeatedControlledX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
Value inverseX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
Value inverseMultipleControlledX(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an X gate (folds to gphase + RX).
Value powHalfX(QCProgramBuilder& b);

/// Creates the reference for powHalfX: sx (X^(1/2) = SX exactly).
Value powHalfXRef(QCProgramBuilder& b);

/// Creates a circuit with pow(-0.5) wrapping an X gate (r == -0.5 → sxdg).
Value powNegHalfX(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an X gate (general: gphase + rx).
Value powThirdX(QCProgramBuilder& b);

/// Creates the reference for powThirdX: gphase(-π/6) + rx(π/3).
Value powThirdXRef(QCProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
Value y(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
Value singleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
Value multipleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
Value nestedControlledY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
Value trivialControlledY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
Value inverseY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
Value inverseMultipleControlledY(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping a Y gate (folds to gphase + RY).
Value powHalfY(QCProgramBuilder& b);

/// Creates the reference for powHalfY: gphase(-π/4) followed by ry(π/2).
Value powHalfYRef(QCProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
Value z(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
Value singleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
Value multipleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
Value nestedControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
Value trivialControlledZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
Value inverseZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
Value inverseMultipleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping a Z gate (folds to P(π/2) = S).
Value powHalfZ(QCProgramBuilder& b);

/// Creates a circuit with pow(1.5) wrapping a Z gate.
/// Exercises normalizeAngle theta -= twoPi (1.5π normalises to -π/2 → sdg).
Value powThreeHalvesZ(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping a Z gate (falls through to P gate).
Value powThirdZ(QCProgramBuilder& b);

/// Creates the reference for powThirdZ: p(π/3).
Value powThirdZRef(QCProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
Value h(QCProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
Value singleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
Value multipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
Value nestedControlledH(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
Value trivialControlledH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
Value inverseH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
Value inverseMultipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
Value hWithoutRegister(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an H gate (even hermitian → erase).
Value powEvenH(QCProgramBuilder& b);

/// Creates a circuit with pow(3) wrapping an H gate (odd hermitian → H).
Value powOddH(QCProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
Value s(QCProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
Value singleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
Value multipleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
Value nestedControlledS(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
Value trivialControlledS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
Value inverseS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
Value inverseMultipleControlledS(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an S gate (folds to P(π) = Z).
Value powTwoS(QCProgramBuilder& b);

/// Creates a circuit with pow(4.0) wrapping an S gate.
/// Exercises tryReplaceWithNamedPhaseGate erase path (angle=2π → identity).
Value powFourS(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an S gate.
/// Exercises tryReplaceWithNamedPhaseGate TOp path (angle=π/4 → t).
Value powHalfS(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an S gate (default: p(π/6)).
Value powThirdS(QCProgramBuilder& b);

/// Creates the reference for powThirdS: p(π/6).
Value powThirdSRef(QCProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
Value sdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
Value singleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
Value multipleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
Value nestedControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
Value trivialControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
Value inverseSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
Value inverseMultipleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an Sdg gate (folds to P(-π) = Z).
Value powTwoSdg(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an Sdg gate.
/// Exercises tryReplaceWithNamedPhaseGate TdgOp path (angle=-π/4 → tdg).
Value powHalfSdg(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an Sdg gate (default: p(-π/6)).
Value powThirdSdg(QCProgramBuilder& b);

/// Creates the reference for powThirdSdg: p(-π/6).
Value powThirdSdgRef(QCProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
Value t_(QCProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
Value singleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
Value multipleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
Value nestedControlledT(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
Value trivialControlledT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
Value inverseT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
Value inverseMultipleControlledT(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping a T gate (folds to P(π/2) = S).
Value powTwoT(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping a T gate (default: p(π/12)).
Value powThirdT(QCProgramBuilder& b);

/// Creates the reference for powThirdT: p(π/12).
Value powThirdTRef(QCProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
Value tdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
Value singleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
Value multipleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
Value nestedControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
Value trivialControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
Value inverseTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
Value inverseMultipleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping a Tdg gate (folds to P(-π/2) = Sdg).
Value powTwoTdg(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping a Tdg gate (default: p(-π/12)).
Value powThirdTdg(QCProgramBuilder& b);

/// Creates the reference for powThirdTdg: p(-π/12).
Value powThirdTdgRef(QCProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
Value sx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
Value singleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
Value multipleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
Value nestedControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
Value trivialControlledSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
Value inverseSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
Value inverseMultipleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an SX gate (folds to X: SX^2 = X).
Value powTwoSx(QCProgramBuilder& b);

/// Creates the reference for powTwoSx: x (SX^2 = X exactly).
Value powTwoSxRef(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an SX gate (default: gphase+rx).
Value powThirdSx(QCProgramBuilder& b);

/// Creates the reference for powThirdSx: gphase(-π/12) + rx(π/6).
Value powThirdSxRef(QCProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
Value sxdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
Value singleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
Value multipleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
Value nestedControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
Value trivialControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
Value inverseSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
Value inverseMultipleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an SXdg gate (folds to X: SXdg^2 =
/// X).
Value powTwoSxdg(QCProgramBuilder& b);

/// Creates the reference for powTwoSxdg: x (SXdg^2 = X exactly).
Value powTwoSxdgRef(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an SXdg gate (default: gphase+rx).
Value powThirdSxdg(QCProgramBuilder& b);

/// Creates the reference for powThirdSxdg: gphase(π/12) + rx(-π/6).
Value powThirdSxdgRef(QCProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
Value rx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
Value singleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
Value multipleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
Value nestedControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
Value trivialControlledRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
Value inverseRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
Value inverseMultipleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping rx(0.123) (folds to rx(0.246)).
Value powRxScaled(QCProgramBuilder& b);

/// Creates the reference for powRxScaled: rx(0.246) directly.
Value rxScaled(QCProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
Value ry(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
Value singleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
Value multipleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
Value nestedControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
Value trivialControlledRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
Value inverseRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
Value inverseMultipleControlledRy(QCProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
Value rz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
Value singleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
Value multipleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
Value nestedControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
Value trivialControlledRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
Value inverseRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
Value inverseMultipleControlledRz(QCProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
Value p(QCProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
Value singleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
Value multipleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
Value nestedControlledP(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
Value trivialControlledP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
Value inverseP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
Value inverseMultipleControlledP(QCProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
Value r(QCProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
Value singleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
Value multipleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
Value nestedControlledR(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
Value trivialControlledR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
Value inverseR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
Value inverseMultipleControlledR(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an R gate (scales θ, preserves φ).
Value powRScaled(QCProgramBuilder& b);

/// Creates the reference for powRScaled: r(3*0.123, 0.456).
Value powRScaledRef(QCProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
Value u2(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
Value singleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
Value multipleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
Value nestedControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
Value trivialControlledU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
Value inverseU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
Value inverseMultipleControlledU2(QCProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
Value u(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
Value singleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
Value multipleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
Value nestedControlledU(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
Value trivialControlledU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
Value inverseU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
Value inverseMultipleControlledU(QCProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
Value swap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
Value singleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
Value multipleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
Value nestedControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
Value trivialControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
Value inverseSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
Value inverseMultipleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping a SWAP gate (even hermitian → erase).
Value powEvenSwap(QCProgramBuilder& b);

/// Creates a circuit with pow(3) wrapping a SWAP gate (odd hermitian → SWAP).
Value powOddSwap(QCProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
Value iswap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
Value singleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
Value multipleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
Value nestedControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
Value trivialControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
Value inverseIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
Value inverseMultipleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an iSWAP gate (folds to
/// xx_plus_yy(-π/2, 0)).
Value powHalfIswap(QCProgramBuilder& b);

/// Creates the reference for powHalfIswap: xx_plus_yy(-π/2, 0) directly.
Value powHalfIswapRef(QCProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
Value dcx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
Value singleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
Value multipleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
Value nestedControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
Value trivialControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
Value inverseDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
Value inverseMultipleControlledDcx(QCProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
Value ecr(QCProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
Value singleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
Value multipleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
Value nestedControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
Value trivialControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
Value inverseEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
Value inverseMultipleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an ECR gate (even hermitian → erase).
Value powEvenEcr(QCProgramBuilder& b);

/// Creates a circuit with pow(3) wrapping an ECR gate (odd hermitian → ECR).
Value powOddEcr(QCProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
Value rxx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
Value singleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
Value multipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
Value nestedControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
Value trivialControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
Value inverseRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
Value inverseMultipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
Value tripleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
Value fourControlledRxx(QCProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
Value ryy(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
Value singleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
Value multipleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
Value nestedControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
Value trivialControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
Value inverseRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
Value inverseMultipleControlledRyy(QCProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
Value rzx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
Value singleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
Value multipleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
Value nestedControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
Value trivialControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
Value inverseRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
Value inverseMultipleControlledRzx(QCProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
Value rzz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
Value singleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
Value multipleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
Value nestedControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
Value trivialControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
Value inverseRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
Value inverseMultipleControlledRzz(QCProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
Value xxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
Value singleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
Value multipleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
Value nestedControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
Value trivialControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
Value inverseXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
Value inverseMultipleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an XX+YY gate (scales θ).
Value powXxPlusYYScaled(QCProgramBuilder& b);

/// Creates the reference for powXxPlusYYScaled: xx_plus_yy(3*0.123, 0.456).
Value powXxPlusYYScaledRef(QCProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
Value xxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
Value singleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
Value multipleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
Value nestedControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
Value trivialControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
Value inverseXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
Value inverseMultipleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an XX-YY gate (scales θ).
Value powXxMinusYYScaled(QCProgramBuilder& b);

/// Creates the reference for powXxMinusYYScaled: xx_minus_yy(3*0.123, 0.456).
Value powXxMinusYYScaledRef(QCProgramBuilder& b);

// --- RCCXOp --------------------------------------------------------------- //

/// Creates a circuit with just an RCCX gate.
Value rccx(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping an RCCX gate.
Value powEvenRccx(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an RCCX gate.
Value powOddRccx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RCCX gate.
Value singleControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RCCX gate.
Value multipleControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RCCX gate.
Value nestedControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RCCX gate.
Value trivialControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RCCX gate.
Value inverseRccx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a multi-controlled
/// RCCX gate.
Value inverseMultipleControlledRccx(QCProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
Value barrier(QCProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
Value barrierTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
Value barrierMultipleQubits(QCProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
Value singleControlledBarrier(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
Value inverseBarrier(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping barrier (should pass through).
Value powBarrier(QCProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
Value trivialCtrl(QCProgramBuilder& b);

/// Creates a circuit with an empty ctrl modifier.
Value emptyCtrl(QCProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
Value nestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
Value tripleNestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
Value doubleNestedCtrlTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
Value ctrlInvSandwich(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to two gates.
Value ctrlTwo(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a controlled and a
/// non-controlled gate.
Value ctrlTwoMixed(QCProgramBuilder& b);

/// Creates a circuit with nested control modifiers applied to two gates.
Value nestedCtrlTwo(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a inverse modifier
/// applied to two gates.
Value ctrlInvTwo(QCProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with an empty inverse modifier.
Value emptyInv(QCProgramBuilder& b);

/// Creates a circuit with an empty power modifier.
Value emptyPow(QCProgramBuilder& b);

/// Creates a circuit with nested inverse modifiers.
Value nestedInv(QCProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
Value tripleNestedInv(QCProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
Value invCtrlSandwich(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to two gates.
Value invTwo(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a control modifier
/// applied to two gates.
Value invCtrlTwo(QCProgramBuilder& b);

// --- PowOp ---------------------------------------------------------------- //

/// Creates a circuit with pow(1.0) modifier (should inline to just the gate).
Value pow1Inline(QCProgramBuilder& b);

/// Creates a circuit with pow(0.0) modifier (should erase to identity).
Value pow0Erase(QCProgramBuilder& b);

/// Creates a circuit with nested pow modifiers (should merge exponents).
Value nestedPow(QCProgramBuilder& b);

/// Creates a circuit with pow(6.0) as the merged reference for nestedPow.
Value powSingleExponent(QCProgramBuilder& b);

/// Creates pow(0.5){pow(2){X}}, which is identity under principal-branch
/// matrix-power semantics and must not be flattened to pow(1){X}.
Value nestedPowBranchCut(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping a two-qubit RXX gate.
Value powRxx(QCProgramBuilder& b);

/// Creates the reference for powRxx: RXX with twice the rotation angle.
Value powRxxRef(QCProgramBuilder& b);

/// Creates a circuit with pow(-2.0) wrapping an RX gate (negative exponent).
Value negPowRx(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping RX(-0.123) (reference for
/// negPowRx and invPowRx — inv folds into angle negation).
Value powRxNeg(QCProgramBuilder& b);

/// Creates a circuit with pow(-0.5) wrapping H (negative non-integer exponent).
/// Expected to remain unchanged: fractional exponent on a unitary with
/// eigenvalue -1 cannot safely apply NegPowToInvPow.
Value negPowH(QCProgramBuilder& b);

/// Creates a circuit with inv wrapping pow(0.5) wrapping H.
/// MovePowOutside emits pow(-0.5){H} (not wrapping in inv).
Value invPowHFrac(QCProgramBuilder& b);

/// Creates a circuit with pow(-0.5) wrapping H (reference for invPowHFrac).
Value powHFracNeg(QCProgramBuilder& b);

/// Creates inv wrapping pow(2){H}. The even power folds to the identity inside
/// the modifier, leaving the inv body empty so it is erased (reference:
/// emptyQC).
Value invPowEvenH(QCProgramBuilder& b);

/// Creates inv wrapping pow(2){SWAP}. The even power folds to the identity
/// inside the modifier, leaving the inv body empty so it is erased (reference:
/// emptyQC).
Value invPowEvenSwap(QCProgramBuilder& b);

/// Creates inv wrapping pow(2){Z}. Z^2 folds to the identity inside the
/// modifier, leaving the inv body empty so it is erased (reference: emptyQC).
Value invPowSquaredZ(QCProgramBuilder& b);

/// Creates a circuit with inv wrapping pow (should reorder to pow wrapping
/// inv).
Value invPowRx(QCProgramBuilder& b);

/// Creates inv(pow(0.5){swap}) whose inner pow aliases the inv's qubits in
/// swapped order.
Value invPowReordered(QCProgramBuilder& b);

/// Creates the reference for invPowReordered: pow(-0.5) over the swapped
/// qubits.
Value invPowReorderedRef(QCProgramBuilder& b);

/// Creates a nested pow with an integral outer exponent whose inner pow aliases
/// the outer pow's qubits in swapped order.
Value mergeNestedPowReordered(QCProgramBuilder& b);

/// Creates the reference for mergeNestedPowReordered: pow(1.0) over the swapped
/// qubits.
Value mergeNestedPowReorderedRef(QCProgramBuilder& b);

/// Creates a circuit with pow wrapping ctrl wrapping RX (should move ctrl
/// outside).
Value powCtrlRx(QCProgramBuilder& b);

/// Creates a circuit with ctrl wrapping pow wrapping RX (reference for
/// powCtrlRx).
Value ctrlPowRx(QCProgramBuilder& b);

/// Creates a circuit with pow(-2) wrapping inv wrapping iSWAP.
/// Exercises NegPowToInvPow: inv{iswap} survives InvOp canonicalization,
/// FoldPowIntoGate fails (inner is InvOp), so NegPowToInvPow fires.
Value negPowInvIswap(QCProgramBuilder& b);

/// Reference for negPowInvIswap: xx_plus_yy(-2π, 0) (the fully folded form).
Value negPowInvIswapRef(QCProgramBuilder& b);

/// Creates a circuit with ctrl wrapping pow(1/3) wrapping SX. Canonicalization
/// expands pow(p){SX} to gphase+rx inside ctrl.
Value ctrlPowSx(QCProgramBuilder& b);

/// Creates the reference for ctrlPowSx: controlled gphase(-pi/12) and RX(pi/6).
Value ctrlPowSxRef(QCProgramBuilder& b);

/// pow(2) with a two-unitary body (x; rxx). The optimizer leaves multi-unitary
/// pow bodies untouched; checks verification and the QC↔QCO round-trip.
Value powTwo(QCProgramBuilder& b);

/// pow(0) with a two-unitary body (x; rxx) — folds to identity (erased at top
/// level).
Value pow0Two(QCProgramBuilder& b);

// --- IfOp ----------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
SmallVector<Value> simpleIf(QCProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
SmallVector<Value> ifElse(QCProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
SmallVector<Value> ifTwoQubits(QCProgramBuilder& b);

/// Creates a circuit that measures a qubit inside an if operation.
SmallVector<Value> ifWithMeasurement(QCProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
Value nestedIfOpForLoop(QCProgramBuilder& b);

// --- IndexSwitchOp -------------------------------------------------------- //

/// Creates a circuit with an index switch operation with one qubit.
SmallVector<Value> simpleIndexSwitch(QCProgramBuilder& b);

/// Creates a circuit with an index switch operation with multiple cases.
Value indexSwitchMultiCase(QCProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
Value simpleWhileReset(QCProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
Value simpleDoWhileReset(QCProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
Value simpleForLoop(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
Value nestedForLoopIfOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
Value nestedForLoopWhileOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested index
/// switch operation.
Value nestedForLoopSwitchOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
Value nestedForLoopCtrlOpWithSeparateQubit(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
Value nestedForLoopCtrlOpWithExtractedQubit(QCProgramBuilder& b);

} // namespace mlir::qc
