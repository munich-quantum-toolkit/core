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

namespace mlir::qc {
class QCProgramBuilder;

/// Creates an empty QC Program.
void emptyQC(QCProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
void allocQubit(QCProgramBuilder& b);

/// Allocates a qubit register of size `2`.
void allocQubitRegister(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
void allocMultipleQubitRegisters(QCProgramBuilder& b);

/// Allocates a large qubit register.
void allocLargeRegister(QCProgramBuilder& b);

/// Allocates two inline qubits.
void staticQubits(QCProgramBuilder& b);

/// Allocates two static qubits and applies operations.
void staticQubitsWithOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
void staticQubitsWithParametricOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
void staticQubitsWithTwoTargetOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
void staticQubitsWithCtrl(QCProgramBuilder& b);

/// Allocates a static qubit and applies an inverse modifier.
void staticQubitsWithInv(QCProgramBuilder& b);

/// Allocates duplicate static qubits and applies operations on both.
void staticQubitsWithDuplicates(QCProgramBuilder& b);

/// Same as `staticQubitsWithDuplicates`, but with canonical static qubit
/// retrievals.
void staticQubitsCanonical(QCProgramBuilder& b);

/// Allocates and explicitly deallocates a single qubit.
void allocDeallocPair(QCProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
void singleMeasurementToSingleBit(QCProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
void repeatedMeasurementToSameBit(QCProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
void repeatedMeasurementToDifferentBits(QCProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
void multipleClassicalRegistersAndMeasurements(QCProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
void measurementWithoutRegisters(QCProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
void resetQubitWithoutOp(QCProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
void resetMultipleQubitsWithoutOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
void repeatedResetWithoutOp(QCProgramBuilder& b);

/// Resets a single qubit after a single operation.
void resetQubitAfterSingleOp(QCProgramBuilder& b);

/// Resets multiple qubits after a single operation.
void resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
void repeatedResetAfterSingleOp(QCProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
void globalPhase(QCProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
void singleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
void multipleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled global phase gate.
void nestedControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled global phase gate.
void trivialControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
void inverseGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
void inverseMultipleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping a global-phase gate (scales θ).
void powGphaseScaled(QCProgramBuilder& b);

/// Creates the reference for powGphaseScaled: gphase(3*0.123).
void powGphaseScaledRef(QCProgramBuilder& b);

/// Creates a circuit with pow(-3.0) wrapping gphase (negative exponent).
void negPowGphase(QCProgramBuilder& b);

/// Reference for negPowGphase: gphase(-3.0 * 0.123).
void negPowGphaseRef(QCProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
void identity(QCProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
void singleControlledIdentity(QCProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
void multipleControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
void nestedControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
void trivialControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
void inverseIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
void inverseMultipleControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping id (should pass through).
void powId(QCProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
void x(QCProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
void singleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
void multipleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
void nestedControlledX(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
void trivialControlledX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
void inverseX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
void inverseMultipleControlledX(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an X gate (folds to gphase + RX).
void powHalfX(QCProgramBuilder& b);

/// Creates the reference for powHalfX: sx (X^(1/2) = SX exactly).
void powHalfXRef(QCProgramBuilder& b);

/// Creates a circuit with pow(-0.5) wrapping an X gate (r == -0.5 → sxdg).
void powNegHalfX(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an X gate (general: gphase + rx).
void powThirdX(QCProgramBuilder& b);

/// Creates the reference for powThirdX: gphase(-π/6) + rx(π/3).
void powThirdXRef(QCProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
void y(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
void singleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
void multipleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
void nestedControlledY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
void trivialControlledY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
void inverseY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
void inverseMultipleControlledY(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping a Y gate (folds to gphase + RY).
void powHalfY(QCProgramBuilder& b);

/// Creates the reference for powHalfY: gphase(-π/4) followed by ry(π/2).
void powHalfYRef(QCProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
void z(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
void singleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
void multipleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
void nestedControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
void trivialControlledZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
void inverseZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
void inverseMultipleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping a Z gate (folds to P(π/2) = S).
void powHalfZ(QCProgramBuilder& b);

/// Creates a circuit with pow(1.5) wrapping a Z gate.
/// Exercises normalizeAngle theta -= twoPi (1.5π normalises to -π/2 → sdg).
void powThreeHalvesZ(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping a Z gate (falls through to P gate).
void powThirdZ(QCProgramBuilder& b);

/// Creates the reference for powThirdZ: p(π/3).
void powThirdZRef(QCProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
void h(QCProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
void singleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
void multipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
void nestedControlledH(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
void trivialControlledH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
void inverseH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
void inverseMultipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
void hWithoutRegister(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an H gate (even hermitian → erase).
void powEvenH(QCProgramBuilder& b);

/// Creates a circuit with pow(3) wrapping an H gate (odd hermitian → H).
void powOddH(QCProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
void s(QCProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
void singleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
void multipleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
void nestedControlledS(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
void trivialControlledS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
void inverseS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
void inverseMultipleControlledS(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an S gate (folds to P(π) = Z).
void powTwoS(QCProgramBuilder& b);

/// Creates a circuit with pow(4.0) wrapping an S gate.
/// Exercises tryReplaceWithNamedPhaseGate erase path (angle=2π → identity).
void powFourS(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an S gate.
/// Exercises tryReplaceWithNamedPhaseGate TOp path (angle=π/4 → t).
void powHalfS(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an S gate (default: p(π/6)).
void powThirdS(QCProgramBuilder& b);

/// Creates the reference for powThirdS: p(π/6).
void powThirdSRef(QCProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
void sdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
void singleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
void multipleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
void nestedControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
void trivialControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
void inverseSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
void inverseMultipleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an Sdg gate (folds to P(-π) = Z).
void powTwoSdg(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an Sdg gate.
/// Exercises tryReplaceWithNamedPhaseGate TdgOp path (angle=-π/4 → tdg).
void powHalfSdg(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an Sdg gate (default: p(-π/6)).
void powThirdSdg(QCProgramBuilder& b);

/// Creates the reference for powThirdSdg: p(-π/6).
void powThirdSdgRef(QCProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
void t_(QCProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
void singleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
void multipleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
void nestedControlledT(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
void trivialControlledT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
void inverseT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
void inverseMultipleControlledT(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping a T gate (folds to P(π/2) = S).
void powTwoT(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping a T gate (default: p(π/12)).
void powThirdT(QCProgramBuilder& b);

/// Creates the reference for powThirdT: p(π/12).
void powThirdTRef(QCProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
void tdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
void singleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
void multipleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
void nestedControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
void trivialControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
void inverseTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
void inverseMultipleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping a Tdg gate (folds to P(-π/2) = Sdg).
void powTwoTdg(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping a Tdg gate (default: p(-π/12)).
void powThirdTdg(QCProgramBuilder& b);

/// Creates the reference for powThirdTdg: p(-π/12).
void powThirdTdgRef(QCProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
void sx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
void singleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
void multipleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
void nestedControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
void trivialControlledSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
void inverseSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
void inverseMultipleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an SX gate (folds to X: SX^2 = X).
void powTwoSx(QCProgramBuilder& b);

/// Creates the reference for powTwoSx: x (SX^2 = X exactly).
void powTwoSxRef(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an SX gate (default: gphase+rx).
void powThirdSx(QCProgramBuilder& b);

/// Creates the reference for powThirdSx: gphase(-π/12) + rx(π/6).
void powThirdSxRef(QCProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
void sxdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
void singleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
void multipleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
void nestedControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
void trivialControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
void inverseSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
void inverseMultipleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an SXdg gate (folds to X: SXdg^2 =
/// X).
void powTwoSxdg(QCProgramBuilder& b);

/// Creates the reference for powTwoSxdg: x (SXdg^2 = X exactly).
void powTwoSxdgRef(QCProgramBuilder& b);

/// Creates a circuit with pow(1/3) wrapping an SXdg gate (default: gphase+rx).
void powThirdSxdg(QCProgramBuilder& b);

/// Creates the reference for powThirdSxdg: gphase(π/12) + rx(-π/6).
void powThirdSxdgRef(QCProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
void rx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
void singleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
void multipleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
void nestedControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
void trivialControlledRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
void inverseRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
void inverseMultipleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping rx(0.123) (folds to rx(0.246)).
void powRxScaled(QCProgramBuilder& b);

/// Creates the reference for powRxScaled: rx(0.246) directly.
void rxScaled(QCProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
void ry(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
void singleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
void multipleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
void nestedControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
void trivialControlledRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
void inverseRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
void inverseMultipleControlledRy(QCProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
void rz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
void singleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
void multipleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
void nestedControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
void trivialControlledRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
void inverseRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
void inverseMultipleControlledRz(QCProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
void p(QCProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
void singleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
void multipleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
void nestedControlledP(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
void trivialControlledP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
void inverseP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
void inverseMultipleControlledP(QCProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
void r(QCProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
void singleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
void multipleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
void nestedControlledR(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
void trivialControlledR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
void inverseR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
void inverseMultipleControlledR(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an R gate (scales θ, preserves φ).
void powRScaled(QCProgramBuilder& b);

/// Creates the reference for powRScaled: r(3*0.123, 0.456).
void powRScaledRef(QCProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
void u2(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
void singleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
void multipleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
void nestedControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
void trivialControlledU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
void inverseU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
void inverseMultipleControlledU2(QCProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
void u(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
void singleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
void multipleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
void nestedControlledU(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
void trivialControlledU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
void inverseU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
void inverseMultipleControlledU(QCProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
void swap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
void singleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
void multipleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
void nestedControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
void trivialControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
void inverseSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
void inverseMultipleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping a SWAP gate (even hermitian → erase).
void powEvenSwap(QCProgramBuilder& b);

/// Creates a circuit with pow(3) wrapping a SWAP gate (odd hermitian → SWAP).
void powOddSwap(QCProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
void iswap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
void singleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
void multipleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
void nestedControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
void trivialControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
void inverseIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
void inverseMultipleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with pow(0.5) wrapping an iSWAP gate (folds to
/// xx_plus_yy(-π/2, 0)).
void powHalfIswap(QCProgramBuilder& b);

/// Creates the reference for powHalfIswap: xx_plus_yy(-π/2, 0) directly.
void powHalfIswapRef(QCProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
void dcx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
void singleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
void multipleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
void nestedControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
void trivialControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
void inverseDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
void inverseMultipleControlledDcx(QCProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
void ecr(QCProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
void singleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
void multipleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
void nestedControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
void trivialControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
void inverseEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
void inverseMultipleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with pow(2) wrapping an ECR gate (even hermitian → erase).
void powEvenEcr(QCProgramBuilder& b);

/// Creates a circuit with pow(3) wrapping an ECR gate (odd hermitian → ECR).
void powOddEcr(QCProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
void rxx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
void singleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
void multipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
void nestedControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
void trivialControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
void inverseRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
void inverseMultipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
void tripleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
void fourControlledRxx(QCProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
void ryy(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
void singleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
void multipleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
void nestedControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
void trivialControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
void inverseRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
void inverseMultipleControlledRyy(QCProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
void rzx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
void singleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
void multipleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
void nestedControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
void trivialControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
void inverseRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
void inverseMultipleControlledRzx(QCProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
void rzz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
void singleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
void multipleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
void nestedControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
void trivialControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
void inverseRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
void inverseMultipleControlledRzz(QCProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
void xxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
void singleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
void multipleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
void nestedControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
void trivialControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
void inverseXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
void inverseMultipleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an XX+YY gate (scales θ).
void powXxPlusYYScaled(QCProgramBuilder& b);

/// Creates the reference for powXxPlusYYScaled: xx_plus_yy(3*0.123, 0.456).
void powXxPlusYYScaledRef(QCProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
void xxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
void singleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
void multipleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
void nestedControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
void trivialControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
void inverseXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
void inverseMultipleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with pow(3.0) wrapping an XX-YY gate (scales θ).
void powXxMinusYYScaled(QCProgramBuilder& b);

/// Creates the reference for powXxMinusYYScaled: xx_minus_yy(3*0.123, 0.456).
void powXxMinusYYScaledRef(QCProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
void barrier(QCProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
void barrierTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
void barrierMultipleQubits(QCProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
void singleControlledBarrier(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
void inverseBarrier(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping barrier (should pass through).
void powBarrier(QCProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
void trivialCtrl(QCProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
void nestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
void tripleNestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
void doubleNestedCtrlTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
void ctrlInvSandwich(QCProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with nested inverse modifiers.
void nestedInv(QCProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
void tripleNestedInv(QCProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
void invCtrlSandwich(QCProgramBuilder& b);

// --- PowOp ---------------------------------------------------------------- //

/// Creates a circuit with pow(1.0) modifier (should inline to just the gate).
void pow1Inline(QCProgramBuilder& b);

/// Creates a circuit with pow(0.0) modifier (should erase to identity).
void pow0Erase(QCProgramBuilder& b);

/// Creates a circuit with nested pow modifiers (should merge exponents).
void nestedPow(QCProgramBuilder& b);

/// Creates a circuit with pow(6.0) as the merged reference for nestedPow.
void powSingleExponent(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping a two-qubit RXX gate.
void powRxx(QCProgramBuilder& b);

/// Creates a circuit with pow(-2.0) wrapping an RX gate (negative exponent).
void negPowRx(QCProgramBuilder& b);

/// Creates a circuit with pow(2.0) wrapping RX(-0.123) (reference for
/// negPowRx and invPowRx — inv folds into angle negation).
void powRxNeg(QCProgramBuilder& b);

/// Creates a circuit with pow(-0.5) wrapping H (negative non-integer exponent).
void negPowH(QCProgramBuilder& b);

/// Reference for negPowH: pow(0.5) wrapping H (NegPowToInvPow + inv{H}=H).
void negPowHRef(QCProgramBuilder& b);

/// Creates a circuit with inv wrapping pow (should reorder to pow wrapping
/// inv).
void invPowRx(QCProgramBuilder& b);

/// Creates a circuit with pow wrapping ctrl wrapping RX (should move ctrl
/// outside).
void powCtrlRx(QCProgramBuilder& b);

/// Creates a circuit with ctrl wrapping pow wrapping RX (reference for
/// powCtrlRx).
void ctrlPowRx(QCProgramBuilder& b);

/// Creates a circuit with ctrl wrapping pow(1/3) wrapping SX. The fold
/// pow(p){SX} → gphase+rx is suppressed inside ctrl (would emit two ops),
/// so the pow survives canonicalization and reaches ConvertQCPowOp.
void ctrlPowSx(QCProgramBuilder& b);

} // namespace mlir::qc
