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

#include <utility>

namespace mlir::qc {
class QCProgramBuilder;

/// Creates an empty QC Program.
std::pair<SmallVector<Value>, SmallVector<Type>>
emptyQC(QCProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubit(QCProgramBuilder& b);

/// Allocates a qubit register of size `1`.
std::pair<SmallVector<Value>, SmallVector<Type>>
alloc1QubitRegister(QCProgramBuilder& b);

/// Allocates a qubit register of size `2`.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubitRegister(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegisters(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3` and applies operations.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegistersWithOps(QCProgramBuilder& b);

/// Allocates a large qubit register.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocLargeRegister(QCProgramBuilder& b);

/// Allocates two inline qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubits(QCProgramBuilder& b);

/// Allocates two static qubits and applies operations.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithParametricOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithTwoTargetOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithCtrl(QCProgramBuilder& b);

/// Allocates a static qubit and applies an inverse modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithInv(QCProgramBuilder& b);

/// Allocates duplicate static qubits and applies operations on both.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithDuplicates(QCProgramBuilder& b);

/// Same as `staticQubitsWithDuplicates`, but with canonical static qubit
/// retrievals.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsCanonical(QCProgramBuilder& b);

/// Allocates and explicitly deallocates a single qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocDeallocPair(QCProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
std::pair<SmallVector<Value>, SmallVector<Type>>
mixedStaticThenDynamicQubit(QCProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: dynamic register then
/// static.
std::pair<SmallVector<Value>, SmallVector<Type>>
mixedDynamicRegisterThenStaticQubit(QCProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleMeasurementToSingleBit(QCProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToSameBit(QCProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToDifferentBits(QCProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleClassicalRegistersAndMeasurements(QCProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
std::pair<SmallVector<Value>, SmallVector<Type>>
measurementWithoutRegisters(QCProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitWithoutOp(QCProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsWithoutOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetWithoutOp(QCProgramBuilder& b);

/// Resets a single qubit after a single operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitAfterSingleOp(QCProgramBuilder& b);

/// Resets multiple qubits after a single operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetAfterSingleOp(QCProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
std::pair<SmallVector<Value>, SmallVector<Type>>
globalPhase(QCProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled global phase gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled global phase gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledGlobalPhase(QCProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>> identity(QCProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIdentity(QCProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIdentity(QCProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
std::pair<SmallVector<Value>, SmallVector<Type>> x(QCProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledX(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledX(QCProgramBuilder& b);

/// Creates a circuit with repeated controlled X gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedControlledX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledX(QCProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>> y(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledY(QCProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>> z(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledZ(QCProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
std::pair<SmallVector<Value>, SmallVector<Type>> h(QCProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledH(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
std::pair<SmallVector<Value>, SmallVector<Type>>
hWithoutRegister(QCProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
std::pair<SmallVector<Value>, SmallVector<Type>> s(QCProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledS(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledS(QCProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSdg(QCProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
t_(QCProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledT(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledT(QCProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> tdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledTdg(QCProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSx(QCProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sxdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSxdg(QCProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRx(QCProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> ry(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRy(QCProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRz(QCProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
std::pair<SmallVector<Value>, SmallVector<Type>> p(QCProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledP(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledP(QCProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
std::pair<SmallVector<Value>, SmallVector<Type>> r(QCProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledR(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledR(QCProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>> u2(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU2(QCProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
std::pair<SmallVector<Value>, SmallVector<Type>> u(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU(QCProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>> swap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSwap(QCProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>> iswap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIswap(QCProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> dcx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledDcx(QCProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>> ecr(QCProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledEcr(QCProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rxx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
tripleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
fourControlledRxx(QCProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> ryy(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRyy(QCProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rzx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzx(QCProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rzz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzz(QCProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> xxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxPlusYY(QCProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> xxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxMinusYY(QCProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
std::pair<SmallVector<Value>, SmallVector<Type>> barrier(QCProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
barrierTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
barrierMultipleQubits(QCProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledBarrier(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseBarrier(QCProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialCtrl(QCProgramBuilder& b);

/// Creates a circuit with an empty ctrl modifier.
std::pair<SmallVector<Value>, SmallVector<Type>> emptyCtrl(QCProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
std::pair<SmallVector<Value>, SmallVector<Type>>
doubleNestedCtrlTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvSandwich(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to two gates.
std::pair<SmallVector<Value>, SmallVector<Type>> ctrlTwo(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a controlled and a
/// non-controlled gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlTwoMixed(QCProgramBuilder& b);

/// Creates a circuit with nested control modifiers applied to two gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrlTwo(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a inverse modifier
/// applied to two gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvTwo(QCProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with an empty inverse modifier.
std::pair<SmallVector<Value>, SmallVector<Type>> emptyInv(QCProgramBuilder& b);

/// Creates a circuit with nested inverse modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>> nestedInv(QCProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedInv(QCProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlSandwich(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to two gates.
std::pair<SmallVector<Value>, SmallVector<Type>> invTwo(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a control modifier
/// applied to two gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlTwo(QCProgramBuilder& b);

// --- IfOp ----------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
std::pair<SmallVector<Value>, SmallVector<Type>> simpleIf(QCProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
std::pair<SmallVector<Value>, SmallVector<Type>> ifElse(QCProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
ifTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedIfOpForLoop(QCProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
std::pair<SmallVector<Value>, SmallVector<Type>>
simpleWhileReset(QCProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
std::pair<SmallVector<Value>, SmallVector<Type>>
simpleDoWhileReset(QCProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
std::pair<SmallVector<Value>, SmallVector<Type>>
simpleForLoop(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopIfOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopWhileOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithSeparateQubit(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithExtractedQubit(QCProgramBuilder& b);

} // namespace mlir::qc
