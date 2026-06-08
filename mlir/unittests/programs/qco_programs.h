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

namespace mlir::qco {
class QCOProgramBuilder;

/// Creates an empty QCO program.
std::pair<SmallVector<Value>, SmallVector<Type>>
emptyQCO(QCOProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubit(QCOProgramBuilder& b);

/// Allocates a single qubit that is not measured.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubitNoMeasure(QCOProgramBuilder& b);

/// Allocates a qubit register of size `1`.
std::pair<SmallVector<Value>, SmallVector<Type>>
alloc1QubitRegister(QCOProgramBuilder& b);

/// Allocates a qubit register of size `2`.
std::pair<SmallVector<Value>, SmallVector<Type>>
alloc2QubitRegister(QCOProgramBuilder& b);

/// Allocates a qubit register of size `3`.
std::pair<SmallVector<Value>, SmallVector<Type>>
alloc3QubitRegister(QCOProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegisters(QCOProgramBuilder& b);

/// Allocates a large qubit register.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocLargeRegister(QCOProgramBuilder& b);

/// Allocates two inline qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubits(QCOProgramBuilder& b);

/// Allocates two inline qubits without measuring them.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsNoMeasure(QCOProgramBuilder& b);

/// Allocates two static qubits and applies operations.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithOps(QCOProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithParametricOps(QCOProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithTwoTargetOps(QCOProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithCtrl(QCOProgramBuilder& b);

/// Allocates a static qubit and applies an inverse modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithInv(QCOProgramBuilder& b);

/// Allocates and explicitly sinks a single qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
allocSinkPair(QCOProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
std::pair<SmallVector<Value>, SmallVector<Type>>
mixedStaticThenDynamicQubit(QCOProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: `qtensor` alloc then
/// static.
std::pair<SmallVector<Value>, SmallVector<Type>>
mixedDynamicRegisterThenStaticQubit(QCOProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleMeasurementToSingleBit(QCOProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToSameBit(QCOProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToDifferentBits(QCOProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
std::pair<SmallVector<Value>, SmallVector<Type>>
measurementWithoutRegisters(QCOProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitWithoutOp(QCOProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsWithoutOp(QCOProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetWithoutOp(QCOProgramBuilder& b);

/// Resets a single qubit after a single operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitAfterSingleOp(QCOProgramBuilder& b);

/// Resets multiple qubits after a single operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetAfterSingleOp(QCOProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
std::pair<SmallVector<Value>, SmallVector<Type>>
globalPhase(QCOProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledGlobalPhase(QCOProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledGlobalPhase(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseGlobalPhase(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>> identity(QCOProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIdentity(QCOProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIdentity(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIdentity(QCOProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
std::pair<SmallVector<Value>, SmallVector<Type>> x(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledX(QCOProgramBuilder& b);

/// Creates a circuit with repeated controlled X gates.
std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedControlledX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with two X gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoX(QCOProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>> y(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with two Y gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoY(QCOProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>> z(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseZ(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with two Z gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoZ(QCOProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
std::pair<SmallVector<Value>, SmallVector<Type>> h(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledH(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseH(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with two H gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoH(QCOProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
std::pair<SmallVector<Value>, SmallVector<Type>>
hWithoutRegister(QCOProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
std::pair<SmallVector<Value>, SmallVector<Type>> s(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledS(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseS(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with an S gate followed by an Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sThenSdg(QCOProgramBuilder& b);

/// Creates a circuit with two S gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoS(QCOProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with an Sdg gate followed an S gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sdgThenS(QCOProgramBuilder& b);

/// Creates a circuit with two Sdg gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoSdg(QCOProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
t_(QCOProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledT(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseT(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a T gate followed by a Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> tThenTdg(QCOProgramBuilder& b);

/// Creates a circuit with two T gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoT(QCOProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> tdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseTdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a Tdg gate followed by a T gate.
std::pair<SmallVector<Value>, SmallVector<Type>> tdgThenT(QCOProgramBuilder& b);

/// Creates a circuit with two Tdg gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoTdg(QCOProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with an SX gate followed by an SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
sxThenSxdg(QCOProgramBuilder& b);

/// Creates a circuit with two SX gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoSx(QCOProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>> sxdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an SXdg gate followed by an SX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
sxdgThenSx(QCOProgramBuilder& b);

/// Creates a circuit with two SXdg gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoSxdg(QCOProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with two RX gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with an RX gate with an angle of pi/2.
std::pair<SmallVector<Value>, SmallVector<Type>>
rxPiOver2(QCOProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> ry(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with two RY gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with an RY gate with an angle of pi/2.
std::pair<SmallVector<Value>, SmallVector<Type>>
ryPiOver2(QCOProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rz(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with two RZ gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzOppositePhase(QCOProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
std::pair<SmallVector<Value>, SmallVector<Type>> p(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledP(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseP(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with two P gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoPOppositePhase(QCOProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
std::pair<SmallVector<Value>, SmallVector<Type>> r(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledR(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseR(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with an R gate that can be canonicalized to an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeRToRx(QCOProgramBuilder& b);

/// Creates a circuit with an R gate that can be canonicalized to an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeRToRy(QCOProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>> u2(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseU2(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an H gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeU2ToH(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeU2ToRx(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeU2ToRy(QCOProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
std::pair<SmallVector<Value>, SmallVector<Type>> u(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
std::pair<SmallVector<Value>, SmallVector<Type>> inverseU(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to a P gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToP(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to an RX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToRx(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to an RY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToRy(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to a U2 gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToU2(QCOProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>> swap(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSwap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with two SWAP gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoSwap(QCOProgramBuilder& b);

/// Creates a circuit with two SWAP gates in a row with swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoSwapSwappedTargets(QCOProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>> iswap(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIswap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIswap(QCOProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> dcx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseDcx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with two DCX gates in a row with identical targets.
std::pair<SmallVector<Value>, SmallVector<Type>> twoDcx(QCOProgramBuilder& b);

/// Creates a circuit with two DCX gates in a row with swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoDcxSwappedTargets(QCOProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>> ecr(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseEcr(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with two ECR gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoEcr(QCOProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rxx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRxx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
tripleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
fourControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoRxx(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxxSwappedTargets(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxxOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with opposite phases and
/// swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxxOppositePhaseSwappedTargets(QCOProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> ryy(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRyy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoRyy(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyySwappedTargets(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyyOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with opposite phases and
/// swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyyOppositePhaseSwappedTargets(QCOProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rzx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with two RZX gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzxOppositePhase(QCOProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>> rzz(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row.
std::pair<SmallVector<Value>, SmallVector<Type>> twoRzz(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzzSwappedTargets(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzzOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with opposite phases and
/// swapped targets.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzzOppositePhaseSwappedTargets(QCOProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>> xxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with two XXPlusYY gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoXxPlusYYOppositePhase(QCOProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
xxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with two XXMinusYY gates in a row with opposite phases.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoXxMinusYYOppositePhase(QCOProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
std::pair<SmallVector<Value>, SmallVector<Type>> barrier(QCOProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
barrierTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
barrierMultipleQubits(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledBarrier(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
std::pair<SmallVector<Value>, SmallVector<Type>>
inverseBarrier(QCOProgramBuilder& b);

/// Creates a circuit with two barriers in a row with overlapping qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
twoBarrier(QCOProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
trivialCtrl(QCOProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrl(QCOProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedCtrl(QCOProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
std::pair<SmallVector<Value>, SmallVector<Type>>
doubleNestedCtrlTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvSandwich(QCOProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with nested inverse modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedInv(QCOProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedInv(QCOProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlSandwich(QCOProgramBuilder& b);

// --- IfOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
std::pair<SmallVector<Value>, SmallVector<Type>> simpleIf(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
std::pair<SmallVector<Value>, SmallVector<Type>>
ifTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
std::pair<SmallVector<Value>, SmallVector<Type>> ifElse(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with one qubit and one register.
std::pair<SmallVector<Value>, SmallVector<Type>>
ifOneQubitOneTensor(QCOProgramBuilder& b);

/// Creates a circuit with an if operation that uses a constant true as
/// condition.
std::pair<SmallVector<Value>, SmallVector<Type>>
constantTrueIf(QCOProgramBuilder& b);

/// Creates a circuit with an if operation that uses a constant false as
/// condition.
std::pair<SmallVector<Value>, SmallVector<Type>>
constantFalseIf(QCOProgramBuilder& b);

/// Creates a circuit with a nested if operation in the then branch that uses
/// the same condition.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedTrueIf(QCOProgramBuilder& b);

/// Creates a circuit with a nested if operation in the else branch that uses
/// the same condition.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedFalseIf(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedIfOpForLoop(QCOProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
std::pair<SmallVector<Value>, SmallVector<Type>>
simpleWhileReset(QCOProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
std::pair<SmallVector<Value>, SmallVector<Type>>
simpleDoWhileReset(QCOProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
std::pair<SmallVector<Value>, SmallVector<Type>>
simpleForLoop(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopIfOp(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopWhileOp(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithSeparateQubit(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithExtractedQubit(QCOProgramBuilder& b);

// --- QTensor Operations -------------------------------------------------- //

/// Allocates a tensor of size `3`.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorAlloc(QCOProgramBuilder& b);

/// Allocates and explicitly deallocates a tensor.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorDealloc(QCOProgramBuilder& b);

/// Constructs a tensor with from_elements.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorFromElements(QCOProgramBuilder& b);

/// Extracts a qubit from a tensor.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorExtract(QCOProgramBuilder& b);

/// Inserts a qubit into a tensor.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorInsert(QCOProgramBuilder& b);

/// Extracts a qubit from a tensor and inserts it immediately at a different
/// index.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorExtractInsertIndexMismatch(QCOProgramBuilder& b);

/// Extracts a qubit from a tensor and inserts it immediately at the same index.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorExtractInsertSameIndex(QCOProgramBuilder& b);

/// Inserts a qubit into a tensor and extracts it immediately at a different
/// index.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorInsertExtractIndexMismatch(QCOProgramBuilder& b);

/// Inserts a qubit into a tensor and extracts it immediately at the same index.
std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorInsertExtractSameIndex(QCOProgramBuilder& b);

} // namespace mlir::qco
