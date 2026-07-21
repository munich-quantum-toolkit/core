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

namespace mlir::qco {
class QCOProgramBuilder;

/// Creates an empty QCO program.
Value emptyQCO(QCOProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
Value allocQubit(QCOProgramBuilder& b);

/// Allocates two individual qubits.
SmallVector<Value> alloc2Qubits(QCOProgramBuilder& b);

/// Allocates a single qubit that is not measured.
Value allocQubitNoMeasure(QCOProgramBuilder& b);

/// Allocates a qubit register of size `1`.
Value alloc1QubitRegister(QCOProgramBuilder& b);

/// Allocates a qubit register of size `2`.
SmallVector<Value> alloc2QubitRegister(QCOProgramBuilder& b);

/// Allocates a qubit register of size `3`.
SmallVector<Value> alloc3QubitRegister(QCOProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
SmallVector<Value> allocMultipleQubitRegisters(QCOProgramBuilder& b);

/// Allocates a large qubit register.
Value allocLargeRegister(QCOProgramBuilder& b);

/// Allocates two inline qubits.
SmallVector<Value> staticQubits(QCOProgramBuilder& b);

/// Allocates two inline qubits without measuring them.
Value staticQubitsNoMeasure(QCOProgramBuilder& b);

/// Allocates two static qubits and applies operations.
SmallVector<Value> staticQubitsWithOps(QCOProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
SmallVector<Value> staticQubitsWithParametricOps(QCOProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
SmallVector<Value> staticQubitsWithTwoTargetOps(QCOProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
SmallVector<Value> staticQubitsWithCtrl(QCOProgramBuilder& b);

/// Allocates a static qubit and applies an inverse modifier.
Value staticQubitsWithInv(QCOProgramBuilder& b);

/// Allocates and explicitly sinks a single qubit.
Value allocSinkPair(QCOProgramBuilder& b);

/// Allocates two qubits and performs a set of dead gates on them.
SmallVector<Value> deadGatesProgram(QCOProgramBuilder& b);

/// Allocates a qubit and uses a bunch of dead gates together with ResetOps.
SmallVector<Value> deadGatesResetProgram(QCOProgramBuilder& b);

/// Allocates two qubits and performs a set of dead gates on them, including
/// `if` operations.
Value deadGatesWithIfOpProgram(QCOProgramBuilder& b);

/// Allocates two qubits and performs only non-dead `if` operations.
Value deadGatesWithIfOpSimplified(QCOProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
SmallVector<Value> mixedStaticThenDynamicQubit(QCOProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: `qtensor` alloc then
/// static.
Value mixedDynamicRegisterThenStaticQubit(QCOProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
Value singleMeasurementToSingleBit(QCOProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
Value repeatedMeasurementToSameBit(QCOProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
SmallVector<Value> repeatedMeasurementToDifferentBits(QCOProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
SmallVector<Value>
multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
Value measurementWithoutRegisters(QCOProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
Value resetQubitWithoutOp(QCOProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
SmallVector<Value> resetMultipleQubitsWithoutOp(QCOProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
Value repeatedResetWithoutOp(QCOProgramBuilder& b);

/// Resets a single qubit after a single operation.
SmallVector<Value> resetQubitAfterSingleOp(QCOProgramBuilder& b);

/// Resets multiple qubits after a single operation.
SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
SmallVector<Value> repeatedResetAfterSingleOp(QCOProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
Value globalPhase(QCOProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
Value singleControlledGlobalPhase(QCOProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
SmallVector<Value> multipleControlledGlobalPhase(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
Value inverseGlobalPhase(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
SmallVector<Value> inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
Value identity(QCOProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
SmallVector<Value> singleControlledIdentity(QCOProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
SmallVector<Value> multipleControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
SmallVector<Value> nestedControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
Value trivialControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
Value inverseIdentity(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
SmallVector<Value> inverseMultipleControlledIdentity(QCOProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
Value x(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
SmallVector<Value> singleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
SmallVector<Value> multipleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
SmallVector<Value> nestedControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
Value trivialControlledX(QCOProgramBuilder& b);

/// Creates a circuit with repeated controlled X gates.
SmallVector<Value> repeatedControlledX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
Value inverseX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
SmallVector<Value> inverseMultipleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with two subsequent X gates.
Value twoX(QCOProgramBuilder& b);

/// Creates a circuit with a control modifier applied to two subsequent X gates.
SmallVector<Value> controlledTwoX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to two subsequent X
/// gates.
Value inverseTwoX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase and an
/// X gate.
Value inverseGphaseX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase and a
/// barrier.
Value inverseGphaseBarrier(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to two consecutive
/// barriers.
Value inverseTwoBarriersInInv(QCOProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
Value y(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
SmallVector<Value> singleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
SmallVector<Value> multipleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
SmallVector<Value> nestedControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
Value trivialControlledY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
Value inverseY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
SmallVector<Value> inverseMultipleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with two Y gates in a row.
Value twoY(QCOProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
Value z(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
SmallVector<Value> singleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
SmallVector<Value> multipleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
SmallVector<Value> nestedControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
Value trivialControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
Value inverseZ(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
SmallVector<Value> inverseMultipleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with two Z gates in a row.
Value twoZ(QCOProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
Value h(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
SmallVector<Value> singleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
SmallVector<Value> multipleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
SmallVector<Value> nestedControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
Value trivialControlledH(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
Value inverseH(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
SmallVector<Value> inverseMultipleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with two H gates in a row.
Value twoH(QCOProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
Value hWithoutRegister(QCOProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
Value s(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
SmallVector<Value> singleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
SmallVector<Value> multipleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
SmallVector<Value> nestedControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
Value trivialControlledS(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
Value inverseS(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
SmallVector<Value> inverseMultipleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with an S gate followed by an Sdg gate.
Value sThenSdg(QCOProgramBuilder& b);

/// Creates a circuit with two S gates in a row.
Value twoS(QCOProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
Value sdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
SmallVector<Value> singleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
SmallVector<Value> multipleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
SmallVector<Value> nestedControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
Value trivialControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
Value inverseSdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
SmallVector<Value> inverseMultipleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with an Sdg gate followed an S gate.
Value sdgThenS(QCOProgramBuilder& b);

/// Creates a circuit with two Sdg gates in a row.
Value twoSdg(QCOProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
Value t_(QCOProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
SmallVector<Value> singleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
SmallVector<Value> multipleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
SmallVector<Value> nestedControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
Value trivialControlledT(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
Value inverseT(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
SmallVector<Value> inverseMultipleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a T gate followed by a Tdg gate.
Value tThenTdg(QCOProgramBuilder& b);

/// Creates a circuit with two T gates in a row.
Value twoT(QCOProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
Value tdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
SmallVector<Value> singleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
SmallVector<Value> multipleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
SmallVector<Value> nestedControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
Value trivialControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
Value inverseTdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
SmallVector<Value> inverseMultipleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a Tdg gate followed by a T gate.
Value tdgThenT(QCOProgramBuilder& b);

/// Creates a circuit with two Tdg gates in a row.
Value twoTdg(QCOProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
Value sx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
SmallVector<Value> singleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
SmallVector<Value> multipleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
SmallVector<Value> nestedControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
Value trivialControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
Value inverseSx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
SmallVector<Value> inverseMultipleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with an SX gate followed by an SXdg gate.
Value sxThenSxdg(QCOProgramBuilder& b);

/// Creates a circuit with two SX gates in a row.
Value twoSx(QCOProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
Value sxdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
SmallVector<Value> singleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
SmallVector<Value> multipleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
SmallVector<Value> nestedControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
Value trivialControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
Value inverseSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
SmallVector<Value> inverseMultipleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an SXdg gate followed by an SX gate.
Value sxdgThenSx(QCOProgramBuilder& b);

/// Creates a circuit with two SXdg gates in a row.
Value twoSxdg(QCOProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
Value rx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
SmallVector<Value> singleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
SmallVector<Value> multipleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
SmallVector<Value> nestedControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
Value trivialControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
Value inverseRx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
SmallVector<Value> inverseMultipleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with two RX gates in a row with opposite phases.
Value twoRxOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with an RX gate with an angle of pi/2.
Value rxPiOver2(QCOProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
Value ry(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
SmallVector<Value> singleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
SmallVector<Value> multipleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
SmallVector<Value> nestedControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
Value trivialControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
Value inverseRy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
SmallVector<Value> inverseMultipleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with two RY gates in a row with opposite phases.
Value twoRyOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with an RY gate with an angle of pi/2.
Value ryPiOver2(QCOProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
Value rz(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
SmallVector<Value> singleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
SmallVector<Value> multipleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
SmallVector<Value> nestedControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
Value trivialControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
Value inverseRz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
SmallVector<Value> inverseMultipleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with two RZ gates in a row with opposite phases.
Value twoRzOppositePhase(QCOProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
Value p(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
SmallVector<Value> singleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
SmallVector<Value> multipleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
SmallVector<Value> nestedControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
Value trivialControlledP(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
Value inverseP(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
SmallVector<Value> inverseMultipleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with two P gates in a row with opposite phases.
Value twoPOppositePhase(QCOProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
Value r(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
SmallVector<Value> singleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
SmallVector<Value> multipleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
SmallVector<Value> nestedControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
Value trivialControlledR(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
Value inverseR(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
SmallVector<Value> inverseMultipleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with an R gate that can be canonicalized to an RX gate.
Value canonicalizeRToRx(QCOProgramBuilder& b);

/// Creates a circuit with an R gate that can be canonicalized to an RY gate.
Value canonicalizeRToRy(QCOProgramBuilder& b);

/// Creates a circuit with two R gates in a row with the same `phi`.
Value twoR(QCOProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
Value u2(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
SmallVector<Value> singleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
SmallVector<Value> multipleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
SmallVector<Value> nestedControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
Value trivialControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
Value inverseU2(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
SmallVector<Value> inverseMultipleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an H gate.
Value canonicalizeU2ToH(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an RX gate.
Value canonicalizeU2ToRx(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an RY gate.
Value canonicalizeU2ToRy(QCOProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
Value u(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
SmallVector<Value> singleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
SmallVector<Value> multipleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
SmallVector<Value> nestedControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
Value trivialControlledU(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
Value inverseU(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
SmallVector<Value> inverseMultipleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to a P gate.
Value canonicalizeUToP(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to an RX gate.
Value canonicalizeUToRx(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to an RY gate.
Value canonicalizeUToRy(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to a U2 gate.
Value canonicalizeUToU2(QCOProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
SmallVector<Value> swap(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
SmallVector<Value> singleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
SmallVector<Value> multipleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
SmallVector<Value> nestedControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
SmallVector<Value> trivialControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
SmallVector<Value> inverseSwap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
SmallVector<Value> inverseMultipleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with two SWAP gates in a row.
SmallVector<Value> twoSwap(QCOProgramBuilder& b);

/// Creates a circuit with two SWAP gates in a row with swapped targets.
SmallVector<Value> twoSwapSwappedTargets(QCOProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
SmallVector<Value> iswap(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
SmallVector<Value> singleControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
SmallVector<Value> multipleControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
SmallVector<Value> nestedControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
SmallVector<Value> trivialControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
SmallVector<Value> inverseIswap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
SmallVector<Value> inverseMultipleControlledIswap(QCOProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
SmallVector<Value> dcx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
SmallVector<Value> singleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
SmallVector<Value> multipleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
SmallVector<Value> nestedControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
SmallVector<Value> trivialControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
SmallVector<Value> inverseDcx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
SmallVector<Value> inverseMultipleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with two DCX gates in a row with identical targets.
SmallVector<Value> twoDcx(QCOProgramBuilder& b);

/// Creates a circuit with two DCX gates in a row with swapped targets.
SmallVector<Value> twoDcxSwappedTargets(QCOProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
SmallVector<Value> ecr(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
SmallVector<Value> singleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
SmallVector<Value> multipleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
SmallVector<Value> nestedControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
SmallVector<Value> trivialControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
SmallVector<Value> inverseEcr(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
SmallVector<Value> inverseMultipleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with two ECR gates in a row.
SmallVector<Value> twoEcr(QCOProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
SmallVector<Value> rxx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
SmallVector<Value> singleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
SmallVector<Value> multipleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
SmallVector<Value> nestedControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
SmallVector<Value> trivialControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
SmallVector<Value> inverseRxx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
SmallVector<Value> inverseMultipleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
SmallVector<Value> tripleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
SmallVector<Value> fourControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row.
SmallVector<Value> twoRxx(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with swapped targets.
SmallVector<Value> twoRxxSwappedTargets(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with opposite phases.
SmallVector<Value> twoRxxOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with opposite phases and
/// swapped targets.
SmallVector<Value> twoRxxOppositePhaseSwappedTargets(QCOProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
SmallVector<Value> ryy(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
SmallVector<Value> singleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
SmallVector<Value> multipleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
SmallVector<Value> nestedControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
SmallVector<Value> trivialControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
SmallVector<Value> inverseRyy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
SmallVector<Value> inverseMultipleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row.
SmallVector<Value> twoRyy(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with swapped targets.
SmallVector<Value> twoRyySwappedTargets(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with opposite phases.
SmallVector<Value> twoRyyOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with opposite phases and
/// swapped targets.
SmallVector<Value> twoRyyOppositePhaseSwappedTargets(QCOProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
SmallVector<Value> rzx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
SmallVector<Value> singleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
SmallVector<Value> multipleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
SmallVector<Value> nestedControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
SmallVector<Value> trivialControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
SmallVector<Value> inverseRzx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
SmallVector<Value> inverseMultipleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with two RZX gates in a row with opposite phases.
SmallVector<Value> twoRzxOppositePhase(QCOProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
SmallVector<Value> rzz(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
SmallVector<Value> singleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
SmallVector<Value> multipleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
SmallVector<Value> nestedControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
SmallVector<Value> trivialControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
SmallVector<Value> inverseRzz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
SmallVector<Value> inverseMultipleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row.
SmallVector<Value> twoRzz(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with swapped targets.
SmallVector<Value> twoRzzSwappedTargets(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with opposite phases.
SmallVector<Value> twoRzzOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with opposite phases and
/// swapped targets.
SmallVector<Value> twoRzzOppositePhaseSwappedTargets(QCOProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
SmallVector<Value> xxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
SmallVector<Value> singleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
SmallVector<Value> multipleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
SmallVector<Value> nestedControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
SmallVector<Value> trivialControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
SmallVector<Value> inverseXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
SmallVector<Value> inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with two XXPlusYY gates in a row with opposite phases.
SmallVector<Value> twoXxPlusYYOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two XXPlusYY gates in a row with swapped targets.
SmallVector<Value> twoXxPlusYYSwappedTargets(QCOProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
SmallVector<Value> xxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
SmallVector<Value> singleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
SmallVector<Value> multipleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
SmallVector<Value> nestedControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
SmallVector<Value> trivialControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
SmallVector<Value> inverseXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
SmallVector<Value> inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with two XXMinusYY gates in a row with opposite phases.
SmallVector<Value> twoXxMinusYYOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with two XXMinusYY gates in a row with swapped targets.
SmallVector<Value> twoXxMinusYYSwappedTargets(QCOProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
Value barrier(QCOProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
SmallVector<Value> barrierTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
SmallVector<Value> barrierMultipleQubits(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
Value singleControlledBarrier(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
Value inverseBarrier(QCOProgramBuilder& b);

/// Creates a circuit with two barriers in a row with overlapping qubits.
SmallVector<Value> twoBarrier(QCOProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
SmallVector<Value> trivialCtrl(QCOProgramBuilder& b);

/// Creates a circuit with an empty ctrl modifier.
SmallVector<Value> emptyCtrl(QCOProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
SmallVector<Value> nestedCtrl(QCOProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
SmallVector<Value> tripleNestedCtrl(QCOProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
SmallVector<Value> doubleNestedCtrlTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
SmallVector<Value> ctrlInvSandwich(QCOProgramBuilder& b);

/// Creates a circuit with a control modifier applied to two gates.
SmallVector<Value> ctrlTwo(QCOProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a controlled and a
/// non-controlled gate.
SmallVector<Value> ctrlTwoMixed(QCOProgramBuilder& b);

/// Creates a circuit with nested control modifiers applied to two gates.
SmallVector<Value> nestedCtrlTwo(QCOProgramBuilder& b);

/// Creates a circuit with a control modifier applied to an inverse modifier
/// applied to two gates.
SmallVector<Value> ctrlInvTwo(QCOProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with an empty inverse modifier.
SmallVector<Value> emptyInv(QCOProgramBuilder& b);

/// Creates a circuit with nested inverse modifiers.
SmallVector<Value> nestedInv(QCOProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
SmallVector<Value> tripleNestedInv(QCOProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
SmallVector<Value> invCtrlSandwich(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to two gates.
SmallVector<Value> invTwo(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a control modifier
/// applied to two gates.
SmallVector<Value> invCtrlTwo(QCOProgramBuilder& b);

// --- IfOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
SmallVector<Value> simpleIf(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
SmallVector<Value> ifTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
SmallVector<Value> ifElse(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with one qubit and one register.
Value ifOneQubitOneTensor(QCOProgramBuilder& b);

/// Creates a circuit with an if operation that uses a constant true as
/// condition.
Value constantTrueIf(QCOProgramBuilder& b);

/// Creates a circuit with an if operation that uses a constant false as
/// condition.
Value constantFalseIf(QCOProgramBuilder& b);

/// Creates a circuit with a nested if operation in the then branch that uses
/// the same condition.
SmallVector<Value> nestedTrueIf(QCOProgramBuilder& b);

/// Creates a circuit with a nested if operation in the else branch that uses
/// the same condition.
SmallVector<Value> nestedFalseIf(QCOProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
Value nestedIfOpForLoop(QCOProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
Value simpleWhileReset(QCOProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
Value simpleDoWhileReset(QCOProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
SmallVector<Value> simpleForLoop(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
Value nestedForLoopIfOp(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
SmallVector<Value> nestedForLoopWhileOp(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
Value nestedForLoopCtrlOpWithSeparateQubit(QCOProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
Value nestedForLoopCtrlOpWithExtractedQubit(QCOProgramBuilder& b);

// --- QTensor Operations -------------------------------------------------- //

/// Allocates a tensor of size `3`.
SmallVector<Value> qtensorAlloc(QCOProgramBuilder& b);

/// Allocates and explicitly deallocates a tensor.
SmallVector<Value> qtensorDealloc(QCOProgramBuilder& b);

/// Constructs a tensor with from_elements.
SmallVector<Value> qtensorFromElements(QCOProgramBuilder& b);

/// Extracts a qubit from a tensor.
Value qtensorExtract(QCOProgramBuilder& b);

/// Inserts a qubit into a tensor.
SmallVector<Value> qtensorInsert(QCOProgramBuilder& b);

/// Extracts a qubit from a tensor and inserts it immediately at a different
/// index.
SmallVector<Value> qtensorExtractInsertIndexMismatch(QCOProgramBuilder& b);

/// Extracts a qubit from a tensor and inserts it immediately at the same index.
SmallVector<Value> qtensorExtractInsertSameIndex(QCOProgramBuilder& b);

/// Inserts a qubit into a tensor and extracts it immediately at a different
/// index.
SmallVector<Value> qtensorInsertExtractIndexMismatch(QCOProgramBuilder& b);

/// Inserts a qubit into a tensor and extracts it immediately at the same index.
SmallVector<Value> qtensorInsertExtractSameIndex(QCOProgramBuilder& b);

/// Extracts three qubits with ascending index (0, 1, 2), performs a
/// computation, and finally inserts the qubits in ascending order (0, 1, 2).
SmallVector<Value> qtensorChain(QCOProgramBuilder& b);

/// Performs the same computation as the `qtensorChain` function, but uses
/// qubits immediately after the extract and inserts the qubits in descending
/// order (2, 1, 0).
SmallVector<Value> qtensorAlternativeChain(QCOProgramBuilder& b);

SmallVector<Value> controlledXH(QCOProgramBuilder& b);

SmallVector<Value> controlledInverseHT(QCOProgramBuilder& b);

SmallVector<Value> inverseTwoRxRy(QCOProgramBuilder& b);

SmallVector<Value> inverseCxThenRz(QCOProgramBuilder& b);

SmallVector<Value> inverseDcxThenRz(QCOProgramBuilder& b);

Value inverseGphaseBarrierX(QCOProgramBuilder& b);

Value inverseNestedInvHAndT(QCOProgramBuilder& b);

SmallVector<Value> inverseNestedInvHAndX(QCOProgramBuilder& b);

SmallVector<Value> inverseThreeWireRxRyRz(QCOProgramBuilder& b);

SmallVector<Value> inverseThreeWireNestedTwoInv(QCOProgramBuilder& b);

SmallVector<Value> inverseWithThreeQubitOpInBody(QCOProgramBuilder& b);

} // namespace mlir::qco
