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
SmallVector<Value> allocQubitRegister(QCProgramBuilder& b);

/// Allocates a qubit register of size `3`.
SmallVector<Value> alloc3QubitRegister(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
SmallVector<Value> allocMultipleQubitRegisters(QCProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3` and applies operations.
SmallVector<Value> allocMultipleQubitRegistersWithOps(QCProgramBuilder& b);

/// Allocates a large qubit register.
SmallVector<Value> allocLargeRegister(QCProgramBuilder& b);

/// Allocates two inline qubits.
SmallVector<Value> staticQubits(QCProgramBuilder& b);

/// Allocates two inline qubits without measuring them.
Value staticQubitsNoMeasure(QCProgramBuilder& b);

/// Allocates two static qubits and applies operations.
SmallVector<Value> staticQubitsWithOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
SmallVector<Value> staticQubitsWithParametricOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
SmallVector<Value> staticQubitsWithTwoTargetOps(QCProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
SmallVector<Value> staticQubitsWithCtrl(QCProgramBuilder& b);

/// Allocates a static qubit and applies an inverse modifier.
Value staticQubitsWithInv(QCProgramBuilder& b);

/// Allocates duplicate static qubits and applies operations on both.
SmallVector<Value> staticQubitsWithDuplicates(QCProgramBuilder& b);

/// Same as `staticQubitsWithDuplicates`, but with canonical static qubit
/// retrievals.
SmallVector<Value> staticQubitsCanonical(QCProgramBuilder& b);

/// Allocates and explicitly deallocates a single qubit.
Value allocDeallocPair(QCProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
SmallVector<Value> mixedStaticThenDynamicQubit(QCProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: dynamic register then
/// static.
SmallVector<Value> mixedDynamicRegisterThenStaticQubit(QCProgramBuilder& b);

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

/// Measures one bit of a two-bit register, returning the register with its
/// second (unmeasured) bit still part of the output.
Value partialMeasurementToRegister(QCProgramBuilder& b);

/// Measures qubits into a classical register at a runtime (dynamic) bit index,
/// namely a loop induction variable.
Value dynamicallyIndexedMeasurement(QCProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
Value measurementWithoutRegisters(QCProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
Value resetQubitWithoutOp(QCProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
SmallVector<Value> resetMultipleQubitsWithoutOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
Value repeatedResetWithoutOp(QCProgramBuilder& b);

/// Resets a single qubit after a single operation.
Value resetQubitAfterSingleOp(QCProgramBuilder& b);

/// Resets multiple qubits after a single operation.
SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
Value repeatedResetAfterSingleOp(QCProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
Value globalPhase(QCProgramBuilder& b);

/// Creates a circuit with just a global phase and a single measured qubit.
Value globalPhaseAndMeasure(QCProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
Value singleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
SmallVector<Value> multipleControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled global phase gate.
SmallVector<Value> nestedControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled global phase gate.
Value trivialControlledGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
Value inverseGlobalPhase(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
SmallVector<Value> inverseMultipleControlledGlobalPhase(QCProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
Value identity(QCProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
SmallVector<Value> singleControlledIdentity(QCProgramBuilder& b);

/// Creates an identity gate on a single qubit in a two-qubit register.
SmallVector<Value> twoQubitsOneIdentity(QCProgramBuilder& b);

/// Creates an identity gate on a single qubit in a three-qubit register.
SmallVector<Value> threeQubitsOneIdentity(QCProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
SmallVector<Value> multipleControlledIdentity(QCProgramBuilder& b);

/// Creates an barrier gate on a single qubit in a two-qubit register.
SmallVector<Value> twoQubitsOneBarrier(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
SmallVector<Value> nestedControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
Value trivialControlledIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
Value inverseIdentity(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
SmallVector<Value> inverseMultipleControlledIdentity(QCProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
Value x(QCProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
SmallVector<Value> singleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
SmallVector<Value> multipleControlledX(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
SmallVector<Value> nestedControlledX(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
Value trivialControlledX(QCProgramBuilder& b);

/// Creates a circuit with repeated controlled X gates.
SmallVector<Value> repeatedControlledX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
Value inverseX(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
SmallVector<Value> inverseMultipleControlledX(QCProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
Value y(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
SmallVector<Value> singleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
SmallVector<Value> multipleControlledY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
SmallVector<Value> nestedControlledY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
Value trivialControlledY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
Value inverseY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
SmallVector<Value> inverseMultipleControlledY(QCProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
Value z(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
SmallVector<Value> singleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
SmallVector<Value> multipleControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
SmallVector<Value> nestedControlledZ(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
Value trivialControlledZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
Value inverseZ(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
SmallVector<Value> inverseMultipleControlledZ(QCProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
Value h(QCProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
SmallVector<Value> singleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
SmallVector<Value> multipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
SmallVector<Value> nestedControlledH(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
Value trivialControlledH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
Value inverseH(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
SmallVector<Value> inverseMultipleControlledH(QCProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
Value hWithoutRegister(QCProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
Value s(QCProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
SmallVector<Value> singleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
SmallVector<Value> multipleControlledS(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
SmallVector<Value> nestedControlledS(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
Value trivialControlledS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
Value inverseS(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
SmallVector<Value> inverseMultipleControlledS(QCProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
Value sdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
SmallVector<Value> singleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
SmallVector<Value> multipleControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
SmallVector<Value> nestedControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
Value trivialControlledSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
Value inverseSdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
SmallVector<Value> inverseMultipleControlledSdg(QCProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
Value t_(QCProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
SmallVector<Value> singleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
SmallVector<Value> multipleControlledT(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
SmallVector<Value> nestedControlledT(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
Value trivialControlledT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
Value inverseT(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
SmallVector<Value> inverseMultipleControlledT(QCProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
Value tdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
SmallVector<Value> singleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
SmallVector<Value> multipleControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
SmallVector<Value> nestedControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
Value trivialControlledTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
Value inverseTdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
SmallVector<Value> inverseMultipleControlledTdg(QCProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
Value sx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
SmallVector<Value> singleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
SmallVector<Value> multipleControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
SmallVector<Value> nestedControlledSx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
Value trivialControlledSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
Value inverseSx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
SmallVector<Value> inverseMultipleControlledSx(QCProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
Value sxdg(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
SmallVector<Value> singleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
SmallVector<Value> multipleControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
SmallVector<Value> nestedControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
Value trivialControlledSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
Value inverseSxdg(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
SmallVector<Value> inverseMultipleControlledSxdg(QCProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
Value rx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
SmallVector<Value> singleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
SmallVector<Value> multipleControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
SmallVector<Value> nestedControlledRx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
Value trivialControlledRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
Value inverseRx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
SmallVector<Value> inverseMultipleControlledRx(QCProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
Value ry(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
SmallVector<Value> singleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
SmallVector<Value> multipleControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
SmallVector<Value> nestedControlledRy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
Value trivialControlledRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
Value inverseRy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
SmallVector<Value> inverseMultipleControlledRy(QCProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
Value rz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
SmallVector<Value> singleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
SmallVector<Value> multipleControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
SmallVector<Value> nestedControlledRz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
Value trivialControlledRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
Value inverseRz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
SmallVector<Value> inverseMultipleControlledRz(QCProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
Value p(QCProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
SmallVector<Value> singleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
SmallVector<Value> multipleControlledP(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
SmallVector<Value> nestedControlledP(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
Value trivialControlledP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
Value inverseP(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
SmallVector<Value> inverseMultipleControlledP(QCProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
Value r(QCProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
SmallVector<Value> singleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
SmallVector<Value> multipleControlledR(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
SmallVector<Value> nestedControlledR(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
Value trivialControlledR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
Value inverseR(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
SmallVector<Value> inverseMultipleControlledR(QCProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
Value u2(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
SmallVector<Value> singleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
SmallVector<Value> multipleControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
SmallVector<Value> nestedControlledU2(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
Value trivialControlledU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
Value inverseU2(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
SmallVector<Value> inverseMultipleControlledU2(QCProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
Value u(QCProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
SmallVector<Value> singleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
SmallVector<Value> multipleControlledU(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
SmallVector<Value> nestedControlledU(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
Value trivialControlledU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
Value inverseU(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
SmallVector<Value> inverseMultipleControlledU(QCProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
SmallVector<Value> swap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
SmallVector<Value> singleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
SmallVector<Value> multipleControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
SmallVector<Value> nestedControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
SmallVector<Value> trivialControlledSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
SmallVector<Value> inverseSwap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
SmallVector<Value> inverseMultipleControlledSwap(QCProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
SmallVector<Value> iswap(QCProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
SmallVector<Value> singleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
SmallVector<Value> multipleControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
SmallVector<Value> nestedControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
SmallVector<Value> trivialControlledIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
SmallVector<Value> inverseIswap(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
SmallVector<Value> inverseMultipleControlledIswap(QCProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
SmallVector<Value> dcx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
SmallVector<Value> singleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
SmallVector<Value> multipleControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
SmallVector<Value> nestedControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
SmallVector<Value> trivialControlledDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
SmallVector<Value> inverseDcx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
SmallVector<Value> inverseMultipleControlledDcx(QCProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
SmallVector<Value> ecr(QCProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
SmallVector<Value> singleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
SmallVector<Value> multipleControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
SmallVector<Value> nestedControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
SmallVector<Value> trivialControlledEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
SmallVector<Value> inverseEcr(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
SmallVector<Value> inverseMultipleControlledEcr(QCProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
SmallVector<Value> rxx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
SmallVector<Value> singleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
SmallVector<Value> multipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
SmallVector<Value> nestedControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
SmallVector<Value> trivialControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
SmallVector<Value> inverseRxx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
SmallVector<Value> inverseMultipleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
SmallVector<Value> tripleControlledRxx(QCProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
SmallVector<Value> fourControlledRxx(QCProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
SmallVector<Value> ryy(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
SmallVector<Value> singleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
SmallVector<Value> multipleControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
SmallVector<Value> nestedControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
SmallVector<Value> trivialControlledRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
SmallVector<Value> inverseRyy(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
SmallVector<Value> inverseMultipleControlledRyy(QCProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
SmallVector<Value> rzx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
SmallVector<Value> singleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
SmallVector<Value> multipleControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
SmallVector<Value> nestedControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
SmallVector<Value> trivialControlledRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
SmallVector<Value> inverseRzx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
SmallVector<Value> inverseMultipleControlledRzx(QCProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
SmallVector<Value> rzz(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
SmallVector<Value> singleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
SmallVector<Value> multipleControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
SmallVector<Value> nestedControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
SmallVector<Value> trivialControlledRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
SmallVector<Value> inverseRzz(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
SmallVector<Value> inverseMultipleControlledRzz(QCProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
SmallVector<Value> xxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
SmallVector<Value> singleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
SmallVector<Value> multipleControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
SmallVector<Value> nestedControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
SmallVector<Value> trivialControlledXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
SmallVector<Value> inverseXxPlusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
SmallVector<Value> inverseMultipleControlledXxPlusYY(QCProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
SmallVector<Value> xxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
SmallVector<Value> singleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
SmallVector<Value> multipleControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
SmallVector<Value> nestedControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
SmallVector<Value> trivialControlledXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
SmallVector<Value> inverseXxMinusYY(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
SmallVector<Value> inverseMultipleControlledXxMinusYY(QCProgramBuilder& b);

// --- RCCXOp --------------------------------------------------------------- //

/// Creates a circuit with just an RCCX gate.
SmallVector<Value> rccx(QCProgramBuilder& b);

/// Creates a circuit with a single controlled RCCX gate.
SmallVector<Value> singleControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with a multi-controlled RCCX gate.
SmallVector<Value> multipleControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with a nested controlled RCCX gate.
SmallVector<Value> nestedControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with a trivial controlled RCCX gate.
SmallVector<Value> trivialControlledRccx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RCCX gate.
SmallVector<Value> inverseRccx(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a multi-controlled
/// RCCX gate.
SmallVector<Value> inverseMultipleControlledRccx(QCProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
Value barrier(QCProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
SmallVector<Value> barrierTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
SmallVector<Value> barrierMultipleQubits(QCProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
SmallVector<Value> singleControlledBarrier(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
Value inverseBarrier(QCProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
SmallVector<Value> trivialCtrl(QCProgramBuilder& b);

/// Creates a circuit with an empty ctrl modifier.
SmallVector<Value> emptyCtrl(QCProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
SmallVector<Value> nestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
SmallVector<Value> tripleNestedCtrl(QCProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
SmallVector<Value> doubleNestedCtrlTwoQubits(QCProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
SmallVector<Value> ctrlInvSandwich(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to two gates.
SmallVector<Value> ctrlTwo(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a controlled and a
/// non-controlled gate.
SmallVector<Value> ctrlTwoMixed(QCProgramBuilder& b);

/// Creates a circuit with nested control modifiers applied to two gates.
SmallVector<Value> nestedCtrlTwo(QCProgramBuilder& b);

/// Creates a circuit with a control modifier applied to a inverse modifier
/// applied to two gates.
SmallVector<Value> ctrlInvTwo(QCProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with an empty inverse modifier.
SmallVector<Value> emptyInv(QCProgramBuilder& b);

/// Creates a circuit with nested inverse modifiers.
SmallVector<Value> nestedInv(QCProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
SmallVector<Value> tripleNestedInv(QCProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
SmallVector<Value> invCtrlSandwich(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to two gates.
SmallVector<Value> invTwo(QCProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a control modifier
/// applied to two gates.
SmallVector<Value> invCtrlTwo(QCProgramBuilder& b);

// --- IfOp ----------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
SmallVector<Value> simpleIf(QCProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
SmallVector<Value> ifElse(QCProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
SmallVector<Value> ifTwoQubits(QCProgramBuilder& b);

/// Creates a circuit that measures a qubit inside an if operation's branch.
SmallVector<Value> measureInIf(QCProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
Value nestedIfOpForLoop(QCProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
Value simpleWhileReset(QCProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
Value simpleDoWhileReset(QCProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
SmallVector<Value> simpleForLoop(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
Value nestedForLoopIfOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
SmallVector<Value> nestedForLoopWhileOp(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
Value nestedForLoopCtrlOpWithSeparateQubit(QCProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
Value nestedForLoopCtrlOpWithExtractedQubit(QCProgramBuilder& b);

} // namespace mlir::qc
