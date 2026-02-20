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

namespace mlir::qco {
class QCOProgramBuilder;

/// Creates an empty QCO program.
void emptyQCO(QCOProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
void allocQubit(QCOProgramBuilder& b);

/// Allocates a qubit register of size `2`.
void allocQubitRegister(QCOProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
void allocMultipleQubitRegisters(QCOProgramBuilder& b);

/// Allocates a large qubit register.
void allocLargeRegister(QCOProgramBuilder& b);

/// Allocates two inline qubits.
void staticQubits(QCOProgramBuilder& b);

/// Allocates and explicitly deallocates a single qubit.
void allocDeallocPair(QCOProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
void singleMeasurementToSingleBit(QCOProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
void repeatedMeasurementToSameBit(QCOProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
void repeatedMeasurementToDifferentBits(QCOProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
void multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
void resetQubitWithoutOp(QCOProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
void resetMultipleQubitsWithoutOp(QCOProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
void repeatedResetWithoutOp(QCOProgramBuilder& b);

/// Resets a single qubit after a single operation.
void resetQubitAfterSingleOp(QCOProgramBuilder& b);

/// Resets multiple qubits after a single operation.
void resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
void repeatedResetAfterSingleOp(QCOProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
void globalPhase(QCOProgramBuilder& b);

/// Creates a controlled global phase gate with a single control qubit.
void singleControlledGlobalPhase(QCOProgramBuilder& b);

/// Creates a multi-controlled global phase gate with multiple control qubits.
void multipleControlledGlobalPhase(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a global phase gate.
void inverseGlobalPhase(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
void inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
void identity(QCOProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
void singleControlledIdentity(QCOProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
void multipleControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled identity gate.
void nestedControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled identity gate.
void trivialControlledIdentity(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an identity gate.
void inverseIdentity(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
void inverseMultipleControlledIdentity(QCOProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
void x(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
void singleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
void multipleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled X gate.
void nestedControlledX(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled X gate.
void trivialControlledX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an X gate.
void inverseX(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
void inverseMultipleControlledX(QCOProgramBuilder& b);

/// Creates a circuit with two X gates in a row.
void twoX(QCOProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
void y(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
void singleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
void multipleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Y gate.
void nestedControlledY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Y gate.
void trivialControlledY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Y gate.
void inverseY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
void inverseMultipleControlledY(QCOProgramBuilder& b);

/// Creates a circuit with two Y gates in a row.
void twoY(QCOProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
void z(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
void singleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
void multipleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Z gate.
void nestedControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Z gate.
void trivialControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Z gate.
void inverseZ(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
void inverseMultipleControlledZ(QCOProgramBuilder& b);

/// Creates a circuit with two Z gates in a row.
void twoZ(QCOProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
void h(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
void singleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
void multipleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled H gate.
void nestedControlledH(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled H gate.
void trivialControlledH(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an H gate.
void inverseH(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
void inverseMultipleControlledH(QCOProgramBuilder& b);

/// Creates a circuit with two H gates in a row.
void twoH(QCOProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
void s(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
void singleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
void multipleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled S gate.
void nestedControlledS(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled S gate.
void trivialControlledS(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an S gate.
void inverseS(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
void inverseMultipleControlledS(QCOProgramBuilder& b);

/// Creates a circuit with an S gate followed by an Sdg gate.
void sThenSdg(QCOProgramBuilder& b);

/// Creates a circuit with two S gates in a row.
void twoS(QCOProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
void sdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
void singleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
void multipleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Sdg gate.
void nestedControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Sdg gate.
void trivialControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
void inverseSdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
void inverseMultipleControlledSdg(QCOProgramBuilder& b);

/// Creates a circuit with an Sdg gate followed an S gate.
void sdgThenS(QCOProgramBuilder& b);

/// Creates a circuit with two Sdg gates in a row.
void twoSdg(QCOProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
void t_(QCOProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
void singleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
void multipleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled T gate.
void nestedControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled T gate.
void trivialControlledT(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a T gate.
void inverseT(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
void inverseMultipleControlledT(QCOProgramBuilder& b);

/// Creates a circuit with a T gate followed by a Tdg gate.
void tThenTdg(QCOProgramBuilder& b);

/// Creates a circuit with two T gates in a row.
void twoT(QCOProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
void tdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
void singleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
void multipleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled Tdg gate.
void nestedControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled Tdg gate.
void trivialControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
void inverseTdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
void inverseMultipleControlledTdg(QCOProgramBuilder& b);

/// Creates a circuit with a Tdg gate followed by a T gate.
void tdgThenT(QCOProgramBuilder& b);

/// Creates a circuit with two Tdg gates in a row.
void twoTdg(QCOProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
void sx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
void singleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
void multipleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SX gate.
void nestedControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SX gate.
void trivialControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SX gate.
void inverseSx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
void inverseMultipleControlledSx(QCOProgramBuilder& b);

/// Creates a circuit with an SX gate followed by an SXdg gate.
void sxThenSxdg(QCOProgramBuilder& b);

/// Creates a circuit with two SX gates in a row.
void twoSx(QCOProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
void sxdg(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
void singleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
void multipleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SXdg gate.
void nestedControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SXdg gate.
void trivialControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
void inverseSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
void inverseMultipleControlledSxdg(QCOProgramBuilder& b);

/// Creates a circuit with an SXdg gate followed by an SX gate.
void sxdgThenSx(QCOProgramBuilder& b);

/// Creates a circuit with two SXdg gates in a row.
void twoSxdg(QCOProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
void rx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
void singleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
void multipleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RX gate.
void nestedControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RX gate.
void trivialControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RX gate.
void inverseRx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
void inverseMultipleControlledRx(QCOProgramBuilder& b);

/// Creates a circuit with two RX gates in a row with opposite phases.
void twoRxOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with an RX gate with an angle of pi/2.
void rxPiOver2(QCOProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
void ry(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
void singleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
void multipleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RY gate.
void nestedControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RY gate.
void trivialControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RY gate.
void inverseRy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
void inverseMultipleControlledRy(QCOProgramBuilder& b);

/// Creates a circuit with two RY gates in a row with opposite phases.
void twoRyOppositePhase(QCOProgramBuilder& b);

/// Creates a circuit with an RY gate with an angle of pi/2.
void ryPiOver2(QCOProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
void rz(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
void singleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
void multipleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZ gate.
void nestedControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZ gate.
void trivialControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZ gate.
void inverseRz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
void inverseMultipleControlledRz(QCOProgramBuilder& b);

/// Creates a circuit with two RZ gates in a row with opposite phases.
void twoRzOppositePhase(QCOProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
void p(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
void singleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
void multipleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled P gate.
void nestedControlledP(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled P gate.
void trivialControlledP(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a P gate.
void inverseP(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
void inverseMultipleControlledP(QCOProgramBuilder& b);

/// Creates a circuit with two P gates in a row with opposite phases.
void twoPOppositePhase(QCOProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
void r(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
void singleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
void multipleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled R gate.
void nestedControlledR(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled R gate.
void trivialControlledR(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an R gate.
void inverseR(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
void inverseMultipleControlledR(QCOProgramBuilder& b);

/// Creates a circuit with an R gate that can be canonicalized to an RX gate.
void canonicalizeRToRx(QCOProgramBuilder& b);

/// Creates a circuit with an R gate that can be canonicalized to an RY gate.
void canonicalizeRToRy(QCOProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
void u2(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
void singleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
void multipleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled U2 gate.
void nestedControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled U2 gate.
void trivialControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U2 gate.
void inverseU2(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
void inverseMultipleControlledU2(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an H gate.
void canonicalizeU2ToH(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an RX gate.
void canonicalizeU2ToRx(QCOProgramBuilder& b);

/// Creates a circuit with a U2 gate that can be canonicalized to an RY gate.
void canonicalizeU2ToRy(QCOProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
void u(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
void singleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
void multipleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled U gate.
void nestedControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled U gate.
void trivialControlledU(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a U gate.
void inverseU(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
void inverseMultipleControlledU(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to a P gate.
void canonicalizeUToP(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to an RX gate.
void canonicalizeUToRx(QCOProgramBuilder& b);

/// Creates a circuit with a U gate that can be canonicalized to an RY gate.
void canonicalizeUToRy(QCOProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
void swap(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
void singleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
void multipleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled SWAP gate.
void nestedControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled SWAP gate.
void trivialControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
void inverseSwap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
void inverseMultipleControlledSwap(QCOProgramBuilder& b);

/// Creates a circuit with two SWAP gates in a row.
void twoSwap(QCOProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
void iswap(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
void singleControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
void multipleControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled iSWAP gate.
void nestedControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled iSWAP gate.
void trivialControlledIswap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
void inverseIswap(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
void inverseMultipleControlledIswap(QCOProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
void dcx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
void singleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
void multipleControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled DCX gate.
void nestedControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled DCX gate.
void trivialControlledDcx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a DCX gate.
void inverseDcx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
void inverseMultipleControlledDcx(QCOProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
void ecr(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
void singleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
void multipleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled ECR gate.
void nestedControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled ECR gate.
void trivialControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an ECR gate.
void inverseEcr(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
void inverseMultipleControlledEcr(QCOProgramBuilder& b);

/// Creates a circuit with two ECR gates in a row.
void twoEcr(QCOProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
void rxx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
void singleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
void multipleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RXX gate.
void nestedControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RXX gate.
void trivialControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RXX gate.
void inverseRxx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
void inverseMultipleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
void tripleControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with a four-controlled RXX gate.
void fourControlledRxx(QCOProgramBuilder& b);

/// Creates a circuit with two RXX gates in a row with opposite phases.
void twoRxxOppositePhase(QCOProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
void ryy(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
void singleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
void multipleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RYY gate.
void nestedControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RYY gate.
void trivialControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RYY gate.
void inverseRyy(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
void inverseMultipleControlledRyy(QCOProgramBuilder& b);

/// Creates a circuit with two RYY gates in a row with opposite phases.
void twoRyyOppositePhase(QCOProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
void rzx(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
void singleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
void multipleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZX gate.
void nestedControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZX gate.
void trivialControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZX gate.
void inverseRzx(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
void inverseMultipleControlledRzx(QCOProgramBuilder& b);

/// Creates a circuit with two RZX gates in a row with opposite phases.
void twoRzxOppositePhase(QCOProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
void rzz(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
void singleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
void multipleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled RZZ gate.
void nestedControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled RZZ gate.
void trivialControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
void inverseRzz(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
void inverseMultipleControlledRzz(QCOProgramBuilder& b);

/// Creates a circuit with two RZZ gates in a row with opposite phases.
void twoRzzOppositePhase(QCOProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
void xxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
void singleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
void multipleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled XXPlusYY gate.
void nestedControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXPlusYY gate.
void trivialControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
void inverseXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
void inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b);

/// Creates a circuit with two XXPlusYY gates in a row with opposite phases.
void twoXxPlusYYOppositePhase(QCOProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
void xxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
void singleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
void multipleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a nested controlled XXMinusYY gate.
void nestedControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with a trivial controlled XXMinusYY gate.
void trivialControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
void inverseXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
void inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b);

/// Creates a circuit with two XXMinusYY gates in a row with opposite phases.
void twoXxMinusYYOppositePhase(QCOProgramBuilder& b);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
void barrier(QCOProgramBuilder& b);

/// Creates a circuit with a barrier on two qubits.
void barrierTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with a barrier on multiple qubits.
void barrierMultipleQubits(QCOProgramBuilder& b);

/// Creates a circuit with a single controlled barrier.
void singleControlledBarrier(QCOProgramBuilder& b);

/// Creates a circuit with an inverse modifier applied to a barrier.
void inverseBarrier(QCOProgramBuilder& b);

/// Creates a circuit with two barriers in a row with overlapping qubits.
void twoBarrier(QCOProgramBuilder& b);

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
void trivialCtrl(QCOProgramBuilder& b);

/// Creates a circuit with nested ctrl modifiers.
void nestedCtrl(QCOProgramBuilder& b);

/// Creates a circuit with triple nested ctrl modifiers.
void tripleNestedCtrl(QCOProgramBuilder& b);

/// Creates a circuit with double nested ctrl modifiers with two qubits each.
void doubleNestedCtrlTwoQubits(QCOProgramBuilder& b);

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
void ctrlInvSandwich(QCOProgramBuilder& b);

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with nested inverse modifiers.
void nestedInv(QCOProgramBuilder& b);

/// Creates a circuit with triple nested inverse modifiers.
void tripleNestedInv(QCOProgramBuilder& b);

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
void invCtrlSandwich(QCOProgramBuilder& b);
} // namespace mlir::qco
