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

namespace mlir::qir {
class QIRProgramBuilder;

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
void allocQubit(QIRProgramBuilder& b);

/// Allocates a qubit register of size `2`.
void allocQubitRegister(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
void allocMultipleQubitRegisters(QIRProgramBuilder& b);

/// Allocates a large qubit register.
void allocLargeRegister(QIRProgramBuilder& b);

/// Allocates two inline qubits.
void staticQubits(QIRProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
void singleMeasurementToSingleBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
void repeatedMeasurementToSameBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
void repeatedMeasurementToDifferentBits(QIRProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
void multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
void resetQubitWithoutOp(QIRProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
void resetMultipleQubitsWithoutOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
void repeatedResetWithoutOp(QIRProgramBuilder& b);

/// Resets a single qubit after a single operation.
void resetQubitAfterSingleOp(QIRProgramBuilder& b);

/// Resets multiple qubits after a single operation.
void resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
void repeatedResetAfterSingleOp(QIRProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
void globalPhase(QIRProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
void identity(QIRProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
void singleControlledIdentity(QIRProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
void multipleControlledIdentity(QIRProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
void x(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
void singleControlledX(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
void multipleControlledX(QIRProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
void y(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
void singleControlledY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
void multipleControlledY(QIRProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
void z(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
void singleControlledZ(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
void multipleControlledZ(QIRProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
void h(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
void singleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
void multipleControlledH(QIRProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
void s(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
void singleControlledS(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
void multipleControlledS(QIRProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
void sdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
void singleControlledSdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
void multipleControlledSdg(QIRProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
void t_(QIRProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
void singleControlledT(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
void multipleControlledT(QIRProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
void tdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
void singleControlledTdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
void multipleControlledTdg(QIRProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
void sx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
void singleControlledSx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
void multipleControlledSx(QIRProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
void sxdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
void singleControlledSxdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
void multipleControlledSxdg(QIRProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
void rx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
void singleControlledRx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
void multipleControlledRx(QIRProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
void ry(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
void singleControlledRy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
void multipleControlledRy(QIRProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
void rz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
void singleControlledRz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
void multipleControlledRz(QIRProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
void p(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
void singleControlledP(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
void multipleControlledP(QIRProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
void r(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
void singleControlledR(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
void multipleControlledR(QIRProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
void u2(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
void singleControlledU2(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
void multipleControlledU2(QIRProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
void u(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
void singleControlledU(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
void multipleControlledU(QIRProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
void swap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
void singleControlledSwap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
void multipleControlledSwap(QIRProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
void iswap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
void singleControlledIswap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
void multipleControlledIswap(QIRProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
void dcx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
void singleControlledDcx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
void multipleControlledDcx(QIRProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
void ecr(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
void singleControlledEcr(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
void multipleControlledEcr(QIRProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
void rxx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
void singleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
void multipleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
void tripleControlledRxx(QIRProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
void ryy(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
void singleControlledRyy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
void multipleControlledRyy(QIRProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
void rzx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
void singleControlledRzx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
void multipleControlledRzx(QIRProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
void rzz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
void singleControlledRzz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
void multipleControlledRzz(QIRProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
void xxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
void singleControlledXxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
void multipleControlledXxPlusYY(QIRProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
void xxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
void singleControlledXxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
void multipleControlledXxMinusYY(QIRProgramBuilder& b);

} // namespace mlir::qir
