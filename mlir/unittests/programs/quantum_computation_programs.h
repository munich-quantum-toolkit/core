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

namespace qc {
class QuantumComputation;

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
void allocQubit(QuantumComputation& comp);

/// Allocates a qubit register of size `2`.
void allocQubitRegister(QuantumComputation& comp);

/// Allocates two qubit registers of size `2` and `3`.
void allocMultipleQubitRegisters(QuantumComputation& comp);

/// Allocates a large qubit register.
void allocLargeRegister(QuantumComputation& comp);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
void singleMeasurementToSingleBit(QuantumComputation& comp);

/// Repeatedly measures a single qubit into the same classical bit.
void repeatedMeasurementToSameBit(QuantumComputation& comp);

/// Repeatedly measures a single qubit into different classical bits.
void repeatedMeasurementToDifferentBits(QuantumComputation& comp);

/// Measures multiple qubits into multiple classical bits.
void multipleClassicalRegistersAndMeasurements(QuantumComputation& comp);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
void resetQubitWithoutOp(QuantumComputation& comp);

/// Resets multiple qubits without any operations being applied.
void resetMultipleQubitsWithoutOp(QuantumComputation& comp);

/// Repeatedly resets a single qubit without any operations being applied.
void repeatedResetWithoutOp(QuantumComputation& comp);

/// Resets a single qubit after a single operation.
void resetQubitAfterSingleOp(QuantumComputation& comp);

/// Resets multiple qubits after a single operation.
void resetMultipleQubitsAfterSingleOp(QuantumComputation& comp);

/// Repeatedly resets a single qubit after a single operation.
void repeatedResetAfterSingleOp(QuantumComputation& comp);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
void globalPhase(QuantumComputation& comp);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
void identity(QuantumComputation& comp);

/// Creates a controlled identity gate with a single control qubit.
void singleControlledIdentity(QuantumComputation& comp);

/// Creates a multi-controlled identity gate with multiple control qubits.
void multipleControlledIdentity(QuantumComputation& comp);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
void x(QuantumComputation& comp);

/// Creates a circuit with a single controlled X gate.
void singleControlledX(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled X gate.
void multipleControlledX(QuantumComputation& comp);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
void y(QuantumComputation& comp);

/// Creates a circuit with a single controlled Y gate.
void singleControlledY(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled Y gate.
void multipleControlledY(QuantumComputation& comp);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
void z(QuantumComputation& comp);

/// Creates a circuit with a single controlled Z gate.
void singleControlledZ(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled Z gate.
void multipleControlledZ(QuantumComputation& comp);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
void h(QuantumComputation& comp);

/// Creates a circuit with a single controlled H gate.
void singleControlledH(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled H gate.
void multipleControlledH(QuantumComputation& comp);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
void s(QuantumComputation& comp);

/// Creates a circuit with a single controlled S gate.
void singleControlledS(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled S gate.
void multipleControlledS(QuantumComputation& comp);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
void sdg(QuantumComputation& comp);

/// Creates a circuit with a single controlled Sdg gate.
void singleControlledSdg(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled Sdg gate.
void multipleControlledSdg(QuantumComputation& comp);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
void t_(QuantumComputation& comp); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
void singleControlledT(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled T gate.
void multipleControlledT(QuantumComputation& comp);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
void tdg(QuantumComputation& comp);

/// Creates a circuit with a single controlled Tdg gate.
void singleControlledTdg(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled Tdg gate.
void multipleControlledTdg(QuantumComputation& comp);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
void sx(QuantumComputation& comp);

/// Creates a circuit with a single controlled SX gate.
void singleControlledSx(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled SX gate.
void multipleControlledSx(QuantumComputation& comp);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
void sxdg(QuantumComputation& comp);

/// Creates a circuit with a single controlled SXdg gate.
void singleControlledSxdg(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled SXdg gate.
void multipleControlledSxdg(QuantumComputation& comp);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
void rx(QuantumComputation& comp);

/// Creates a circuit with a single controlled RX gate.
void singleControlledRx(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RX gate.
void multipleControlledRx(QuantumComputation& comp);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
void ry(QuantumComputation& comp);

/// Creates a circuit with a single controlled RY gate.
void singleControlledRy(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RY gate.
void multipleControlledRy(QuantumComputation& comp);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
void rz(QuantumComputation& comp);

/// Creates a circuit with a single controlled RZ gate.
void singleControlledRz(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RZ gate.
void multipleControlledRz(QuantumComputation& comp);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
void p(QuantumComputation& comp);

/// Creates a circuit with a single controlled P gate.
void singleControlledP(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled P gate.
void multipleControlledP(QuantumComputation& comp);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
void r(QuantumComputation& comp);

/// Creates a circuit with a single controlled R gate.
void singleControlledR(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled R gate.
void multipleControlledR(QuantumComputation& comp);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
void u2(QuantumComputation& comp);

/// Creates a circuit with a single controlled U2 gate.
void singleControlledU2(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled U2 gate.
void multipleControlledU2(QuantumComputation& comp);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
void u(QuantumComputation& comp);

/// Creates a circuit with a single controlled U gate.
void singleControlledU(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled U gate.
void multipleControlledU(QuantumComputation& comp);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
void swap(QuantumComputation& comp);

/// Creates a circuit with a single controlled SWAP gate.
void singleControlledSwap(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled SWAP gate.
void multipleControlledSwap(QuantumComputation& comp);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
void iswap(QuantumComputation& comp);

/// Creates a circuit with a single controlled iSWAP gate.
void singleControlledIswap(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled iSWAP gate.
void multipleControlledIswap(QuantumComputation& comp);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
void dcx(QuantumComputation& comp);

/// Creates a circuit with a single controlled DCX gate.
void singleControlledDcx(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled DCX gate.
void multipleControlledDcx(QuantumComputation& comp);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
void ecr(QuantumComputation& comp);

/// Creates a circuit with a single controlled ECR gate.
void singleControlledEcr(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled ECR gate.
void multipleControlledEcr(QuantumComputation& comp);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
void rxx(QuantumComputation& comp);

/// Creates a circuit with a single controlled RXX gate.
void singleControlledRxx(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RXX gate.
void multipleControlledRxx(QuantumComputation& comp);

/// Creates a circuit with a triple-controlled RXX gate.
void tripleControlledRxx(QuantumComputation& comp);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
void ryy(QuantumComputation& comp);

/// Creates a circuit with a single controlled RYY gate.
void singleControlledRyy(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RYY gate.
void multipleControlledRyy(QuantumComputation& comp);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
void rzx(QuantumComputation& comp);

/// Creates a circuit with a single controlled RZX gate.
void singleControlledRzx(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RZX gate.
void multipleControlledRzx(QuantumComputation& comp);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
void rzz(QuantumComputation& comp);

/// Creates a circuit with a single controlled RZZ gate.
void singleControlledRzz(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled RZZ gate.
void multipleControlledRzz(QuantumComputation& comp);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
void xxPlusYY(QuantumComputation& comp);

/// Creates a circuit with a single controlled XXPlusYY gate.
void singleControlledXxPlusYY(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
void multipleControlledXxPlusYY(QuantumComputation& comp);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
void xxMinusYY(QuantumComputation& comp);

/// Creates a circuit with a single controlled XXMinusYY gate.
void singleControlledXxMinusYY(QuantumComputation& comp);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
void multipleControlledXxMinusYY(QuantumComputation& comp);

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with just a barrier.
void barrier(QuantumComputation& comp);

/// Creates a circuit with a barrier on two qubits.
void barrierTwoQubits(QuantumComputation& comp);

/// Creates a circuit with a barrier on multiple qubits.
void barrierMultipleQubits(QuantumComputation& comp);

} // namespace qc
