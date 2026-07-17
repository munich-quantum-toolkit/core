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

namespace mlir::qir {
class QIRProgramBuilder;

/// Creates an empty QIR program.
template <bool IntoRegister = false> Value emptyQIR(QIRProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
template <bool IntoRegister = false> Value allocQubit(QIRProgramBuilder& b);

/// Allocates a qubit register of size `1`.
template <bool IntoRegister = false>
Value alloc1QubitRegister(QIRProgramBuilder& b);

/// Allocates a qubit register of size `2`.
template <bool IntoRegister = false>
Value allocQubitRegister(QIRProgramBuilder& b);

/// Allocates a qubit register of size `3`.
template <bool IntoRegister = false>
Value alloc3QubitRegister(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
template <bool IntoRegister = false>
Value allocMultipleQubitRegisters(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3` and applies operations.
template <bool IntoRegister = false>
Value allocMultipleQubitRegistersWithOps(QIRProgramBuilder& b);

/// Allocates a large qubit register.
template <bool IntoRegister = false>
Value allocLargeRegister(QIRProgramBuilder& b);

/// Allocates two inline qubits.
Value staticQubits(QIRProgramBuilder& b);

/// Allocates two static qubits and applies operations.
Value staticQubitsWithOps(QIRProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
Value staticQubitsWithParametricOps(QIRProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
Value staticQubitsWithTwoTargetOps(QIRProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
Value staticQubitsWithCtrl(QIRProgramBuilder& b);

/// Allocates a static qubit and applies the inverse of a T gate (Tdg).
Value staticQubitsWithInv(QIRProgramBuilder& b);

/// Allocates duplicate static qubits and applies operations on both.
Value staticQubitsWithDuplicates(QIRProgramBuilder& b);

/// Same as `staticQubitsWithDuplicates`, but with canonical static qubit
/// retrievals.
Value staticQubitsCanonical(QIRProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
Value mixedStaticThenDynamicQubit(QIRProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: dynamic register then
/// static.
Value mixedDynamicRegisterThenStaticQubit(QIRProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
template <bool IntoRegister = false>
Value singleMeasurementToSingleBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
template <bool IntoRegister = false>
Value repeatedMeasurementToSameBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
template <bool IntoRegister = false>
Value repeatedMeasurementToDifferentBits(QIRProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
template <bool IntoRegister = false>
Value multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b);

/// Measures one bit of a two-bit register, leaving the second bit unmeasured
/// but still output-recorded.
template <bool IntoRegister = false>
Value partialMeasurementToRegister(QIRProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
template <bool IntoRegister = false>
Value measurementWithoutRegisters(QIRProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
template <bool IntoRegister = false>
Value resetQubitWithoutOp(QIRProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
template <bool IntoRegister = false>
Value resetMultipleQubitsWithoutOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
template <bool IntoRegister = false>
Value repeatedResetWithoutOp(QIRProgramBuilder& b);

/// Resets a single qubit after a single operation.
template <bool IntoRegister = false>
Value resetQubitAfterSingleOp(QIRProgramBuilder& b);

/// Resets multiple qubits after a single operation.
template <bool IntoRegister = false>
Value resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
template <bool IntoRegister = false>
Value repeatedResetAfterSingleOp(QIRProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
template <bool IntoRegister = false> Value globalPhase(QIRProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
template <bool IntoRegister = false> Value identity(QIRProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
template <bool IntoRegister = false>
Value singleControlledIdentity(QIRProgramBuilder& b);

/// Creates an identity gate on a single qubit in a two-qubit register.
template <bool IntoRegister = false>
Value twoQubitsOneIdentity(QIRProgramBuilder& b);

/// Creates an identity gate on a single qubit in a three-qubit register.
template <bool IntoRegister = false>
Value threeQubitsOneIdentity(QIRProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
template <bool IntoRegister = false>
Value multipleControlledIdentity(QIRProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
template <bool IntoRegister = false> Value x(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
template <bool IntoRegister = false>
Value singleControlledX(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
template <bool IntoRegister = false>
Value multipleControlledX(QIRProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
template <bool IntoRegister = false> Value y(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
template <bool IntoRegister = false>
Value singleControlledY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
template <bool IntoRegister = false>
Value multipleControlledY(QIRProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
template <bool IntoRegister = false> Value z(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
template <bool IntoRegister = false>
Value singleControlledZ(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
template <bool IntoRegister = false>
Value multipleControlledZ(QIRProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
template <bool IntoRegister = false> Value h(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
template <bool IntoRegister = false>
Value singleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
template <bool IntoRegister = false>
Value multipleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
template <bool IntoRegister = false>
Value hWithoutRegister(QIRProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
template <bool IntoRegister = false> Value s(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
template <bool IntoRegister = false>
Value singleControlledS(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
template <bool IntoRegister = false>
Value multipleControlledS(QIRProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
template <bool IntoRegister = false> Value sdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
template <bool IntoRegister = false>
Value singleControlledSdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
template <bool IntoRegister = false>
Value multipleControlledSdg(QIRProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
template <bool IntoRegister = false>
Value t_(QIRProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
template <bool IntoRegister = false>
Value singleControlledT(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
template <bool IntoRegister = false>
Value multipleControlledT(QIRProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
template <bool IntoRegister = false> Value tdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
template <bool IntoRegister = false>
Value singleControlledTdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
template <bool IntoRegister = false>
Value multipleControlledTdg(QIRProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
template <bool IntoRegister = false> Value sx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
template <bool IntoRegister = false>
Value singleControlledSx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
template <bool IntoRegister = false>
Value multipleControlledSx(QIRProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
template <bool IntoRegister = false> Value sxdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
template <bool IntoRegister = false>
Value singleControlledSxdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
template <bool IntoRegister = false>
Value multipleControlledSxdg(QIRProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
template <bool IntoRegister = false> Value rx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
template <bool IntoRegister = false>
Value singleControlledRx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
template <bool IntoRegister = false>
Value multipleControlledRx(QIRProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
template <bool IntoRegister = false> Value ry(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
template <bool IntoRegister = false>
Value singleControlledRy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
template <bool IntoRegister = false>
Value multipleControlledRy(QIRProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
template <bool IntoRegister = false> Value rz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
template <bool IntoRegister = false>
Value singleControlledRz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
template <bool IntoRegister = false>
Value multipleControlledRz(QIRProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
template <bool IntoRegister = false> Value p(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
template <bool IntoRegister = false>
Value singleControlledP(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
template <bool IntoRegister = false>
Value multipleControlledP(QIRProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
template <bool IntoRegister = false> Value r(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
template <bool IntoRegister = false>
Value singleControlledR(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
template <bool IntoRegister = false>
Value multipleControlledR(QIRProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
template <bool IntoRegister = false> Value u2(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
template <bool IntoRegister = false>
Value singleControlledU2(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
template <bool IntoRegister = false>
Value multipleControlledU2(QIRProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
template <bool IntoRegister = false> Value u(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
template <bool IntoRegister = false>
Value singleControlledU(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
template <bool IntoRegister = false>
Value multipleControlledU(QIRProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
template <bool IntoRegister = false> Value swap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
template <bool IntoRegister = false>
Value singleControlledSwap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
template <bool IntoRegister = false>
Value multipleControlledSwap(QIRProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
template <bool IntoRegister = false> Value iswap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
template <bool IntoRegister = false>
Value singleControlledIswap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
template <bool IntoRegister = false>
Value multipleControlledIswap(QIRProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
template <bool IntoRegister = false> Value dcx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
template <bool IntoRegister = false>
Value singleControlledDcx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
template <bool IntoRegister = false>
Value multipleControlledDcx(QIRProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
template <bool IntoRegister = false> Value ecr(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
template <bool IntoRegister = false>
Value singleControlledEcr(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
template <bool IntoRegister = false>
Value multipleControlledEcr(QIRProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
template <bool IntoRegister = false> Value rxx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
template <bool IntoRegister = false>
Value singleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
template <bool IntoRegister = false>
Value multipleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
template <bool IntoRegister = false>
Value tripleControlledRxx(QIRProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
template <bool IntoRegister = false> Value ryy(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
template <bool IntoRegister = false>
Value singleControlledRyy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
template <bool IntoRegister = false>
Value multipleControlledRyy(QIRProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
template <bool IntoRegister = false> Value rzx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
template <bool IntoRegister = false>
Value singleControlledRzx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
template <bool IntoRegister = false>
Value multipleControlledRzx(QIRProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
template <bool IntoRegister = false> Value rzz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
template <bool IntoRegister = false>
Value singleControlledRzz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
template <bool IntoRegister = false>
Value multipleControlledRzz(QIRProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
template <bool IntoRegister = false> Value xxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
template <bool IntoRegister = false>
Value singleControlledXxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
template <bool IntoRegister = false>
Value multipleControlledXxPlusYY(QIRProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
template <bool IntoRegister = false> Value xxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
template <bool IntoRegister = false>
Value singleControlledXxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
template <bool IntoRegister = false>
Value multipleControlledXxMinusYY(QIRProgramBuilder& b);

// --- RCCXOp --------------------------------------------------------------- //

/// Creates a circuit with just an RCCX gate.
template <bool IntoRegister = false> Value rccx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RCCX gate.
template <bool IntoRegister = false>
Value singleControlledRccx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RCCX gate.
template <bool IntoRegister = false>
Value multipleControlledRccx(QIRProgramBuilder& b);

// --- IfOp ----------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
template <bool IntoRegister = false> Value simpleIf(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
template <bool IntoRegister = false> Value ifElse(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
template <bool IntoRegister = false> Value ifTwoQubits(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
template <bool IntoRegister = false>
Value nestedIfOpForLoop(QIRProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
template <bool IntoRegister = false>
Value simpleWhileReset(QIRProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
template <bool IntoRegister = false>
Value simpleDoWhileReset(QIRProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
template <bool IntoRegister = false> Value simpleForLoop(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
template <bool IntoRegister = false>
Value nestedForLoopIfOp(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
template <bool IntoRegister = false>
Value nestedForLoopWhileOp(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
template <bool IntoRegister = false>
Value nestedForLoopCtrlOpWithSeparateQubit(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
template <bool IntoRegister = false>
Value nestedForLoopCtrlOpWithExtractedQubit(QIRProgramBuilder& b);
// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a control modifier applied to two gates.
template <bool IntoRegister = false> Value ctrlTwo(QIRProgramBuilder& b);

} // namespace mlir::qir
