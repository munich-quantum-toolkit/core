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

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <utility>

namespace mlir::qir {
class QIRProgramBuilder;

/// Creates an empty QIR program.
template <bool IntoRegister = true>
std::pair<Value, Type> emptyQIR(QIRProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
template <bool IntoRegister = true>
std::pair<Value, Type> allocQubit(QIRProgramBuilder& b);

/// Allocates a qubit register of size `1`.
template <bool IntoRegister = true>
std::pair<Value, Type> alloc1QubitRegister(QIRProgramBuilder& b);

/// Allocates a qubit register of size `2`.
template <bool IntoRegister = true>
std::pair<Value, Type> allocQubitRegister(QIRProgramBuilder& b);

/// Allocates a qubit register of size `3`.
template <bool IntoRegister = true>
std::pair<Value, Type> alloc3QubitRegister(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
template <bool IntoRegister = true>
std::pair<Value, Type> allocMultipleQubitRegisters(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3` and applies operations.
template <bool IntoRegister = true>
std::pair<Value, Type> allocMultipleQubitRegistersWithOps(QIRProgramBuilder& b);

/// Allocates a large qubit register.
template <bool IntoRegister = true>
std::pair<Value, Type> allocLargeRegister(QIRProgramBuilder& b);

/// Allocates two inline qubits.
std::pair<Value, Type> staticQubits(QIRProgramBuilder& b);

/// Allocates two static qubits and applies operations.
std::pair<Value, Type> staticQubitsWithOps(QIRProgramBuilder& b);

/// Allocates two static qubits and applies parametric gates.
std::pair<Value, Type> staticQubitsWithParametricOps(QIRProgramBuilder& b);

/// Allocates two static qubits and applies a two-target gate.
std::pair<Value, Type> staticQubitsWithTwoTargetOps(QIRProgramBuilder& b);

/// Allocates two static qubits and applies a controlled gate.
std::pair<Value, Type> staticQubitsWithCtrl(QIRProgramBuilder& b);

/// Allocates a static qubit and applies the inverse of a T gate (Tdg).
std::pair<Value, Type> staticQubitsWithInv(QIRProgramBuilder& b);

/// Allocates duplicate static qubits and applies operations on both.
std::pair<Value, Type> staticQubitsWithDuplicates(QIRProgramBuilder& b);

/// Same as `staticQubitsWithDuplicates`, but with canonical static qubit
/// retrievals.
std::pair<Value, Type> staticQubitsCanonical(QIRProgramBuilder& b);

// --- Invalid / mixed addressing (unit tests) --------------------------------

/// @pre `builder.initialize()`. Fatal mixed addressing: static then dynamic
/// alloc.
std::pair<Value, Type> mixedStaticThenDynamicQubit(QIRProgramBuilder& b);

/// @pre `builder.initialize()`. Fatal mixed addressing: dynamic register then
/// static.
std::pair<Value, Type>
mixedDynamicRegisterThenStaticQubit(QIRProgramBuilder& b);

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
template <bool IntoRegister = true>
std::pair<Value, Type> singleMeasurementToSingleBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
template <bool IntoRegister = true>
std::pair<Value, Type> repeatedMeasurementToSameBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
template <bool IntoRegister = true>
std::pair<Value, Type> repeatedMeasurementToDifferentBits(QIRProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
template <bool IntoRegister = true>
std::pair<Value, Type>
multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
template <bool IntoRegister = true>
std::pair<Value, Type> measurementWithoutRegisters(QIRProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
template <bool IntoRegister = true>
std::pair<Value, Type> resetQubitWithoutOp(QIRProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
template <bool IntoRegister = true>
std::pair<Value, Type> resetMultipleQubitsWithoutOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
template <bool IntoRegister = true>
std::pair<Value, Type> repeatedResetWithoutOp(QIRProgramBuilder& b);

/// Resets a single qubit after a single operation.
template <bool IntoRegister = true>
std::pair<Value, Type> resetQubitAfterSingleOp(QIRProgramBuilder& b);

/// Resets multiple qubits after a single operation.
template <bool IntoRegister = true>
std::pair<Value, Type> resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
template <bool IntoRegister = true>
std::pair<Value, Type> repeatedResetAfterSingleOp(QIRProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
template <bool IntoRegister = true>
std::pair<Value, Type> globalPhase(QIRProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
template <bool IntoRegister = true>
std::pair<Value, Type> identity(QIRProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledIdentity(QIRProgramBuilder& b);

/// Creates an identity gate on a single qubit in a two-qubit register.
template <bool IntoRegister = true>
std::pair<Value, Type> twoQubitsOneIdentity(QIRProgramBuilder& b);

/// Creates an identity gate on a single qubit in a three-qubit register.
template <bool IntoRegister = true>
std::pair<Value, Type> threeQubitsOneIdentity(QIRProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledIdentity(QIRProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
template <bool IntoRegister = true>
std::pair<Value, Type> x(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledX(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledX(QIRProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
template <bool IntoRegister = true>
std::pair<Value, Type> y(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledY(QIRProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
template <bool IntoRegister = true>
std::pair<Value, Type> z(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledZ(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledZ(QIRProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
template <bool IntoRegister = true>
std::pair<Value, Type> h(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
template <bool IntoRegister = true>
std::pair<Value, Type> hWithoutRegister(QIRProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
template <bool IntoRegister = true>
std::pair<Value, Type> s(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledS(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledS(QIRProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> sdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledSdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledSdg(QIRProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
template <bool IntoRegister = true>
std::pair<Value, Type> t_(QIRProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledT(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledT(QIRProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> tdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledTdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledTdg(QIRProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> sx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledSx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledSx(QIRProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> sxdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledSxdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledSxdg(QIRProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> rx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRx(QIRProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> ry(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRy(QIRProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
template <bool IntoRegister = true>
std::pair<Value, Type> rz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRz(QIRProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
template <bool IntoRegister = true>
std::pair<Value, Type> p(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledP(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledP(QIRProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
template <bool IntoRegister = true>
std::pair<Value, Type> r(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledR(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledR(QIRProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
template <bool IntoRegister = true>
std::pair<Value, Type> u2(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledU2(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledU2(QIRProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
template <bool IntoRegister = true>
std::pair<Value, Type> u(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledU(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledU(QIRProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
template <bool IntoRegister = true>
std::pair<Value, Type> swap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledSwap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledSwap(QIRProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
template <bool IntoRegister = true>
std::pair<Value, Type> iswap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledIswap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledIswap(QIRProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> dcx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledDcx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledDcx(QIRProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
template <bool IntoRegister = true>
std::pair<Value, Type> ecr(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledEcr(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledEcr(QIRProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> rxx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> tripleControlledRxx(QIRProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> ryy(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRyy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRyy(QIRProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> rzx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRzx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRzx(QIRProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
template <bool IntoRegister = true>
std::pair<Value, Type> rzz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledRzz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledRzz(QIRProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> xxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledXxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledXxPlusYY(QIRProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> xxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> singleControlledXxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
template <bool IntoRegister = true>
std::pair<Value, Type> multipleControlledXxMinusYY(QIRProgramBuilder& b);

// --- IfOp ----------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
template <bool IntoRegister = true>
std::pair<Value, Type> simpleIf(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
template <bool IntoRegister = true>
std::pair<Value, Type> ifElse(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
template <bool IntoRegister = true>
std::pair<Value, Type> ifTwoQubits(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
template <bool IntoRegister = true>
std::pair<Value, Type> nestedIfOpForLoop(QIRProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
template <bool IntoRegister = true>
std::pair<Value, Type> simpleWhileReset(QIRProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
template <bool IntoRegister = true>
std::pair<Value, Type> simpleDoWhileReset(QIRProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
template <bool IntoRegister = true>
std::pair<Value, Type> simpleForLoop(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
template <bool IntoRegister = true>
std::pair<Value, Type> nestedForLoopIfOp(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
template <bool IntoRegister = true>
std::pair<Value, Type> nestedForLoopWhileOp(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
template <bool IntoRegister = true>
std::pair<Value, Type>
nestedForLoopCtrlOpWithSeparateQubit(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
template <bool IntoRegister = true>
std::pair<Value, Type>
nestedForLoopCtrlOpWithExtractedQubit(QIRProgramBuilder& b);
// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a control modifier applied to two gates.
template <bool IntoRegister = true>
std::pair<Value, Type> ctrlTwo(QIRProgramBuilder& b);

} // namespace mlir::qir
