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
std::pair<Value, Type> emptyQIR(QIRProgramBuilder& builder);

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
std::pair<Value, Type> allocQubit(QIRProgramBuilder& b);

/// Allocates a qubit register of size `2`.
std::pair<Value, Type> allocQubitRegister(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3`.
std::pair<Value, Type> allocMultipleQubitRegisters(QIRProgramBuilder& b);

/// Allocates two qubit registers of size `2` and `3` and applies operations.
std::pair<Value, Type> allocMultipleQubitRegistersWithOps(QIRProgramBuilder& b);

/// Allocates a large qubit register.
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
std::pair<Value, Type> singleMeasurementToSingleBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into the same classical bit.
std::pair<Value, Type> repeatedMeasurementToSameBit(QIRProgramBuilder& b);

/// Repeatedly measures a single qubit into different classical bits.
std::pair<Value, Type> repeatedMeasurementToDifferentBits(QIRProgramBuilder& b);

/// Measures multiple qubits into multiple classical bits.
std::pair<Value, Type>
multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b);

/// Measures a single qubit into a single classical bit, without explicitly
/// allocating a quantum or classical register.
std::pair<Value, Type> measurementWithoutRegisters(QIRProgramBuilder& b);

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
std::pair<Value, Type> resetQubitWithoutOp(QIRProgramBuilder& b);

/// Resets multiple qubits without any operations being applied.
std::pair<Value, Type> resetMultipleQubitsWithoutOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit without any operations being applied.
std::pair<Value, Type> repeatedResetWithoutOp(QIRProgramBuilder& b);

/// Resets a single qubit after a single operation.
std::pair<Value, Type> resetQubitAfterSingleOp(QIRProgramBuilder& b);

/// Resets multiple qubits after a single operation.
std::pair<Value, Type> resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b);

/// Repeatedly resets a single qubit after a single operation.
std::pair<Value, Type> repeatedResetAfterSingleOp(QIRProgramBuilder& b);

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
std::pair<Value, Type> globalPhase(QIRProgramBuilder& b);

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
std::pair<Value, Type> identity(QIRProgramBuilder& b);

/// Creates a controlled identity gate with a single control qubit.
std::pair<Value, Type> singleControlledIdentity(QIRProgramBuilder& b);

/// Creates a multi-controlled identity gate with multiple control qubits.
std::pair<Value, Type> multipleControlledIdentity(QIRProgramBuilder& b);

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
std::pair<Value, Type> x(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled X gate.
std::pair<Value, Type> singleControlledX(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled X gate.
std::pair<Value, Type> multipleControlledX(QIRProgramBuilder& b);

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
std::pair<Value, Type> y(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Y gate.
std::pair<Value, Type> singleControlledY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Y gate.
std::pair<Value, Type> multipleControlledY(QIRProgramBuilder& b);

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
std::pair<Value, Type> z(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Z gate.
std::pair<Value, Type> singleControlledZ(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Z gate.
std::pair<Value, Type> multipleControlledZ(QIRProgramBuilder& b);

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
std::pair<Value, Type> h(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled H gate.
std::pair<Value, Type> singleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled H gate.
std::pair<Value, Type> multipleControlledH(QIRProgramBuilder& b);

/// Creates a circuit with just an H gate and no qubit register.
std::pair<Value, Type> hWithoutRegister(QIRProgramBuilder& b);

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
std::pair<Value, Type> s(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled S gate.
std::pair<Value, Type> singleControlledS(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled S gate.
std::pair<Value, Type> multipleControlledS(QIRProgramBuilder& b);

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
std::pair<Value, Type> sdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Sdg gate.
std::pair<Value, Type> singleControlledSdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Sdg gate.
std::pair<Value, Type> multipleControlledSdg(QIRProgramBuilder& b);

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
std::pair<Value, Type> t_(QIRProgramBuilder& b); // NOLINT(*-identifier-naming)

/// Creates a circuit with a single controlled T gate.
std::pair<Value, Type> singleControlledT(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled T gate.
std::pair<Value, Type> multipleControlledT(QIRProgramBuilder& b);

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
std::pair<Value, Type> tdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled Tdg gate.
std::pair<Value, Type> singleControlledTdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled Tdg gate.
std::pair<Value, Type> multipleControlledTdg(QIRProgramBuilder& b);

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
std::pair<Value, Type> sx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SX gate.
std::pair<Value, Type> singleControlledSx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SX gate.
std::pair<Value, Type> multipleControlledSx(QIRProgramBuilder& b);

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
std::pair<Value, Type> sxdg(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SXdg gate.
std::pair<Value, Type> singleControlledSxdg(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SXdg gate.
std::pair<Value, Type> multipleControlledSxdg(QIRProgramBuilder& b);

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
std::pair<Value, Type> rx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RX gate.
std::pair<Value, Type> singleControlledRx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RX gate.
std::pair<Value, Type> multipleControlledRx(QIRProgramBuilder& b);

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
std::pair<Value, Type> ry(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RY gate.
std::pair<Value, Type> singleControlledRy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RY gate.
std::pair<Value, Type> multipleControlledRy(QIRProgramBuilder& b);

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
std::pair<Value, Type> rz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZ gate.
std::pair<Value, Type> singleControlledRz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZ gate.
std::pair<Value, Type> multipleControlledRz(QIRProgramBuilder& b);

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
std::pair<Value, Type> p(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled P gate.
std::pair<Value, Type> singleControlledP(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled P gate.
std::pair<Value, Type> multipleControlledP(QIRProgramBuilder& b);

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
std::pair<Value, Type> r(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled R gate.
std::pair<Value, Type> singleControlledR(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled R gate.
std::pair<Value, Type> multipleControlledR(QIRProgramBuilder& b);

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
std::pair<Value, Type> u2(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U2 gate.
std::pair<Value, Type> singleControlledU2(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U2 gate.
std::pair<Value, Type> multipleControlledU2(QIRProgramBuilder& b);

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
std::pair<Value, Type> u(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled U gate.
std::pair<Value, Type> singleControlledU(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled U gate.
std::pair<Value, Type> multipleControlledU(QIRProgramBuilder& b);

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
std::pair<Value, Type> swap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled SWAP gate.
std::pair<Value, Type> singleControlledSwap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled SWAP gate.
std::pair<Value, Type> multipleControlledSwap(QIRProgramBuilder& b);

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
std::pair<Value, Type> iswap(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled iSWAP gate.
std::pair<Value, Type> singleControlledIswap(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled iSWAP gate.
std::pair<Value, Type> multipleControlledIswap(QIRProgramBuilder& b);

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
std::pair<Value, Type> dcx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled DCX gate.
std::pair<Value, Type> singleControlledDcx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled DCX gate.
std::pair<Value, Type> multipleControlledDcx(QIRProgramBuilder& b);

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
std::pair<Value, Type> ecr(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled ECR gate.
std::pair<Value, Type> singleControlledEcr(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled ECR gate.
std::pair<Value, Type> multipleControlledEcr(QIRProgramBuilder& b);

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
std::pair<Value, Type> rxx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RXX gate.
std::pair<Value, Type> singleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RXX gate.
std::pair<Value, Type> multipleControlledRxx(QIRProgramBuilder& b);

/// Creates a circuit with a triple-controlled RXX gate.
std::pair<Value, Type> tripleControlledRxx(QIRProgramBuilder& b);

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
std::pair<Value, Type> ryy(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RYY gate.
std::pair<Value, Type> singleControlledRyy(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RYY gate.
std::pair<Value, Type> multipleControlledRyy(QIRProgramBuilder& b);

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
std::pair<Value, Type> rzx(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZX gate.
std::pair<Value, Type> singleControlledRzx(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZX gate.
std::pair<Value, Type> multipleControlledRzx(QIRProgramBuilder& b);

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
std::pair<Value, Type> rzz(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled RZZ gate.
std::pair<Value, Type> singleControlledRzz(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled RZZ gate.
std::pair<Value, Type> multipleControlledRzz(QIRProgramBuilder& b);

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
std::pair<Value, Type> xxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXPlusYY gate.
std::pair<Value, Type> singleControlledXxPlusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXPlusYY gate.
std::pair<Value, Type> multipleControlledXxPlusYY(QIRProgramBuilder& b);

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
std::pair<Value, Type> xxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a single controlled XXMinusYY gate.
std::pair<Value, Type> singleControlledXxMinusYY(QIRProgramBuilder& b);

/// Creates a circuit with a multi-controlled XXMinusYY gate.
std::pair<Value, Type> multipleControlledXxMinusYY(QIRProgramBuilder& b);

// --- IfOp ----------------------------------------------------------------- //

/// Creates a circuit with a simple if operation with one qubit.
std::pair<Value, Type> simpleIf(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with an else branch.
std::pair<Value, Type> ifElse(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with two qubits.
std::pair<Value, Type> ifTwoQubits(QIRProgramBuilder& b);

/// Creates a circuit with an if operation with a nested for operation with
/// a register.
std::pair<Value, Type> nestedIfOpForLoop(QIRProgramBuilder& b);

// --- WhileOp -------------------------------------------------------------- //

/// Creates a circuit with a while operation using a while loop.
std::pair<Value, Type> simpleWhileReset(QIRProgramBuilder& b);

/// Creates a circuit with a while operation using a do-while loop.
std::pair<Value, Type> simpleDoWhileReset(QIRProgramBuilder& b);

// --- ForOp ---------------------------------------------------------------- //

/// Creates a circuit with a simple for operation with a register.
std::pair<Value, Type> simpleForLoop(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested if operation.
std::pair<Value, Type> nestedForLoopIfOp(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a nested while
/// operation.
std::pair<Value, Type> nestedForLoopWhileOp(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is separately allocated from the
/// register.
std::pair<Value, Type>
nestedForLoopCtrlOpWithSeparateQubit(QIRProgramBuilder& b);

/// Creates a circuit with a for operation with a register and a qubit and a
/// nested ctrl operation where the qubit is extracted from the register.
std::pair<Value, Type>
nestedForLoopCtrlOpWithExtractedQubit(QIRProgramBuilder& b);
// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a control modifier applied to two gates.
std::pair<Value, Type> ctrlTwo(QIRProgramBuilder& b);

} // namespace mlir::qir
