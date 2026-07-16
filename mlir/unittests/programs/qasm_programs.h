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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <string>

// NOLINTBEGIN(readability-identifier-naming)
namespace mlir::qasm {

struct OpenQASMProgram {
  llvm::StringRef name;
  llvm::StringRef source;
};

/// OpenQASM programs expected to traverse QC, optimized QCO, reconstructed QC,
/// and Adaptive QIR.
[[nodiscard]] llvm::ArrayRef<OpenQASMProgram> standardPipelinePrograms();

/// OpenQASM programs that additionally round-trip through jeff.
[[nodiscard]] llvm::ArrayRef<OpenQASMProgram> jeffCompatiblePrograms();

/// OpenQASM programs accepted by the standard pipeline but rejected when QCO
/// is converted to jeff.
[[nodiscard]] llvm::ArrayRef<OpenQASMProgram> jeffIncompatiblePrograms();

/// Straight-line compiler programs that additionally support Base QIR.
[[nodiscard]] llvm::ArrayRef<OpenQASMProgram> baseProfilePrograms();

/// Allocates a single qubit.
extern const std::string allocQubit;

/// Allocates a qubit register of size `2`.
extern const std::string allocQubitRegister;

/// Allocates two qubit registers of size `2` and `3`.
extern const std::string allocMultipleQubitRegisters;

/// Allocates a large qubit register.
extern const std::string allocLargeRegister;

/// Measures a single qubit into a single classical bit.
extern const std::string singleMeasurementToSingleBit;

/// Repeatedly measures a single qubit into the same classical bit.
extern const std::string repeatedMeasurementToSameBit;

/// Repeatedly measures a single qubit into different classical bits.
extern const std::string repeatedMeasurementToDifferentBits;

/// Measures multiple qubits into multiple classical bits.
extern const std::string multipleClassicalRegistersAndMeasurements;

/// Resets a single qubit after a single operation.
extern const std::string resetQubitAfterSingleOp;

/// Resets multiple qubits after a single operation.
extern const std::string resetMultipleQubitsAfterSingleOp;

/// Repeatedly resets a single qubit after a single operation.
extern const std::string repeatedResetAfterSingleOp;

/// Creates a circuit with just a global phase.
extern const std::string globalPhase;

/// Creates a circuit with an inverse modifier applied to a global phase gate.
extern const std::string inverseGlobalPhase;

/// Creates a circuit with just an identity gate.
extern const std::string identity;

/// Creates a controlled identity gate with a single control qubit.
extern const std::string singleControlledIdentity;

/// Creates a multi-controlled identity gate with multiple control qubits.
extern const std::string multipleControlledIdentity;

/// Creates a circuit with just an X gate.
extern const std::string x;

/// Creates a circuit with two X gates.
extern const std::string twoX;

/// Creates a circuit with a single controlled X gate.
extern const std::string singleControlledX;

/// Creates a circuit with a single negatively controlled X gate.
extern const std::string singleNegControlledX;

/// Creates a circuit with a multi-controlled X gate.
extern const std::string multipleControlledX;

/// Creates a circuit with a triple-controlled X gate in OpenQASM 2.
extern const std::string tripleControlledXOpenQASM2;

/// Creates a circuit with an X gate that is positively and negatively
/// controlled.
extern const std::string mixedControlledX;

/// Creates a circuit with two X gates that are positively and negatively
/// controlled.
extern const std::string twoMixedControlledX;

/// Creates a circuit with an inverse modifier applied to an X gate.
extern const std::string inverseX;

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
extern const std::string inverseMultipleControlledX;

/// Creates a circuit with just a Y gate.
extern const std::string y;

/// Creates a circuit with a single controlled Y gate.
extern const std::string singleControlledY;

/// Creates a circuit with a multi-controlled Y gate.
extern const std::string multipleControlledY;

/// Creates a circuit with just a Z gate.
extern const std::string z;

/// Creates a circuit with a single controlled Z gate.
extern const std::string singleControlledZ;

/// Creates a circuit with a multi-controlled Z gate.
extern const std::string multipleControlledZ;

/// Creates a circuit with just an H gate.
extern const std::string h;

/// Creates a circuit with a single controlled H gate.
extern const std::string singleControlledH;

/// Creates a circuit with a multi-controlled H gate.
extern const std::string multipleControlledH;

/// Creates a circuit with just an S gate.
extern const std::string s;

/// Creates a circuit with a single controlled S gate.
extern const std::string singleControlledS;

/// Creates a circuit with a multi-controlled S gate.
extern const std::string multipleControlledS;

/// Creates a circuit with just an Sdg gate.
extern const std::string sdg;

/// Creates a circuit with a single controlled Sdg gate.
extern const std::string singleControlledSdg;

/// Creates a circuit with a multi-controlled Sdg gate.
extern const std::string multipleControlledSdg;

/// Creates a circuit with just a T gate.
extern const std::string t_;

/// Creates a circuit with a single controlled T gate.
extern const std::string singleControlledT;

/// Creates a circuit with a multi-controlled T gate.
extern const std::string multipleControlledT;

/// Creates a circuit with just a Tdg gate.
extern const std::string tdg;

/// Creates a circuit with a single controlled Tdg gate.
extern const std::string singleControlledTdg;

/// Creates a circuit with a multi-controlled Tdg gate.
extern const std::string multipleControlledTdg;

/// Creates a circuit with just an SX gate.
extern const std::string sx;

/// Creates a circuit with a single controlled SX gate.
extern const std::string singleControlledSx;

/// Creates a circuit with a multi-controlled SX gate.
extern const std::string multipleControlledSx;

/// Creates a circuit with just an SXdg gate.
extern const std::string sxdg;

/// Creates a circuit with a single controlled SXdg gate.
extern const std::string singleControlledSxdg;

/// Creates a circuit with a multi-controlled SXdg gate.
extern const std::string multipleControlledSxdg;

/// Creates a circuit with just an RX gate.
extern const std::string rx;

/// Creates a circuit with a single controlled RX gate.
extern const std::string singleControlledRx;

/// Creates a circuit with a multi-controlled RX gate.
extern const std::string multipleControlledRx;

/// Creates a circuit with just an RY gate.
extern const std::string ry;

/// Creates a circuit with a single controlled RY gate.
extern const std::string singleControlledRy;

/// Creates a circuit with a multi-controlled RY gate.
extern const std::string multipleControlledRy;

/// Creates a circuit with just an RZ gate.
extern const std::string rz;

/// Creates a circuit with a single controlled RZ gate.
extern const std::string singleControlledRz;

/// Creates a circuit with a multi-controlled RZ gate.
extern const std::string multipleControlledRz;

/// Creates a circuit with just a P gate.
extern const std::string p;

/// Creates a circuit with a single controlled P gate.
extern const std::string singleControlledP;

/// Creates a circuit with a multi-controlled P gate.
extern const std::string multipleControlledP;

/// Creates a circuit with just an R gate.
extern const std::string r;

/// Creates a circuit with a single controlled R gate.
extern const std::string singleControlledR;

/// Creates a circuit with a multi-controlled R gate.
extern const std::string multipleControlledR;

/// Creates a circuit with just a U2 gate.
extern const std::string u2;

/// Creates a circuit with a single controlled U2 gate.
extern const std::string singleControlledU2;

/// Creates a circuit with a multi-controlled U2 gate.
extern const std::string multipleControlledU2;

/// Creates a circuit with just a U gate.
extern const std::string u;

/// Creates a circuit with a single controlled U gate.
extern const std::string singleControlledU;

/// Creates a circuit with a multi-controlled U gate.
extern const std::string multipleControlledU;

/// Creates a circuit with just a SWAP gate.
extern const std::string swap;

/// Creates a circuit with a single controlled SWAP gate.
extern const std::string singleControlledSwap;

/// Creates a circuit with a multi-controlled SWAP gate.
extern const std::string multipleControlledSwap;

/// Creates a circuit with just an iSWAP gate.
extern const std::string iswap;

/// Creates a circuit with a single controlled iSWAP gate.
extern const std::string singleControlledIswap;

/// Creates a circuit with a multi-controlled iSWAP gate.
extern const std::string multipleControlledIswap;

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
extern const std::string inverseIswap;

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
extern const std::string inverseMultipleControlledIswap;

/// Creates a circuit with just a DCX gate.
extern const std::string dcx;

/// Creates a circuit with a single controlled DCX gate.
extern const std::string singleControlledDcx;

/// Creates a circuit with a multi-controlled DCX gate.
extern const std::string multipleControlledDcx;

/// Creates a circuit with just an ECR gate.
extern const std::string ecr;

/// Creates a circuit with a single controlled ECR gate.
extern const std::string singleControlledEcr;

/// Creates a circuit with a multi-controlled ECR gate.
extern const std::string multipleControlledEcr;

/// Creates a circuit with just an RXX gate.
extern const std::string rxx;

/// Creates a circuit with a single controlled RXX gate.
extern const std::string singleControlledRxx;

/// Creates a circuit with a multi-controlled RXX gate.
extern const std::string multipleControlledRxx;

/// Creates a circuit with a triple-controlled RXX gate.
extern const std::string tripleControlledRxx;

/// Creates a circuit with just an RYY gate.
extern const std::string ryy;

/// Creates a circuit with a single controlled RYY gate.
extern const std::string singleControlledRyy;

/// Creates a circuit with a multi-controlled RYY gate.
extern const std::string multipleControlledRyy;

/// Creates a circuit with just an RZX gate.
extern const std::string rzx;

/// Creates a circuit with a single controlled RZX gate.
extern const std::string singleControlledRzx;

/// Creates a circuit with a multi-controlled RZX gate.
extern const std::string multipleControlledRzx;

/// Creates a circuit with just an RZZ gate.
extern const std::string rzz;

/// Creates a circuit with a single controlled RZZ gate.
extern const std::string singleControlledRzz;

/// Creates a circuit with a multi-controlled RZZ gate.
extern const std::string multipleControlledRzz;

/// Creates a circuit with just an XXPlusYY gate.
extern const std::string xxPlusYY;

/// Creates a circuit with a single controlled XXPlusYY gate.
extern const std::string singleControlledXxPlusYY;

/// Creates a circuit with a multi-controlled XXPlusYY gate.
extern const std::string multipleControlledXxPlusYY;

/// Creates a circuit with just an XXMinusYY gate.
extern const std::string xxMinusYY;

/// Creates a circuit with a single controlled XXMinusYY gate.
extern const std::string singleControlledXxMinusYY;

/// Creates a circuit with a multi-controlled XXMinusYY gate.
extern const std::string multipleControlledXxMinusYY;

/// Creates a circuit with a barrier.
extern const std::string barrier;

/// Creates a circuit with a barrier on two qubits.
extern const std::string barrierTwoQubits;

/// Creates a circuit with a barrier on multiple qubits.
extern const std::string barrierMultipleQubits;

/// Creates a circuit with a control modifier applied to two gates.
extern const std::string ctrlTwo;

/// Creates a circuit with a control modifier applied to a controlled and a
/// non-controlled gate.
extern const std::string ctrlTwoMixed;

/// Creates a circuit with a simple if operation with one qubit.
extern const std::string simpleIf;

/// Creates a circuit with an if operation with a negated condition.
extern const std::string ifNot;

/// Creates a circuit with an if operation with two qubits.
extern const std::string ifTwoQubits;

/// Creates a circuit with an if operation with an empty then branch.
extern const std::string ifEmptyThen;

/// Creates a circuit with an if operation with an else branch.
extern const std::string ifElse;

/// Creates an if operation with a nested register loop.
extern const std::string nestedIfOpForLoop;

/// Creates a measurement-controlled while loop.
extern const std::string simpleWhileReset;

/// Applies a gate to every qubit selected by a source-level for loop.
extern const std::string simpleForLoop;

/// Creates a for loop containing a measurement-controlled branch.
extern const std::string nestedForLoopIfOp;

/// Creates two register loops, one containing a while loop.
extern const std::string nestedForLoopWhileOp;

/// Uses a separately allocated control in a register loop.
extern const std::string nestedForLoopCtrlOpWithSeparateQubit;

/// Uses one register element as a control for dynamically selected elements.
extern const std::string nestedForLoopCtrlOpWithExtractedQubit;

/// Broadcasts a controlled gate over a register and scalar qubit.
extern const std::string broadcastRegisterAndQubit;

/// Broadcasts a custom compound gate over a register and scalar qubit.
extern const std::string broadcastCompoundGate;

/// Exercises arithmetic precedence in gate parameters.
extern const std::string expressionArithmetic;

/// Exercises unary negation in a gate parameter.
extern const std::string expressionUnaryMinus;

/// Exercises the built-in pi, tau, and Euler constants.
extern const std::string expressionBuiltinConstants;

/// Exercises scalar mathematical functions in gate parameters.
extern const std::string expressionMathFunctions;

/// Exercises nested scalar mathematical functions.
extern const std::string expressionNestedMathFunctions;

/// Uses a constant floating-point variable as a gate parameter.
extern const std::string expressionConstFloat;

/// Reassigns a mutable floating-point variable used by a gate.
extern const std::string expressionMutableFloat;

/// Exercises constant integer arithmetic used as an index.
extern const std::string expressionConstIntArithmetic;

/// Uses a mutable integer as a dynamic qubit index.
extern const std::string expressionDynamicIntIndex;

/// Uses the integer modulo operator to derive a dynamic qubit index.
extern const std::string expressionModIndex;

/// Creates a conditional from a Boolean literal.
extern const std::string conditionLiteral;

/// Creates a conditional directly from a measurement.
extern const std::string conditionMeasurement;

/// Combines two measurement conditions with logical conjunction.
extern const std::string conditionAnd;

/// Combines two measurement conditions with logical disjunction.
extern const std::string conditionOr;

/// Exercises precedence among negation, conjunction, and disjunction.
extern const std::string conditionNotAndOr;

/// Creates a conditional from a mutable Boolean variable.
extern const std::string conditionBoolVariable;

/// Creates a conditional from an indexed classical bit.
extern const std::string conditionIndexedBit;

/// Combines a while loop with a compound measurement condition.
extern const std::string conditionWhileAnd;

} // namespace mlir::qasm
// NOLINTEND(readability-identifier-naming)
