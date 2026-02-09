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

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
inline void allocQubit(mlir::qc::QCProgramBuilder& b) { b.allocQubit(); }

/// Allocates a qubit register of size `2`.
inline void allocQubitRegister(mlir::qc::QCProgramBuilder& b) {
  b.allocQubitRegister(2);
}

/// Allocates two qubit registers of size `2` and `3`.
inline void allocMultipleQubitRegisters(mlir::qc::QCProgramBuilder& b) {
  b.allocQubitRegister(2, "reg0");
  b.allocQubitRegister(3, "reg1");
}

/// Allocates a large qubit register.
inline void allocLargeRegister(mlir::qc::QCProgramBuilder& b) {
  b.allocQubitRegister(100);
}

/// Allocates two inline qubits.
inline void staticQubits(mlir::qc::QCProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
}

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
inline void singleMeasurementToSingleBit(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
}

/// Repeatedly measures a single qubit into the same classical bit.
inline void repeatedMeasurementToSameBit(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
}

/// Repeatedly measures a single qubit into different classical bits.
inline void repeatedMeasurementToDifferentBits(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[1]);
  b.measure(q[0], c[2]);
}

/// Measures multiple qubits into multiple classical bits.
inline void
multipleClassicalRegistersAndMeasurements(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  b.measure(q[0], c0[0]);
  b.measure(q[1], c1[0]);
  b.measure(q[2], c1[1]);
}

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
inline void resetQubitWithoutOp(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.reset(q);
}

/// Resets multiple qubits without any operations being applied.
inline void resetMultipleQubitsWithoutOp(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
}

/// Repeatedly resets a single qubit without any operations being applied.
inline void repeatedResetWithoutOp(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.reset(q);
  b.reset(q);
  b.reset(q);
}

/// Resets a single qubit after a single operation.
inline void resetQubitAfterSingleOp(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.reset(q);
}

/// Resets multiple qubits after a single operation.
inline void resetMultipleQubitsAfterSingleOp(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
}

/// Repeatedly resets a single qubit after a single operation.
inline void repeatedResetAfterSingleOp(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.reset(q);
  b.reset(q);
  b.reset(q);
}

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
inline void globalPhase(mlir::qc::QCProgramBuilder& b) { b.gphase(1.23); }

/// Creates a controlled global phase gate with a single control qubit.
inline void singleControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
}

/// Canonicalized version of `singleControlledGlobalPhase`.
inline void
singleControlledGlobalPhaseCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
}

/// Creates a multi-controlled global phase gate with multiple control qubits.
inline void multipleControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
}

/// Canonicalized version of `multipleControlledGlobalPhase`.
inline void
multipleControlledGlobalPhaseCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with an inverse modifier applied to a global phase gate.
inline void inverseGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  // The angle needs to be created outside of the `inv` block to avoid failing
  // the verifier for the `inv` block that must only contain two operations.
  auto angle = b.doubleConstant(1.23);
  b.inv([&]() { b.gphase(angle); });
}

/// Canonicalized version of `inverseGlobalPhase`.
inline void inverseGlobalPhaseCanonicalized(mlir::qc::QCProgramBuilder& b) {
  b.gphase(-1.23);
}

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
inline void identity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
}

/// Creates a controlled identity gate with a single control qubit.
inline void singleControlledIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[0], q[1]);
}

/// Canonicalized version of `singleControlledIdentity`.
inline void
singleControlledIdentityCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.id(q[1]);
}

/// Creates a multi-controlled identity gate with multiple control qubits.
inline void multipleControlledIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[0], q[1]}, q[2]);
}

/// Canonicalized version of `multipleControlledIdentity`.
inline void
multipleControlledIdentityCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.id(q[2]);
}

/// Creates a circuit with an inverse modifier applied to an identity gate.
inline void inverseIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.id(q[0]); });
}

/// Canonicalized version of `inverseIdentity`.
inline void inverseIdentityCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
}

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
inline void x(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

/// Creates a circuit with a single controlled X gate.
inline void singleControlledX(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled X gate.
inline void multipleControlledX(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled X gate.
inline void nestedControlledX(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cx(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled X gate.
inline void trivialControlledX(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an X gate.
inline void inverseX(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.x(q[0]); });
}

/// Canonicalized version of `inverseX`.
inline void inverseXCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
inline void inverseMultipleControlledX(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcx({q[0], q[1]}, q[2]); });
}

/// Canonicalized version of `inverseMultipleControlledX`.
inline void
inverseMultipleControlledXCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
}

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
inline void y(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
}

/// Creates a circuit with a single controlled Y gate.
inline void singleControlledY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Y gate.
inline void multipleControlledY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Y gate.
inline void nestedControlledY(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cy(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled Y gate.
inline void trivialControlledY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a Y gate.
inline void inverseY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.y(q[0]); });
}

/// Canonicalized version of `inverseY`.
inline void inverseYCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
}

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
inline void inverseMultipleControlledY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcy({q[0], q[1]}, q[2]); });
}

/// Canonicalized version of `inverseMultipleControlledY`.
inline void
inverseMultipleControlledYCanonicalized(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
}
