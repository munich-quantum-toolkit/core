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

#include <numbers>

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

/// Allocates and explicitly deallocates a single qubit.
inline void allocDeallocPair(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
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
inline void globalPhase(mlir::qc::QCProgramBuilder& b) { b.gphase(0.123); }

/// Creates a controlled global phase gate with a single control qubit.
inline void singleControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
}

/// Creates a multi-controlled global phase gate with multiple control qubits.
inline void multipleControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
}

/// Creates a circuit with a nested controlled global phase gate.
inline void nestedControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cgphase(0.123, reg[1]); });
}

/// Creates a circuit with a trivial controlled global phase gate.
inline void trivialControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcgphase(0.123, {});
}

/// Creates a circuit with an inverse modifier applied to a global phase gate.
inline void inverseGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  b.inv([&]() { b.gphase(-0.123); });
}

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
inline void
inverseMultipleControlledGlobalPhase(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcgphase(-0.123, {q[0], q[1], q[2]}); });
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
  b.cid(q[1], q[0]);
}

/// Creates a multi-controlled identity gate with multiple control qubits.
inline void multipleControlledIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[2], q[1]}, q[0]);
}

/// Creates a circuit with a nested controlled identity gate.
inline void nestedControlledIdentity(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[2], [&] { b.cid(reg[1], reg[0]); });
  ;
}

/// Creates a circuit with a trivial controlled identity gate.
inline void trivialControlledIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an identity gate.
inline void inverseIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.id(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
inline void inverseMultipleControlledIdentity(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcid({q[2], q[1]}, q[0]); });
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

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
inline void inverseMultipleControlledX(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcx({q[0], q[1]}, q[2]); });
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

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
inline void inverseMultipleControlledY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcy({q[0], q[1]}, q[2]); });
}

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
inline void z(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
}

/// Creates a circuit with a single controlled Z gate.
inline void singleControlledZ(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Z gate.
inline void multipleControlledZ(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Z gate.
inline void nestedControlledZ(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cz(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled Z gate.
inline void trivialControlledZ(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a Z gate.
inline void inverseZ(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.z(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
inline void inverseMultipleControlledZ(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcz({q[0], q[1]}, q[2]); });
}

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
inline void h(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
}

/// Creates a circuit with a single controlled H gate.
inline void singleControlledH(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled H gate.
inline void multipleControlledH(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled H gate.
inline void nestedControlledH(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.ch(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled H gate.
inline void trivialControlledH(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an H gate.
inline void inverseH(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.h(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
inline void inverseMultipleControlledH(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mch({q[0], q[1]}, q[2]); });
}

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
inline void s(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
}

/// Creates a circuit with a single controlled S gate.
inline void singleControlledS(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled S gate.
inline void multipleControlledS(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled S gate.
inline void nestedControlledS(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cs(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled S gate.
inline void trivialControlledS(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an S gate.
inline void inverseS(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.s(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
inline void inverseMultipleControlledS(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcs({q[0], q[1]}, q[2]); });
}

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
inline void sdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
}

/// Creates a circuit with a single controlled Sdg gate.
inline void singleControlledSdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Sdg gate.
inline void multipleControlledSdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Sdg gate.
inline void nestedControlledSdg(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.csdg(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled Sdg gate.
inline void trivialControlledSdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
inline void inverseSdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.sdg(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
inline void inverseMultipleControlledSdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcsdg({q[0], q[1]}, q[2]); });
}

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
inline void t_(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
}

/// Creates a circuit with a single controlled T gate.
inline void singleControlledT(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled T gate.
inline void multipleControlledT(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled T gate.
inline void nestedControlledT(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.ct(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled T gate.
inline void trivialControlledT(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a T gate.
inline void inverseT(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.t(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
inline void inverseMultipleControlledT(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mct({q[0], q[1]}, q[2]); });
}

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
inline void tdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
}

/// Creates a circuit with a single controlled Tdg gate.
inline void singleControlledTdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Tdg gate.
inline void multipleControlledTdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Tdg gate.
inline void nestedControlledTdg(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.ctdg(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled Tdg gate.
inline void trivialControlledTdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
inline void inverseTdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.tdg(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
inline void inverseMultipleControlledTdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mctdg({q[0], q[1]}, q[2]); });
}

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
inline void sx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
}

/// Creates a circuit with a single controlled SX gate.
inline void singleControlledSx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled SX gate.
inline void multipleControlledSx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled SX gate.
inline void nestedControlledSx(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.csx(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled SX gate.
inline void trivialControlledSx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an SX gate.
inline void inverseSx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.sx(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
inline void inverseMultipleControlledSx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcsx({q[0], q[1]}, q[2]); });
}

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
inline void sxdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
}

/// Creates a circuit with a single controlled SXdg gate.
inline void singleControlledSxdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled SXdg gate.
inline void multipleControlledSxdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled SXdg gate.
inline void nestedControlledSxdg(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.csxdg(reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled SXdg gate.
inline void trivialControlledSxdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
inline void inverseSxdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.sxdg(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
inline void inverseMultipleControlledSxdg(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcsxdg({q[0], q[1]}, q[2]); });
}

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
inline void rx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
}

/// Creates a circuit with a single controlled RX gate.
inline void singleControlledRx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled RX gate.
inline void multipleControlledRx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled RX gate.
inline void nestedControlledRx(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.crx(0.123, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled RX gate.
inline void trivialControlledRx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an RX gate.
inline void inverseRx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.rx(-0.123, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
inline void inverseMultipleControlledRx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcrx(-0.123, {q[0], q[1]}, q[2]); });
}

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
inline void ry(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
}

/// Creates a circuit with a single controlled RY gate.
inline void singleControlledRy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled RY gate.
inline void multipleControlledRy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled RY gate.
inline void nestedControlledRy(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cry(0.456, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled RY gate.
inline void trivialControlledRy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an RY gate.
inline void inverseRy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.ry(-0.456, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
inline void inverseMultipleControlledRy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcry(-0.456, {q[0], q[1]}, q[2]); });
}

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
inline void rz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
}

/// Creates a circuit with a single controlled RZ gate.
inline void singleControlledRz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled RZ gate.
inline void multipleControlledRz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled RZ gate.
inline void nestedControlledRz(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.crz(0.789, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled RZ gate.
inline void trivialControlledRz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an RZ gate.
inline void inverseRz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.rz(-0.789, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
inline void inverseMultipleControlledRz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcrz(-0.789, {q[0], q[1]}, q[2]); });
}

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
inline void p(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
}

/// Creates a circuit with a single controlled P gate.
inline void singleControlledP(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled P gate.
inline void multipleControlledP(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled P gate.
inline void nestedControlledP(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cp(0.123, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled P gate.
inline void trivialControlledP(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a P gate.
inline void inverseP(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.p(-0.123, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
inline void inverseMultipleControlledP(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcp(-0.123, {q[0], q[1]}, q[2]); });
}

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
inline void r(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
}

/// Creates a circuit with a single controlled R gate.
inline void singleControlledR(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled R gate.
inline void multipleControlledR(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled R gate.
inline void nestedControlledR(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cr(0.123, 0.456, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled R gate.
inline void trivialControlledR(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an R gate.
inline void inverseR(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.r(-0.123, 0.456, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
inline void inverseMultipleControlledR(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcr(-0.123, 0.456, {q[0], q[1]}, q[2]); });
}

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
inline void u2(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
}

/// Creates a circuit with a single controlled U2 gate.
inline void singleControlledU2(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled U2 gate.
inline void multipleControlledU2(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled U2 gate.
inline void nestedControlledU2(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cu2(0.234, 0.567, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled U2 gate.
inline void trivialControlledU2(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a U2 gate.
inline void inverseU2(mlir::qc::QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.u2(-0.567 + pi, -0.234 - pi, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
inline void inverseMultipleControlledU2(mlir::qc::QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcu2(-0.567 + pi, -0.234 - pi, {q[0], q[1]}, q[2]); });
}

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
inline void u(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
}

/// Creates a circuit with a single controlled U gate.
inline void singleControlledU(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled U gate.
inline void multipleControlledU(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled U gate.
inline void nestedControlledU(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cu(0.1, 0.2, 0.3, reg[1], reg[2]); });
}

/// Creates a circuit with a trivial controlled U gate.
inline void trivialControlledU(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a U gate.
inline void inverseU(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.u(-0.1, -0.3, -0.2, q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
inline void inverseMultipleControlledU(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcu(-0.1, -0.3, -0.2, {q[0], q[1]}, q[2]); });
}

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
inline void swap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
}

/// Creates a circuit with a single controlled SWAP gate.
inline void singleControlledSwap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled SWAP gate.
inline void multipleControlledSwap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled SWAP gate.
inline void nestedControlledSwap(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cswap(reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled SWAP gate.
inline void trivialControlledSwap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
inline void inverseSwap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.swap(q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
inline void inverseMultipleControlledSwap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcswap({q[0], q[1]}, q[2], q[3]); });
}

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
inline void iswap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
}

/// Creates a circuit with a single controlled iSWAP gate.
inline void singleControlledIswap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled iSWAP gate.
inline void multipleControlledIswap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled iSWAP gate.
inline void nestedControlledIswap(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.ciswap(reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled iSWAP gate.
inline void trivialControlledIswap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
inline void inverseIswap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.iswap(q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
inline void inverseMultipleControlledIswap(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mciswap({q[0], q[1]}, q[2], q[3]); });
}

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
inline void dcx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
}

/// Creates a circuit with a single controlled DCX gate.
inline void singleControlledDcx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled DCX gate.
inline void multipleControlledDcx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled DCX gate.
inline void nestedControlledDcx(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cdcx(reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled DCX gate.
inline void trivialControlledDcx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to a DCX gate.
inline void inverseDcx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.dcx(q[1], q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
inline void inverseMultipleControlledDcx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcdcx({q[0], q[1]}, q[3], q[2]); });
}

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
inline void ecr(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
}

/// Creates a circuit with a single controlled ECR gate.
inline void singleControlledEcr(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled ECR gate.
inline void multipleControlledEcr(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled ECR gate.
inline void nestedControlledEcr(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cecr(reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled ECR gate.
inline void trivialControlledEcr(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an ECR gate.
inline void inverseEcr(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.ecr(q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
inline void inverseMultipleControlledEcr(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcecr({q[0], q[1]}, q[2], q[3]); });
}

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
inline void rxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RXX gate.
inline void singleControlledRxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RXX gate.
inline void multipleControlledRxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RXX gate.
inline void nestedControlledRxx(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.crxx(0.123, reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled RXX gate.
inline void trivialControlledRxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RXX gate.
inline void inverseRxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.rxx(-0.123, q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
inline void inverseMultipleControlledRxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcrxx(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

/// Creates a circuit with a triple-controlled RXX gate.
inline void tripleControlledRxx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
}

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
inline void ryy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RYY gate.
inline void singleControlledRyy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RYY gate.
inline void multipleControlledRyy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RYY gate.
inline void nestedControlledRyy(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cryy(0.123, reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled RYY gate.
inline void trivialControlledRyy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RYY gate.
inline void inverseRyy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.ryy(-0.123, q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
inline void inverseMultipleControlledRyy(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcryy(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
inline void rzx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RZX gate.
inline void singleControlledRzx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RZX gate.
inline void multipleControlledRzx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RZX gate.
inline void nestedControlledRzx(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.crzx(0.123, reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled RZX gate.
inline void trivialControlledRzx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RZX gate.
inline void inverseRzx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.rzx(-0.123, q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
inline void inverseMultipleControlledRzx(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcrzx(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
inline void rzz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RZZ gate.
inline void singleControlledRzz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RZZ gate.
inline void multipleControlledRzz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RZZ gate.
inline void nestedControlledRzz(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.crzz(0.123, reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled RZZ gate.
inline void trivialControlledRzz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
inline void inverseRzz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.rzz(-0.123, q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
inline void inverseMultipleControlledRzz(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcrzz(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
inline void xxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
}

/// Creates a circuit with a single controlled XXPlusYY gate.
inline void singleControlledXxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled XXPlusYY gate.
inline void multipleControlledXxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled XXPlusYY gate.
inline void nestedControlledXxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cxx_plus_yy(0.123, 0.456, reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled XXPlusYY gate.
inline void trivialControlledXxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
inline void inverseXxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.xx_plus_yy(-0.123, 0.456, q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
inline void inverseMultipleControlledXxPlusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcxx_plus_yy(-0.123, 0.456, {q[0], q[1]}, q[2], q[3]); });
}

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
inline void xxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
}

/// Creates a circuit with a single controlled XXMinusYY gate.
inline void singleControlledXxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled XXMinusYY gate.
inline void multipleControlledXxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled XXMinusYY gate.
inline void nestedControlledXxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cxx_minus_yy(0.123, 0.456, reg[1], reg[2], reg[3]); });
}

/// Creates a circuit with a trivial controlled XXMinusYY gate.
inline void trivialControlledXxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
inline void inverseXxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.xx_minus_yy(-0.123, 0.456, q[0], q[1]); });
}

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
inline void inverseMultipleControlledXxMinusYY(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcxx_minus_yy(-0.123, 0.456, {q[0], q[1]}, q[2], q[3]); });
}

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
inline void barrier(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.barrier(q[0]);
}

/// Creates a circuit with a barrier on two qubits.
inline void barrierTwoQubits(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier({q[0], q[1]});
}

/// Creates a circuit with a barrier on multiple qubits.
inline void barrierMultipleQubits(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.barrier({q[0], q[1], q[2]});
}

/// Creates a circuit with a single controlled barrier.
inline void singleControlledBarrier(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[1], [&] { b.barrier(q[0]); });
}

/// Creates a circuit with an inverse modifier applied to a barrier.
inline void inverseBarrier(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.barrier(q[0]); });
}

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
inline void trivialCtrl(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, [&]() { b.rxx(0.123, q[0], q[1]); });
}

/// Creates a circuit with nested ctrl modifiers.
inline void nestedCtrl(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], [&]() { b.ctrl(q[1], [&]() { b.rxx(0.123, q[2], q[3]); }); });
}

/// Creates a circuit with triple nested ctrl modifiers.
inline void tripleNestedCtrl(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.ctrl(q[0], [&]() {
    b.ctrl(q[1], [&]() { b.ctrl(q[2], [&]() { b.rxx(0.123, q[3], q[4]); }); });
  });
}

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
inline void ctrlInvSandwich(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], [&]() {
    b.inv([&]() { b.ctrl(q[1], [&]() { b.rxx(-0.123, q[2], q[3]); }); });
  });
}

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with nested inverse modifiers.
inline void nestedInv(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.inv([&]() { b.rxx(0.123, q[0], q[1]); }); });
}

/// Creates a circuit with triple nested inverse modifiers.
inline void tripleNestedInv(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv(
      [&]() { b.inv([&]() { b.inv([&]() { b.rxx(-0.123, q[0], q[1]); }); }); });
}

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
inline void invCtrlSandwich(mlir::qc::QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() {
    b.ctrl(q[0], [&]() { b.inv([&]() { b.rxx(0.123, q[1], q[2]); }); });
  });
}
