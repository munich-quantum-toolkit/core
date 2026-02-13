/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir_programs.h"

#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

namespace mlir::qir {

void allocQubit(QIRProgramBuilder& b) { b.allocQubitRegister(1); }

void allocQubitRegister(QIRProgramBuilder& b) { b.allocQubitRegister(2); }

void allocMultipleQubitRegisters(QIRProgramBuilder& b) {
  b.allocQubitRegister(2);
  b.allocQubitRegister(3);
}

void allocLargeRegister(QIRProgramBuilder& b) { b.allocQubitRegister(100); }

void staticQubits(QIRProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
}

void singleMeasurementToSingleBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
}

void repeatedMeasurementToSameBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
}

void repeatedMeasurementToDifferentBits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(3);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[1]);
  b.measure(q[0], c[2]);
}

void multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto c0 = b.allocClassicalBitRegister(1, "c0");
  const auto c1 = b.allocClassicalBitRegister(2, "c1");
  b.measure(q[0], c0[0]);
  b.measure(q[1], c1[0]);
  b.measure(q[2], c1[1]);
}

void resetQubitWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
}

void resetMultipleQubitsWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
}

void repeatedResetWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
}

void resetQubitAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
}

void resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
}

void repeatedResetAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
}

void globalPhase(QIRProgramBuilder& b) { b.gphase(0.123); }

void identity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
}

void singleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[0], q[1]);
}

void multipleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[0], q[1]}, q[2]);
}

void x(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

void singleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
}

void multipleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
}

void y(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
}

void singleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
}

void multipleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
}

void z(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
}

void singleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
}

void multipleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
}

void h(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
}

void singleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
}

void multipleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
}

void s(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
}

void singleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
}

void multipleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
}

void sdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
}

void singleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
}

void multipleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
}

void t_(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
}

void singleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
}

void multipleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
}

void tdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
}

void singleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
}

void multipleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
}

void sx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
}

void singleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
}

void multipleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
}

void sxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
}

void singleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
}

void multipleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
}

void rx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
}

void singleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
}

void multipleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
}

void ry(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
}

void singleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
}

void multipleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
}

void rz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
}

void singleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
}

void multipleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
}

void p(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
}

void singleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
}

void multipleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
}

void r(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
}

void singleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
}

void multipleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
}

void u2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
}

void singleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
}

void multipleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
}

void u(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
}

void singleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
}

void multipleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
}

void swap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
}

void singleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
}

void multipleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
}

void iswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
}

void singleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
}

void multipleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
}

void dcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
}

void singleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
}

void multipleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
}

void ecr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
}

void singleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
}

void multipleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
}

void rxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
}

void singleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
}

void multipleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
}

void tripleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
}

void ryy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
}

void singleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
}

void multipleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
}

void rzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
}

void singleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
}

void multipleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
}

void rzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
}

void singleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
}

void multipleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
}

void xxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
}

void singleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

void multipleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

void xxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
}

void singleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

void multipleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

} // namespace mlir::qir
