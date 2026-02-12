/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qc_programs.h"

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <numbers>

namespace mlir::qc {

void allocQubit(QCProgramBuilder& b) { b.allocQubit(); }

void allocQubitRegister(QCProgramBuilder& b) { b.allocQubitRegister(2); }

void allocMultipleQubitRegisters(QCProgramBuilder& b) {
  b.allocQubitRegister(2, "reg0");
  b.allocQubitRegister(3, "reg1");
}

void allocLargeRegister(QCProgramBuilder& b) { b.allocQubitRegister(100); }

void staticQubits(QCProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
}

void allocDeallocPair(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
}

void singleMeasurementToSingleBit(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
}

void repeatedMeasurementToSameBit(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
}

void repeatedMeasurementToDifferentBits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[1]);
  b.measure(q[0], c[2]);
}

void multipleClassicalRegistersAndMeasurements(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  b.measure(q[0], c0[0]);
  b.measure(q[1], c1[0]);
  b.measure(q[2], c1[1]);
}

void resetQubitWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.reset(q);
}

void resetMultipleQubitsWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
}

void repeatedResetWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.reset(q);
  b.reset(q);
  b.reset(q);
}

void resetQubitAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.reset(q);
}

void resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
}

void repeatedResetAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.reset(q);
  b.reset(q);
  b.reset(q);
}

void globalPhase(QCProgramBuilder& b) { b.gphase(0.123); }

void singleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
}

void multipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
}

void nestedControlledGlobalPhase(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cgphase(0.123, reg[1]); });
}

void trivialControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcgphase(0.123, {});
}

void inverseGlobalPhase(QCProgramBuilder& b) {
  b.inv([&]() { b.gphase(-0.123); });
}

void inverseMultipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcgphase(-0.123, {q[0], q[1], q[2]}); });
}

void identity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
}

void singleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[1], q[0]);
}

void multipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[2], q[1]}, q[0]);
}

void nestedControlledIdentity(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[2], [&] { b.cid(reg[1], reg[0]); });
}

void trivialControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
}

void inverseIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.id(q[0]); });
}

void inverseMultipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcid({q[2], q[1]}, q[0]); });
}

void x(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

void singleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
}

void multipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
}

void nestedControlledX(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cx(reg[1], reg[2]); });
}

void trivialControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
}

void inverseX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.x(q[0]); });
}

void inverseMultipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcx({q[0], q[1]}, q[2]); });
}

void y(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
}

void singleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
}

void multipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
}

void nestedControlledY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cy(reg[1], reg[2]); });
}

void trivialControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
}

void inverseY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.y(q[0]); });
}

void inverseMultipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcy({q[0], q[1]}, q[2]); });
}

void z(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
}

void singleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
}

void multipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
}

void nestedControlledZ(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cz(reg[1], reg[2]); });
}

void trivialControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
}

void inverseZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.z(q[0]); });
}

void inverseMultipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcz({q[0], q[1]}, q[2]); });
}

void h(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
}

void singleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
}

void multipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
}

void nestedControlledH(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.ch(reg[1], reg[2]); });
}

void trivialControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
}

void inverseH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.h(q[0]); });
}

void inverseMultipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mch({q[0], q[1]}, q[2]); });
}

void s(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
}

void singleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
}

void multipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
}

void nestedControlledS(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cs(reg[1], reg[2]); });
}

void trivialControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
}

void inverseS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.s(q[0]); });
}

void inverseMultipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcs({q[0], q[1]}, q[2]); });
}

void sdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
}

void singleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
}

void multipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
}

void nestedControlledSdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.csdg(reg[1], reg[2]); });
}

void trivialControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
}

void inverseSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.sdg(q[0]); });
}

void inverseMultipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcsdg({q[0], q[1]}, q[2]); });
}

void t_(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
}

void singleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
}

void multipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
}

void nestedControlledT(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.ct(reg[1], reg[2]); });
}

void trivialControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
}

void inverseT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.t(q[0]); });
}

void inverseMultipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mct({q[0], q[1]}, q[2]); });
}

void tdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
}

void singleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
}

void multipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
}

void nestedControlledTdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.ctdg(reg[1], reg[2]); });
}

void trivialControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
}

void inverseTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.tdg(q[0]); });
}

void inverseMultipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mctdg({q[0], q[1]}, q[2]); });
}

void sx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
}

void singleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
}

void multipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
}

void nestedControlledSx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.csx(reg[1], reg[2]); });
}

void trivialControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
}

void inverseSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.sx(q[0]); });
}

void inverseMultipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcsx({q[0], q[1]}, q[2]); });
}

void sxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
}

void singleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
}

void multipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
}

void nestedControlledSxdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.csxdg(reg[1], reg[2]); });
}

void trivialControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
}

void inverseSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.sxdg(q[0]); });
}

void inverseMultipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcsxdg({q[0], q[1]}, q[2]); });
}

void rx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
}

void singleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
}

void multipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
}

void nestedControlledRx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.crx(0.123, reg[1], reg[2]); });
}

void trivialControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
}

void inverseRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.rx(-0.123, q[0]); });
}

void inverseMultipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcrx(-0.123, {q[0], q[1]}, q[2]); });
}

void ry(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
}

void singleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
}

void multipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
}

void nestedControlledRy(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cry(0.456, reg[1], reg[2]); });
}

void trivialControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
}

void inverseRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.ry(-0.456, q[0]); });
}

void inverseMultipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcry(-0.456, {q[0], q[1]}, q[2]); });
}

void rz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
}

void singleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
}

void multipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
}

void nestedControlledRz(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.crz(0.789, reg[1], reg[2]); });
}

void trivialControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
}

void inverseRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.rz(-0.789, q[0]); });
}

void inverseMultipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcrz(-0.789, {q[0], q[1]}, q[2]); });
}

void p(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
}

void singleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
}

void multipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
}

void nestedControlledP(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cp(0.123, reg[1], reg[2]); });
}

void trivialControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
}

void inverseP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.p(-0.123, q[0]); });
}

void inverseMultipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcp(-0.123, {q[0], q[1]}, q[2]); });
}

void r(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
}

void singleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
}

void multipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
}

void nestedControlledR(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cr(0.123, 0.456, reg[1], reg[2]); });
}

void trivialControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
}

void inverseR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.r(-0.123, 0.456, q[0]); });
}

void inverseMultipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcr(-0.123, 0.456, {q[0], q[1]}, q[2]); });
}

void u2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
}

void singleControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
}

void multipleControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
}

void nestedControlledU2(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cu2(0.234, 0.567, reg[1], reg[2]); });
}

void trivialControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
}

void inverseU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.u2(-0.567 + pi, -0.234 - pi, q[0]); });
}

void inverseMultipleControlledU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcu2(-0.567 + pi, -0.234 - pi, {q[0], q[1]}, q[2]); });
}

void u(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
}

void singleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
}

void multipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
}

void nestedControlledU(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], [&] { b.cu(0.1, 0.2, 0.3, reg[1], reg[2]); });
}

void trivialControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
}

void inverseU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.u(-0.1, -0.3, -0.2, q[0]); });
}

void inverseMultipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() { b.mcu(-0.1, -0.3, -0.2, {q[0], q[1]}, q[2]); });
}

void swap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
}

void singleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
}

void multipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledSwap(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cswap(reg[1], reg[2], reg[3]); });
}

void trivialControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
}

void inverseSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.swap(q[0], q[1]); });
}

void inverseMultipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcswap({q[0], q[1]}, q[2], q[3]); });
}

void iswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
}

void singleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
}

void multipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledIswap(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.ciswap(reg[1], reg[2], reg[3]); });
}

void trivialControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
}

void inverseIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.iswap(q[0], q[1]); });
}

void inverseMultipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mciswap({q[0], q[1]}, q[2], q[3]); });
}

void dcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
}

void singleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
}

void multipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledDcx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cdcx(reg[1], reg[2], reg[3]); });
}

void trivialControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
}

void inverseDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.dcx(q[1], q[0]); });
}

void inverseMultipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcdcx({q[0], q[1]}, q[3], q[2]); });
}

void ecr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
}

void singleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
}

void multipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledEcr(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cecr(reg[1], reg[2], reg[3]); });
}

void trivialControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
}

void inverseEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.ecr(q[0], q[1]); });
}

void inverseMultipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcecr({q[0], q[1]}, q[2], q[3]); });
}

void rxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
}

void singleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
}

void multipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRxx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.crxx(0.123, reg[1], reg[2], reg[3]); });
}

void trivialControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
}

void inverseRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.rxx(-0.123, q[0], q[1]); });
}

void inverseMultipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcrxx(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

void tripleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
}

void ryy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
}

void singleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
}

void multipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRyy(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cryy(0.123, reg[1], reg[2], reg[3]); });
}

void trivialControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
}

void inverseRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.ryy(-0.123, q[0], q[1]); });
}

void inverseMultipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcryy(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

void rzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
}

void singleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
}

void multipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRzx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.crzx(0.123, reg[1], reg[2], reg[3]); });
}

void trivialControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
}

void inverseRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.rzx(-0.123, q[0], q[1]); });
}

void inverseMultipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcrzx(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

void rzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
}

void singleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
}

void multipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRzz(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.crzz(0.123, reg[1], reg[2], reg[3]); });
}

void trivialControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
}

void inverseRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.rzz(-0.123, q[0], q[1]); });
}

void inverseMultipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcrzz(-0.123, {q[0], q[1]}, q[2], q[3]); });
}

void xxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
}

void singleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

void multipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledXxPlusYY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cxx_plus_yy(0.123, 0.456, reg[1], reg[2], reg[3]); });
}

void trivialControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
}

void inverseXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.xx_plus_yy(-0.123, 0.456, q[0], q[1]); });
}

void inverseMultipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcxx_plus_yy(-0.123, 0.456, {q[0], q[1]}, q[2], q[3]); });
}

void xxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
}

void singleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

void multipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledXxMinusYY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], [&] { b.cxx_minus_yy(0.123, 0.456, reg[1], reg[2], reg[3]); });
}

void trivialControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
}

void inverseXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.xx_minus_yy(-0.123, 0.456, q[0], q[1]); });
}

void inverseMultipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv([&]() { b.mcxx_minus_yy(-0.123, 0.456, {q[0], q[1]}, q[2], q[3]); });
}

void barrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.barrier(q[0]);
}

void barrierTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier({q[0], q[1]});
}

void barrierMultipleQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.barrier({q[0], q[1], q[2]});
}

void singleControlledBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[1], [&] { b.barrier(q[0]); });
}

void inverseBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv([&]() { b.barrier(q[0]); });
}

void trivialCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, [&]() { b.rxx(0.123, q[0], q[1]); });
}

void nestedCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], [&]() { b.ctrl(q[1], [&]() { b.rxx(0.123, q[2], q[3]); }); });
}

void tripleNestedCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.ctrl(q[0], [&]() {
    b.ctrl(q[1], [&]() { b.ctrl(q[2], [&]() { b.rxx(0.123, q[3], q[4]); }); });
  });
}

void ctrlInvSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], [&]() {
    b.inv([&]() { b.ctrl(q[1], [&]() { b.rxx(-0.123, q[2], q[3]); }); });
  });
}

void nestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv([&]() { b.inv([&]() { b.rxx(0.123, q[0], q[1]); }); });
}

void tripleNestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv(
      [&]() { b.inv([&]() { b.inv([&]() { b.rxx(-0.123, q[0], q[1]); }); }); });
}

void invCtrlSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv([&]() {
    b.ctrl(q[0], [&]() { b.inv([&]() { b.rxx(0.123, q[1], q[2]); }); });
  });
}

} // namespace mlir::qc
