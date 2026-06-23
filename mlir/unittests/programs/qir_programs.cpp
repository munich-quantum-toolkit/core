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

#include <numbers>

namespace mlir::qir {

std::pair<SmallVector<Value>, SmallVector<Type>>
emptyQIR([[maybe_unused]] QIRProgramBuilder& builder) {
  return {{builder.intConstant(0)}, {builder.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubit(QIRProgramBuilder& b) {
  b.allocQubitRegister(1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubitRegister(QIRProgramBuilder& b) {
  b.allocQubitRegister(2);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegisters(QIRProgramBuilder& b) {
  b.allocQubitRegister(2);
  b.allocQubitRegister(3);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegistersWithOps(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  b.h(q0[0]);
  b.h(q0[1]);
  b.h(q1[0]);
  b.h(q1[1]);
  b.h(q1[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocLargeRegister(QIRProgramBuilder& b) {
  b.allocQubitRegister(100);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubits(QIRProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.h(q0);
  b.h(q1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithParametricOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithTwoTargetOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rzz(0.123, q0, q1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithCtrl(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithInv(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  b.tdg(q0);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithDuplicates(QIRProgramBuilder& b) {
  auto q0a = b.staticQubit(0);
  auto q1a = b.staticQubit(1);
  auto q0b = b.staticQubit(0);
  auto q1b = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0a);
  b.p(std::numbers::pi / 2., q1a);
  b.rzz(0.123, q0b, q1b);
  b.cx(q0b, q1b);
  b.tdg(q0a);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsCanonical(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  b.rzz(0.123, q0, q1);
  b.cx(q0, q1);
  b.tdg(q0);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
mixedStaticThenDynamicQubit(QIRProgramBuilder& b) {
  b.staticQubit(0);
  b.allocQubit();
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
mixedDynamicRegisterThenStaticQubit(QIRProgramBuilder& b) {
  b.allocQubitRegister(2);
  b.staticQubit(0);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleMeasurementToSingleBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToSameBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToDifferentBits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(3);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[1]);
  b.measure(q[0], c[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  b.measure(q[0], c0[0]);
  b.measure(q[1], c1[0]);
  b.measure(q[2], c1[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
measurementWithoutRegisters(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.measure(q, 0);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
globalPhase(QIRProgramBuilder& b) {
  b.gphase(0.123);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
identity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> x(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> y(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> z(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> h(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
hWithoutRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> s(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> sdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> t_(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> tdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> sx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> sxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> rx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> ry(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> rz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> p(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> r(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> u2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> u(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> swap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> iswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> dcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> ecr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> rxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> ryy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> rzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> rzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
xxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
xxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleIf(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0], 0);
  b.scfIf(cond, [&] { b.x(q[0]); });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> ifElse(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0], 0);
  b.scfIf(cond, [&] { b.x(q[0]); }, [&] { b.z(q[0]); });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ifTwoQubits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto cond = b.measure(q[0], 0);
  b.scfIf(cond, [&] {
    b.x(q[0]);
    b.x(q[1]);
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedIfOpForLoop(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q0 = b.allocQubit();
  b.h(q0);
  auto cond = b.measure(q0, 0);
  b.scfIf(
      cond, [&] { b.h(q0); },
      [&] {
        b.scfFor(0, 3, 1, [&](Value iv) {
          auto q1 = b.load(reg.value, iv);
          b.h(q1);
        });
      });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleWhileReset(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.scfWhile(
      [&] {
        auto measureResult = b.measure(q, 0);
        return measureResult;
      },
      [&] { b.h(q); });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleDoWhileReset(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.scfWhile([&] {
    b.h(q);
    auto measureResult = b.measure(q, 0);
    return measureResult;
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleForLoop(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.load(reg.value, iv);
    b.h(q);
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
};

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopIfOp(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto qCond = b.allocQubit();
  b.scfFor(0, 2, 1, [&](Value iv) {
    b.h(qCond);
    auto cond = b.measure(qCond, 0);
    b.scfIf(cond, [&] {
      auto q = b.load(reg.value, iv);
      b.h(q);
    });
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopWhileOp(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.load(reg.value, iv);
    b.h(q);
  });
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.load(reg.value, iv);
    b.scfWhile(
        [&] {
          auto measureResult = b.measure(q, 0);
          return measureResult;
        },
        [&] { b.h(q); });
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithSeparateQubit(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control = b.allocQubit();
  b.h(control);
  b.scfFor(0, 3, 1, [&](Value iv) {
    auto q0 = b.load(reg.value, iv);
    b.h(q0);
    b.cx(control, q0);
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithExtractedQubit(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.h(reg[0]);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto q0 = b.load(reg.value, iv);
    b.h(q0);
    b.cx(reg[0], q0);
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>> ctrlTwo(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcx({q[0], q[1]}, q[2]);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

} // namespace mlir::qir
