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
#include "mlir/IR/Value.h"

#include <numbers>

namespace mlir::qc {

void emptyQC([[maybe_unused]] QCProgramBuilder& builder) {}

void allocQubit(QCProgramBuilder& b) { b.allocQubit(); }

void allocQubitRegister(QCProgramBuilder& b) { b.allocQubitRegister(2); }

void allocMultipleQubitRegisters(QCProgramBuilder& b) {
  b.allocQubitRegister(2);
  b.allocQubitRegister(3);
}

void allocMultipleQubitRegistersWithOps(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  b.h(q0[0]);
  b.h(q0[1]);
  b.h(q1[0]);
  b.h(q1[1]);
  b.h(q1[2]);
}

void allocLargeRegister(QCProgramBuilder& b) { b.allocQubitRegister(100); }

void staticQubits(QCProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
}

void staticQubitsWithOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.h(q0);
  b.h(q1);
}

void staticQubitsWithParametricOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
}

void staticQubitsWithTwoTargetOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rzz(0.123, q0, q1);
}

void staticQubitsWithCtrl(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
}

void staticQubitsWithInv(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  b.inv({q0}, [&](ValueRange qubits) { b.t(qubits[0]); });
}

void staticQubitsWithDuplicates(QCProgramBuilder& b) {
  const auto q0a = b.staticQubit(0);
  const auto q1a = b.staticQubit(1);
  const auto q0b = b.staticQubit(0);
  const auto q1b = b.staticQubit(1);

  b.rx(std::numbers::pi / 4., q0a);
  b.p(std::numbers::pi / 2., q1a);
  b.rzz(0.123, q0b, q1b);
  b.cx(q0b, q1b);
  b.inv({q0a}, [&](ValueRange qubits) { b.t(qubits[0]); });
}

void staticQubitsCanonical(QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);

  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  b.rzz(0.123, q0, q1);
  b.cx(q0, q1);
  b.inv({q0}, [&](ValueRange qubits) { b.t(qubits[0]); });
}

void allocDeallocPair(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
}

void mixedStaticThenDynamicQubit(QCProgramBuilder& b) {
  b.staticQubit(0);
  b.allocQubit();
}

void mixedDynamicRegisterThenStaticQubit(QCProgramBuilder& b) {
  b.allocQubitRegister(2);
  b.staticQubit(0);
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

void measurementWithoutRegisters(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.measure(q);
}

void resetQubitWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
}

void resetMultipleQubitsWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
}

void repeatedResetWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
}

void resetQubitAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
}

void resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
}

void repeatedResetAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
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
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[0], {q[1]},
         [&](ValueRange targets) { b.cgphase(0.123, targets[0]); });
}

void trivialControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcgphase(0.123, {});
}

void inverseGlobalPhase(QCProgramBuilder& b) {
  b.inv({}, [&](ValueRange /*qubits*/) { b.gphase(-0.123); });
}

void inverseMultipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcgphase(-0.123, qubits); });
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
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[2], {q[0], q[1]},
         [&](ValueRange targets) { b.cid(targets[1], targets[0]); });
}

void trivialControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
}

void inverseIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.id(qubits[0]); });
}

void inverseMultipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcid({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cx(targets[0], targets[1]); });
}

void trivialControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
}

void repeatedControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(64);
  b.h(q[0]);
  for (auto i = 1; i < 64; i++) {
    b.cx(q[0], q[i]);
  }
}

void inverseX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.x(qubits[0]); });
}

void inverseMultipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcx({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cy(targets[0], targets[1]); });
}

void trivialControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
}

void inverseY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.y(qubits[0]); });
}

void inverseMultipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcy({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cz(targets[0], targets[1]); });
}

void trivialControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
}

void inverseZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.z(qubits[0]); });
}

void inverseMultipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcz({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ch(targets[0], targets[1]); });
}

void trivialControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
}

void inverseH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.h(qubits[0]); });
}

void inverseMultipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mch({qubits[0], qubits[1]}, qubits[2]); });
}

void hWithoutRegister(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cs(targets[0], targets[1]); });
}

void trivialControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
}

void inverseS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.s(qubits[0]); });
}

void inverseMultipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcs({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csdg(targets[0], targets[1]); });
}

void trivialControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
}

void inverseSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.sdg(qubits[0]); });
}

void inverseMultipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcsdg({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ct(targets[0], targets[1]); });
}

void trivialControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
}

void inverseT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.t(qubits[0]); });
}

void inverseMultipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mct({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ctdg(targets[0], targets[1]); });
}

void trivialControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
}

void inverseTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.tdg(qubits[0]); });
}

void inverseMultipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mctdg({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csx(targets[0], targets[1]); });
}

void trivialControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
}

void inverseSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.sx(qubits[0]); });
}

void inverseMultipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcsx({qubits[0], qubits[1]}, qubits[2]); });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csxdg(targets[0], targets[1]); });
}

void trivialControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
}

void inverseSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.sxdg(qubits[0]); });
}

void inverseMultipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcsxdg({qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.crx(0.123, targets[0], targets[1]); });
}

void trivialControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
}

void inverseRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.rx(-0.123, qubits[0]); });
}

void inverseMultipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcrx(-0.123, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cry(0.456, targets[0], targets[1]); });
}

void trivialControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
}

void inverseRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.ry(-0.456, qubits[0]); });
}

void inverseMultipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcry(-0.456, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.crz(0.789, targets[0], targets[1]); });
}

void trivialControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
}

void inverseRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.rz(-0.789, qubits[0]); });
}

void inverseMultipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcrz(-0.789, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cp(0.123, targets[0], targets[1]); });
}

void trivialControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
}

void inverseP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.p(-0.123, qubits[0]); });
}

void inverseMultipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcp(-0.123, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cr(0.123, 0.456, targets[0], targets[1]);
  });
}

void trivialControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
}

void inverseR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.r(-0.123, 0.456, qubits[0]); });
}

void inverseMultipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcr(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cu2(0.234, 0.567, targets[0], targets[1]);
  });
}

void trivialControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
}

void inverseU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv(q[0],
        [&](ValueRange qubits) { b.u2(-0.567 + pi, -0.234 - pi, qubits[0]); });
}

void inverseMultipleControlledU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcu2(-0.567 + pi, -0.234 - pi, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cu(0.1, 0.2, 0.3, targets[0], targets[1]);
  });
}

void trivialControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
}

void inverseU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.u(-0.1, -0.3, -0.2, qubits[0]); });
}

void inverseMultipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcu(-0.1, -0.3, -0.2, {qubits[0], qubits[1]}, qubits[2]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cswap(targets[0], targets[1], targets[2]);
  });
}

void trivialControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
}

void inverseSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.swap(qubits[0], qubits[1]); });
}

void inverseMultipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.ciswap(targets[0], targets[1], targets[2]);
  });
}

void trivialControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
}

void inverseIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.iswap(qubits[0], qubits[1]); });
}

void inverseMultipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mciswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cdcx(targets[0], targets[1], targets[2]);
  });
}

void trivialControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
}

void inverseDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.dcx(qubits[1], qubits[0]); });
}

void inverseMultipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[3], q[2]}, [&](ValueRange qubits) {
    b.mcdcx({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cecr(targets[0], targets[1], targets[2]);
  });
}

void trivialControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
}

void inverseEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.ecr(qubits[0], qubits[1]); });
}

void inverseMultipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcecr({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crxx(0.123, targets[0], targets[1], targets[2]);
  });
}

void trivialControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
}

void inverseRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rxx(-0.123, qubits[0], qubits[1]); });
}

void inverseMultipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrxx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
}

void tripleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
}

void fourControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cryy(0.123, targets[0], targets[1], targets[2]);
  });
}

void trivialControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
}

void inverseRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.ryy(-0.123, qubits[0], qubits[1]); });
}

void inverseMultipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcryy(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crzx(0.123, targets[0], targets[1], targets[2]);
  });
}

void trivialControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
}

void inverseRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rzx(-0.123, qubits[0], qubits[1]); });
}

void inverseMultipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrzx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crzz(0.123, targets[0], targets[1], targets[2]);
  });
}

void trivialControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
}

void inverseRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rzz(-0.123, qubits[0], qubits[1]); });
}

void inverseMultipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrzz(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cxx_plus_yy(0.123, 0.456, targets[0], targets[1], targets[2]);
  });
}

void trivialControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
}

void inverseXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
  });
}

void inverseMultipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcxx_plus_yy(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
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
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cxx_minus_yy(0.123, 0.456, targets[0], targets[1], targets[2]);
  });
}

void trivialControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
}

void inverseXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
  });
}

void inverseMultipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcxx_minus_yy(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2],
                    qubits[3]);
  });
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
  b.ctrl(q[1], q[0], [&](ValueRange targets) { b.barrier(targets[0]); });
}

void inverseBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.barrier(qubits[0]); });
}

void trivialCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, {q[0], q[1]},
         [&](ValueRange targets) { b.rxx(0.123, targets[0], targets[1]); });
}

void emptyCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.ctrl({q[0]}, {q[1]}, [&](ValueRange /*targets*/) {});
}

void nestedCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
      b.rxx(0.123, innerTargets[0], innerTargets[1]);
    });
  });
}

void tripleNestedCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.ctrl(q[0], {q[1], q[2], q[3], q[4]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2], targets[3]},
           [&](ValueRange innerTargets) {
             b.ctrl(innerTargets[0], {innerTargets[1], innerTargets[2]},
                    [&](ValueRange innerInnerTargets) {
                      b.rxx(0.123, innerInnerTargets[0], innerInnerTargets[1]);
                    });
           });
  });
}

void doubleNestedCtrlTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.ctrl({q[0], q[1]}, {q[2], q[3], q[4], q[5]}, [&](ValueRange targets) {
    b.ctrl({targets[0], targets[1]}, {targets[2], targets[3]},
           [&](ValueRange innerTargets) {
             b.rxx(0.123, innerTargets[0], innerTargets[1]);
           });
  });
}

void ctrlInvSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.inv(targets, [&](ValueRange qubits) {
      b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange innerTargets) {
        b.rxx(-0.123, innerTargets[0], innerTargets[1]);
      });
    });
  });
}

void ctrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    b.x(targets[0]);
    b.rxx(0.123, targets[0], targets[1]);
  });
}

void ctrlTwoMixed(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    b.cx(targets[0], targets[1]);
    b.rxx(0.123, targets[0], targets[1]);
  });
}

void nestedCtrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
      b.x(innerTargets[0]);
      b.rxx(0.123, innerTargets[0], innerTargets[1]);
    });
  });
}

void ctrlInvTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    b.inv(targets, [&](ValueRange qubits) {
      b.x(qubits[0]);
      b.rxx(0.123, qubits[0], qubits[1]);
    });
  });
}

void emptyInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.inv({q[0], q[1]}, [&](ValueRange /*targets*/) {});
}

void nestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.inv(qubits, [&](ValueRange innerQubits) {
      b.rxx(0.123, innerQubits[0], innerQubits[1]);
    });
  });
}

void tripleNestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.inv(qubits, [&](ValueRange innerQubits) {
      b.inv(innerQubits, [&](ValueRange innerInnerQubits) {
        b.rxx(-0.123, innerInnerQubits[0], innerInnerQubits[1]);
      });
    });
  });
}

void invCtrlSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange targets) {
      b.inv({targets[0], targets[1]}, [&](ValueRange innerQubits) {
        b.rxx(0.123, innerQubits[0], innerQubits[1]);
      });
    });
  });
}

void invTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.x(qubits[0]);
    b.rxx(0.123, qubits[0], qubits[1]);
  });
}

void invCtrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange targets) {
      b.x(targets[0]);
      b.rxx(0.123, targets[0], targets[1]);
    });
  });
}

void simpleIf(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] { b.x(q[0]); });
}

void ifTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] {
    b.x(q[0]);
    b.x(q[1]);
  });
}

void ifElse(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] { b.x(q[0]); }, [&] { b.z(q[0]); });
}

void nestedIfOpForLoop(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q0 = b.allocQubit();
  b.h(q0);
  auto cond = b.measure(q0);
  b.scfIf(
      cond, [&] { b.h(q0); },
      [&] {
        b.scfFor(0, 3, 1, [&](Value iv) {
          auto q1 = b.memrefLoad(reg.value, iv);
          b.h(q1);
        });
      });
}

void simpleWhileReset(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.scfWhile(
      [&] {
        auto measureResult = b.measure(q);
        b.scfCondition(measureResult);
      },
      [&] { b.h(q); });
}

void simpleDoWhileReset(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.scfWhile(
      [&] {
        b.h(q);
        auto measureResult = b.measure(q);
        b.scfCondition(measureResult);
      },
      [&] {});
}

void simpleForLoop(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.memrefLoad(reg.value, iv);
    b.h(q);
  });
};

void nestedForLoopIfOp(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto qCond = b.allocQubit();
  b.scfFor(0, 2, 1, [&](Value iv) {
    b.h(qCond);
    auto cond = b.measure(qCond);
    b.scfIf(cond, [&] {
      auto q = b.memrefLoad(reg.value, iv);
      b.h(q);
    });
  });
}

void nestedForLoopWhileOp(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.memrefLoad(reg.value, iv);
    b.h(q);
  });
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.memrefLoad(reg.value, iv);
    b.scfWhile(
        [&] {
          auto measureResult = b.measure(q);
          b.scfCondition(measureResult);
        },
        [&] { b.h(q); });
  });
}

void nestedForLoopCtrlOpWithSeparateQubit(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control = b.allocQubit();
  b.h(control);
  b.scfFor(0, 3, 1, [&](Value iv) {
    auto q0 = b.memrefLoad(reg.value, iv);
    b.h(q0);
    b.cx(control, q0);
  });
}

void nestedForLoopCtrlOpWithExtractedQubit(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.h(reg[0]);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto q0 = b.memrefLoad(reg.value, iv);
    b.h(q0);
    b.cx(reg[0], q0);
  });
}

static void emitNativeSynthControlledPhase(QCProgramBuilder& b,
                                           const double theta, mlir::Value ctrl,
                                           mlir::Value tgt) {
  b.p(theta / 2.0, ctrl);
  b.cx(ctrl, tgt);
  b.p(-theta / 2.0, tgt);
  b.cx(ctrl, tgt);
  b.p(theta / 2.0, tgt);
}

static void emitNativeSynthToffoli(QCProgramBuilder& b, mlir::Value c1,
                                   mlir::Value c2, mlir::Value t) {
  b.h(t);
  b.cx(c2, t);
  b.tdg(t);
  b.cx(c1, t);
  b.t(t);
  b.cx(c2, t);
  b.tdg(t);
  b.cx(c1, t);
  b.t(c2);
  b.t(t);
  b.h(t);
  b.cx(c1, c2);
  b.t(c1);
  b.tdg(c2);
  b.cx(c1, c2);
}

/// Shared by ``nativeSynthBroadOneQCanonicalization`` and
/// ``nativeSynthIbmFractionalAllGateFamilies``: wide 1q sweep on two qubits,
/// ending before any two-qubit primitive.
static void emitNativeSynthFixtureBroad1qPrefix(QCProgramBuilder& b,
                                                mlir::Value q0,
                                                mlir::Value q1) {
  b.id(q0);
  b.x(q0);
  b.y(q1);
  b.z(q0);
  b.h(q1);
  b.s(q0);
  b.sdg(q1);
  b.t(q0);
  b.tdg(q1);
  b.sx(q0);
  b.sxdg(q1);
  b.rx(0.13, q0);
  b.ry(-0.47, q1);
  b.rz(0.29, q0);
  b.p(-0.38, q1);
  b.r(0.61, -0.22, q0);
}

static void emitNativeSynthFiveQStressLayers(QCProgramBuilder& b,
                                             const int numLayers) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  const auto q3 = b.allocQubit();
  const auto q4 = b.allocQubit();
  b.h(q0);
  b.s(q1);
  b.t(q2);
  b.y(q3);
  b.h(q4);
  b.cx(q0, q1);
  b.cz(q1, q2);
  b.swap(q2, q3);
  b.cx(q3, q4);
  for (int layer = 0; layer < numLayers; ++layer) {
    b.h(q0);
    b.s(q0);
    b.t(q0);
    b.y(q1);
    b.h(q2);
    b.s(q3);
    b.t(q4);
    b.cx(q0, q2);
    b.cz(q1, q3);
    b.cx(q2, q4);
    if ((layer % 2) == 0) {
      b.swap(q0, q1);
      b.swap(q3, q4);
    } else {
      b.cx(q4, q0);
      b.cz(q2, q1);
    }
  }
  b.p(0.25, q0);
  b.p(-0.5, q2);
  b.p(0.75, q4);
}

static void emitNativeSynthTwoQRzx(QCProgramBuilder& b, const double theta,
                                   const bool controlOnFirstWire) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  if (controlOnFirstWire) {
    b.rzx(theta, q0, q1);
  } else {
    b.rzx(theta, q1, q0);
  }
}

void nativeSynthBroadOneQCanonicalization(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  emitNativeSynthFixtureBroad1qPrefix(b, q0, q1);
  b.cz(q0, q1);
}

void nativeSynthZeroAngleCanonicalization(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.rx(0.0, q0);
  b.ry(0.0, q1);
  b.rz(0.0, q0);
  b.p(0.0, q1);
  b.r(0.0, 0.0, q0);
  b.cz(q0, q1);
}

void nativeSynthIbmFractionalAllGateFamilies(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  emitNativeSynthFixtureBroad1qPrefix(b, q0, q1);
  b.cx(q0, q1);
  b.cz(q1, q0);
  b.swap(q0, q1);
  b.iswap(q0, q1);
  b.dcx(q0, q1);
  b.ecr(q0, q1);
  b.rxx(0.17, q0, q1);
  b.ryy(-0.21, q0, q1);
  b.rzx(0.41, q0, q1);
  b.rzz(-0.33, q0, q1);
  b.xx_plus_yy(0.52, -0.14, q0, q1);
  b.xx_minus_yy(-0.37, 0.26, q0, q1);
}

void nativeSynthFusionHadamardZHadamard(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.h(q0);
  b.z(q0);
  b.h(q0);
}

void nativeSynthFusionHadamardHadamard(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.h(q0);
  b.h(q0);
}

void nativeSynthFusionMixedChainHSTYSX(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.h(q0);
  b.s(q0);
  b.t(q0);
  b.y(q0);
  b.sx(q0);
}

void nativeSynthFusionHadamardCxHadamard(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.h(q0);
}

void nativeSynthFusionHadamardBarrierHadamard(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.h(q0);
  b.barrier({q0});
  b.h(q0);
}

void nativeSynthFusionRzSxRz(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.rz(0.4, q0);
  b.sx(q0);
  b.rz(-0.9, q0);
}

void nativeSynthFusionUUTwoQGenericU3(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.u(0.3, 0.1, -0.2, q0);
  b.u(-0.5, 0.7, 0.4, q0);
}

void nativeSynthFusionTS(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.t(q0);
  b.s(q0);
}

void nativeSynthFusionUUTwoQDet1(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  b.u(0.3, 0.2, -0.2, q0);
  b.u(0.5, 0.4, -0.4, q0);
}

void nativeSynthFusionLongMixedTenOpCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.t(q0);
  b.rx(0.37, q0);
  b.s(q0);
  b.ry(-0.21, q0);
  b.h(q0);
  b.z(q0);
  b.rz(0.52, q0);
  b.sx(q0);
  b.y(q0);
  b.cx(q0, q1);
}

void nativeSynthFusionCxCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q0, q1);
}

void nativeSynthFusionHCxInterleavedTCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.t(q1);
  b.s(q0);
  b.cx(q0, q1);
}

void nativeSynthFusionThreeLineCx01Cx12Cx01(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q1, q2);
  b.cx(q0, q1);
}

void nativeSynthFusionCxBarrierCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.barrier({q0, q1});
  b.cx(q0, q1);
}

void nativeSynthFusionSwapCxPattern(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.cx(q1, q0);
  b.cx(q0, q1);
}

void nativeSynthFusionHDcxSCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.dcx(q0, q1);
  b.s(q1);
  b.cx(q0, q1);
}

void nativeSynthFusionXRzxTCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.rzx(0.41, q0, q1);
  b.t(q1);
  b.cx(q0, q1);
}

void nativeSynthFusionHRzzSRzz(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.rzz(-0.29, q0, q1);
  b.s(q1);
  b.rzz(0.17, q0, q1);
}

void nativeSynthFusionRzx041Q0First(QCProgramBuilder& b) {
  emitNativeSynthTwoQRzx(b, 0.41, /*controlOnFirstWire=*/true);
}

void nativeSynthFusionRzx041Q1First(QCProgramBuilder& b) {
  emitNativeSynthTwoQRzx(b, 0.41, /*controlOnFirstWire=*/false);
}

void nativeSynthFusionRzxPiHalfQ0First(QCProgramBuilder& b) {
  emitNativeSynthTwoQRzx(b, std::numbers::pi / 2.0,
                         /*controlOnFirstWire=*/true);
}

void nativeSynthFusionRzxPiHalfQ1First(QCProgramBuilder& b) {
  emitNativeSynthTwoQRzx(b, std::numbers::pi / 2.0,
                         /*controlOnFirstWire=*/false);
}

void nativeSynthProfilesHstycxTwoQ(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.s(q0);
  b.t(q0);
  b.y(q0);
  b.cx(q0, q1);
}

void nativeSynthProfilesHCxTOnQ1(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q1);
  b.cx(q0, q1);
  b.t(q1);
}

void nativeSynthProfilesXYSXCz(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.y(q0);
  b.sx(q0);
  b.cz(q0, q1);
}

void nativeSynthProfilesFractionalChain(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.ry(0.37, q0);
  b.sxdg(q0);
  b.cx(q0, q1);
  b.rzz(0.23, q0, q1);
}

void nativeSynthProfilesHYcx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q0);
  b.cx(q0, q1);
}

void nativeSynthProfilesZCx(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.z(q0);
  b.cx(q0, q1);
}

void nativeSynthProfilesXHCz(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.h(q0);
  b.cz(q0, q1);
}

void nativeSynthProfilesCxYOnQ1(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.y(q1);
}

void nativeSynthProfilesHq0Yq1CxSq0(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q1);
  b.cx(q0, q1);
  b.s(q0);
}

void nativeSynthProfilesHCxSq1(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.s(q1);
}

void nativeSynthProfilesHYSameWireCxSq1(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q0);
  b.cx(q0, q1);
  b.s(q1);
}

void nativeSynthProfilesPhaseHCxPhase(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.p(0.13, q0);
  b.h(q0);
  b.cx(q0, q1);
  b.p(-0.27, q1);
}

void nativeSynthProfilesLargeFiveQStressEightLayers(QCProgramBuilder& b) {
  emitNativeSynthFiveQStressLayers(b, /*numLayers=*/8);
}

void nativeSynthMultiQThreeQGhz(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.cx(q1, q2);
}

void nativeSynthMultiQThreeQToffoli(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  emitNativeSynthToffoli(b, q0, q1, q2);
}

void nativeSynthMultiQThreeQQft(QCProgramBuilder& b) {
  using std::numbers::pi;
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.h(q2);
  emitNativeSynthControlledPhase(b, pi / 2.0, q1, q2);
  b.h(q1);
  emitNativeSynthControlledPhase(b, pi / 4.0, q0, q2);
  emitNativeSynthControlledPhase(b, pi / 2.0, q0, q1);
  b.h(q0);
  b.cx(q0, q2);
  b.cx(q2, q0);
  b.cx(q0, q2);
}

void nativeSynthMultiQThreeQCliffordTMix(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.h(q0);
  b.t(q1);
  b.x(q2);
  b.cx(q0, q1);
  b.rz(0.37, q2);
  b.cz(q1, q2);
  b.sdg(q0);
  b.ry(-0.42, q1);
  b.cx(q2, q0);
  b.y(q1);
  b.tdg(q2);
  b.cx(q0, q1);
  b.p(0.21, q2);
  b.h(q2);
  b.cz(q0, q2);
  b.rx(-0.13, q1);
  b.s(q0);
}

void nativeSynthMultiQFiveQStressFourLayers(QCProgramBuilder& b) {
  emitNativeSynthFiveQStressLayers(b, /*numLayers=*/4);
}

void nativeSynthCustomMenusIbmFractionalTwoQStress(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.sxdg(q0);
  b.ry(-0.22, q1);
  b.swap(q0, q1);
  b.rxx(0.53, q0, q1);
  b.ecr(q0, q1);
  b.p(0.31, q0);
  b.rzz(-0.44, q0, q1);
}

void nativeSynthCustomMenusXxPlusYyChain(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.sx(q1);
  b.xx_plus_yy(0.52, -0.14, q0, q1);
  b.rz(0.31, q0);
}

void nativeSynthCustomMenusXxMinusYyOnly(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.xx_minus_yy(-0.37, 0.26, q0, q1);
}

void nativeSynthDeterminismTwoQubitSwap(QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.swap(q0, q1);
  b.dealloc(q0);
  b.dealloc(q1);
}

void nativeSynthAllSingleControlledGateFamiliesOneCtrlOneTarget(
    QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const mlir::Value c = q[0];
  const mlir::Value t = q[1];

  b.cgphase(0.07, c);

  b.cid(c, t);
  b.cx(c, t);
  b.cy(c, t);
  b.cz(c, t);
  b.ch(c, t);
  b.cs(c, t);
  b.csdg(c, t);
  b.ct(c, t);
  b.ctdg(c, t);
  b.csx(c, t);
  b.csxdg(c, t);

  b.crx(0.11, c, t);
  b.cry(0.12, c, t);
  b.crz(0.13, c, t);
  b.cp(0.14, c, t);
  b.cr(0.15, 0.16, c, t);
  b.cu2(0.17, 0.18, c, t);
  b.cu(0.19, 0.2, 0.21, c, t);
}
} // namespace mlir::qc
