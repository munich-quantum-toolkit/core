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

void powGphaseScaled(QCProgramBuilder& b) {
  b.pow(3.0, [&] { b.gphase(0.123); });
}

void powGphaseScaledRef(QCProgramBuilder& b) { b.gphase(3.0 * 0.123); }

void negPowGphase(QCProgramBuilder& b) {
  b.pow(-3.0, [&] { b.gphase(0.123); });
}

void negPowGphaseRef(QCProgramBuilder& b) { b.gphase(-3.0 * 0.123); }

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

void powId(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.id(q[0]); });
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

void powHalfX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, [&] { b.x(q[0]); });
}

void powHalfXRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
}

void powNegHalfX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, [&] { b.x(q[0]); });
}

void powThirdX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.x(q[0]); });
}

void powThirdXRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 2.0);
  b.rx(1.0 / 3.0 * std::numbers::pi, q[0]);
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

void powHalfY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, [&] { b.y(q[0]); });
}

void powHalfYRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-std::numbers::pi / 4.0);
  b.ry(std::numbers::pi / 2.0, q[0]);
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

void powHalfZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, [&] { b.z(q[0]); });
}

void powThreeHalvesZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.5, [&] { b.z(q[0]); });
}

void powThirdZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.z(q[0]); });
}

void powThirdZRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(1.0 / 3.0 * std::numbers::pi, q[0]);
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

void powEvenH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.h(q[0]); });
}

void powOddH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, [&] { b.h(q[0]); });
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

void powTwoS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.s(q[0]); });
}

void powFourS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(4.0, [&] { b.s(q[0]); });
}

void powHalfS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, [&] { b.s(q[0]); });
}

void powThirdS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.s(q[0]); });
}

void powThirdSRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
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

void powTwoSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.sdg(q[0]); });
}

void powHalfSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, [&] { b.sdg(q[0]); });
}

void powThirdSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.sdg(q[0]); });
}

void powThirdSdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
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

void powTwoT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.t(q[0]); });
}

void powThirdT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.t(q[0]); });
}

void powThirdTRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
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

void powTwoTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.tdg(q[0]); });
}

void powThirdTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.tdg(q[0]); });
}

void powThirdTdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(-1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
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

void powTwoSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.sx(q[0]); });
}

void powTwoSxRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

void powThirdSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.sx(q[0]); });
}

void powThirdSxRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 4.0);
  b.rx(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
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

void powTwoSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.sxdg(q[0]); });
}

void powTwoSxdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

void powThirdSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, [&] { b.sxdg(q[0]); });
}

void powThirdSxdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(1.0 / 3.0 * std::numbers::pi / 4.0);
  b.rx(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
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

void powRxScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.rx(0.123, q[0]); });
}

void rxScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.246, q[0]);
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

void powRScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, [&] { b.r(0.123, 0.456, q[0]); });
}

void powRScaledRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(3.0 * 0.123, 0.456, q[0]);
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

void powEvenSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, [&] { b.swap(q[0], q[1]); });
}

void powOddSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, [&] { b.swap(q[0], q[1]); });
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

void powHalfIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.5, [&] { b.iswap(q[0], q[1]); });
}

void powHalfIswapRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(-std::numbers::pi / 2.0, 0.0, q[0], q[1]);
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

void powEvenEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, [&] { b.ecr(q[0], q[1]); });
}

void powOddEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, [&] { b.ecr(q[0], q[1]); });
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

void powXxPlusYYScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, [&] { b.xx_plus_yy(0.123, 0.456, q[0], q[1]); });
}

void powXxPlusYYScaledRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
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

void powXxMinusYYScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, [&] { b.xx_minus_yy(0.123, 0.456, q[0], q[1]); });
}

void powXxMinusYYScaledRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
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

void powBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.barrier(q[0]); });
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

void pow1Inline(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0, [&] { b.rx(0.123, q[0]); });
}

void pow0Erase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.0, [&] { b.rx(0.123, q[0]); });
}

void nestedPow(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, [&] { b.pow(2.0, [&] { b.rx(0.123, q[0]); }); });
}

void powSingleExponent(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(6.0, [&] { b.rx(0.123, q[0]); });
}

void powRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, [&] { b.rxx(0.123, q[0], q[1]); });
}

void negPowRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-2.0, [&] { b.rx(0.123, q[0]); });
}

void powRxNeg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, [&] { b.rx(-0.123, q[0]); });
}

void negPowH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, [&] { b.h(q[0]); });
}

void invPowHFrac(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange args) { b.pow(0.5, [&] { b.h(args[0]); }); });
}

void powHFracNeg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, [&] { b.h(q[0]); });
}

void invPowRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0],
        [&](ValueRange args) { b.pow(2.0, [&] { b.rx(0.123, args[0]); }); });
}

void powCtrlRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, [&] {
    b.ctrl(q[0], q[1], [&](ValueRange args) { b.rx(0.123, args[0]); });
  });
}

void ctrlPowRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1],
         [&](ValueRange args) { b.pow(2.0, [&] { b.rx(0.123, args[0]); }); });
}

void negPowInvIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  SmallVector<Value> qubits{q[0], q[1]};
  b.pow(-2.0, [&] {
    b.inv(qubits, [&](ValueRange args) { b.iswap(args[0], args[1]); });
  });
}

void negPowInvIswapRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(-2.0 * std::numbers::pi, 0.0, q[0], q[1]);
}

void ctrlPowSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1],
         [&](ValueRange args) { b.pow(1.0 / 3.0, [&] { b.sx(args[0]); }); });
}

void ctrlPowSxRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1],
         [&](ValueRange args) { b.pow(1.0 / 3.0, [&] { b.sx(args[0]); }); });
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

} // namespace mlir::qc
