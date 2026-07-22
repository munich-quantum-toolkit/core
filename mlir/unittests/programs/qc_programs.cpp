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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <numbers>

namespace mlir::qc {

/**
 * @brief Measures the given qubits and returns the measurement outcomes.
 * @param b The `ProgramBuilder` used to perform the measurements.
 * @param qubits The qubits to be measured.
 * @return The result values.
 */
static SmallVector<Value> measureAndReturn(QCProgramBuilder& b,
                                           ValueRange qubits) {
  return llvm::to_vector(
      llvm::map_range(qubits, [&](Value q) { return b.measure(q); }));
}

Value emptyQC(QCProgramBuilder& b) { return b.intConstant(0); }

Value allocQubit(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  return b.measure(q);
}

Value allocQubitNoMeasure(QCProgramBuilder& b) {
  b.allocQubit();
  return b.intConstant(0);
}

SmallVector<Value> allocMultipleQubitRegistersWithOps(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  b.h(q0[0]);
  b.h(q0[1]);
  b.h(q1[0]);
  b.h(q1[1]);
  b.h(q1[2]);
  return measureAndReturn(b, {q0[0], q0[1], q1[0], q1[1], q1[2]});
}

Value alloc1QubitRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  return b.measure(q[0]);
}

SmallVector<Value> allocQubitRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> alloc3QubitRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> allocMultipleQubitRegisters(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  return measureAndReturn(b, {q0[0], q0[1], q1[0], q1[1], q1[2]});
}

SmallVector<Value> allocLargeRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(100);
  return measureAndReturn(b, {q[0], q[99]});
}

SmallVector<Value> staticQubits(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsNoMeasure(QCProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
  return b.intConstant(0);
}

SmallVector<Value> staticQubitsWithOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.h(q0);
  b.h(q1);
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> staticQubitsWithParametricOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> staticQubitsWithTwoTargetOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rzz(0.123, q0, q1);
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> staticQubitsWithCtrl(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsWithInv(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  b.inv(q0, [&](Value qubit) { b.t(qubit); });
  return b.measure(q0);
}

SmallVector<Value> staticQubitsWithDuplicates(QCProgramBuilder& b) {
  const auto q0a = b.staticQubit(0);
  const auto q1a = b.staticQubit(1);
  const auto q0b = b.staticQubit(0);
  const auto q1b = b.staticQubit(1);

  b.rx(std::numbers::pi / 4., q0a);
  b.p(std::numbers::pi / 2., q1a);
  b.rzz(0.123, q0b, q1b);
  b.cx(q0b, q1b);
  b.inv(q0a, [&](Value qubit) { b.t(qubit); });
  return measureAndReturn(b, {q0b, q1b});
}

SmallVector<Value> staticQubitsCanonical(QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);

  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  b.rzz(0.123, q0, q1);
  b.cx(q0, q1);
  b.inv(q0, [&](Value qubit) { b.t(qubit); });
  return measureAndReturn(b, {q0, q1});
}

Value allocDeallocPair(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
  return b.intConstant(0);
}

SmallVector<Value> mixedStaticThenDynamicQubit(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> mixedDynamicRegisterThenStaticQubit(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.staticQubit(0);
  return measureAndReturn(b, {q0[0], q0[1], q1});
}

Value singleMeasurementToSingleBit(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  const auto outcome = b.measure(q[0], c[0]);
  return outcome;
}

Value repeatedMeasurementToSameBit(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  auto c3 = b.measure(q[0], c[0]);
  return c3;
}

SmallVector<Value> repeatedMeasurementToDifferentBits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  auto c1 = b.measure(q[0], c[0]);
  auto c2 = b.measure(q[0], c[1]);
  auto c3 = b.measure(q[0], c[2]);
  return {c1, c2, c3};
}

SmallVector<Value>
multipleClassicalRegistersAndMeasurements(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  const auto b1 = b.measure(q[0], c0[0]);
  const auto b2 = b.measure(q[1], c1[0]);
  const auto b3 = b.measure(q[2], c1[1]);
  return {b1, b2, b3};
}

Value measurementWithoutRegisters(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  auto c = b.measure(q);
  return c;
}

Value resetQubitWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> resetMultipleQubitsWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
  return measureAndReturn(b, q.qubits);
}

Value repeatedResetWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return b.measure(q[0]);
}

Value resetQubitAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
  return measureAndReturn(b, q.qubits);
}

Value repeatedResetAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return b.measure(q[0]);
}

Value globalPhase(QCProgramBuilder& b) {
  b.gphase(0.123);
  return b.intConstant(0);
}

Value globalPhaseAndMeasure(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(0.123);
  return b.measure(q[0]);
}

Value singleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> multipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](Value target) { b.cgphase(0.123, target); });
  return measureAndReturn(b, q.qubits);
}

Value trivialControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcgphase(0.123, {});
  return b.measure(q[0]);
}

Value inverseGlobalPhase(QCProgramBuilder& b) {
  b.inv(ValueRange{}, [&](ValueRange /*qubits*/) { b.gphase(-0.123); });
  return b.intConstant(0);
}

SmallVector<Value> inverseMultipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcgphase(-0.123, qubits); });
  return measureAndReturn(b, q.qubits);
}

Value powGphaseScaled(QCProgramBuilder& b) {
  b.pow(3.0, {}, [&](ValueRange) { b.gphase(0.123); });
  return b.intConstant(0);
}

Value powGphaseScaledRef(QCProgramBuilder& b) {
  b.gphase(3.0 * 0.123);
  return b.intConstant(0);
}

Value negPowGphase(QCProgramBuilder& b) {
  b.pow(-3.0, {}, [&](ValueRange) { b.gphase(0.123); });
  return b.intConstant(0);
}

Value negPowGphaseRef(QCProgramBuilder& b) {
  b.gphase(-3.0 * 0.123);
  return b.intConstant(0);
}

Value identity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoQubitsOneIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.id(q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> threeQubitsOneIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.id(q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoQubitsOneBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[0], {q[1], q[2]},
         [&](ValueRange targets) { b.cid(targets[0], targets[1]); });
  return measureAndReturn(b, q.qubits);
}

Value trivialControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
  return b.measure(q[0]);
}

Value inverseIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.id(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcid({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powId(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.id(qubits[0]); });
  return b.measure(q[0]);
}

Value x(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledX(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cx(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> repeatedControlledX(QCProgramBuilder& b) {
  auto control = b.allocQubit();
  b.h(control);
  SmallVector<Value> qubits;
  for (auto i = 0; i < 50; i++) {
    auto qubit = b.allocQubit();
    b.cx(control, qubit);
    qubits.push_back(qubit);
  }
  qubits.push_back(control);
  return measureAndReturn(b, qubits);
}

Value inverseX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.x(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcx({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powHalfX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, q[0], [&](ValueRange qubits) { b.x(qubits[0]); });
  return b.measure(q[0]);
}

Value powHalfXRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
  return b.measure(q[0]);
}

Value powNegHalfX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, q[0], [&](ValueRange qubits) { b.x(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.x(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdXRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 2.0);
  b.rx(1.0 / 3.0 * std::numbers::pi, q[0]);
  return b.measure(q[0]);
}

Value y(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cy(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
  return b.measure(q[0]);
}

Value inverseY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.y(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcy({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powHalfY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, q[0], [&](ValueRange qubits) { b.y(qubits[0]); });
  return b.measure(q[0]);
}

Value powHalfYRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-std::numbers::pi / 4.0);
  b.ry(std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]);
}

Value z(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledZ(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cz(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
  return b.measure(q[0]);
}

Value inverseZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.z(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcz({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powHalfZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, q[0], [&](ValueRange qubits) { b.z(qubits[0]); });
  return b.measure(q[0]);
}

Value powThreeHalvesZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.5, q[0], [&](ValueRange qubits) { b.z(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.z(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdZRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(1.0 / 3.0 * std::numbers::pi, q[0]);
  return b.measure(q[0]);
}

Value h(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledH(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ch(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
  return b.measure(q[0]);
}

Value inverseH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.h(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mch({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value hWithoutRegister(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  return b.measure(q);
}

Value powEvenH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.h(qubits[0]); });
  return b.measure(q[0]);
}

Value powOddH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, q[0], [&](ValueRange qubits) { b.h(qubits[0]); });
  return b.measure(q[0]);
}

Value s(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledS(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cs(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
  return b.measure(q[0]);
}

Value inverseS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.s(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcs({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powTwoS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.s(qubits[0]); });
  return b.measure(q[0]);
}

Value powFourS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(4.0, q[0], [&](ValueRange qubits) { b.s(qubits[0]); });
  return b.measure(q[0]);
}

Value powHalfS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, q[0], [&](ValueRange qubits) { b.s(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.s(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdSRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]);
}

Value sdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csdg(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
  return b.measure(q[0]);
}

Value inverseSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.sdg(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcsdg({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powTwoSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.sdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powHalfSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, q[0], [&](ValueRange qubits) { b.sdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.sdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdSdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]);
}

Value t_(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledT(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ct(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
  return b.measure(q[0]);
}

Value inverseT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.t(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mct({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powTwoT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.t(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.t(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdTRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
  return b.measure(q[0]);
}

Value tdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledTdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ctdg(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
  return b.measure(q[0]);
}

Value inverseTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.tdg(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mctdg({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powTwoTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.tdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.tdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdTdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(-1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
  return b.measure(q[0]);
}

Value sx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csx(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
  return b.measure(q[0]);
}

Value inverseSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.sx(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcsx({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

Value powTwoSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.sx(qubits[0]); });
  return b.measure(q[0]);
}

Value powTwoSxRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return b.measure(q[0]);
}

Value powThirdSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.sx(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdSxRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 4.0);
  b.rx(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]);
}

Value sxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSxdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csxdg(targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
  return b.measure(q[0]);
}

Value inverseSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.sxdg(qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcsxdg({qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value powTwoSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.sxdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powTwoSxdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return b.measure(q[0]);
}

Value powThirdSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, q[0], [&](ValueRange qubits) { b.sxdg(qubits[0]); });
  return b.measure(q[0]);
}

Value powThirdSxdgRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(1.0 / 3.0 * std::numbers::pi / 4.0);
  b.rx(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]);
}

Value rx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.crx(0.123, targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.rx(-0.123, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcrx(-0.123, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value powRxScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.rx(0.123, qubits[0]); });
  return b.measure(q[0]);
}

Value rxScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.246, q[0]);
  return b.measure(q[0]);
}

Value ry(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRy(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cry(0.456, targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.ry(-0.456, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcry(-0.456, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value rz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRz(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.crz(0.789, targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.rz(-0.789, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcrz(-0.789, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value p(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledP(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cp(0.123, targets[0], targets[1]); });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.p(-0.123, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcp(-0.123, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value r(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledR(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cr(0.123, 0.456, targets[0], targets[1]);
  });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.r(-0.123, 0.456, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcr(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value powRScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, q[0], [&](ValueRange qubits) { b.r(0.123, 0.456, qubits[0]); });
  return b.measure(q[0]);
}

Value powRScaledRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(3.0 * 0.123, 0.456, q[0]);
  return b.measure(q[0]);
}

Value u2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledU2(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cu2(0.234, 0.567, targets[0], targets[1]);
  });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.u2(-0.567 + pi, -0.234 - pi, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcu2(-0.567 + pi, -0.234 - pi, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

Value u(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> singleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledU(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cu(0.1, 0.2, 0.3, targets[0], targets[1]);
  });
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
  return b.measure(q[0]);
}

Value inverseU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.u(-0.1, -0.3, -0.2, qubit); });
  return b.measure(q[0]);
}

SmallVector<Value> inverseMultipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcu(-0.1, -0.3, -0.2, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> swap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSwap(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cswap(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.swap(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powEvenSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]},
        [&](ValueRange qubits) { b.swap(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powOddSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]},
        [&](ValueRange qubits) { b.swap(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> iswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledIswap(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.ciswap(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.iswap(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mciswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powHalfIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.5, {q[0], q[1]},
        [&](ValueRange qubits) { b.iswap(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powHalfIswapRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(-std::numbers::pi / 2.0, 0.0, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> dcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledDcx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cdcx(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.dcx(qubits[1], qubits[0]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[3], q[2]}, [&](ValueRange qubits) {
    b.mcdcx({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ecr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledEcr(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cecr(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.ecr(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcecr({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powEvenEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]},
        [&](ValueRange qubits) { b.ecr(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powOddEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]},
        [&](ValueRange qubits) { b.ecr(qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRxx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crxx(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rxx(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrxx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> tripleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> fourControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ryy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRyy(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cryy(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.ryy(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcryy(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRzx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crzx(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rzx(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrzx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRzz(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crzz(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rzz(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrzz(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> xxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledXxPlusYY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cxx_plus_yy(0.123, 0.456, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcxx_plus_yy(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powXxPlusYYScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_plus_yy(0.123, 0.456, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powXxPlusYYScaledRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> xxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledXxMinusYY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cxx_minus_yy(0.123, 0.456, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcxx_minus_yy(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2],
                    qubits[3]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powXxMinusYYScaled(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_minus_yy(0.123, 0.456, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powXxMinusYYScaledRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rccx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.rccx(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRccx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.crccx(q[0], q[1], q[2], q[3]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRccx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrccx({q[0], q[1]}, q[2], q[3], q[4]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRccx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(5);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3], reg[4]}, [&](ValueRange targets) {
    b.crccx(targets[0], targets[1], targets[2], targets[3]);
  });
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRccx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrccx({}, q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRccx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.rccx(qubits[0], qubits[1], qubits[2]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRccx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.inv({q[0], q[1], q[2], q[3], q[4]}, [&](ValueRange qubits) {
    b.mcrccx({qubits[0], qubits[1]}, qubits[2], qubits[3], qubits[4]);
  });
  return measureAndReturn(b, q.qubits);
}
Value barrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.barrier(q[0]);
  return b.measure(q[0]);
}

SmallVector<Value> barrierTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier({q[0], q[1]});
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> barrierMultipleQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.barrier({q[0], q[1], q[2]});
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[1], q[0], [&](Value target) { b.barrier({target}); });
  return measureAndReturn(b, q.qubits);
}

Value inverseBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](Value qubit) { b.barrier(qubit); });
  return b.measure(q[0]);
}

Value powBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.barrier(qubits[0]); });
  return b.measure(q[0]);
}

SmallVector<Value> trivialCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, {q[0], q[1]},
         [&](ValueRange targets) { b.rxx(0.123, targets[0], targets[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> emptyCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.ctrl(q[0], q[1], [&](Value /*target*/) {});
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
      b.rxx(0.123, innerTargets[0], innerTargets[1]);
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> tripleNestedCtrl(QCProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> doubleNestedCtrlTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.ctrl({q[0], q[1]}, {q[2], q[3], q[4], q[5]}, [&](ValueRange targets) {
    b.ctrl({targets[0], targets[1]}, {targets[2], targets[3]},
           [&](ValueRange innerTargets) {
             b.rxx(0.123, innerTargets[0], innerTargets[1]);
           });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ctrlInvSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.inv(targets, [&](ValueRange qubits) {
      b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange innerTargets) {
        b.rxx(-0.123, innerTargets[0], innerTargets[1]);
      });
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ctrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    b.x(targets[0]);
    b.rxx(0.123, targets[0], targets[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ctrlTwoMixed(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    b.cx(targets[0], targets[1]);
    b.rxx(0.123, targets[0], targets[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedCtrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
      b.x(innerTargets[0]);
      b.rxx(0.123, innerTargets[0], innerTargets[1]);
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ctrlInvTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    b.inv(targets, [&](ValueRange qubits) {
      b.x(qubits[0]);
      b.rxx(0.123, qubits[0], qubits[1]);
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> emptyInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.inv({q[0], q[1]}, [&](ValueRange /*targets*/) {});
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> emptyPow(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.pow(2.0, {q[0], q[1]}, [&](ValueRange /*qubits*/) {});
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.inv(qubits, [&](ValueRange innerQubits) {
      b.rxx(0.123, innerQubits[0], innerQubits[1]);
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> tripleNestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.inv(qubits, [&](ValueRange innerQubits) {
      b.inv(innerQubits, [&](ValueRange innerInnerQubits) {
        b.rxx(-0.123, innerInnerQubits[0], innerInnerQubits[1]);
      });
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> invCtrlSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange targets) {
      b.inv({targets[0], targets[1]}, [&](ValueRange innerQubits) {
        b.rxx(0.123, innerQubits[0], innerQubits[1]);
      });
    });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> invTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.x(qubits[0]);
    b.rxx(0.123, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> invCtrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange targets) {
      b.x(targets[0]);
      b.rxx(0.123, targets[0], targets[1]);
    });
  });
  return measureAndReturn(b, q.qubits);
}

Value pow1Inline(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0, q[0], [&](ValueRange qubits) { b.rx(0.123, qubits[0]); });
  return b.measure(q[0]);
}

Value pow0Erase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.0, q[0], [&](ValueRange qubits) { b.rx(0.123, qubits[0]); });
  return b.measure(q[0]);
}

Value nestedPow(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, q[0], [&](ValueRange qubits) {
    b.pow(2.0, qubits[0], [&](ValueRange inner) { b.rx(0.123, inner[0]); });
  });
  return b.measure(q[0]);
}

Value powSingleExponent(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(6.0, q[0], [&](ValueRange qubits) { b.rx(0.123, qubits[0]); });
  return b.measure(q[0]);
}

Value nestedPowBranchCut(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, q[0], [&](ValueRange outer) {
    b.pow(2.0, outer[0], [&](ValueRange inner) { b.x(inner[0]); });
  });
  return b.measure(q[0]);
}

SmallVector<Value> powRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]},
        [&](ValueRange qubits) { b.rxx(0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, q.qubits);
}

Value negPowRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-2.0, q[0], [&](ValueRange qubits) { b.rx(0.123, qubits[0]); });
  return b.measure(q[0]);
}

Value powRxNeg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, q[0], [&](ValueRange qubits) { b.rx(-0.123, qubits[0]); });
  return b.measure(q[0]);
}

Value negPowH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, q[0], [&](ValueRange qubits) { b.h(qubits[0]); });
  return b.measure(q[0]);
}

Value invPowHFrac(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange args) {
    b.pow(0.5, args[0], [&](ValueRange p) { b.h(p[0]); });
  });
  return b.measure(q[0]);
}

Value powHFracNeg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, q[0], [&](ValueRange qubits) { b.h(qubits[0]); });
  return b.measure(q[0]);
}

Value invPowEvenH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange args) {
    b.pow(2.0, args[0], [&](ValueRange p) { b.h(p[0]); });
  });
  return b.measure(q[0]);
}

SmallVector<Value> invPowEvenSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange args) {
    b.pow(2.0, {args[0], args[1]}, [&](ValueRange p) { b.swap(p[0], p[1]); });
  });
  return measureAndReturn(b, q.qubits);
}

Value invPowSquaredZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange args) {
    b.pow(2.0, args[0], [&](ValueRange p) { b.z(p[0]); });
  });
  return b.measure(q[0]);
}

Value invPowRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange args) {
    b.pow(2.0, args[0], [&](ValueRange p) { b.rx(0.123, p[0]); });
  });
  return b.measure(q[0]);
}

SmallVector<Value> invPowReordered(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange args) {
    b.pow(0.5, {args[1], args[0]}, [&](ValueRange p) { b.swap(p[0], p[1]); });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> invPowReorderedRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(-0.5, {q[1], q[0]}, [&](ValueRange p) { b.swap(p[0], p[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> mergeNestedPowReordered(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](ValueRange o) {
    b.pow(0.5, {o[1], o[0]}, [&](ValueRange p) { b.swap(p[0], p[1]); });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> mergeNestedPowReorderedRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(1.0, {q[1], q[0]}, [&](ValueRange p) { b.swap(p[0], p[1]); });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powCtrlRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], qubits[1],
           [&](ValueRange args) { b.rx(0.123, args[0]); });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ctrlPowRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](ValueRange args) {
    b.pow(2.0, args[0], [&](ValueRange p) { b.rx(0.123, p[0]); });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> negPowInvIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(-2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    b.inv({qubits[0], qubits[1]},
          [&](ValueRange args) { b.iswap(args[0], args[1]); });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> negPowInvIswapRef(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(-2.0 * std::numbers::pi, 0.0, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ctrlPowSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[0], q[1], [&](ValueRange args) {
    b.pow(1.0 / 3.0, args[0], [&](ValueRange p) { b.sx(p[0]); });
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> powTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    b.x(qubits[0]);
    b.rxx(0.123, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> pow0Two(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.0, {q[0], q[1]}, [&](ValueRange qubits) {
    b.x(qubits[0]);
    b.rxx(0.123, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> simpleIf(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] { b.x(q[0]); });
  auto res = b.measure(q[0]);
  return {cond, res};
}

SmallVector<Value> ifElse(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] { b.x(q[0]); }, [&] { b.z(q[0]); });
  auto bit = b.measure(q[0]);
  return {cond, bit};
}

SmallVector<Value> ifTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] {
    b.x(q[0]);
    b.x(q[1]);
  });
  auto c0 = b.measure(q[0]);
  auto c1 = b.measure(q[1]);
  return {cond, c0, c1};
}

Value nestedIfOpForLoop(QCProgramBuilder& b) {
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
  return b.measure(q0);
}

Value simpleWhileReset(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.scfWhile(
      [&] {
        auto measureResult = b.measure(q);
        b.scfCondition(measureResult);
      },
      [&] { b.h(q); });
  return b.measure(q);
}

Value simpleDoWhileReset(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.scfWhile(
      [&] {
        b.h(q);
        auto measureResult = b.measure(q);
        b.scfCondition(measureResult);
      },
      [&] {});
  return b.measure(q);
}

SmallVector<Value> simpleForLoop(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.memrefLoad(reg.value, iv);
    b.h(q);
  });
  return measureAndReturn(b, reg.qubits);
};

Value nestedForLoopIfOp(QCProgramBuilder& b) {
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
  return b.measure(qCond);
}

SmallVector<Value> nestedForLoopWhileOp(QCProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

Value nestedForLoopCtrlOpWithSeparateQubit(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control = b.allocQubit();
  b.h(control);
  b.scfFor(0, 3, 1, [&](Value iv) {
    auto q0 = b.memrefLoad(reg.value, iv);
    b.h(q0);
    b.cx(control, q0);
  });
  return b.measure(control);
}

Value nestedForLoopCtrlOpWithExtractedQubit(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.h(reg[0]);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto q0 = b.memrefLoad(reg.value, iv);
    b.h(q0);
    b.cx(reg[0], q0);
  });
  return b.measure(reg[0]);
}

} // namespace mlir::qc
