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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

#include <numbers>
#include <tuple>
#include <utility>

static std::pair<mlir::SmallVector<mlir::Value>, mlir::SmallVector<mlir::Type>>
measureAndReturn(mlir::qc::QCProgramBuilder& b,
                 mlir::SmallVector<mlir::Value> qubits) {
  mlir::SmallVector<mlir::Value> bits;
  mlir::SmallVector<mlir::Type> bitTypes;
  auto i1Type = b.getI1Type();
  for (const auto& q : qubits) {
    bits.push_back(b.measure(q));
    bitTypes.push_back(i1Type);
  }
  return {bits, bitTypes};
}

namespace mlir::qc {

std::pair<SmallVector<Value>, SmallVector<Type>>
emptyQC([[maybe_unused]] QCProgramBuilder& builder) {
  return measureAndReturn(builder, {});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubit(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegistersWithOps(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  b.h(q0[0]);
  b.h(q0[1]);
  b.h(q1[0]);
  b.h(q1[1]);
  b.h(q1[2]);
  return measureAndReturn(b, {q0[0], q0[1], q1[0], q1[1], q1[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
alloc1QubitRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubitRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegisters(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  return measureAndReturn(b, {q0[0], q0[1], q1[0], q1[1], q1[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocLargeRegister(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(100);
  return measureAndReturn(b, {q[0], q[99]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubits(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.h(q0);
  b.h(q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithParametricOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithTwoTargetOps(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rzz(0.123, q0, q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithCtrl(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithInv(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  b.inv({q0}, [&](ValueRange qubits) { b.t(qubits[0]); });
  return measureAndReturn(b, {q0});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithDuplicates(QCProgramBuilder& b) {
  const auto q0a = b.staticQubit(0);
  const auto q1a = b.staticQubit(1);
  const auto q0b = b.staticQubit(0);
  const auto q1b = b.staticQubit(1);

  b.rx(std::numbers::pi / 4., q0a);
  b.p(std::numbers::pi / 2., q1a);
  b.rzz(0.123, q0b, q1b);
  b.cx(q0b, q1b);
  b.inv({q0a}, [&](ValueRange qubits) { b.t(qubits[0]); });
  return measureAndReturn(b, {q0b, q1b});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsCanonical(QCProgramBuilder& b) {
  const auto q0 = b.staticQubit(0);
  const auto q1 = b.staticQubit(1);

  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  b.rzz(0.123, q0, q1);
  b.cx(q0, q1);
  b.inv({q0}, [&](ValueRange qubits) { b.t(qubits[0]); });
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocDeallocPair(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
  return measureAndReturn(b, {});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
mixedStaticThenDynamicQubit(QCProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
mixedDynamicRegisterThenStaticQubit(QCProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.staticQubit(0);
  return measureAndReturn(b, {q0[0], q0[1], q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleMeasurementToSingleBit(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  const auto outcome = b.measure(q[0], c[0]);
  return {{outcome}, {b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToSameBit(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  auto c3 = b.measure(q[0], c[0]);
  return {{c3}, {b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToDifferentBits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  auto c1 = b.measure(q[0], c[0]);
  auto c2 = b.measure(q[0], c[1]);
  auto c3 = b.measure(q[0], c[2]);
  return {{c1, c2, c3}, {b.getI1Type(), b.getI1Type(), b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleClassicalRegistersAndMeasurements(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  const auto b1 = b.measure(q[0], c0[0]);
  const auto b2 = b.measure(q[1], c1[0]);
  const auto b3 = b.measure(q[2], c1[1]);
  return {{b1, b2, b3}, {b.getI1Type(), b.getI1Type(), b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
measurementWithoutRegisters(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  auto c = b.measure(q);
  return {{c}, {b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetWithoutOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetAfterSingleOp(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
globalPhase(QCProgramBuilder& b) {
  b.gphase(0.123);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[0], {q[1]},
         [&](ValueRange targets) { b.cgphase(0.123, targets[0]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcgphase(0.123, {});
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseGlobalPhase(QCProgramBuilder& b) {
  b.inv({}, [&](ValueRange /*qubits*/) { b.gphase(-0.123); });
  return measureAndReturn(b, {});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledGlobalPhase(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcgphase(-0.123, qubits); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> identity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[2], q[1]}, q[0]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[2], {q[0], q[1]},
         [&](ValueRange targets) { b.cid(targets[1], targets[0]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.id(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIdentity(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcid({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> x(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledX(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cx(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedControlledX(QCProgramBuilder& b) {
  auto control = b.allocQubit();
  b.h(control);
  mlir::SmallVector<mlir::Value> qubits = {control};
  for (auto i = 0; i < 50; i++) {
    auto qubit = b.allocQubit();
    b.cx(control, qubit);
    qubits.push_back(qubit);
  }
  return measureAndReturn(b, qubits);
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.x(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledX(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcx({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> y(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cy(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.y(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcy({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> z(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledZ(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cz(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.z(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledZ(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcz({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> h(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledH(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ch(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.h(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledH(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mch({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
hWithoutRegister(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>> s(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledS(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cs(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.s(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledS(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcs({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> sdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csdg(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.sdg(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcsdg({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> t_(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledT(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ct(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.t(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledT(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mct({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> tdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledTdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.ctdg(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.tdg(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledTdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mctdg({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> sx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csx(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.sx(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]},
        [&](ValueRange qubits) { b.mcsx({qubits[0], qubits[1]}, qubits[2]); });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> sxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSxdg(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.csxdg(targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.sxdg(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSxdg(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcsxdg({qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.crx(0.123, targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.rx(-0.123, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcrx(-0.123, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ry(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRy(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cry(0.456, targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.ry(-0.456, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcry(-0.456, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRz(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.crz(0.789, targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.rz(-0.789, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcrz(-0.789, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> p(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledP(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]},
         [&](ValueRange targets) { b.cp(0.123, targets[0], targets[1]); });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.p(-0.123, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledP(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcp(-0.123, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> r(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledR(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cr(0.123, 0.456, targets[0], targets[1]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.r(-0.123, 0.456, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledR(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcr(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> u2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU2(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cu2(0.234, 0.567, targets[0], targets[1]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU2(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv(q[0],
        [&](ValueRange qubits) { b.u2(-0.567 + pi, -0.234 - pi, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU2(QCProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcu2(-0.567 + pi, -0.234 - pi, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> u(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl(reg[0], {reg[1], reg[2]}, [&](ValueRange targets) {
    b.cu(0.1, 0.2, 0.3, targets[0], targets[1]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> inverseU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.u(-0.1, -0.3, -0.2, qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.mcu(-0.1, -0.3, -0.2, {qubits[0], qubits[1]}, qubits[2]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> swap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSwap(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cswap(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.swap(qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSwap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> iswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIswap(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.ciswap(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.iswap(qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIswap(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mciswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> dcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledDcx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cdcx(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.dcx(qubits[1], qubits[0]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledDcx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[3], q[2]}, [&](ValueRange qubits) {
    b.mcdcx({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ecr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledEcr(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cecr(targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) { b.ecr(qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledEcr(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcecr({qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRxx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crxx(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rxx(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrxx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
fourControlledRxx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4], q[5]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ryy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRyy(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cryy(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.ryy(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRyy(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcryy(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzx(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crzx(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rzx(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzx(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrzx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzz(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.crzz(0.123, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]},
        [&](ValueRange qubits) { b.rzz(-0.123, qubits[0], qubits[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzz(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcrzz(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> xxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxPlusYY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cxx_plus_yy(0.123, 0.456, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxPlusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcxx_plus_yy(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
xxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxMinusYY(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl(reg[0], {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
    b.cxx_minus_yy(0.123, 0.456, targets[0], targets[1], targets[2]);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxMinusYY(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    b.mcxx_minus_yy(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2],
                    qubits[3]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> barrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.barrier(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
barrierTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier({q[0], q[1]});
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
barrierMultipleQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.barrier({q[0], q[1], q[2]});
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl(q[1], q[0], [&](ValueRange targets) { b.barrier(targets[0]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseBarrier(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv(q[0], [&](ValueRange qubits) { b.barrier(qubits[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, {q[0], q[1]},
         [&](ValueRange targets) { b.rxx(0.123, targets[0], targets[1]); });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
emptyCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.ctrl({q[0]}, {q[1]}, [&](ValueRange /*targets*/) {});
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrl(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
      b.rxx(0.123, innerTargets[0], innerTargets[1]);
    });
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedCtrl(QCProgramBuilder& b) {
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
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
doubleNestedCtrlTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.ctrl({q[0], q[1]}, {q[2], q[3], q[4], q[5]}, [&](ValueRange targets) {
    b.ctrl({targets[0], targets[1]}, {targets[2], targets[3]},
           [&](ValueRange innerTargets) {
             b.rxx(0.123, innerTargets[0], innerTargets[1]);
           });
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4], q[5]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.inv(targets, [&](ValueRange qubits) {
      b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange innerTargets) {
        b.rxx(-0.123, innerTargets[0], innerTargets[1]);
      });
    });
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ctrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    b.x(targets[0]);
    b.rxx(0.123, targets[0], targets[1]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlTwoMixed(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    b.cx(targets[0], targets[1]);
    b.rxx(0.123, targets[0], targets[1]);
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    b.ctrl(targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
      b.x(innerTargets[0]);
      b.rxx(0.123, innerTargets[0], innerTargets[1]);
    });
  });
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    b.inv(targets, [&](ValueRange qubits) {
      b.x(qubits[0]);
      b.rxx(0.123, qubits[0], qubits[1]);
    });
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> emptyInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  b.inv({q[0], q[1]}, [&](ValueRange /*targets*/) {});
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.inv(qubits, [&](ValueRange innerQubits) {
      b.rxx(0.123, innerQubits[0], innerQubits[1]);
    });
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedInv(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.inv(qubits, [&](ValueRange innerQubits) {
      b.inv(innerQubits, [&](ValueRange innerInnerQubits) {
        b.rxx(-0.123, innerInnerQubits[0], innerInnerQubits[1]);
      });
    });
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlSandwich(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange targets) {
      b.inv({targets[0], targets[1]}, [&](ValueRange innerQubits) {
        b.rxx(0.123, innerQubits[0], innerQubits[1]);
      });
    });
  });
}

std::pair<SmallVector<Value>, SmallVector<Type>> invTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    b.x(qubits[0]);
    b.rxx(0.123, qubits[0], qubits[1]);
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlTwo(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    b.ctrl(qubits[0], {qubits[1], qubits[2]}, [&](ValueRange targets) {
      b.x(targets[0]);
      b.rxx(0.123, targets[0], targets[1]);
    });
  });
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> simpleIf(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] { b.x(q[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ifElse(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] { b.x(q[0]); }, [&] { b.z(q[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ifTwoQubits(QCProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto cond = b.measure(q[0]);
  b.scfIf(cond, [&] {
    b.x(q[0]);
    b.x(q[1]);
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedIfOpForLoop(QCProgramBuilder& b) {
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
  return measureAndReturn(b, {reg[0], reg[1], reg[2], q0});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleWhileReset(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.scfWhile(
      [&] {
        auto measureResult = b.measure(q);
        b.scfCondition(measureResult);
      },
      [&] { b.h(q); });
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleDoWhileReset(QCProgramBuilder& b) {
  auto q = b.allocQubit();
  b.scfWhile(
      [&] {
        b.h(q);
        auto measureResult = b.measure(q);
        b.scfCondition(measureResult);
      },
      [&] {});
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleForLoop(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.memrefLoad(reg.value, iv);
    b.h(q);
  });
  return measureAndReturn(b, {reg[0], reg[1]});
};

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopIfOp(QCProgramBuilder& b) {
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
  return measureAndReturn(b, {reg[0], reg[1], qCond});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopWhileOp(QCProgramBuilder& b) {
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
  return measureAndReturn(b, {reg[0], reg[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithSeparateQubit(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control = b.allocQubit();
  b.h(control);
  b.scfFor(0, 3, 1, [&](Value iv) {
    auto q0 = b.memrefLoad(reg.value, iv);
    b.h(q0);
    b.cx(control, q0);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], control});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithExtractedQubit(QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.h(reg[0]);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto q0 = b.memrefLoad(reg.value, iv);
    b.h(q0);
    b.cx(reg[0], q0);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

} // namespace mlir::qc
