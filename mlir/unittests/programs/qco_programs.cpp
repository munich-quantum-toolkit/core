/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qco_programs.h"

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

#include <numbers>
#include <tuple>

static std::pair<mlir::SmallVector<mlir::Value>, mlir::SmallVector<mlir::Type>>
measureAndReturnQTensor(mlir::qco::QCOProgramBuilder& b, mlir::Value qTensor,
                        int64_t size) {
  mlir::SmallVector<mlir::Value> bits;
  mlir::SmallVector<mlir::Type> bitTypes;
  auto i1Type = b.getI1Type();
  for (auto i = 0; i < size; ++i) {
    auto [qTensorOut, qubit] = b.qtensorExtract(qTensor, i);
    auto [q2, bit] = b.measure(qubit);
    bits.push_back(bit);
    bitTypes.push_back(i1Type);
    qTensor = b.qtensorInsert(q2, qTensorOut, i);
  }
  return {bits, bitTypes};
}

static std::pair<mlir::SmallVector<mlir::Value>, mlir::SmallVector<mlir::Type>>
measureAndReturn(mlir::qco::QCOProgramBuilder& b,
                 mlir::SmallVector<mlir::Value> qubits) {
  mlir::SmallVector<mlir::Value> bits;
  mlir::SmallVector<mlir::Type> bitTypes;
  auto i1Type = b.getI1Type();
  for (const auto& q : qubits) {
    auto [q2, bit] = b.measure(q);
    bits.push_back(bit);
    bitTypes.push_back(i1Type);
  }
  return {bits, bitTypes};
}

namespace mlir::qco {

std::pair<SmallVector<Value>, SmallVector<Type>>
emptyQCO(QCOProgramBuilder& b) {
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubit(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocQubitNoMeasure(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
alloc1QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(1);
  return measureAndReturn(b, {reg[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
alloc2QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  return measureAndReturn(b, {reg[0], reg[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
alloc3QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocMultipleQubitRegisters(QCOProgramBuilder& b) {
  auto r1 = b.allocQubitRegister(2);
  auto r2 = b.allocQubitRegister(3);
  return measureAndReturn(b, {r1[0], r1[1], r2[0], r2[1], r2[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocLargeRegister(QCOProgramBuilder& b) {
  auto r = b.allocQubitRegister(100);
  return measureAndReturn(b, {r[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsNoMeasure(QCOProgramBuilder& b) {
  auto q1 = b.staticQubit(0);
  auto q2 = b.staticQubit(1);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubits(QCOProgramBuilder& b) {
  auto q1 = b.staticQubit(0);
  auto q2 = b.staticQubit(1);
  return measureAndReturn(b, {q1, q2});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  q0 = b.h(q0);
  q1 = b.h(q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithParametricOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  q0 = b.rx(std::numbers::pi / 4., q0);
  q1 = b.p(std::numbers::pi / 2., q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithTwoTargetOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  std::tie(q0, q1) = b.rzz(0.123, q0, q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithCtrl(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  std::tie(q0, q1) = b.cx(q0, q1);
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
staticQubitsWithInv(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  q0 = b.inv({q0}, [&](auto targets) -> SmallVector<Value> {
    return {b.t(targets[0])};
  })[0];
  return measureAndReturn(b, {q0});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
allocSinkPair(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  b.sink(q);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
mixedStaticThenDynamicQubit(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
mixedDynamicRegisterThenStaticQubit(QCOProgramBuilder& b) {
  b.qtensorAlloc(2);
  auto q1 = b.staticQubit(0);
  return measureAndReturn(b, {q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleMeasurementToSingleBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  const auto [q1, bit] = b.measure(q[0], c[0]);
  return {{bit}, {b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToSameBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  auto [q1, _c1] = b.measure(q[0], c[0]);
  auto [q2, _c2] = b.measure(q1, c[0]);
  auto [q3, c3] = b.measure(q2, c[0]);
  return {{c3}, {b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedMeasurementToDifferentBits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  auto [q1, c1] = b.measure(q[0], c[0]);
  auto [q2, c2] = b.measure(q1, c[1]);
  auto [q3, c3] = b.measure(q2, c[2]);
  return {{c1, c2, c3}, {b.getI1Type(), b.getI1Type(), b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  auto [q0, bit1] = b.measure(q[0], c0[0]);
  auto [q1, bit2] = b.measure(q[1], c1[0]);
  auto [q2, bit3] = b.measure(q[2], c1[1]);
  return {{bit1, bit2, bit3}, {b.getI1Type(), b.getI1Type(), b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
measurementWithoutRegisters(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto [q1, c] = b.measure(q);
  return {{c}, {b.getI1Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.reset(q[0]);
  q[1] = b.reset(q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetQubitAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[1] = b.h(q[1]);
  q[1] = b.reset(q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedResetAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
globalPhase(QCOProgramBuilder& b) {
  b.gphase(0.123);
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.cgphase(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto qs = b.mcgphase(0.123, {q[0], q[1], q[2]});
  return measureAndReturn(b, {qs[0], qs[1], qs[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseGlobalPhase(QCOProgramBuilder& b) {
  b.inv({}, [&](ValueRange /*qubits*/) {
    b.gphase(-0.123);
    return SmallVector<Value>{};
  });
  return {{b.intConstant(0)}, {b.getI64Type()}};
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto qs = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    SmallVector controls{qubits[0], qubits[1], qubits[2]};
    auto controlsOut = b.mcgphase(-0.123, controls);
    return SmallVector<Value>(controlsOut.begin(), controlsOut.end());
  });
  return measureAndReturn(b, {qs[0], qs[1], qs[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
identity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.id(q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[1], q[0]) = b.cid(q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcid({q[2], q[1]}, q[0]);
  q[2] = res.first[0];
  q[1] = res.first[1];
  q[0] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIdentity(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.id(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto res = b.mcid({}, q);
  q = res.second;
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto res = b.inv(
      {q}, [&](ValueRange qubits) { return SmallVector{b.id(qubits[0])}; });
  q = res[0];
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcid({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> x(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcx({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledX(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.x(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcx({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
repeatedControlledX(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto control = b.h(q0);
  std::vector<Value> targets;
  for (auto i = 0; i < 50; i++) {
    auto qubit = b.allocQubit();
    auto res = b.cx(control, qubit);
    control = res.first;
    targets.push_back(res.second);
  }
  targets.push_back(control);
  return measureAndReturn(b,
                          SmallVector<Value>(targets.begin(), targets.end()));
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.x(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcx({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  q[0] = b.x(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
controlledTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctrl(q[0], q[1], [&](ValueRange targets) {
    auto q = b.x(targets[0]);
    q = b.x(q);
    return SmallVector{q};
  });
  return measureAndReturn(b, {res.first[0], res.second[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](ValueRange qubits) {
    auto q = b.x(qubits[0]);
    q = b.x(q);
    return SmallVector{q};
  });
  return measureAndReturn(b, {res[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> y(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cy(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcy({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.y(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcy({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.y(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcy({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  q[0] = b.y(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> z(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cz(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcz({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledZ(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.z(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcz({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.z(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcz({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  q[0] = b.z(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> h(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ch(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mch({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledH(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.h(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mch({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.h(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mch({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoH(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.h(q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
hWithoutRegister(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>> s(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.cs(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcs({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledS(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.s(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcs({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.s(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcs({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
sThenSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.sdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.s(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> sdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.sdg(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsdg({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.sdg(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
sdgThenS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.s(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.sdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> t_(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ct(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mct({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledT(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.t(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mct({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.t(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mct({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tThenTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.tdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.t(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> tdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mctdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledTdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.tdg(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mctdg({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.tdg(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mctdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tdgThenT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.t(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.tdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> sx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csx(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsx({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.sx(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsx({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      {q[0]}, [&](ValueRange qubits) { return SmallVector{b.sx(qubits[0])}; });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsx({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
sxThenSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sxdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> sxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csxdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsxdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSxdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.sxdg(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsxdg({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.sxdg(qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsxdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
sxdgThenSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sxdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.crx(0.123, q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcrx(0.123, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.rx(0.123, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcrx(0.123, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.rx(-0.123, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcrx(-0.123, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  q[0] = b.rx(-0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
rxPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ry(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cry(0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcry(0.456, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRy(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.ry(0.456, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcry(0.456, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.ry(-0.456, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcry(-0.456, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  q[0] = b.ry(-0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ryPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.crz(0.789, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcrz(0.789, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRz(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.rz(0.789, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcrz(0.789, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.rz(-0.789, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcrz(-0.789, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  q[0] = b.rz(-0.789, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> p(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cp(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcp(0.123, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledP(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.p(0.123, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcp(0.123, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.p(-0.123, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcp(-0.123, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoPOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.p(0.123, q);
  q = b.p(-0.123, q);
  return measureAndReturn(b, {q});
}

std::pair<SmallVector<Value>, SmallVector<Type>> r(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledR(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.r(0.123, 0.456, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcr(0.123, 0.456, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.r(-0.123, 0.456, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcr(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeRToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0., q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeRToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.456, std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> u2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0.234, 0.567, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU2(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.u2(0.234, 0.567, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcu2(0.234, 0.567, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.u2(-0.567 + pi, -0.234 - pi, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcu2(-0.567 + pi, -0.234 - pi, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeU2ToH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., std::numbers::pi, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeU2ToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(-std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeU2ToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., 0., q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> u(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.1, 0.2, 0.3, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledU(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](ValueRange innerTargets) {
          return SmallVector{b.u(0.1, 0.2, 0.3, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, {reg[0], reg[1], reg[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcu(0.1, 0.2, 0.3, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.u(-0.1, -0.3, -0.2, qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcu(-0.1, -0.3, -0.2, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<Value>(controlsOut, ValueRange{targetOut}));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0., 0., 0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.123, -std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.456, 0., 0., q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
canonicalizeUToU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(std::numbers::pi / 2, 0.234, 0.567, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> swap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cswap(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcswap({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledSwap(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.swap(innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcswap({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoSwapSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  std::tie(q[1], q[0]) = b.swap(q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> iswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.iswap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.ciswap(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mciswap({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledIswap(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.iswap(innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mciswap({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.iswap(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mciswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> dcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cdcx(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcdcx({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledDcx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.dcx(innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcdcx({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[1], q[0]}, [&](ValueRange qubits) {
    auto res = b.dcx(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[1] = res[0];
  q[0] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[3], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcdcx({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[3] = res[2];
  q[2] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoDcxSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  std::tie(q[1], q[0]) = b.dcx(q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ecr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cecr(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcecr({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledEcr(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.ecr(innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcecr({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcecr({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.crxx(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRxx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.rxx(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcrxx(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.rxx(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrxx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  auto res = b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.first[2];
  q[3] = res.second.first;
  q[4] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
fourControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  auto res = b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.first[2];
  q[3] = res.first[3];
  q[4] = res.second.first;
  q[5] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4], q[5]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rxx(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rxx(0.078, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxxSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rxx(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rxx(0.078, q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rxx(-0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRxxOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rxx(-0.123, q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ryy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cryy(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRyy(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.ryy(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcryy(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.ryy(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcryy(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.ryy(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.ryy(0.078, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyyOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.ryy(-0.123, q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyyOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.ryy(-0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRyySwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.ryy(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.ryy(0.078, q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.crzx(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.rzx(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcrzx(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.rzx(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrzx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzx(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzx(-0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> rzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.crzz(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledRzz(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.rzz(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcrzz(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.rzz(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrzz(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> twoRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rzz(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzz(0.078, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzzSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rzz(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rzz(0.078, q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzzOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzz(-0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoRzzOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rzz(-0.123, q[1], q[0]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
xxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxPlusYY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.xx_plus_yy(0.123, 0.456, innerTargets[0],
                                             innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.mcxx_plus_yy(
        -0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoXxPlusYYOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  std::tie(q[0], q[1]) = b.xx_plus_yy(-0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
xxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
multipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedControlledXxMinusYY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto res =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto res = b.xx_minus_yy(0.123, 0.456, innerTargets[0],
                                              innerTargets[1]);
                     return SmallVector{res.first, res.second};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  reg[3] = res.second[2];
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.inv({q[0], q[1], q[2], q[3]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.mcxx_minus_yy(
        -0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    SmallVector<Value, 2> targets{targetsOut.first, targetsOut.second};
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targets));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoXxMinusYYOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  std::tie(q[0], q[1]) = b.xx_minus_yy(-0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> barrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q1 = b.barrier(q[0]);
  return measureAndReturn(b, {q1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
barrierTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.barrier({q[0], q[1]});
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
barrierMultipleQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.barrier({q[0], q[1], q[2]});
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
singleControlledBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctrl({q[1]}, {q[0]}, [&](ValueRange targets) {
    return SmallVector<Value>{b.barrier(targets[0])};
  });
  q[1] = res.first[0];
  q[0] = res.second[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
inverseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv({q[0]}, [&](ValueRange qubits) {
    return SmallVector<Value>{b.barrier(qubits[0])};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
twoBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto b1 = b.barrier({q[0], q[1]});
  q[0] = b1[0];
  q[1] = b1[1];
  auto b2 = b.barrier({q[0], q[1]});
  q[0] = b2[0];
  q[1] = b2[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
trivialCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [_, q01] = b.ctrl({}, {q[0], q[1]}, [&](ValueRange targets) {
    auto [q0, q1] = b.rxx(0.123, targets[0], targets[1]);
    return SmallVector{q0, q1};
  });
  return measureAndReturn(b, {q01[0], q01[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
emptyCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  auto [res0, res1] =
      b.ctrl(q[0], q[1], [&](ValueRange targets) { return targets; });
  return measureAndReturn(b, {res0[0], res1[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.ctrl({q[0]}, {q[1], q[2], q[3]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        {targets[0]}, {targets[1], targets[2]}, [&](ValueRange innerTargets) {
          auto [q0, q1] = b.rxx(0.123, innerTargets[0], innerTargets[1]);
          return SmallVector{q0, q1};
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  q[0] = res.first[0];
  q[1] = res.second[0];
  q[2] = res.second[1];
  q[3] = res.second[2];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  auto res = b.ctrl({q[0]}, {q[1], q[2], q[3], q[4]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        {targets[0]}, {targets[1], targets[2], targets[3]},
        [&](ValueRange innerTargets) {
          const auto& [innerInnerControlsOut, innerInnerTargetsOut] =
              b.ctrl({innerTargets[0]}, {innerTargets[1], innerTargets[2]},
                     [&](ValueRange innerInnerTargets) {
                       auto [q0, q1] = b.rxx(0.123, innerInnerTargets[0],
                                             innerInnerTargets[1]);
                       return SmallVector{q0, q1};
                     });
          return llvm::to_vector(
              llvm::concat<Value>(innerInnerControlsOut, innerInnerTargetsOut));
        });
    return llvm::to_vector(
        llvm::concat<Value>(innerControlsOut, innerTargetsOut));
  });
  q[0] = res.first[0];
  q[1] = res.second[0];
  q[2] = res.second[1];
  q[3] = res.second[2];
  q[4] = res.second[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
doubleNestedCtrlTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  auto res =
      b.ctrl({q[0], q[1]}, {q[2], q[3], q[4], q[5]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0], targets[1]}, {targets[2], targets[3]},
                   [&](ValueRange innerTargets) {
                     auto [q0, q1] =
                         b.rxx(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{q0, q1};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second[0];
  q[3] = res.second[1];
  q[4] = res.second[2];
  q[5] = res.second[3];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4], q[5]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvSandwich(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.ctrl({q[0]}, {q[1], q[2], q[3]}, [&](ValueRange targets) {
    auto inner = b.inv(
        {targets[0], targets[1], targets[2]}, [&](ValueRange innerTargets) {
          auto [innerControlsOut, innerTargetsOut] =
              b.ctrl({innerTargets[0]}, {innerTargets[1], innerTargets[2]},
                     [&](ValueRange innerInnerTargets) {
                       auto [q0, q1] = b.rxx(-0.123, innerInnerTargets[0],
                                             innerInnerTargets[1]);
                       return SmallVector{q0, q1};
                     });
          return llvm::to_vector(
              llvm::concat<Value>(innerControlsOut, innerTargetsOut));
        });
    return llvm::to_vector(inner);
  });
  q[0] = res.first[0];
  q[1] = res.second[0];
  q[2] = res.second[1];
  q[3] = res.second[2];
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ctrlTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    auto i0 = targets[0];
    auto i1 = targets[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
  return measureAndReturn(
      b, {res.first[0], res.first[1], res.second[0], res.second[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlTwoMixed(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
    auto i0 = targets[0];
    auto i1 = targets[1];
    std::tie(i0, i1) = b.cx(i0, i1);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
  return measureAndReturn(
      b, {res.first[0], res.first[1], res.second[0], res.second[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedCtrlTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.ctrl(q[0], {q[1], q[2], q[3]}, [&](ValueRange targets) {
    const auto& [controlsOut, targetsOut] = b.ctrl(
        targets[0], {targets[1], targets[2]}, [&](ValueRange innerTargets) {
          auto i0 = innerTargets[0];
          auto i1 = innerTargets[1];
          i0 = b.x(i0);
          std::tie(i0, i1) = b.rxx(0.123, i0, i1);
          return SmallVector{i0, i1};
        });
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targetsOut));
  });
  return measureAndReturn(
      b, {res.first[0], res.second[0], res.second[1], res.second[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ctrlInvTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    auto inner = b.inv(targets, [&](ValueRange qubits) {
      auto i0 = qubits[0];
      auto i1 = qubits[1];
      i0 = b.x(i0);
      std::tie(i0, i1) = b.rxx(0.123, i0, i1);
      return SmallVector{i0, i1};
    });
    return llvm::to_vector(inner);
  });
  return measureAndReturn(b, {res.first[0], res.second[0], res.second[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
emptyInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) { return qubits; });
  return measureAndReturn(b, {res[0], res[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto inner = b.inv({qubits[0], qubits[1]}, [&](ValueRange innerQubits) {
      auto [q0, q1] = b.rxx(0.123, innerQubits[0], innerQubits[1]);
      return SmallVector{q0, q1};
    });
    return llvm::to_vector(inner);
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
tripleNestedInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto inner1 = b.inv({qubits[0], qubits[1]}, [&](ValueRange innerQubits) {
      auto inner2 = b.inv(
          {innerQubits[0], innerQubits[1]}, [&](ValueRange innerInnerQubits) {
            auto [q0, q1] =
                b.rxx(-0.123, innerInnerQubits[0], innerInnerQubits[1]);
            return SmallVector{q0, q1};
          });
      return llvm::to_vector(inner2);
    });
    return llvm::to_vector(inner1);
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlSandwich(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.ctrl({qubits[0]}, {qubits[1], qubits[2]}, [&](ValueRange targets) {
          auto inner =
              b.inv({targets[0], targets[1]}, [&](ValueRange innerQubits) {
                auto [q0, q1] = b.rxx(0.123, innerQubits[0], innerQubits[1]);
                return SmallVector{q0, q1};
              });
          return llvm::to_vector(inner);
        });
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targetsOut));
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> invTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto i0 = qubits[0];
    auto i1 = qubits[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
  return measureAndReturn(b, {res[0], res[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
invCtrlTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.ctrl({qubits[0]}, {qubits[1], qubits[2]}, [&](ValueRange targets) {
          auto i0 = targets[0];
          auto i1 = targets[1];
          i0 = b.x(i0);
          std::tie(i0, i1) = b.rxx(0.123, i0, i1);
          return SmallVector{i0, i1};
        });
    return llvm::to_vector(llvm::concat<Value>(controlsOut, targetsOut));
  });
  return measureAndReturn(b, {res[0], res[1], res[2]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  auto res = b.qcoIf(measureResult, measuredQubit, [&](ValueRange args) {
    auto innerQubit = b.x(args[0]);
    return SmallVector{innerQubit};
  });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ifWithAngle(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto theta = b.floatConstant(0.123);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  b.qcoIf(measureResult, measuredQubit, [&](ValueRange args) {
    auto innerQubit = b.rx(theta, args[0]);
    return SmallVector{innerQubit};
  });
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ifTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  auto res =
      b.qcoIf(measureResult, {measuredQubit, q[1]}, [&](ValueRange args) {
        auto innerQubit0 = b.x(args[0]);
        auto innerQubit1 = b.x(args[1]);
        return SmallVector{innerQubit0, innerQubit1};
      });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>> ifElse(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  auto res = b.qcoIf(
      measureResult, {measuredQubit},
      [&](ValueRange args) {
        auto innerQubit = b.x(args[0]);
        return SmallVector{innerQubit};
      },
      [&](ValueRange args) {
        auto innerQubit = b.z(args[0]);
        return SmallVector{innerQubit};
      });
  q[0] = res[0];
  return measureAndReturn(b, {q[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
ifOneQubitOneTensor(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto t0 = b.allocQubitRegister(1);
  auto q1 = b.h(q0);
  auto [measuredQubit, measureResult] = b.measure(q1);
  auto ifRes =
      b.qcoIf(measureResult, {measuredQubit, t0.value}, [&](ValueRange args) {
        auto innerQubit0 = b.x(args[0]);
        auto [t1, innerQubit1] = b.qtensorExtract(args[1], 0);
        auto innerQubit2 = b.x(innerQubit1);
        auto t2 = b.qtensorInsert(innerQubit2, t1, 0);
        return SmallVector{innerQubit0, t2};
      });
  return measureAndReturn(b, {ifRes[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
constantTrueIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto ifRes = b.qcoIf(
      true, q.qubits,
      [&](ValueRange args) {
        auto innerQubit = b.x(args[0]);
        return SmallVector{innerQubit};
      },
      [&](ValueRange args) {
        auto innerQubit = b.z(args[0]);
        return SmallVector{innerQubit};
      });
  return measureAndReturn(b, {ifRes[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
constantFalseIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto ifRes = b.qcoIf(
      false, q.qubits,
      [&](ValueRange args) {
        auto innerQubit = b.x(args[0]);
        return SmallVector{innerQubit};
      },
      [&](ValueRange args) {
        auto innerQubit = b.z(args[0]);
        return SmallVector{innerQubit};
      });
  return measureAndReturn(b, {ifRes[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedTrueIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  auto ifRes = b.qcoIf(measureResult, measuredQubit, [&](ValueRange outerArgs) {
    auto innerResult =
        b.qcoIf(measureResult, outerArgs, [&](ValueRange innerArgs) {
          auto innerQubit = b.x(innerArgs[0]);
          return SmallVector{innerQubit};
        });
    return llvm::to_vector(innerResult);
  });
  return measureAndReturn(b, {ifRes[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedFalseIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  auto ifRes = b.qcoIf(
      measureResult, measuredQubit,
      [&](ValueRange args) {
        auto innerQubit = b.x(args[0]);
        return SmallVector{innerQubit};
      },
      [&](ValueRange outerArgs) {
        auto innerResult = b.qcoIf(
            measureResult, outerArgs,
            [&](ValueRange innerArgs) { return llvm::to_vector(innerArgs); },
            [&](ValueRange innerArgs) {
              auto innerQubit = b.z(innerArgs[0]);
              return SmallVector{innerQubit};
            });
        return llvm::to_vector(innerResult);
      });
  return measureAndReturn(b, {ifRes[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorAlloc(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  return measureAndReturn(b, {qtensor});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorDealloc(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  b.qtensorDealloc(qtensor);
  return measureAndReturn(b, {});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorFromElements(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  auto q2 = b.allocQubit();
  auto t = b.qtensorFromElements({q0, q1, q2});
  return measureAndReturn(b, {t});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorExtract(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [t, q] = b.qtensorExtract(qtensor, 0);
  return measureAndReturn(b, {t, q});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorInsert(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  return measureAndReturn(b, {insertOutTensor});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorExtractInsertIndexMismatch(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto insertOutTensor = b.qtensorInsert(q0, extractOutTensor, 1);
  return measureAndReturn(b, {insertOutTensor});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorExtractInsertSameIndex(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto insertOutTensor = b.qtensorInsert(q0, extractOutTensor, 0);
  return measureAndReturn(b, {insertOutTensor});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorInsertExtractIndexMismatch(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  auto [extractOutTensor1, q2] = b.qtensorExtract(insertOutTensor, 1);
  auto insertOutTensor1 = b.qtensorInsert(q2, extractOutTensor1, 0);
  return measureAndReturn(b, {insertOutTensor1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorInsertExtractSameIndex(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  auto [extractOutTensor1, q2] = b.qtensorExtract(insertOutTensor, 0);
  auto insertOutTensor1 = b.qtensorInsert(q2, extractOutTensor1, 0);
  return measureAndReturn(b, {insertOutTensor1});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorChain(QCOProgramBuilder& b) {
  Value q0;
  Value q1;
  Value q2;
  auto qtensor = b.qtensorAlloc(3);
  std::tie(qtensor, q0) = b.qtensorExtract(qtensor, 0);
  std::tie(qtensor, q1) = b.qtensorExtract(qtensor, 1);
  std::tie(qtensor, q2) = b.qtensorExtract(qtensor, 2);
  q0 = b.h(q0);
  q1 = b.h(q1);
  std::tie(q1, q2) = b.cx(q1, q2);

  qtensor = b.qtensorInsert(q2, qtensor, 2);
  qtensor = b.qtensorInsert(q1, qtensor, 1);
  qtensor = b.qtensorInsert(q0, qtensor, 0);
  b.qtensorDealloc(qtensor);

  return measureAndReturn(b, {});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
qtensorAlternativeChain(QCOProgramBuilder& b) {
  Value q0;
  Value q1;
  Value q2;
  auto qtensor = b.qtensorAlloc(3);
  std::tie(qtensor, q0) = b.qtensorExtract(qtensor, 0);
  q0 = b.h(q0);
  std::tie(qtensor, q1) = b.qtensorExtract(qtensor, 1);
  q1 = b.h(q1);
  std::tie(qtensor, q2) = b.qtensorExtract(qtensor, 2);
  std::tie(q1, q2) = b.cx(q1, q2);

  qtensor = b.qtensorInsert(q0, qtensor, 0);
  qtensor = b.qtensorInsert(q1, qtensor, 1);
  qtensor = b.qtensorInsert(q2, qtensor, 2);
  b.qtensorDealloc(qtensor);

  return measureAndReturn(b, {});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleWhileReset(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.h(q0);
  auto scfWhile = b.scfWhile(
      ValueRange{q1},
      [&](ValueRange iterArgs) {
        auto [q2, measureResult] = b.measure(iterArgs[0]);
        b.scfCondition(measureResult, q2);
        return SmallVector{q2};
      },
      [&](ValueRange iterArgs) {
        auto q3 = b.h(iterArgs[0]);
        return SmallVector{q3};
      });
  return measureAndReturn(b, {scfWhile[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleDoWhileReset(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto scfWhile = b.scfWhile(
      ValueRange{q0},
      [&](ValueRange iterArgs) {
        auto q1 = b.h(iterArgs[0]);
        auto [q2, measureResult] = b.measure(q1);
        b.scfCondition(measureResult, q2);
        return SmallVector{q2};
      },
      [&](ValueRange iterArgs) { return llvm::to_vector(iterArgs); });
  return measureAndReturn(b, {scfWhile[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
simpleForLoop(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto scfFor =
      b.scfFor(0, 2, 1, {reg.value}, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto q1 = b.h(q0);
        auto insert = b.qtensorInsert(q1, t0, iv);
        return SmallVector{insert};
      });
  return measureAndReturnQTensor(b, scfFor[0], 2);
};

std::pair<SmallVector<Value>, SmallVector<Type>>
forLoopWithAngle(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto theta = b.floatConstant(0.123);
  auto scfFor =
      b.scfFor(0, 2, 1, {reg.value}, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto q1 = b.rx(theta, q0);
        auto insert = b.qtensorInsert(q1, t0, iv);
        return SmallVector{insert};
      });
  return measureAndReturn(b, {scfFor[0]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopIfOp(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto q0 = b.allocQubit();
  auto scfFor =
      b.scfFor(0, 2, 1, {reg.value, q0}, [&](Value iv, ValueRange iterArgs) {
        auto q1 = b.h(iterArgs[1]);
        auto [q2, cond] = b.measure(q1);
        auto ifOp = b.qcoIf(cond, iterArgs[0], [&](ValueRange args) {
          auto [t0, q3] = b.qtensorExtract(args[0], iv);
          auto q4 = b.h(q3);
          auto insert = b.qtensorInsert(q4, t0, iv);
          return SmallVector{insert};
        });
        return SmallVector{ifOp[0], q2};
      });
  return measureAndReturn(b, {scfFor[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopWhileOp(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto loopResult =
      b.scfFor(0, 2, 1, {reg.value}, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto q1 = b.h(q0);
        auto insert = b.qtensorInsert(q1, t0, iv);
        return SmallVector{insert};
      });
  auto scfFor =
      b.scfFor(0, 2, 1, loopResult, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto whileResult = b.scfWhile(
            q0,
            [&](ValueRange iterArgs) {
              auto [q1, measureResult] = b.measure(iterArgs[0]);
              b.scfCondition(measureResult, q1);
              return SmallVector{q1};
            },
            [&](ValueRange iterArgs) {
              auto q2 = b.h(iterArgs[0]);
              return SmallVector{q2};
            });
        auto insert = b.qtensorInsert(whileResult[0], t0, iv);
        return SmallVector{insert};
      });
  return measureAndReturnQTensor(b, scfFor[0], 2);
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithSeparateQubit(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control0 = b.allocQubit();
  auto control1 = b.h(control0);
  auto scfFor = b.scfFor(0, 3, 1, {reg.value, control1},
                         [&](Value iv, ValueRange iterArgs) {
                           auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
                           auto q1 = b.h(q0);
                           auto [controls, targets] =
                               b.ctrl(iterArgs[1], q1, [&](ValueRange args) {
                                 auto q2 = b.x(args[0]);
                                 return SmallVector{q2};
                               });
                           auto insert = b.qtensorInsert(targets[0], t0, iv);
                           return SmallVector{insert, controls[0]};
                         });
  return measureAndReturn(b, {scfFor[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedForLoopCtrlOpWithExtractedQubit(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto control = b.h(reg[0]);
  auto scfFor = b.scfFor(1, 4, 1, {reg.value, control},
                         [&](Value iv, ValueRange iterArgs) {
                           auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
                           auto q1 = b.h(q0);
                           auto [controls, targets] =
                               b.ctrl(iterArgs[1], q1, [&](ValueRange args) {
                                 auto q2 = b.x(args[0]);
                                 return SmallVector{q2};
                               });
                           auto insert = b.qtensorInsert(targets[0], t0, iv);
                           return SmallVector{insert, controls[0]};
                         });
  return measureAndReturn(b, {scfFor[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedIfOpForLoop(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q0 = b.allocQubit();
  auto q1 = b.h(q0);
  auto [q2, cond] = b.measure(q1);
  auto ifRes = b.qcoIf(
      cond, {reg.value, q2},
      [&](ValueRange args) {
        auto q3 = b.h(args[1]);
        return SmallVector{args[0], q3};
      },
      [&](ValueRange args) {
        auto scfFor =
            b.scfFor(0, 3, 1, args[0], [&](Value iv, ValueRange iterArgs) {
              auto [t0, q4] = b.qtensorExtract(iterArgs[0], iv);
              auto q5 = b.h(q4);
              auto insert = b.qtensorInsert(q5, t0, iv);
              return SmallVector{insert};
            });
        return SmallVector{scfFor[0], args[1]};
      });
  return measureAndReturn(b, {ifRes[1]});
}

std::pair<SmallVector<Value>, SmallVector<Type>>
nestedIfOpForLoopWithAngle(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q0 = b.allocQubit();
  auto theta1 = b.floatConstant(0.123);
  auto theta2 = b.floatConstant(0.456);
  auto q1 = b.h(q0);
  auto [q2, cond] = b.measure(q1);
  auto res = b.qcoIf(
      cond, {reg.value, q2},
      [&](ValueRange args) {
        auto q3 = b.rx(theta1, args[1]);
        return SmallVector{args[0], q3};
      },
      [&](ValueRange args) {
        auto scfFor =
            b.scfFor(0, 3, 1, args[0], [&](Value iv, ValueRange iterArgs) {
              auto [t0, q4] = b.qtensorExtract(iterArgs[0], iv);
              auto q5 = b.rx(theta2, q4);
              auto insert = b.qtensorInsert(q5, t0, iv);
              return SmallVector{insert};
            });
        return SmallVector{scfFor[0], args[1]};
      });
  return measureAndReturn(b, {res[0]});
}

} // namespace mlir::qco
