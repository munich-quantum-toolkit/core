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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>
#include <tuple>
#include <vector>

namespace mlir::qco {

static Value measureAndReturnQTensor(QCOProgramBuilder& b, Value qTensor,
                                     const int64_t size) {
  auto c = b.allocClassicalBitRegister(size);
  for (auto i = 0; i < size; ++i) {
    auto [qTensorOut, qubit] = b.qtensorExtract(qTensor, i);
    auto [q2, bit] = b.measure(qubit, c, i);
    qTensor = b.qtensorInsert(q2, qTensorOut, i);
  }
  return c;
}

static Value measureToRegister(QCOProgramBuilder& b, ValueRange qubits) {
  auto c = b.allocClassicalBitRegister(static_cast<int64_t>(qubits.size()));
  for (auto [i, q] : llvm::enumerate(qubits)) {
    b.measure(q, c, static_cast<int64_t>(i));
  }
  return c;
}

static Value measureToRegister(QCOProgramBuilder& b, Value qubit) {
  return measureToRegister(b, ValueRange(qubit));
}

static Value measureAndReturn(QCOProgramBuilder& b, ValueRange qubits) {
  if (qubits.empty()) {
    return b.intConstant(0);
  }
  return measureToRegister(b, qubits);
}

Value emptyQCO(QCOProgramBuilder& b) { return b.intConstant(0); }

Value allocQubit(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  return measureToRegister(b, q);
}

Value alloc2Qubits(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

Value allocQubitNoMeasure(QCOProgramBuilder& b) {
  (void)b.allocQubit();
  return b.intConstant(0);
}

Value alloc1QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(1);
  return measureToRegister(b, reg[0]);
}

Value alloc2QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  return measureAndReturn(b, reg.qubits);
}

Value alloc3QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  return measureAndReturn(b, reg.qubits);
}

Value allocMultipleQubitRegisters(QCOProgramBuilder& b) {
  auto r1 = b.allocQubitRegister(2);
  auto r2 = b.allocQubitRegister(3);
  return measureAndReturn(b, {r1[0], r1[1], r2[0], r2[1], r2[2]});
}

Value allocLargeRegister(QCOProgramBuilder& b) {
  auto r = b.allocQubitRegister(100);
  return measureToRegister(b, {r[0], r[99]});
}

Value staticQubitsNoMeasure(QCOProgramBuilder& b) {
  (void)b.staticQubit(0);
  (void)b.staticQubit(1);
  return b.intConstant(0);
}

Value staticQubits(QCOProgramBuilder& b) {
  auto q1 = b.staticQubit(0);
  auto q2 = b.staticQubit(1);
  return measureAndReturn(b, {q1, q2});
}

Value staticQubitsWithOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  q0 = b.h(q0);
  q1 = b.h(q1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsWithParametricOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  q0 = b.rx(std::numbers::pi / 4., q0);
  q1 = b.p(std::numbers::pi / 2., q1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsWithTwoTargetOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  std::tie(q0, q1) = b.rzz(0.123, q0, q1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsWithCtrl(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  std::tie(q0, q1) = b.cx(q0, q1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsWithInv(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  q0 = b.inv(q0, [&](Value qubit) { return b.t(qubit); });
  return measureToRegister(b, q0);
}

Value allocSinkPair(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  b.sink(q);
  return b.intConstant(0);
}

SmallVector<Value> deadGatesProgram(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  auto c = b.allocClassicalBitRegister(2);

  auto [q0M, m0] = b.measure(q0, c, 0);
  auto [q1M, m1] = b.measure(q1, c, 1);

  q0 = b.h(q0M);
  auto [res0, res1] = b.cx(q0, q1M);
  auto [_, c1] = b.measure(res1);
  q0 = b.reset(res0);

  return {c};
}

SmallVector<Value> deadGatesResetProgram(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto c = b.allocClassicalBitRegister(1);

  q0 = b.h(q0);
  q0 = b.reset(q0);
  q0 = b.measure(q0, c, 0).first;
  q0 = b.reset(q0);

  return {c};
}

Value deadGatesWithIfOpProgram(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  q0 = b.h(q0);
  auto [r0, c0] = b.measure(q0);
  q0 = r0;

  // This is an `if` with memory effects - it can't be removed.
  q1 = b.qcoIf(
      c0, {q1},
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Then = b.x(qubits[0]);
        b.gphase(0.5); // This adds memory effects to the `IfOp`.
        return {q1Then};
      },
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Else = b.h(qubits[0]);
        return {q1Else};
      })[0];

  // This is an `if` without memory effects - it can be removed.
  q1 = b.qcoIf(
      c0, {q1},
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Then = b.x(qubits[0]);
        return {q1Then};
      },
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Else = b.h(qubits[0]);
        return {q1Else};
      })[0];

  return c0;
}

Value deadGatesWithIfOpSimplified(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  q0 = b.h(q0);
  auto [r0, c0] = b.measure(q0);
  q0 = r0;

  // This is an `if` with memory effects - it can't be removed.
  q1 = b.qcoIf(
      c0, {q1},
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Then = b.x(qubits[0]);
        b.gphase(0.5); // Due to memory effect, the `IfOp` stays.
        return {q1Then};
      },
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Else = b.h(qubits[0]);
        return {q1Else};
      })[0];

  return c0;
}

Value mixedStaticThenDynamicQubit(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

Value mixedDynamicRegisterThenStaticQubit(QCOProgramBuilder& b) {
  b.qtensorAlloc(2);
  auto q1 = b.staticQubit(0);
  return measureToRegister(b, q1);
}

Value singleMeasurementToSingleBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c, 0);
  return c;
}

Value repeatedMeasurementToSameBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(1);
  auto q1 = b.measure(q[0], c, 0).first;
  auto q2 = b.measure(q1, c, 0).first;
  b.measure(q2, c, 0);
  return c;
}

SmallVector<Value> repeatedMeasurementToDifferentBits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(3);
  auto q1 = b.measure(q[0], c, 0).first;
  auto q2 = b.measure(q1, c, 1).first;
  b.measure(q2, c, 2);
  return {c};
}

SmallVector<Value>
multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(2);
  b.measure(q[0], c0, 0);
  b.measure(q[1], c1, 0);
  b.measure(q[2], c1, 1);
  return {c0, c1};
}

Value partialMeasurementToRegister(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(2);
  b.measure(q[0], c, 0);
  return c;
}

Value dynamicallyIndexedMeasurement(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto c = b.allocClassicalBitRegister(2);
  b.scfFor(0, 2, 1, {q.value}, [&](Value iv, ValueRange iterArgs) {
    auto [t0, qubit] = b.qtensorExtract(iterArgs[0], iv);
    auto q1 = b.measure(qubit, c, iv).first;
    auto insert = b.qtensorInsert(q1, t0, iv);
    return SmallVector{insert};
  });
  return c;
}

Value measurementWithoutRegisters(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto [q1, c] = b.measure(q);
  return c;
}

Value resetQubitWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  return measureToRegister(b, q);
}

Value resetMultipleQubitsWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.reset(q[0]);
  q[1] = b.reset(q[1]);
  return measureAndReturn(b, q.qubits);
}

Value repeatedResetWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
  return measureToRegister(b, q);
}

SmallVector<Value> resetQubitAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.measure(q[0], c, 0).first;
  q[0] = b.reset(q[0]);
  q[0] = b.measure(q[0], c, 1).first;
  return {c};
}

SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto c = b.allocClassicalBitRegister(4);
  q[0] = b.h(q[0]);
  q[0] = b.measure(q[0], c, 0).first;
  q[0] = b.reset(q[0]);
  q[0] = b.measure(q[0], c, 1).first;
  q[1] = b.h(q[1]);
  q[1] = b.measure(q[1], c, 2).first;
  q[1] = b.reset(q[1]);
  q[1] = b.measure(q[1], c, 3).first;
  return {c};
}

SmallVector<Value> repeatedResetAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c = b.allocClassicalBitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.measure(q[0], c, 0).first;
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.measure(q[0], c, 1).first;
  return {c};
}

Value globalPhase(QCOProgramBuilder& b) {
  b.gphase(0.123);
  return b.intConstant(0);
}

Value singleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.cgphase(0.123, q[0]);
  return measureToRegister(b, q[0]);
}

Value multipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto qs = b.mcgphase(0.123, {q[0], q[1], q[2]});
  return measureAndReturn(b, qs);
}

Value inverseGlobalPhase(QCOProgramBuilder& b) {
  b.inv(ValueRange{}, [&](ValueRange /*qubits*/) {
    b.gphase(-0.123);
    return SmallVector<Value>{};
  });
  return b.intConstant(0);
}

Value inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto qs = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    SmallVector controls{qubits[0], qubits[1], qubits[2]};
    auto controlsOut = b.mcgphase(-0.123, controls);
    return SmallVector<Value>(controlsOut.begin(), controlsOut.end());
  });
  return measureAndReturn(b, qs);
}

Value powGphaseScaled(QCOProgramBuilder& b) {
  b.pow(3.0, ValueRange{}, [&](mlir::ValueRange /*qubits*/) {
    b.gphase(0.123);
    return llvm::SmallVector<mlir::Value>{};
  });
  return b.intConstant(0);
}

Value powGphaseScaledRef(QCOProgramBuilder& b) {
  b.gphase(3.0 * 0.123);
  return b.intConstant(0);
}

Value negPowGphase(QCOProgramBuilder& b) {
  b.pow(-3.0, ValueRange{}, [&](mlir::ValueRange /*qubits*/) {
    b.gphase(0.123);
    return llvm::SmallVector<mlir::Value>{};
  });
  return b.intConstant(0);
}

Value negPowGphaseRef(QCOProgramBuilder& b) {
  b.gphase(-3.0 * 0.123);
  return b.intConstant(0);
}

Value identity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.id(q);
  return measureToRegister(b, q);
}

Value singleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[1], q[0]) = b.cid(q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcid({q[2], q[1]}, q[0]);
  q[2] = res.first[0];
  q[1] = res.first[1];
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledIdentity(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.id(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto res = b.mcid({}, q);
  q = res.second;
  return measureToRegister(b, q);
}

Value inverseIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.id(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledIdentity(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value powId(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.id(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value x(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcx({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledX(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.x(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcx({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value repeatedControlledX(QCOProgramBuilder& b) {
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

Value inverseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.x(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledX(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  q[0] = b.x(q[0]);
  return measureToRegister(b, q[0]);
}

Value controlledTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctrl(q[0], q[1], [&](Value target) {
    target = b.x(target);
    return b.x(target);
  });
  return measureAndReturn(b, {res.first, res.second});
}

Value inverseTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    qubit = b.x(qubit);
    qubit = b.x(qubit);
    return qubit;
  });
  return measureToRegister(b, res);
}
Value powHalfX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.5, q[0], [&](Value qubits) {
    auto q0 = b.x(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powHalfXRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  return b.measure(q[0]).second;
}

Value powNegHalfX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(-0.5, q[0], [&](Value qubits) {
    auto q0 = b.x(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThirdX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.x(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdXRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 2.0);
  q[0] = b.rx(1.0 / 3.0 * std::numbers::pi, q[0]);
  return b.measure(q[0]).second;
}

Value inverseGphaseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    b.gphase(-0.123);
    return b.x(qubit);
  });
  return measureToRegister(b, res);
}

Value inverseGphaseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    b.gphase(0.123);
    return b.barrier({qubit})[0];
  });
  return measureToRegister(b, res);
}

Value inverseTwoBarriersInInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    qubit = b.barrier({qubit})[0];
    return b.barrier({qubit})[0];
  });
  return measureToRegister(b, res);
}

Value y(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cy(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcy({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.y(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcy({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.y(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledY(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  q[0] = b.y(q[0]);
  return measureToRegister(b, q[0]);
}

Value powHalfY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.5, q[0], [&](Value qubits) {
    auto q0 = b.y(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powHalfYRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-std::numbers::pi / 4.0);
  q[0] = b.ry(std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]).second;
}

Value z(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cz(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcz({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledZ(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.z(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcz({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.z(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledZ(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  q[0] = b.z(q[0]);
  return measureToRegister(b, q[0]);
}

Value powHalfZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.5, q[0], [&](Value qubits) {
    auto q0 = b.z(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThreeHalvesZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.5, q[0], [&](Value qubits) {
    auto q0 = b.z(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThirdZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.z(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdZRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(1.0 / 3.0 * std::numbers::pi, q[0]);
  return b.measure(q[0]).second;
}

Value h(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ch(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mch({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledH(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.h(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mch({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.h(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledH(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoH(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.h(q);
  return measureToRegister(b, q);
}

Value powEvenH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.h(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powOddH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(3.0, q[0], [&](Value qubits) {
    auto q0 = b.h(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value hWithoutRegister(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  return b.measure(q).second;
}

Value s(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.cs(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcs({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledS(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.s(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcs({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.s(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledS(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value sThenSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.sdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value twoS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.s(q[0]);
  return measureToRegister(b, q[0]);
}

Value powTwoS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.s(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powFourS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(4.0, q[0], [&](Value qubits) {
    auto q0 = b.s(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powHalfS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.5, q[0], [&](Value qubits) {
    auto q0 = b.s(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThirdS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.s(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdSRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]).second;
}

Value sdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledSdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.sdg(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsdg({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sdg(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledSdg(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value sdgThenS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.s(q[0]);
  return measureToRegister(b, q[0]);
}

Value twoSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.sdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value powTwoSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.sdg(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powHalfSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.5, q[0], [&](Value qubits) {
    auto q0 = b.sdg(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThirdSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.sdg(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdSdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]).second;
}

Value t_(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ct(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mct({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledT(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.t(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mct({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.t(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledT(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value tThenTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.tdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value twoT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.t(q[0]);
  return measureToRegister(b, q[0]);
}

Value powTwoT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.t(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThirdT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.t(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdTRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
  return b.measure(q[0]).second;
}

Value tdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mctdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledTdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.tdg(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mctdg({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.tdg(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledTdg(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value tdgThenT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.t(q[0]);
  return measureToRegister(b, q[0]);
}

Value twoTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.tdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value powTwoTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.tdg(qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value powThirdTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.tdg(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdTdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(-1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
  return b.measure(q[0]).second;
}

Value sx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csx(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsx({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledSx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.sx(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsx({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sx(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledSx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value sxThenSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sxdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value twoSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sx(q[0]);
  return measureToRegister(b, q[0]);
}

Value powTwoSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.sx(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powTwoSxRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return b.measure(q[0]).second;
}

Value powThirdSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.sx(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdSxRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 4.0);
  q[0] = b.rx(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]).second;
}

Value sxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csxdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsxdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledSxdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        targets[0], targets[1], [&](Value target) { return b.sxdg(target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsxdg({}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sxdg(qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledSxdg(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value sxdgThenSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sx(q[0]);
  return measureToRegister(b, q[0]);
}

Value twoSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sxdg(q[0]);
  return measureToRegister(b, q[0]);
}

Value powTwoSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.sxdg(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powTwoSxdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return b.measure(q[0]).second;
}

Value powThirdSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0 / 3.0, q[0], [&](Value qubits) {
    auto q0 = b.sxdg(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powThirdSxdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(1.0 / 3.0 * std::numbers::pi / 4.0);
  q[0] = b.rx(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
  return b.measure(q[0]).second;
}

Value rx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.crx(0.123, q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcrx(0.123, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.rx(0.123, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcrx(0.123, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.rx(-0.123, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledRx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoRxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  q[0] = b.rx(-0.123, q[0]);
  return measureToRegister(b, q[0]);
}

Value rxPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(std::numbers::pi / 2, q[0]);
  return measureToRegister(b, q[0]);
}

Value powRxScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.rx(0.123, qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value rxScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.246, q[0]);
  return b.measure(q[0]).second;
}

Value ry(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cry(0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcry(0.456, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRy(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.ry(0.456, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcry(0.456, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.ry(-0.456, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledRy(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoRyOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  q[0] = b.ry(-0.456, q[0]);
  return measureToRegister(b, q[0]);
}

Value ryPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(std::numbers::pi / 2, q[0]);
  return measureToRegister(b, q[0]);
}

Value rz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.crz(0.789, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcrz(0.789, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRz(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.rz(0.789, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcrz(0.789, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.rz(-0.789, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledRz(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoRzOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  q[0] = b.rz(-0.789, q[0]);
  return measureToRegister(b, q[0]);
}

Value p(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(0.123, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cp(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcp(0.123, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledP(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.p(0.123, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcp(0.123, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.p(-0.123, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledP(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoPOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.p(0.123, q);
  q = b.p(-0.123, q);
  return measureToRegister(b, q);
}

Value r(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0.456, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledR(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.r(0.123, 0.456, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcr(0.123, 0.456, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.inv(q[0], [&](Value qubit) { return b.r(-0.123, 0.456, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledR(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value powRScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(3.0, q[0], [&](Value qubits) {
    auto q0 = b.r(0.123, 0.456, qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powRScaledRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(3.0 * 0.123, 0.456, q[0]);
  return b.measure(q[0]).second;
}

Value canonicalizeRToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0., q[0]);
  return measureToRegister(b, q[0]);
}

Value canonicalizeRToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.456, std::numbers::pi / 2, q[0]);
  return measureToRegister(b, q[0]);
}

Value twoR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.045, 0.456, q[0]);
  q[0] = b.r(0.078, 0.456, q[0]);
  return measureToRegister(b, q[0]);
}

Value u2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0.234, 0.567, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledU2(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.u2(0.234, 0.567, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcu2(0.234, 0.567, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      q[0], [&](Value qubit) { return b.u2(-0.567 + pi, -0.234 - pi, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledU2(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value canonicalizeU2ToH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., std::numbers::pi, q[0]);
  return measureToRegister(b, q[0]);
}

Value canonicalizeU2ToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(-std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return measureToRegister(b, q[0]);
}

Value canonicalizeU2ToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., 0., q[0]);
  return measureToRegister(b, q[0]);
}

Value u(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.1, 0.2, 0.3, q[0]);
  return measureToRegister(b, q[0]);
}

Value singleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledU(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto res = b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl(targets[0], targets[1],
               [&](Value target) { return b.u(0.1, 0.2, 0.3, target); });
    return SmallVector{innerControlsOut, innerTargetsOut};
  });
  reg[0] = res.first[0];
  reg[1] = res.second[0];
  reg[2] = res.second[1];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcu(0.1, 0.2, 0.3, {}, q[0]);
  q[0] = res.second;
  return measureToRegister(b, q[0]);
}

Value inverseU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.inv(q[0], [&](Value qubit) { return b.u(-0.1, -0.3, -0.2, qubit); });
  return measureToRegister(b, res);
}

Value inverseMultipleControlledU(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value canonicalizeUToP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0., 0., 0.123, q[0]);
  return measureToRegister(b, q[0]);
}

Value canonicalizeUToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.123, -std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return measureToRegister(b, q[0]);
}

Value canonicalizeUToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.456, 0., 0., q[0]);
  return measureToRegister(b, q[0]);
}

Value canonicalizeUToU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(std::numbers::pi / 2, 0.234, 0.567, q[0]);
  return measureToRegister(b, q[0]);
}

Value swap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cswap(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcswap({q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledSwap(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] = b.swap(innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcswap({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.swap(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledSwap(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoSwapSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  std::tie(q[1], q[0]) = b.swap(q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value powEvenSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  return measureAndReturn(b, powOut);
}

Value powOddSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  return measureAndReturn(b, powOut);
}

Value iswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.iswap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.ciswap(q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mciswap({q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledIswap(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] =
                         b.iswap(innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mciswap({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.iswap(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledIswap(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value powHalfIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(0.5, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.iswap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  return measureAndReturn(b, powOut);
}

Value powHalfIswapRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [rq0, rq1] = b.xx_plus_yy(-std::numbers::pi / 2.0, 0.0, q[0], q[1]);
  return measureAndReturn(b, {rq0, rq1});
}

Value dcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cdcx(q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcdcx({q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledDcx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] = b.dcx(innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcdcx({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[1], q[0]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.dcx(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[1] = res[0];
  q[0] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledDcx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoDcxSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  std::tie(q[1], q[0]) = b.dcx(q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value ecr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cecr(q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [fst, snd] = b.mcecr({q[0], q[1]}, q[2], q[3]);
  q[0] = fst[0];
  q[1] = fst[1];
  q[2] = snd.first;
  q[3] = snd.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledEcr(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] = b.ecr(innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcecr({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.ecr(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledEcr(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value powEvenEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  return measureAndReturn(b, powOut);
}

Value powOddEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  return measureAndReturn(b, powOut);
}

Value rxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.crxx(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRxx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] =
                         b.rxx(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcrxx(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.rxx(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledRxx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value tripleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  auto [c, t] = b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = c[2];
  q[3] = t.first;
  q[4] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value fourControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  auto [c, t] = b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = c[2];
  q[3] = c[3];
  q[4] = t.first;
  q[5] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value twoRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rxx(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rxx(0.078, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoRxxSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rxx(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rxx(0.078, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value twoRxxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rxx(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoRxxOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rxx(-0.123, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value ryy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cryy(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRyy(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] =
                         b.ryy(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcryy(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.ryy(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledRyy(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.ryy(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.ryy(0.078, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoRyyOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.ryy(-0.123, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value twoRyyOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.ryy(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoRyySwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.ryy(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.ryy(0.078, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value rzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.crzx(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRzx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] =
                         b.rzx(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcrzx(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.rzx(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledRzx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoRzxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzx(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzx(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value rzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.crzz(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRzz(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] =
                         b.rzz(0.123, innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcrzz(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.rzz(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledRzz(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value twoRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rzz(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzz(0.078, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoRzzSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rzz(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rzz(0.078, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value twoRzzOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzz(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoRzzOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rzz(-0.123, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value xxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledXxPlusYY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] = b.xx_plus_yy(
                         0.123, 0.456, innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value powXxPlusYYScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto [q0, q1] = b.xx_plus_yy(0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
  return measureAndReturn(b, powOut);
}

Value powXxPlusYYScaledRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [rq0, rq1] = b.xx_plus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {rq0, rq1});
}

Value twoXxPlusYYOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  std::tie(q[0], q[1]) = b.xx_plus_yy(-0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoXxPlusYYSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.045, 0.456, q[0], q[1]);
  std::tie(q[1], q[0]) = b.xx_plus_yy(0.078, 0.456, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value xxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledXxMinusYY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto [c, t] =
      b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](ValueRange targets) {
        const auto& [innerControlsOut, innerTargetsOut] =
            b.ctrl({targets[0]}, {targets[1], targets[2]},
                   [&](ValueRange innerTargets) {
                     auto [fst, snd] = b.xx_minus_yy(
                         0.123, 0.456, innerTargets[0], innerTargets[1]);
                     return SmallVector{fst, snd};
                   });
        return llvm::to_vector(
            llvm::concat<Value>(innerControlsOut, innerTargetsOut));
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

Value inverseXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value powXxMinusYYScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto [q0, q1] = b.xx_minus_yy(0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
  return measureAndReturn(b, powOut);
}

Value powXxMinusYYScaledRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [rq0, rq1] = b.xx_minus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {rq0, rq1});
}

Value twoXxMinusYYOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  std::tie(q[0], q[1]) = b.xx_minus_yy(-0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

Value twoXxMinusYYSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.045, 0.456, q[0], q[1]);
  std::tie(q[1], q[0]) = b.xx_minus_yy(0.078, 0.456, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value rccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  std::tie(q[0], q[1], q[2]) = b.rccx(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

Value powEvenRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto powOut = b.pow(2.0, q.qubits, [&](ValueRange args) {
    auto [q0, q1, q2] = b.rccx(args[0], args[1], args[2]);
    return SmallVector<Value>{q0, q1, q2};
  });
  return measureAndReturn(b, powOut);
}

Value powOddRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto powOut = b.pow(3.0, q.qubits, [&](ValueRange args) {
    auto [q0, q1, q2] = b.rccx(args[0], args[1], args[2]);
    return SmallVector<Value>{q0, q1, q2};
  });
  return measureAndReturn(b, powOut);
}

Value twoRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  std::tie(q[0], q[1], q[2]) = b.rccx(q[0], q[1], q[2]);
  std::tie(q[0], q[1], q[2]) = b.rccx(q[0], q[1], q[2]);
  return measureAndReturn(b, q.qubits);
}

Value singleControlledRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.crccx(q[0], q[1], q[2], q[3]);
  const auto& [q0, q1, q2] = t;
  q[0] = c;
  q[1] = q0;
  q[2] = q1;
  q[3] = q2;
  return measureAndReturn(b, q.qubits);
}

Value multipleControlledRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  auto [c, t] = b.mcrccx({q[0], q[1]}, q[2], q[3], q[4]);
  const auto& [q0, q1, q2] = t;
  q[0] = c[0];
  q[1] = c[1];
  q[2] = q0;
  q[3] = q1;
  q[4] = q2;
  return measureAndReturn(b, q.qubits);
}

Value nestedControlledRccx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(5);
  auto [c, t] = b.ctrl(
      {reg[0]}, {reg[1], reg[2], reg[3], reg[4]}, [&](ValueRange targets) {
        const auto& [controlOut, targetsOut] =
            b.crccx(targets[0], targets[1], targets[2], targets[3]);
        const auto& [q0, q1, q2] = targetsOut;
        return SmallVector<Value>{controlOut, q0, q1, q2};
      });
  reg[0] = c[0];
  reg[1] = t[0];
  reg[2] = t[1];
  reg[3] = t[2];
  reg[4] = t[3];
  return measureAndReturn(b, reg.qubits);
}

Value trivialControlledRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.mcrccx({}, q[0], q[1], q[2]);
  const auto& [q0, q1, q2] = t;
  q[0] = q0;
  q[1] = q1;
  q[2] = q2;
  return measureAndReturn(b, q.qubits);
}

Value inverseRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    const auto [q0, q1, q2] = b.rccx(qubits[0], qubits[1], qubits[2]);
    return SmallVector<Value>{q0, q1, q2};
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, q.qubits);
}

Value inverseMultipleControlledRccx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  auto res = b.inv({q[0], q[1], q[2], q[3], q[4]}, [&](ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrccx({qubits[0], qubits[1]}, qubits[2], qubits[3], qubits[4]);
    const auto& [q0, q1, q2] = targetsOut;
    return SmallVector<Value>{controlsOut[0], controlsOut[1], q0, q1, q2};
  });
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  q[3] = res[3];
  q[4] = res[4];
  return measureAndReturn(b, q.qubits);
}

Value barrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.barrier(q[0])[0];
  return measureToRegister(b, q[0]);
}

Value barrierTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.barrier({q[0], q[1]});
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

Value barrierMultipleQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.barrier({q[0], q[1], q[2]});
  q[0] = res[0];
  q[1] = res[1];
  q[2] = res[2];
  return measureAndReturn(b, q.qubits);
}

Value singleControlledBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res =
      b.ctrl(q[1], q[0], [&](Value target) { return b.barrier({target})[0]; });
  return measureToRegister(b, res.second);
}

Value inverseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.barrier({qubit})[0]; });
  return measureToRegister(b, res);
}

Value powBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut =
      b.pow(2.0, q[0], [&](Value qubit) { return b.barrier(qubit).front(); });
  return measureToRegister(b, powOut);
}

Value twoBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto b1 = b.barrier({q[0], q[1]});
  q[0] = b1[0];
  q[1] = b1[1];
  auto b2 = b.barrier({q[0], q[1]});
  q[0] = b2[0];
  q[1] = b2[1];
  return measureAndReturn(b, q.qubits);
}

Value trivialCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [_, q01] = b.ctrl({}, {q[0], q[1]}, [&](ValueRange targets) {
    auto [q0, q1] = b.rxx(0.123, targets[0], targets[1]);
    return SmallVector{q0, q1};
  });
  return measureAndReturn(b, q01);
}

Value emptyCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  auto [res0, res1] = b.ctrl(q[0], q[1], [&](Value target) { return target; });
  return measureAndReturn(b, {res0, res1});
}

Value nestedCtrl(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value tripleNestedCtrl(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value doubleNestedCtrlTwoQubits(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value ctrlInvSandwich(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value ctrlTwo(QCOProgramBuilder& b) {
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

Value ctrlTwoMixed(QCOProgramBuilder& b) {
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

Value nestedCtrlTwo(QCOProgramBuilder& b) {
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

Value ctrlInvTwo(QCOProgramBuilder& b) {
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

Value emptyInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) { return qubits; });
  return measureAndReturn(b, res);
}

Value emptyPow(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  const auto powOut =
      b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) { return qubits; });
  return measureAndReturn(b, powOut);
}

Value nestedInv(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value tripleNestedInv(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value invCtrlSandwich(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, q.qubits);
}

Value invTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto i0 = qubits[0];
    auto i1 = qubits[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
  return measureAndReturn(b, res);
}

Value powTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    auto i0 = qubits[0];
    auto i1 = qubits[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
  return measureAndReturn(b, powOut);
}

Value invCtrlTwo(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, res);
}

Value pow1Inline(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(1.0, q[0], [&](Value qubits) {
    auto q0 = b.rx(0.123, qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value pow0Erase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.0, q[0], [&](Value qubits) {
    auto q0 = b.rx(0.123, qubits);
    return q0;
  });
  return measureToRegister(b, powOut);
}

Value pow0Two(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(0.0, {q[0], q[1]}, [&](ValueRange qubits) {
    auto i0 = qubits[0];
    auto i1 = qubits[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
  return measureAndReturn(b, powOut);
}

Value nestedPow(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(3.0, q[0], [&](Value qubits) {
    return b.pow(2.0, qubits,
                 [&](Value innerQubit) { return b.rx(0.123, innerQubit); });
  });
  return b.measure(powOut).second;
}

Value powSingleExponent(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(6.0, q[0], [&](Value qubits) {
    auto q0 = b.rx(0.123, qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value nestedPowBranchCut(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(0.5, q[0], [&](Value outer) {
    return b.pow(2.0, outer, [&](Value inner) { return b.x(inner); });
  });
  return b.measure(powOut).second;
}

Value powRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto [q0, q1] = b.rxx(0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
  return measureAndReturn(b, powOut);
}

Value negPowRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(-2.0, q[0], [&](Value qubits) {
    auto q0 = b.rx(0.123, qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value negPowH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(-0.5, q[0], [&](Value qubits) {
    auto q0 = b.h(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value invPowHFrac(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto invOut = b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(0.5, invArgs[0], [&](Value powArgs) {
      auto q0 = b.h(powArgs);
      return q0;
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
  return b.measure(invOut[0]).second;
}

Value powHFracNeg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(-0.5, q[0], [&](Value qubits) {
    auto q0 = b.h(qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value invPowEvenH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto invOut = b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(2.0, invArgs[0], [&](Value powArgs) {
      auto q0 = b.h(powArgs);
      return q0;
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
  return measureToRegister(b, invOut[0]);
}

Value invPowEvenSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto invOut = b.inv({q[0], q[1]}, [&](mlir::ValueRange invArgs) {
    auto inner =
        b.pow(2.0, {invArgs[0], invArgs[1]}, [&](mlir::ValueRange powArgs) {
          auto res = b.swap(powArgs[0], powArgs[1]);
          return llvm::SmallVector<mlir::Value>{res.first, res.second};
        });
    return llvm::SmallVector<mlir::Value>{inner};
  });
  return measureAndReturn(b, invOut);
}

Value invPowSquaredZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto invOut = b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(2.0, invArgs[0], [&](Value powArgs) {
      auto q0 = b.z(powArgs);
      return q0;
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
  return measureToRegister(b, invOut[0]);
}

Value invPowRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto invOut = b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(2.0, invArgs[0], [&](Value powArgs) {
      auto q0 = b.rx(0.123, powArgs);
      return q0;
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
  return b.measure(invOut[0]).second;
}

Value invPowReordered(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto invOut = b.inv({q[0], q[1]}, [&](mlir::ValueRange invArgs) {
    auto inner =
        b.pow(0.5, {invArgs[1], invArgs[0]}, [&](mlir::ValueRange powArgs) {
          auto res = b.swap(powArgs[0], powArgs[1]);
          return llvm::SmallVector<mlir::Value>{res.first, res.second};
        });
    return llvm::SmallVector<mlir::Value>{inner[1], inner[0]};
  });
  return measureAndReturn(b, invOut);
}

Value invPowReorderedRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(-0.5, {q[1], q[0]}, [&](mlir::ValueRange powArgs) {
    auto res = b.swap(powArgs[0], powArgs[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  // The pow operates on {q[1], q[0]}, so its outputs follow that order; write
  // them back to their register slots before measuring in natural order.
  q[1] = powOut[0];
  q[0] = powOut[1];
  return measureAndReturn(b, q.qubits);
}

Value mergeNestedPowReordered(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange outerArgs) {
    auto inner =
        b.pow(0.5, {outerArgs[1], outerArgs[0]}, [&](mlir::ValueRange powArgs) {
          auto res = b.swap(powArgs[0], powArgs[1]);
          return llvm::SmallVector<mlir::Value>{res.first, res.second};
        });
    return llvm::SmallVector<mlir::Value>{inner[1], inner[0]};
  });
  return measureAndReturn(b, powOut);
}

Value mergeNestedPowReorderedRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(1.0, {q[1], q[0]}, [&](mlir::ValueRange powArgs) {
    auto res = b.swap(powArgs[0], powArgs[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
  // The pow operates on {q[1], q[0]}, so its outputs follow that order; write
  // them back to their register slots before measuring in natural order.
  q[1] = powOut[0];
  q[0] = powOut[1];
  return measureAndReturn(b, q.qubits);
}

Value powRxNeg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto powOut = b.pow(2.0, q[0], [&](Value qubits) {
    auto q0 = b.rx(-0.123, qubits);
    return q0;
  });
  return b.measure(powOut).second;
}

Value powCtrlRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange powArgs) {
    const auto& [controlsOut, targetsOut] =
        b.ctrl({powArgs[0]}, {powArgs[1]}, [&](mlir::ValueRange targets) {
          return llvm::SmallVector<mlir::Value>{b.rx(0.123, targets[0])};
        });
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targetsOut));
  });
  return measureAndReturn(b, powOut);
}

Value ctrlPowRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto& [controlsOut, targetsOut] =
      b.ctrl({q[0]}, {q[1]}, [&](mlir::ValueRange targets) {
        auto inner = b.pow(2.0, targets[0], [&](Value powArgs) {
          auto q0 = b.rx(0.123, powArgs);
          return q0;
        });
        return llvm::SmallVector<mlir::Value>{inner};
      });
  return measureAndReturn(
      b, llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targetsOut)));
}

Value negPowInvIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto powOut = b.pow(-2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    return b.inv({qubits[0], qubits[1]}, [&](mlir::ValueRange invArgs) {
      auto [q0, q1] = b.iswap(invArgs[0], invArgs[1]);
      return llvm::SmallVector<mlir::Value>{q0, q1};
    });
  });
  return measureAndReturn(b, powOut);
}

Value negPowInvIswapRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [rq0, rq1] = b.xx_plus_yy(-2.0 * std::numbers::pi, 0.0, q[0], q[1]);
  return measureAndReturn(b, {rq0, rq1});
}

Value ctrlPowSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto& [controlsOut, targetsOut] =
      b.ctrl({q[0]}, {q[1]}, [&](mlir::ValueRange targets) {
        auto inner = b.pow(1.0 / 3.0, targets[0], [&](Value powArgs) {
          auto q0 = b.sx(powArgs);
          return q0;
        });
        return llvm::SmallVector<mlir::Value>{inner};
      });
  return measureAndReturn(
      b, llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targetsOut)));
}

SmallVector<Value> simpleIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0, c0, 0);
  auto res = b.qcoIf(measureResult, measuredQubit, [&](ValueRange args) {
    auto innerQubit = b.x(args[0]);
    return SmallVector{innerQubit};
  });
  q[0] = res[0];
  b.measure(q[0], c1, 0);
  return {c0, c1};
}

SmallVector<Value> ifElse(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0, c0, 0);
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
  b.measure(q[0], c1, 0);
  return {c0, c1};
}

SmallVector<Value> ifTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(2);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0, c0, 0);
  auto res =
      b.qcoIf(measureResult, {measuredQubit, q[1]}, [&](ValueRange args) {
        auto innerQubit0 = b.x(args[0]);
        auto innerQubit1 = b.x(args[1]);
        return SmallVector{innerQubit0, innerQubit1};
      });
  q[0] = res[0];
  q[1] = res[1];
  b.measure(q[0], c1, 0);
  b.measure(q[1], c1, 1);
  return {c0, c1};
}

SmallVector<Value> ifWithMeasurement(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0, c0, 0);
  auto res = b.qcoIf(measureResult, measuredQubit, [&](ValueRange args) {
    auto innerQubit = b.measure(args[0], c1, 0).first;
    return SmallVector{innerQubit};
  });
  q[0] = res[0];
  return {c0, c1};
}

SmallVector<Value> ifWithCreg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto q0 = b.h(q[0]);
  auto measuredQubit = b.measure(q0, c0, 0).first;
  auto res = b.qcoIf(c0, 0, {measuredQubit}, [&](ValueRange args) {
    auto innerQubit = b.x(args[0]);
    return SmallVector{innerQubit};
  });
  q[0] = res[0];
  b.measure(q[0], c1, 0);
  return {c0, c1};
}

Value ifOneQubitOneTensor(QCOProgramBuilder& b) {
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
  return b.measure(ifRes[0]).second;
}

Value ifOneTensor(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto result = b.qcoIf(true, q.value, [&](Value tensor) {
    auto [updatedTensor, qubit] = b.qtensorExtract(tensor, 0);
    qubit = b.x(qubit);
    return b.qtensorInsert(qubit, updatedTensor, 0);
  });
  return measureAndReturnQTensor(b, result, 1);
}

Value constantTrueIf(QCOProgramBuilder& b) {
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
  return measureToRegister(b, ifRes[0]);
}

Value constantFalseIf(QCOProgramBuilder& b) {
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
  return measureToRegister(b, ifRes[0]);
}

SmallVector<Value> nestedTrueIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0, c0, 0);
  auto ifRes = b.qcoIf(measureResult, measuredQubit, [&](ValueRange outerArgs) {
    auto innerResult =
        b.qcoIf(measureResult, outerArgs, [&](ValueRange innerArgs) {
          auto innerQubit = b.x(innerArgs[0]);
          return SmallVector{innerQubit};
        });
    return llvm::to_vector(innerResult);
  });
  b.measure(ifRes[0], c1, 0);
  return {c0, c1};
}

SmallVector<Value> nestedFalseIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0, c0, 0);
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
  b.measure(ifRes[0], c1, 0);
  return {c0, c1};
}

SmallVector<Value> simpleIndexSwitch(QCOProgramBuilder& b) {
  Value q;
  Value bit0;
  Value c0;
  Value bit1;

  auto reg = b.allocQubitRegister(1);

  q = b.h(reg[0]);
  std::tie(q, bit0) = b.measure(q);
  c0 = arith::IndexCastUIOp::create(b, b.getIndexType(), bit0).getOut();
  q = b.qcoIndexSwitch(c0, q, SmallVector<int64_t>{0},
                       SmallVector<function_ref<Value(Value)>>{
                           [&](Value arg) { return b.x(arg); }},
                       [&](Value arg) { return b.z(arg); });

  std::tie(q, bit1) = b.measure(q);

  return {bit0, bit1};
}

Value indexSwitchMultiCase(QCOProgramBuilder& b) {
  constexpr int64_t size = 2;

  auto reg = b.allocQubitRegister(size);
  auto c1 = arith::ConstantOp::create(b, b.getIndexType(), b.getIndexAttr(1))
                .getResult();
  auto condition =
      arith::ConstantOp::create(b, b.getIndexType(), b.getIndexAttr(0))
          .getResult();
  for (int64_t i = 0; i < size; ++i) {
    Value bit;

    reg[i] = b.h(reg[i]);
    std::tie(reg[i], bit) = b.measure(reg[i]);
    const auto index =
        arith::IndexCastUIOp::create(b, b.getIndexType(), bit).getOut();
    condition = arith::OrIOp::create(b, {condition, index}).getResult();
    condition = arith::ShLIOp::create(b, {condition, c1});
  }

  reg.qubits = b.qcoIndexSwitch(
      condition, reg.qubits, SmallVector<int64_t>{1, 2, 3},
      SmallVector<function_ref<SmallVector<Value>(ValueRange)>>{
          [&](ValueRange args) {
            SmallVector<Value> qs(args);
            qs[1] = b.x(qs[1]);
            return qs;
          },
          [&](ValueRange args) {
            SmallVector<Value> qs(args);
            qs[0] = b.x(qs[0]);
            return qs;
          },
          [&](ValueRange args) {
            SmallVector<Value> qs(args);
            qs[0] = b.x(qs[0]);
            qs[1] = b.x(qs[1]);
            return qs;
          }},
      [&](ValueRange args) { return args; });

  return measureAndReturn(b, reg.qubits);
}

Value qtensorAlloc(QCOProgramBuilder& b) {
  (void)b.qtensorAlloc(3);
  return measureAndReturn(b, {});
}

Value qtensorDealloc(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  b.qtensorDealloc(qtensor);
  return measureAndReturn(b, {});
}

Value qtensorFromElements(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  auto q2 = b.allocQubit();
  (void)b.qtensorFromElements({q0, q1, q2});
  return measureAndReturn(b, {});
}

Value qtensorExtract(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [t, q] = b.qtensorExtract(qtensor, 0);
  return measureToRegister(b, q);
}

Value qtensorInsert(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  (void)b.qtensorInsert(q1, extractOutTensor, 0);
  return measureAndReturn(b, {});
}

Value qtensorExtractInsertIndexMismatch(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  (void)b.qtensorInsert(q0, extractOutTensor, 1);
  return measureAndReturn(b, {});
}

Value qtensorExtractInsertSameIndex(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  (void)b.qtensorInsert(q0, extractOutTensor, 0);
  return measureAndReturn(b, {});
}

Value qtensorInsertExtractIndexMismatch(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  auto [extractOutTensor1, q2] = b.qtensorExtract(insertOutTensor, 1);
  (void)b.qtensorInsert(q2, extractOutTensor1, 0);
  return measureAndReturn(b, {});
}

Value qtensorInsertExtractSameIndex(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  auto [extractOutTensor1, q2] = b.qtensorExtract(insertOutTensor, 0);
  (void)b.qtensorInsert(q2, extractOutTensor1, 0);
  return measureAndReturn(b, {});
}

Value qtensorChain(QCOProgramBuilder& b) {
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

Value qtensorAlternativeChain(QCOProgramBuilder& b) {
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

Value simpleWhileReset(QCOProgramBuilder& b) {
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
  return measureToRegister(b, scfWhile[0]);
}

Value simpleDoWhileReset(QCOProgramBuilder& b) {
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
  return measureToRegister(b, scfWhile[0]);
}

Value simpleForLoop(QCOProgramBuilder& b) {
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

Value nestedForLoopIfOp(QCOProgramBuilder& b) {
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
  return measureToRegister(b, scfFor[1]);
}

Value nestedForLoopWhileOp(QCOProgramBuilder& b) {
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
            [&](ValueRange innerIterArgs) {
              auto [q1, measureResult] = b.measure(innerIterArgs[0]);
              b.scfCondition(measureResult, q1);
              return SmallVector{q1};
            },
            [&](ValueRange innerIterArgs) {
              auto q2 = b.h(innerIterArgs[0]);
              return SmallVector{q2};
            });
        auto insert = b.qtensorInsert(whileResult[0], t0, iv);
        return SmallVector{insert};
      });
  return measureAndReturnQTensor(b, scfFor[0], 2);
}

Value nestedForLoopSwitchOp(QCOProgramBuilder& b) {
  constexpr int64_t n = 3;
  auto reg = b.allocQubitRegister(n);
  auto c3 = arith::ConstantOp::create(b, b.getIndexAttr(3));

  reg.value = b.scfFor(0, n, 1, reg.value, [&](Value iv, ValueRange iterArgs) {
    auto rem = arith::RemUIOp::create(b, {iv, c3}).getResult();
    auto [t, q] = b.qtensorExtract(iterArgs[0], iv);
    q = b.qcoIndexSwitch(
        rem, {q}, SmallVector<int64_t>{0, 1, 2},
        SmallVector<function_ref<SmallVector<Value>(ValueRange)>>{
            [&](ValueRange args) {
              SmallVector<Value> qs(args);
              qs[0] = b.x(qs[0]);
              return qs;
            },
            [&](ValueRange args) {
              SmallVector<Value> qs(args);
              qs[0] = b.y(qs[0]);
              return qs;
            },
            [&](ValueRange args) {
              SmallVector<Value> qs(args);
              qs[0] = b.x(qs[0]);
              qs[0] = b.y(qs[0]);
              return qs;
            }},
        [&](ValueRange args) { return args; })[0];
    auto insert = b.qtensorInsert(q, t, iv);
    return SmallVector{insert};
  })[0];

  return measureAndReturnQTensor(b, reg.value, n);
}

Value nestedForLoopCtrlOpWithSeparateQubit(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control0 = b.allocQubit();
  auto control1 = b.h(control0);
  auto scfFor = b.scfFor(
      0, 3, 1, {reg.value, control1}, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto q1 = b.h(q0);
        auto [controlOut, targetOut] =
            b.ctrl(iterArgs[1], q1, [&](Value target) { return b.x(target); });
        auto insert = b.qtensorInsert(targetOut, t0, iv);
        return SmallVector{insert, controlOut};
      });
  return measureToRegister(b, scfFor[1]);
}

Value nestedForLoopCtrlOpWithExtractedQubit(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto control = b.h(reg[0]);
  auto scfFor = b.scfFor(
      1, 4, 1, {reg.value, control}, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto q1 = b.h(q0);
        auto [controlOut, targetOut] =
            b.ctrl(iterArgs[1], q1, [&](Value target) { return b.x(target); });
        auto insert = b.qtensorInsert(targetOut, t0, iv);
        return SmallVector{insert, controlOut};
      });
  return measureToRegister(b, scfFor[1]);
}

Value nestedIfOpForLoop(QCOProgramBuilder& b) {
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
  return measureToRegister(b, ifRes[1]);
}

Value controlledXH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [ctrl, targ] = b.ctrl(q[0], q[1], [&](Value target) {
    target = b.x(target);
    return b.h(target);
  });
  return measureAndReturn(b, {ctrl, targ});
}

Value controlledInverseHT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [ctrl, targ] = b.ctrl(q[0], q[1], [&](ValueRange targets) {
    auto wire = b.inv({targets[0]}, [&](ValueRange innerTargets) {
      auto inner = b.h(innerTargets[0]);
      inner = b.t(inner);
      return SmallVector{inner};
    })[0];
    return SmallVector{wire};
  });
  return measureAndReturn(b, {ctrl[0], targ[0]});
}

Value inverseTwoRxRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange targets) {
    auto w0 = b.rx(0.2, targets[0]);
    auto w1 = b.ry(0.3, targets[1]);
    return SmallVector{w0, w1};
  });
  return measureAndReturn(b, res);
}

Value inverseCxThenRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange targets) {
    auto w0 = targets[0];
    auto w1 = targets[1];
    std::tie(w0, w1) = b.cx(w0, w1);
    w1 = b.rz(0.4, w1);
    return SmallVector{w0, w1};
  });
  return measureAndReturn(b, res);
}

Value inverseDcxThenRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange targets) {
    auto w0 = targets[0];
    auto w1 = targets[1];
    std::tie(w0, w1) = b.dcx(w0, w1);
    w1 = b.rz(0.4, w1);
    return SmallVector{w0, w1};
  });
  return measureAndReturn(b, res);
}

Value inverseGphaseBarrierX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value target) {
    b.gphase(0.25);
    auto wire = b.barrier({target})[0];
    wire = b.x(wire);
    return wire;
  });
  return measureToRegister(b, res);
}

Value inverseNestedInvHAndT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value target) {
    auto wire = b.inv(target, [&](Value inner) { return b.h(inner); });
    return b.t(wire);
  });
  return measureToRegister(b, res);
}

Value inverseNestedInvHAndX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange targets) {
    auto w0 = b.inv(targets[0], [&](Value inner) { return b.h(inner); });
    auto w1 = b.x(targets[1]);
    return SmallVector{w0, w1};
  });
  return measureAndReturn(b, res);
}

Value inverseThreeWireRxRyRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange targets) {
    auto w0 = b.rx(0.2, targets[0]);
    auto w1 = b.ry(0.3, targets[1]);
    auto w2 = b.rz(0.4, targets[2]);
    return SmallVector{w0, w1, w2};
  });
  return measureAndReturn(b, res);
}

Value inverseThreeWireNestedTwoInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange targets) {
    auto inner = b.inv({targets[0], targets[1]}, [&](ValueRange innerTargets) {
      auto w0 = b.rx(0.2, innerTargets[0]);
      auto w1 = b.ry(0.3, innerTargets[1]);
      return SmallVector{w0, w1};
    });
    auto w2 = b.rz(0.4, targets[2]);
    return SmallVector{inner[0], inner[1], w2};
  });
  return measureAndReturn(b, res);
}

Value inverseWithThreeQubitOpInBody(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange targets) {
    auto [controls, innerTarget] =
        b.ctrl({targets[0], targets[1]}, targets[2],
               [&](Value inner) { return b.x(inner); });
    return SmallVector<Value>{controls[0], controls[1], innerTarget};
  });
  return measureAndReturn(b, res);
}

} // namespace mlir::qco
