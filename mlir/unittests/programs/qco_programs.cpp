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
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>
#include <tuple>
#include <vector>

namespace mlir::qco {

/**
 * @brief Measures the given `qtensor` and returns the measurement outcomes.
 * @param b The `ProgramBuilder` used to perform the measurements.
 * @param qTensor The `qtensor` to be measured.
 * @param size The number of qubits in the `qtensor`.
 * @return The result values.
 */
static SmallVector<Value> measureAndReturnQTensor(QCOProgramBuilder& b,
                                                  Value qTensor,
                                                  const int64_t size) {
  SmallVector<Value> bits;
  for (auto i = 0; i < size; ++i) {
    auto [qTensorOut, qubit] = b.qtensorExtract(qTensor, i);
    auto [q2, bit] = b.measure(qubit);
    bits.push_back(bit);
    qTensor = b.qtensorInsert(q2, qTensorOut, i);
  }
  return bits;
}

/**
 * @brief Measures the given qubits and returns the measurement outcomes.
 * @param b The `ProgramBuilder` used to perform the measurements.
 * @param qubits The qubits to be measured.
 * @return The result values.
 */
static SmallVector<Value> measureAndReturn(QCOProgramBuilder& b,
                                           ValueRange qubits) {
  return llvm::to_vector(
      llvm::map_range(qubits, [&](Value q) { return b.measure(q).second; }));
}

Value emptyQCO(QCOProgramBuilder& b) { return b.intConstant(0); }

Value allocQubit(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  return b.measure(q).second;
}

SmallVector<Value> alloc2Qubits(QCOProgramBuilder& b) {
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
  return b.measure(reg[0]).second;
}

SmallVector<Value> alloc2QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> alloc3QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> allocMultipleQubitRegisters(QCOProgramBuilder& b) {
  auto r1 = b.allocQubitRegister(2);
  auto r2 = b.allocQubitRegister(3);
  return measureAndReturn(b, {r1[0], r1[1], r2[0], r2[1], r2[2]});
}

Value allocLargeRegister(QCOProgramBuilder& b) {
  auto r = b.allocQubitRegister(100);
  return b.measure(r[0]).second;
}

Value staticQubitsNoMeasure(QCOProgramBuilder& b) {
  (void)b.staticQubit(0);
  (void)b.staticQubit(1);
  return b.intConstant(0);
}

SmallVector<Value> staticQubits(QCOProgramBuilder& b) {
  auto q1 = b.staticQubit(0);
  auto q2 = b.staticQubit(1);
  return measureAndReturn(b, {q1, q2});
}

SmallVector<Value> staticQubitsWithOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  q0 = b.h(q0);
  q1 = b.h(q1);
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> staticQubitsWithParametricOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  q0 = b.rx(std::numbers::pi / 4., q0);
  q1 = b.p(std::numbers::pi / 2., q1);
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> staticQubitsWithTwoTargetOps(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  std::tie(q0, q1) = b.rzz(0.123, q0, q1);
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> staticQubitsWithCtrl(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  std::tie(q0, q1) = b.cx(q0, q1);
  return measureAndReturn(b, {q0, q1});
}

Value staticQubitsWithInv(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  q0 = b.inv(q0, [&](Value qubit) { return b.t(qubit); });
  return b.measure(q0).second;
}

Value allocSinkPair(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  b.sink(q);
  return b.intConstant(0);
}

SmallVector<Value> deadGatesProgram(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();

  auto [q0M, m0] = b.measure(q0);
  auto [q1M, m1] = b.measure(q1);

  q0 = b.h(q0M);
  auto [res0, res1] = b.cx(q0, q1M);
  auto [_, c1] = b.measure(res1);
  q0 = b.reset(res0);

  return {m0, m1};
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

SmallVector<Value> mixedStaticThenDynamicQubit(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

Value mixedDynamicRegisterThenStaticQubit(QCOProgramBuilder& b) {
  b.qtensorAlloc(2);
  auto q1 = b.staticQubit(0);
  return b.measure(q1).second;
}

Value singleMeasurementToSingleBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  const auto [q1, bit] = b.measure(q[0], c[0]);
  return bit;
}

Value repeatedMeasurementToSameBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  auto [q1, _c1] = b.measure(q[0], c[0]);
  auto [q2, _c2] = b.measure(q1, c[0]);
  auto [q3, c3] = b.measure(q2, c[0]);
  return c3;
}

SmallVector<Value> repeatedMeasurementToDifferentBits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  auto [q1, c1] = b.measure(q[0], c[0]);
  auto [q2, c2] = b.measure(q1, c[1]);
  auto [q3, c3] = b.measure(q2, c[2]);
  return {c1, c2, c3};
}

SmallVector<Value>
multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  auto [q0, bit1] = b.measure(q[0], c0[0]);
  auto [q1, bit2] = b.measure(q[1], c1[0]);
  auto [q2, bit3] = b.measure(q[2], c1[1]);
  return {bit1, bit2, bit3};
}

Value measurementWithoutRegisters(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto [q1, c] = b.measure(q);
  return c;
}

Value resetQubitWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  return b.measure(q).second;
}

SmallVector<Value> resetMultipleQubitsWithoutOp(QCOProgramBuilder& b) {
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
  return b.measure(q).second;
}

Value resetQubitAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[1] = b.h(q[1]);
  q[1] = b.reset(q[1]);
  return measureAndReturn(b, q.qubits);
}

Value repeatedResetAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  return b.measure(q[0]).second;
}

Value globalPhase(QCOProgramBuilder& b) {
  b.gphase(0.123);
  return b.intConstant(0);
}

Value singleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.cgphase(0.123, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> multipleControlledGlobalPhase(QCOProgramBuilder& b) {
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

SmallVector<Value> inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto qs = b.inv({q[0], q[1], q[2]}, [&](ValueRange qubits) {
    SmallVector controls{qubits[0], qubits[1], qubits[2]};
    auto controlsOut = b.mcgphase(-0.123, controls);
    return SmallVector<Value>(controlsOut.begin(), controlsOut.end());
  });
  return measureAndReturn(b, qs);
}

void powGphaseScaled(QCOProgramBuilder& b) {
  b.pow(3.0, {}, [&](mlir::ValueRange /*qubits*/) {
    b.gphase(0.123);
    return llvm::SmallVector<mlir::Value>{};
  });
}

void powGphaseScaledRef(QCOProgramBuilder& b) { b.gphase(3.0 * 0.123); }

void negPowGphase(QCOProgramBuilder& b) {
  b.pow(-3.0, {}, [&](mlir::ValueRange /*qubits*/) {
    b.gphase(0.123);
    return llvm::SmallVector<mlir::Value>{};
  });
}

void negPowGphaseRef(QCOProgramBuilder& b) { b.gphase(-3.0 * 0.123); }

Value identity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.id(q);
  return b.measure(q).second;
}

SmallVector<Value> singleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[1], q[0]) = b.cid(q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcid({q[2], q[1]}, q[0]);
  q[2] = res.first[0];
  q[1] = res.first[1];
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledIdentity(QCOProgramBuilder& b) {
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
  return b.measure(q).second;
}

Value inverseIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.id(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledIdentity(QCOProgramBuilder& b) {
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

void powId(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.id(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

Value x(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcx({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledX(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

SmallVector<Value> repeatedControlledX(QCOProgramBuilder& b) {
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
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledX(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

SmallVector<Value> controlledTwoX(QCOProgramBuilder& b) {
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
  return b.measure(res).second;
}
void powHalfX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.x(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powHalfXRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
}

void powNegHalfX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.x(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.x(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdXRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 2.0);
  q[0] = b.rx(1.0 / 3.0 * std::numbers::pi, q[0]);
}

Value inverseGphaseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    b.gphase(-0.123);
    return b.x(qubit);
  });
  return b.measure(res).second;
}

Value inverseGphaseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    b.gphase(0.123);
    return b.barrier({qubit})[0];
  });
  return b.measure(res).second;
}

Value inverseTwoBarriersInInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    qubit = b.barrier({qubit})[0];
    return b.barrier({qubit})[0];
  });
  return b.measure(res).second;
}

Value y(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cy(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcy({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledY(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.y(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledY(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

void powHalfY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.y(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powHalfYRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-std::numbers::pi / 4.0);
  q[0] = b.ry(std::numbers::pi / 2.0, q[0]);
}

Value z(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cz(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcz({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledZ(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.z(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledZ(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

void powHalfZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.z(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThreeHalvesZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.z(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.z(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdZRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(1.0 / 3.0 * std::numbers::pi, q[0]);
}

Value h(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ch(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mch({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledH(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.h(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledH(QCOProgramBuilder& b) {
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
  return b.measure(q).second;
}

void powEvenH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.h(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powOddH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.h(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

Value hWithoutRegister(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  return b.measure(q).second;
}

Value s(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.cs(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcs({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledS(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.s(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledS(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value twoS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.s(q[0]);
  return b.measure(q[0]).second;
}

void powTwoS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.s(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powFourS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(4.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.s(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powHalfS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.s(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.s(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdSRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
}

Value sdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSdg(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sdg(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledSdg(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value twoSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.sdg(q[0]);
  return b.measure(q[0]).second;
}

void powTwoSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powHalfSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdSdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
}

Value t_(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ct(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mct({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledT(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.t(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledT(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value twoT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.t(q[0]);
  return b.measure(q[0]).second;
}

void powTwoT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.t(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.t(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdTRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
}

Value tdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mctdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledTdg(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.tdg(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledTdg(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value twoTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.tdg(q[0]);
  return b.measure(q[0]).second;
}

void powTwoTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.tdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.tdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdTdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(-1.0 / 3.0 * std::numbers::pi / 4.0, q[0]);
}

Value sx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csx(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsx({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSx(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sx(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledSx(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value twoSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sx(q[0]);
  return b.measure(q[0]).second;
}

void powTwoSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sx(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powTwoSxRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
}

void powThirdSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sx(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdSxRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(-1.0 / 3.0 * std::numbers::pi / 4.0);
  q[0] = b.rx(1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
}

Value sxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.csxdg(q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcsxdg({q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSxdg(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sxdg(qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledSxdg(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value twoSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sxdg(q[0]);
  return b.measure(q[0]).second;
}

void powTwoSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sxdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powTwoSxdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
}

void powThirdSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0 / 3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.sxdg(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powThirdSxdgRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.gphase(1.0 / 3.0 * std::numbers::pi / 4.0);
  q[0] = b.rx(-1.0 / 3.0 * std::numbers::pi / 2.0, q[0]);
}

Value rx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.crx(0.123, q[0], q[1]);
  q[0] = res.first;
  q[1] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcrx(0.123, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRx(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.rx(-0.123, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledRx(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value rxPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(std::numbers::pi / 2, q[0]);
  return b.measure(q[0]).second;
}

void powRxScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.rx(0.123, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void rxScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.246, q[0]);
}

Value ry(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cry(0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcry(0.456, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRy(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.ry(-0.456, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledRy(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value ryPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(std::numbers::pi / 2, q[0]);
  return b.measure(q[0]).second;
}

Value rz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.crz(0.789, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcrz(0.789, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRz(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.rz(-0.789, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledRz(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value p(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(0.123, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cp(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcp(0.123, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledP(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.p(-0.123, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledP(QCOProgramBuilder& b) {
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
  return b.measure(q).second;
}

Value r(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0.456, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledR(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.inv(q[0], [&](Value qubit) { return b.r(-0.123, 0.456, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledR(QCOProgramBuilder& b) {
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

void powRScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.r(0.123, 0.456, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powRScaledRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(3.0 * 0.123, 0.456, q[0]);
}

Value canonicalizeRToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0., q[0]);
  return b.measure(q[0]).second;
}

Value canonicalizeRToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.456, std::numbers::pi / 2, q[0]);
  return b.measure(q[0]).second;
}

Value twoR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.045, 0.456, q[0]);
  q[0] = b.r(0.078, 0.456, q[0]);
  return b.measure(q[0]).second;
}

Value u2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0.234, 0.567, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledU2(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      q[0], [&](Value qubit) { return b.u2(-0.567 + pi, -0.234 - pi, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledU2(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value canonicalizeU2ToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(-std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return b.measure(q[0]).second;
}

Value canonicalizeU2ToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., 0., q[0]);
  return b.measure(q[0]).second;
}

Value u(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.1, 0.2, 0.3, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> singleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledU(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value inverseU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.inv(q[0], [&](Value qubit) { return b.u(-0.1, -0.3, -0.2, qubit); });
  return b.measure(res).second;
}

SmallVector<Value> inverseMultipleControlledU(QCOProgramBuilder& b) {
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
  return b.measure(q[0]).second;
}

Value canonicalizeUToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.123, -std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return b.measure(q[0]).second;
}

Value canonicalizeUToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.456, 0., 0., q[0]);
  return b.measure(q[0]).second;
}

Value canonicalizeUToU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(std::numbers::pi / 2, 0.234, 0.567, q[0]);
  return b.measure(q[0]).second;
}

SmallVector<Value> swap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cswap(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcswap({q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSwap(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcswap({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.swap(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledSwap(QCOProgramBuilder& b) {
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

SmallVector<Value> twoSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoSwapSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.swap(q[0], q[1]);
  std::tie(q[1], q[0]) = b.swap(q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

void powEvenSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void powOddSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

SmallVector<Value> iswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.iswap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.ciswap(q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mciswap({q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledIswap(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mciswap({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.iswap(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledIswap(QCOProgramBuilder& b) {
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

void powHalfIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.5, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.iswap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void powHalfIswapRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(-std::numbers::pi / 2.0, 0.0, q[0], q[1]);
}

SmallVector<Value> dcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cdcx(q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcdcx({q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledDcx(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcdcx({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[1], q[0]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.dcx(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[1] = res[0];
  q[0] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledDcx(QCOProgramBuilder& b) {
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

SmallVector<Value> twoDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoDcxSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  std::tie(q[1], q[0]) = b.dcx(q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ecr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cecr(q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [fst, snd] = b.mcecr({q[0], q[1]}, q[2], q[3]);
  q[0] = fst[0];
  q[1] = fst[1];
  q[2] = snd.first;
  q[3] = snd.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledEcr(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcecr({}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.ecr(qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledEcr(QCOProgramBuilder& b) {
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

SmallVector<Value> twoEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  std::tie(q[0], q[1]) = b.ecr(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

void powEvenEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void powOddEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

SmallVector<Value> rxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.crxx(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRxx(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcrxx(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.rxx(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRxx(QCOProgramBuilder& b) {
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

SmallVector<Value> tripleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  auto [c, t] = b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = c[2];
  q[3] = t.first;
  q[4] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> fourControlledRxx(QCOProgramBuilder& b) {
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

SmallVector<Value> twoRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rxx(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rxx(0.078, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRxxSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rxx(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rxx(0.078, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRxxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rxx(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRxxOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rxx(-0.123, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ryy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cryy(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRyy(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcryy(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.ryy(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRyy(QCOProgramBuilder& b) {
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

SmallVector<Value> twoRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.ryy(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.ryy(0.078, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRyyOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.ryy(-0.123, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRyyOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.ryy(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.ryy(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRyySwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.ryy(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.ryy(0.078, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.crzx(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRzx(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcrzx(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.rzx(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRzx(QCOProgramBuilder& b) {
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

SmallVector<Value> twoRzxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzx(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzx(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.crzz(0.123, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRzz(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcrzz(0.123, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.rzz(-0.123, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledRzz(QCOProgramBuilder& b) {
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

SmallVector<Value> twoRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rzz(0.045, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzz(0.078, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRzzSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  // 0.045 + 0.078 = 0.123
  std::tie(q[0], q[1]) = b.rzz(0.045, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rzz(0.078, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRzzOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  std::tie(q[0], q[1]) = b.rzz(-0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoRzzOppositePhaseSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rzz(0.123, q[0], q[1]);
  std::tie(q[1], q[0]) = b.rzz(-0.123, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> xxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledXxPlusYY(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b) {
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

void powXxPlusYYScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto [q0, q1] = b.xx_plus_yy(0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
}

void powXxPlusYYScaledRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
}

SmallVector<Value> twoXxPlusYYOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  std::tie(q[0], q[1]) = b.xx_plus_yy(-0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoXxPlusYYSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_plus_yy(0.045, 0.456, q[0], q[1]);
  std::tie(q[1], q[0]) = b.xx_plus_yy(0.078, 0.456, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> xxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto [c, t] = b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = c;
  q[1] = t.first;
  q[2] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [c, t] = b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = c[0];
  q[1] = c[1];
  q[2] = t.first;
  q[3] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledXxMinusYY(QCOProgramBuilder& b) {
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

SmallVector<Value> trivialControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [c, t] = b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = t.first;
  q[1] = t.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto [fst, snd] = b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{fst, snd};
  });
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b) {
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

void powXxMinusYYScaled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(3.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto [q0, q1] = b.xx_minus_yy(0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
}

void powXxMinusYYScaledRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(3.0 * 0.123, 0.456, q[0], q[1]);
}

SmallVector<Value> twoXxMinusYYOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  std::tie(q[0], q[1]) = b.xx_minus_yy(-0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoXxMinusYYSwappedTargets(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.xx_minus_yy(0.045, 0.456, q[0], q[1]);
  std::tie(q[1], q[0]) = b.xx_minus_yy(0.078, 0.456, q[1], q[0]);
  return measureAndReturn(b, q.qubits);
}

Value barrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.barrier(q[0])[0];
  return b.measure(q[0]).second;
}

SmallVector<Value> barrierTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.barrier({q[0], q[1]});
  q[0] = res[0];
  q[1] = res[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> barrierMultipleQubits(QCOProgramBuilder& b) {
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
  return b.measure(res.second).second;
}

Value inverseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.barrier({qubit})[0]; });
  return b.measure(res).second;
}

void powBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.barrier(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

SmallVector<Value> twoBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto b1 = b.barrier({q[0], q[1]});
  q[0] = b1[0];
  q[1] = b1[1];
  auto b2 = b.barrier({q[0], q[1]});
  q[0] = b2[0];
  q[1] = b2[1];
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> trivialCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [_, q01] = b.ctrl({}, {q[0], q[1]}, [&](ValueRange targets) {
    auto [q0, q1] = b.rxx(0.123, targets[0], targets[1]);
    return SmallVector{q0, q1};
  });
  return measureAndReturn(b, q01);
}

SmallVector<Value> emptyCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  auto [res0, res1] = b.ctrl(q[0], q[1], [&](Value target) { return target; });
  return measureAndReturn(b, {res0, res1});
}

SmallVector<Value> nestedCtrl(QCOProgramBuilder& b) {
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

SmallVector<Value> tripleNestedCtrl(QCOProgramBuilder& b) {
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

SmallVector<Value> doubleNestedCtrlTwoQubits(QCOProgramBuilder& b) {
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

SmallVector<Value> ctrlInvSandwich(QCOProgramBuilder& b) {
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

SmallVector<Value> ctrlTwo(QCOProgramBuilder& b) {
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

SmallVector<Value> ctrlTwoMixed(QCOProgramBuilder& b) {
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

SmallVector<Value> nestedCtrlTwo(QCOProgramBuilder& b) {
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

SmallVector<Value> ctrlInvTwo(QCOProgramBuilder& b) {
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

SmallVector<Value> emptyInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) { return qubits; });
  return measureAndReturn(b, res);
}

void emptyPow(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) { return qubits; });
}

SmallVector<Value> nestedInv(QCOProgramBuilder& b) {
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

SmallVector<Value> tripleNestedInv(QCOProgramBuilder& b) {
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

SmallVector<Value> invCtrlSandwich(QCOProgramBuilder& b) {
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

SmallVector<Value> invTwo(QCOProgramBuilder& b) {
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

void powTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    auto i0 = qubits[0];
    auto i1 = qubits[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
}

SmallVector<Value> invCtrlTwo(QCOProgramBuilder& b) {
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

void pow1Inline(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(1.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.rx(0.123, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void pow0Erase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(0.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.rx(0.123, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void pow0Two(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.0, {q[0], q[1]}, [&](ValueRange qubits) {
    auto i0 = qubits[0];
    auto i1 = qubits[1];
    i0 = b.x(i0);
    std::tie(i0, i1) = b.rxx(0.123, i0, i1);
    return SmallVector{i0, i1};
  });
}

void nestedPow(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(3.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto inner = b.pow(2.0, {qubits[0]}, [&](mlir::ValueRange innerQubits) {
      auto q0 = b.rx(0.123, innerQubits[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void powSingleExponent(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(6.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.rx(0.123, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto [q0, q1] = b.rxx(0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
}

void negPowRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.rx(0.123, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void negPowH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.h(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void invPowHFrac(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(0.5, {invArgs[0]}, [&](mlir::ValueRange powArgs) {
      auto q0 = b.h(powArgs[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void powHFracNeg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(-0.5, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.h(qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void invPowEvenH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(2.0, {invArgs[0]}, [&](mlir::ValueRange powArgs) {
      auto q0 = b.h(powArgs[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void invPowEvenSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange invArgs) {
    auto inner =
        b.pow(2.0, {invArgs[0], invArgs[1]}, [&](mlir::ValueRange powArgs) {
          auto res = b.swap(powArgs[0], powArgs[1]);
          return llvm::SmallVector<mlir::Value>{res.first, res.second};
        });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void invPowSquaredZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(2.0, {invArgs[0]}, [&](mlir::ValueRange powArgs) {
      auto q0 = b.z(powArgs[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void invPowRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange invArgs) {
    auto inner = b.pow(2.0, {invArgs[0]}, [&](mlir::ValueRange powArgs) {
      auto q0 = b.rx(0.123, powArgs[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void invPowReordered(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange invArgs) {
    auto inner =
        b.pow(0.5, {invArgs[1], invArgs[0]}, [&](mlir::ValueRange powArgs) {
          auto res = b.swap(powArgs[0], powArgs[1]);
          return llvm::SmallVector<mlir::Value>{res.first, res.second};
        });
    return llvm::SmallVector<mlir::Value>{inner[1], inner[0]};
  });
}

void invPowReorderedRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(-0.5, {q[1], q[0]}, [&](mlir::ValueRange powArgs) {
    auto res = b.swap(powArgs[0], powArgs[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void mergeNestedPowReordered(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.5, {q[0], q[1]}, [&](mlir::ValueRange outerArgs) {
    auto inner =
        b.pow(0.5, {outerArgs[1], outerArgs[0]}, [&](mlir::ValueRange powArgs) {
          auto res = b.swap(powArgs[0], powArgs[1]);
          return llvm::SmallVector<mlir::Value>{res.first, res.second};
        });
    return llvm::SmallVector<mlir::Value>{inner[1], inner[0]};
  });
}

void mergeNestedPowReorderedRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(0.25, {q[1], q[0]}, [&](mlir::ValueRange powArgs) {
    auto res = b.swap(powArgs[0], powArgs[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void powRxNeg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.pow(2.0, {q[0]}, [&](mlir::ValueRange qubits) {
    auto q0 = b.rx(-0.123, qubits[0]);
    return llvm::SmallVector<mlir::Value>{q0};
  });
}

void powCtrlRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(2.0, {q[0], q[1]}, [&](mlir::ValueRange powArgs) {
    const auto& [controlsOut, targetsOut] =
        b.ctrl({powArgs[0]}, {powArgs[1]}, [&](mlir::ValueRange targets) {
          return llvm::SmallVector<mlir::Value>{b.rx(0.123, targets[0])};
        });
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targetsOut));
  });
}

void ctrlPowRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({q[0]}, {q[1]}, [&](mlir::ValueRange targets) {
    auto inner = b.pow(2.0, {targets[0]}, [&](mlir::ValueRange powArgs) {
      auto q0 = b.rx(0.123, powArgs[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void negPowInvIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.pow(-2.0, {q[0], q[1]}, [&](mlir::ValueRange qubits) {
    return b.inv({qubits[0], qubits[1]}, [&](mlir::ValueRange invArgs) {
      auto [q0, q1] = b.iswap(invArgs[0], invArgs[1]);
      return llvm::SmallVector<mlir::Value>{q0, q1};
    });
  });
}

void negPowInvIswapRef(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(-2.0 * std::numbers::pi, 0.0, q[0], q[1]);
}

void ctrlPowSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({q[0]}, {q[1]}, [&](mlir::ValueRange targets) {
    auto inner = b.pow(1.0 / 3.0, {targets[0]}, [&](mlir::ValueRange powArgs) {
      auto q0 = b.sx(powArgs[0]);
      return llvm::SmallVector<mlir::Value>{q0};
    });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

SmallVector<Value> simpleIf(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  auto res = b.qcoIf(measureResult, measuredQubit, [&](ValueRange args) {
    auto innerQubit = b.x(args[0]);
    return SmallVector{innerQubit};
  });
  q[0] = res[0];
  auto [q1, bit] = b.measure(q[0]);
  return {measureResult, bit};
}

SmallVector<Value> ifTwoQubits(QCOProgramBuilder& b) {
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
  auto [q0_, c0] = b.measure(q[0]);
  auto [q1, c1] = b.measure(q[1]);
  return {measureResult, c0, c1};
}

SmallVector<Value> ifElse(QCOProgramBuilder& b) {
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
  auto [q0_, c0] = b.measure(q[0]);
  return {measureResult, c0};
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
  return b.measure(ifRes[0]).second;
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
  return b.measure(ifRes[0]).second;
}

SmallVector<Value> nestedTrueIf(QCOProgramBuilder& b) {
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
  auto [q1, c] = b.measure(ifRes[0]);
  return {measureResult, c};
}

SmallVector<Value> nestedFalseIf(QCOProgramBuilder& b) {
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
  auto [q1, c] = b.measure(ifRes[0]);
  return {measureResult, c};
}

SmallVector<Value> qtensorAlloc(QCOProgramBuilder& b) {
  (void)b.qtensorAlloc(3);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorDealloc(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  b.qtensorDealloc(qtensor);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorFromElements(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  auto q2 = b.allocQubit();
  (void)b.qtensorFromElements({q0, q1, q2});
  return measureAndReturn(b, {});
}

Value qtensorExtract(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [t, q] = b.qtensorExtract(qtensor, 0);
  return b.measure(q).second;
}

SmallVector<Value> qtensorInsert(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  (void)b.qtensorInsert(q1, extractOutTensor, 0);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorExtractInsertIndexMismatch(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  (void)b.qtensorInsert(q0, extractOutTensor, 1);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorExtractInsertSameIndex(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  (void)b.qtensorInsert(q0, extractOutTensor, 0);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorInsertExtractIndexMismatch(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  auto [extractOutTensor1, q2] = b.qtensorExtract(insertOutTensor, 1);
  (void)b.qtensorInsert(q2, extractOutTensor1, 0);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorInsertExtractSameIndex(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [extractOutTensor, q0] = b.qtensorExtract(qtensor, 0);
  auto q1 = b.h(q0);
  auto insertOutTensor = b.qtensorInsert(q1, extractOutTensor, 0);
  auto [extractOutTensor1, q2] = b.qtensorExtract(insertOutTensor, 0);
  (void)b.qtensorInsert(q2, extractOutTensor1, 0);
  return measureAndReturn(b, {});
}

SmallVector<Value> qtensorChain(QCOProgramBuilder& b) {
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

SmallVector<Value> qtensorAlternativeChain(QCOProgramBuilder& b) {
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
  return b.measure(scfWhile[0]).second;
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
  return b.measure(scfWhile[0]).second;
}

SmallVector<Value> simpleForLoop(QCOProgramBuilder& b) {
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
  return b.measure(scfFor[1]).second;
}

SmallVector<Value> nestedForLoopWhileOp(QCOProgramBuilder& b) {
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
  return b.measure(scfFor[1]).second;
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
  return b.measure(scfFor[1]).second;
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
  return b.measure(ifRes[1]).second;
}

SmallVector<Value> controlledXH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto [ctrl, targ] = b.ctrl(q[0], q[1], [&](Value target) {
    target = b.x(target);
    return b.h(target);
  });
  return measureAndReturn(b, {ctrl, targ});
}

SmallVector<Value> controlledInverseHT(QCOProgramBuilder& b) {
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

SmallVector<Value> inverseTwoRxRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange targets) {
    auto w0 = b.rx(0.2, targets[0]);
    auto w1 = b.ry(0.3, targets[1]);
    return SmallVector{w0, w1};
  });
  return measureAndReturn(b, res);
}

SmallVector<Value> inverseCxThenRz(QCOProgramBuilder& b) {
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

SmallVector<Value> inverseDcxThenRz(QCOProgramBuilder& b) {
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
  return b.measure(res).second;
}

Value inverseNestedInvHAndT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value target) {
    auto wire = b.inv(target, [&](Value inner) { return b.h(inner); });
    return b.t(wire);
  });
  return b.measure(res).second;
}

SmallVector<Value> inverseNestedInvHAndX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange targets) {
    auto w0 = b.inv(targets[0], [&](Value inner) { return b.h(inner); });
    auto w1 = b.x(targets[1]);
    return SmallVector{w0, w1};
  });
  return measureAndReturn(b, res);
}

SmallVector<Value> inverseThreeWireRxRyRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.inv({q[0], q[1], q[2]}, [&](ValueRange targets) {
    auto w0 = b.rx(0.2, targets[0]);
    auto w1 = b.ry(0.3, targets[1]);
    auto w2 = b.rz(0.4, targets[2]);
    return SmallVector{w0, w1, w2};
  });
  return measureAndReturn(b, res);
}

SmallVector<Value> inverseThreeWireNestedTwoInv(QCOProgramBuilder& b) {
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

SmallVector<Value> inverseWithThreeQubitOpInBody(QCOProgramBuilder& b) {
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
