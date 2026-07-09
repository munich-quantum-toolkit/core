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

SmallVector<Value> emptyQCO(QCOProgramBuilder& b) { return {b.intConstant(0)}; }

SmallVector<Value> allocQubit(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  return measureAndReturn(b, {q});
}

SmallVector<Value> alloc2Qubits(QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> allocQubitNoMeasure(QCOProgramBuilder& b) {
  (void)b.allocQubit();
  return {b.intConstant(0)};
}

SmallVector<Value> alloc1QubitRegister(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(1);
  return measureAndReturn(b, reg.qubits);
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

SmallVector<Value> allocLargeRegister(QCOProgramBuilder& b) {
  auto r = b.allocQubitRegister(100);
  return measureAndReturn(b, {r[0]});
}

SmallVector<Value> staticQubitsNoMeasure(QCOProgramBuilder& b) {
  (void)b.staticQubit(0);
  (void)b.staticQubit(1);
  return {b.intConstant(0)};
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

SmallVector<Value> staticQubitsWithInv(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  q0 = b.inv(q0, [&](Value qubit) { return b.t(qubit); });
  return measureAndReturn(b, {q0});
}

SmallVector<Value> allocSinkPair(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  b.sink(q);
  return {b.intConstant(0)};
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

SmallVector<Value> deadGatesWithIfOpProgram(QCOProgramBuilder& b) {
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
        return SmallVector<Value>{q1Then};
      },
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Else = b.h(qubits[0]);
        return SmallVector<Value>{q1Else};
      })[0];

  // This is an `if` without memory effects - it can be removed.
  q1 = b.qcoIf(
      c0, {q1},
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Then = b.x(qubits[0]);
        return SmallVector<Value>{q1Then};
      },
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Else = b.h(qubits[0]);
        return SmallVector<Value>{q1Else};
      })[0];

  return {c0};
}

SmallVector<Value> deadGatesWithIfOpSimplified(QCOProgramBuilder& b) {
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
        return SmallVector<Value>{q1Then};
      },
      [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Else = b.h(qubits[0]);
        return SmallVector<Value>{q1Else};
      })[0];

  return {c0};
}

SmallVector<Value> mixedStaticThenDynamicQubit(QCOProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1});
}

SmallVector<Value> mixedDynamicRegisterThenStaticQubit(QCOProgramBuilder& b) {
  b.qtensorAlloc(2);
  auto q1 = b.staticQubit(0);
  return measureAndReturn(b, {q1});
}

SmallVector<Value> singleMeasurementToSingleBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  const auto [q1, bit] = b.measure(q[0], c[0]);
  return {bit};
}

SmallVector<Value> repeatedMeasurementToSameBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  auto [q1, _c1] = b.measure(q[0], c[0]);
  auto [q2, _c2] = b.measure(q1, c[0]);
  auto [q3, c3] = b.measure(q2, c[0]);
  return {c3};
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

SmallVector<Value> measurementWithoutRegisters(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto [q1, c] = b.measure(q);
  return {c};
}

SmallVector<Value> resetQubitWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  return measureAndReturn(b, {q});
}

SmallVector<Value> resetMultipleQubitsWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.reset(q[0]);
  q[1] = b.reset(q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> repeatedResetWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
  return measureAndReturn(b, {q});
}

SmallVector<Value> resetQubitAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[1] = b.h(q[1]);
  q[1] = b.reset(q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> repeatedResetAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  q[0] = b.reset(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> globalPhase(QCOProgramBuilder& b) {
  b.gphase(0.123);
  return {b.intConstant(0)};
}

SmallVector<Value> singleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.cgphase(0.123, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto qs = b.mcgphase(0.123, {q[0], q[1], q[2]});
  return measureAndReturn(b, qs);
}

SmallVector<Value> inverseGlobalPhase(QCOProgramBuilder& b) {
  b.inv(ValueRange{}, [&](ValueRange /*qubits*/) {
    b.gphase(-0.123);
    return SmallVector<Value>{};
  });
  return {b.intConstant(0)};
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

SmallVector<Value> identity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.id(q);
  return measureAndReturn(b, {q});
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

SmallVector<Value> trivialControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  auto res = b.mcid({}, q);
  q = res.second;
  return measureAndReturn(b, {q});
}

SmallVector<Value> inverseIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.id(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> x(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcx({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> inverseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.x(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.x(q[0]);
  q[0] = b.x(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> controlledTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.ctrl(q[0], q[1], [&](Value target) {
    target = b.x(target);
    return b.x(target);
  });
  return measureAndReturn(b, {res.first, res.second});
}

SmallVector<Value> inverseTwoX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    qubit = b.x(qubit);
    qubit = b.x(qubit);
    return qubit;
  });
  return measureAndReturn(b, {res});
}

SmallVector<Value> inverseGphaseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    b.gphase(-0.123);
    return b.x(qubit);
  });
  return measureAndReturn(b, {res});
}

SmallVector<Value> inverseGphaseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    b.gphase(0.123);
    return b.barrier({qubit})[0];
  });
  return measureAndReturn(b, {res});
}

SmallVector<Value> inverseTwoBarriersInInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) {
    qubit = b.barrier({qubit})[0];
    return b.barrier({qubit})[0];
  });
  return measureAndReturn(b, {res});
}

SmallVector<Value> y(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcy({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.y(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.y(q[0]);
  q[0] = b.y(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> z(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcz({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.z(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.z(q[0]);
  q[0] = b.z(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> h(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.h(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mch({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.h(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoH(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.h(q);
  return measureAndReturn(b, {q});
}

SmallVector<Value> hWithoutRegister(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  return measureAndReturn(b, {q});
}

SmallVector<Value> s(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcs({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.s(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> sThenSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.sdg(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.s(q[0]);
  q[0] = b.s(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> sdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsdg({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sdg(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> sdgThenS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.s(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sdg(q[0]);
  q[0] = b.sdg(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> t_(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mct({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.t(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> tThenTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.tdg(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.t(q[0]);
  q[0] = b.t(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> tdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mctdg({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.tdg(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> tdgThenT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.t(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.tdg(q[0]);
  q[0] = b.tdg(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> sx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsx({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sx(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> sxThenSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sxdg(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sx(q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> sxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcsxdg({}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.sxdg(qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> sxdgThenSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sx(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.sxdg(q[0]);
  q[0] = b.sxdg(q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcrx(0.123, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.rx(-0.123, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoRxOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(0.123, q[0]);
  q[0] = b.rx(-0.123, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rxPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rx(std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ry(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcry(0.456, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.ry(-0.456, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoRyOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(0.456, q[0]);
  q[0] = b.ry(-0.456, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> ryPiOver2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.ry(std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> rz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcrz(0.789, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.rz(-0.789, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoRzOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.rz(0.789, q[0]);
  q[0] = b.rz(-0.789, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> p(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.p(0.123, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcp(0.123, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.p(-0.123, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> twoPOppositePhase(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.p(0.123, q);
  q = b.p(-0.123, q);
  return measureAndReturn(b, {q});
}

SmallVector<Value> r(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0.456, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcr(0.123, 0.456, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.inv(q[0], [&](Value qubit) { return b.r(-0.123, 0.456, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> canonicalizeRToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.123, 0., q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> canonicalizeRToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.456, std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> twoR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.r(0.045, 0.456, q[0]);
  q[0] = b.r(0.078, 0.456, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> u2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0.234, 0.567, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcu2(0.234, 0.567, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(
      q[0], [&](Value qubit) { return b.u2(-0.567 + pi, -0.234 - pi, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> canonicalizeU2ToH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., std::numbers::pi, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> canonicalizeU2ToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(-std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> canonicalizeU2ToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u2(0., 0., q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> u(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.1, 0.2, 0.3, q[0]);
  return measureAndReturn(b, q.qubits);
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

SmallVector<Value> trivialControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.mcu(0.1, 0.2, 0.3, {}, q[0]);
  q[0] = res.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res =
      b.inv(q[0], [&](Value qubit) { return b.u(-0.1, -0.3, -0.2, qubit); });
  return measureAndReturn(b, {res});
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

SmallVector<Value> canonicalizeUToP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0., 0., 0.123, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> canonicalizeUToRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.123, -std::numbers::pi / 2, std::numbers::pi / 2, q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> canonicalizeUToRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(0.456, 0., 0., q[0]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> canonicalizeUToU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  q[0] = b.u(std::numbers::pi / 2, 0.234, 0.567, q[0]);
  return measureAndReturn(b, q.qubits);
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
  auto res = b.mcswap({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledSwap(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcswap({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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

SmallVector<Value> iswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.iswap(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.ciswap(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mciswap({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledIswap(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mciswap({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.iswap(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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

SmallVector<Value> dcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.dcx(q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.cdcx(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcdcx({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledDcx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcdcx({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[1], q[0]}, [&](ValueRange qubits) {
    auto res = b.dcx(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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
  auto res = b.cecr(q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcecr({q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledEcr(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcecr({}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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

SmallVector<Value> rxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  std::tie(q[0], q[1]) = b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> singleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.crxx(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRxx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcrxx(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.rxx(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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
  auto res = b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.first[2];
  q[3] = res.second.first;
  q[4] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> fourControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  auto res = b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.first[2];
  q[3] = res.first[3];
  q[4] = res.second.first;
  q[5] = res.second.second;
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
  auto res = b.cryy(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRyy(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcryy(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.ryy(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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
  auto res = b.crzx(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRzx(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcrzx(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.rzx(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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
  auto res = b.crzz(0.123, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledRzz(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcrzz(0.123, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.rzz(-0.123, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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
  auto res = b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledXxPlusYY(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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
  auto res = b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  q[0] = res.first;
  q[1] = res.second.first;
  q[2] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> multipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto res = b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  q[0] = res.first[0];
  q[1] = res.first[1];
  q[2] = res.second.first;
  q[3] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> nestedControlledXxMinusYY(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, reg.qubits);
}

SmallVector<Value> trivialControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
  q[0] = res.second.first;
  q[1] = res.second.second;
  return measureAndReturn(b, q.qubits);
}

SmallVector<Value> inverseXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res = b.inv({q[0], q[1]}, [&](ValueRange qubits) {
    auto res = b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return SmallVector{res.first, res.second};
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

SmallVector<Value> barrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto q1 = b.barrier(q[0]);
  return measureAndReturn(b, {q1});
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

SmallVector<Value> singleControlledBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto res =
      b.ctrl(q[1], q[0], [&](Value target) { return b.barrier({target})[0]; });
  return measureAndReturn(b, {res.second});
}

SmallVector<Value> inverseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value qubit) { return b.barrier({qubit})[0]; });
  return measureAndReturn(b, {res});
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

SmallVector<Value> ifWithAngle(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto theta = b.floatConstant(0.123);
  auto q0 = b.h(q[0]);
  auto [measuredQubit, measureResult] = b.measure(q0);
  q0 = b.qcoIf(measureResult, measuredQubit, [&](ValueRange args) {
    auto innerQubit = b.rx(theta, args[0]);
    return SmallVector{innerQubit};
  })[0];
  return measureAndReturn(b, {q0});
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

SmallVector<Value> ifOneQubitOneTensor(QCOProgramBuilder& b) {
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

SmallVector<Value> constantTrueIf(QCOProgramBuilder& b) {
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

SmallVector<Value> constantFalseIf(QCOProgramBuilder& b) {
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

SmallVector<Value> qtensorExtract(QCOProgramBuilder& b) {
  auto qtensor = b.qtensorAlloc(3);
  auto [t, q] = b.qtensorExtract(qtensor, 0);
  return measureAndReturn(b, {q});
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

SmallVector<Value> simpleWhileReset(QCOProgramBuilder& b) {
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

SmallVector<Value> simpleDoWhileReset(QCOProgramBuilder& b) {
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

SmallVector<Value> forLoopWithAngle(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto theta = b.floatConstant(0.123);
  auto scfFor =
      b.scfFor(0, 2, 1, {reg.value}, [&](Value iv, ValueRange iterArgs) {
        auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
        auto q1 = b.rx(theta, q0);
        auto insert = b.qtensorInsert(q1, t0, iv);
        return SmallVector{insert};
      });
  auto [newReg, q] = b.qtensorExtract(scfFor[0], 0);
  return measureAndReturn(b, {q});
}

SmallVector<Value> nestedForLoopIfOp(QCOProgramBuilder& b) {
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

SmallVector<Value> nestedForLoopCtrlOpWithSeparateQubit(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, {scfFor[1]});
}

SmallVector<Value> nestedForLoopCtrlOpWithExtractedQubit(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, {scfFor[1]});
}

SmallVector<Value> nestedIfOpForLoop(QCOProgramBuilder& b) {
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

SmallVector<Value> nestedIfOpForLoopWithAngle(QCOProgramBuilder& b) {
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
  return measureAndReturn(b, {res[1]});
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

SmallVector<Value> inverseGphaseBarrierX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value target) {
    b.gphase(0.25);
    auto wire = b.barrier({target})[0];
    wire = b.x(wire);
    return wire;
  });
  return measureAndReturn(b, {res});
}

SmallVector<Value> inverseNestedInvHAndT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto res = b.inv(q[0], [&](Value target) {
    auto wire = b.inv(target, [&](Value inner) { return b.h(inner); });
    return b.t(wire);
  });
  return measureAndReturn(b, {res});
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
