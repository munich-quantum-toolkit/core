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
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <mlir/IR/Value.h>

#include <cstdint>
#include <numbers>

namespace mlir::qir {

/**
 * @brief Measures the given qubits, records the outcomes and returns
 * a single `i64` exit code with the value 0.
 * @param b The QIRProgramBuilder used to perform the measurements and create
 * the struct.
 * @param qubits The qubits to be measured.
 * @param inRegister Whether to store the results in a classical result array or
 * not.
 * @param startIndex The starting index for measurement outcomes.
 * @return The result value.
 */
static Value measureAndRecord(QIRProgramBuilder& b, ValueRange qubits,
                              const bool inRegister,
                              const int64_t startIndex = 0) {
  if (qubits.empty()) {
    return b.intConstant(0);
  }

  ClassicalRegister resultArray;
  if (inRegister) {
    resultArray =
        b.allocClassicalBitRegister(static_cast<int64_t>(qubits.size()));
  }

  for (auto i = 0L; i < qubits.size(); ++i) {
    inRegister ? b.measure(qubits[i], resultArray, i)
               : b.measure(qubits[i], startIndex + i);
  }

  return b.intConstant(0);
}

template <bool IntoRegister> Value emptyQIR(QIRProgramBuilder& b) {
  return measureAndRecord(b, {}, IntoRegister);
}

template <bool IntoRegister> Value allocQubit(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  return measureAndRecord(b, {q}, IntoRegister);
}

template <bool IntoRegister> Value alloc1QubitRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value allocQubitRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value alloc3QubitRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value allocMultipleQubitRegisters(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  return measureAndRecord(b, {q0[0], q0[1], q1[0], q1[1], q1[2]}, IntoRegister);
}

template <bool IntoRegister>
Value allocMultipleQubitRegistersWithOps(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  b.h(q0[0]);
  b.h(q0[1]);
  b.h(q1[0]);
  b.h(q1[1]);
  b.h(q1[2]);
  return measureAndRecord(b, {q0[0], q0[1], q1[0], q1[1], q1[2]}, IntoRegister);
}

template <bool IntoRegister> Value allocLargeRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(100);
  return measureAndRecord(b, {q[0], q[99]}, IntoRegister);
}

Value staticQubits(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  return measureAndRecord(b, {q0, q1}, true);
}

Value staticQubitsWithOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.h(q0);
  b.h(q1);
  return measureAndRecord(b, {q0, q1}, true);
}

Value staticQubitsWithParametricOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  return measureAndRecord(b, {q0, q1}, true);
}

Value staticQubitsWithTwoTargetOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rzz(0.123, q0, q1);
  return measureAndRecord(b, {q0, q1}, true);
}

Value staticQubitsWithCtrl(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  return measureAndRecord(b, {q0, q1}, true);
}

Value staticQubitsWithInv(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  b.tdg(q0);
  return measureAndRecord(b, {q0}, true);
}

Value staticQubitsWithDuplicates(QIRProgramBuilder& b) {
  auto q0a = b.staticQubit(0);
  auto q1a = b.staticQubit(1);
  auto q0b = b.staticQubit(0);
  auto q1b = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0a);
  b.p(std::numbers::pi / 2., q1a);
  b.rzz(0.123, q0b, q1b);
  b.cx(q0b, q1b);
  b.tdg(q0a);
  return measureAndRecord(b, {q0b, q1b}, true);
}

Value staticQubitsCanonical(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  b.rzz(0.123, q0, q1);
  b.cx(q0, q1);
  b.tdg(q0);
  return measureAndRecord(b, {q0, q1}, true);
}

Value mixedStaticThenDynamicQubit(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndRecord(b, {q0, q1}, true);
}

Value mixedDynamicRegisterThenStaticQubit(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.staticQubit(0);
  return measureAndRecord(b, {q0[0], q0[1], q1}, true);
}

Value singleMeasurementToSingleBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c, 0);
  return b.intConstant(0);
}

Value repeatedMeasurementToSameBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c, 0);
  b.measure(q[0], c, 0);
  b.measure(q[0], c, 0);
  return b.intConstant(0);
}

Value repeatedMeasurementToDifferentBits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(3);
  b.measure(q[0], c, 0);
  b.measure(q[0], c, 1);
  b.measure(q[0], c, 2);
  return b.intConstant(0);
}

Value multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1);
  const auto& c1 = b.allocClassicalBitRegister(2);
  b.measure(q[0], c0, 0);
  b.measure(q[1], c1, 0);
  b.measure(q[2], c1, 1);
  return b.intConstant(0);
}

Value partialMeasurementToRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(2);
  // Only the first bit is measured; the second is still output-recorded.
  b.measure(q[0], c, 0);
  return b.intConstant(0);
}

Value dynamicallyIndexedMeasurement(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto c = b.allocClassicalBitRegister(2);
  // The bit index is the loop induction variable, i.e. only known at runtime.
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto qubit = b.loadQubit(q.value, iv);
    b.measure(qubit, c, iv);
  });
  return b.intConstant(0);
}

Value measurementWithoutRegisters(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.measure(q, 0);
  return b.intConstant(0);
}

template <bool IntoRegister> Value resetQubitWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value resetMultipleQubitsWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value repeatedResetWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value resetQubitAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value repeatedResetAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value globalPhase(QIRProgramBuilder& b) {
  b.gphase(0.123);
  return measureAndRecord(b, {}, IntoRegister);
}

template <bool IntoRegister> Value identity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value singleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value twoQubitsOneIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.id(q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value threeQubitsOneIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.id(q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value x(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value y(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value z(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value h(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

Value hWithoutRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  return measureAndRecord(b, {q}, false);
}

template <bool IntoRegister> Value s(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value sdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value t_(QIRProgramBuilder& b) { // NOLINT(*-identifier-naming)
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value tdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value sx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value sxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value rx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value ry(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value rz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value p(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value r(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value u2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value u(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value swap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value iswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value dcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value ecr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value rxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value tripleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value ryy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value rzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value rzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value multipleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value xxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value singleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value xxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value singleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value rccx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.rccx(q[0], q[1], q[2]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister> Value singleControlledRccx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.crccx(q[0], q[1], q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

template <bool IntoRegister>
Value multipleControlledRccx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrccx({q[0], q[1]}, q[2], q[3], q[4]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

Value simpleIf(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c0 = b.allocClassicalBitRegister(1);
  const auto c1 = b.allocClassicalBitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0], c0, 0);
  b.scfIf(cond, [&] { b.x(q[0]); });
  b.measure(q[0], c1, 0);
  return b.intConstant(0);
}

Value ifElse(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c0 = b.allocClassicalBitRegister(1);
  const auto c1 = b.allocClassicalBitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0], c0, 0);
  b.scfIf(cond, [&] { b.x(q[0]); }, [&] { b.z(q[0]); });
  b.measure(q[0], c1, 0);
  return b.intConstant(0);
}

Value ifTwoQubits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  const auto c0 = b.allocClassicalBitRegister(1);
  const auto c1 = b.allocClassicalBitRegister(2);
  b.h(q[0]);
  auto cond = b.measure(q[0], c0, 0);
  b.scfIf(cond, [&] {
    b.x(q[0]);
    b.x(q[1]);
  });
  b.measure(q[0], c1, 0);
  b.measure(q[1], c1, 1);
  return b.intConstant(0);
}

template <bool IntoRegister> Value nestedIfOpForLoop(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q0 = b.allocQubit();
  b.h(q0);
  auto cond = b.measure(q0, 0, false);
  b.scfIf(
      cond, [&] { b.h(q0); },
      [&] {
        b.scfFor(0, 3, 1, [&](Value iv) {
          auto q1 = b.loadQubit(reg.value, iv);
          b.h(q1);
        });
      });
  return measureAndRecord(b, {q0}, IntoRegister, 1);
}

template <bool IntoRegister> Value simpleWhileReset(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.scfWhile(
      [&] {
        auto measureResult = b.measure(q, 0, false);
        return measureResult;
      },
      [&] { b.h(q); });
  return measureAndRecord(b, {q}, IntoRegister, 1);
}

template <bool IntoRegister> Value simpleDoWhileReset(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.scfWhile([&] {
    b.h(q);
    auto measureResult = b.measure(q, 0, false);
    return measureResult;
  });
  return measureAndRecord(b, {q}, IntoRegister, 1);
}

template <bool IntoRegister> Value simpleForLoop(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.loadQubit(reg.value, iv);
    b.h(q);
  });
  return measureAndRecord(b, reg.qubits, IntoRegister);
};

template <bool IntoRegister> Value nestedForLoopIfOp(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto qCond = b.allocQubit();
  b.scfFor(0, 2, 1, [&](Value iv) {
    b.h(qCond);
    auto cond = b.measure(qCond, 0, false);
    b.scfIf(cond, [&] {
      auto q = b.loadQubit(reg.value, iv);
      b.h(q);
    });
  });
  return measureAndRecord(b, {qCond}, IntoRegister, 1);
}

template <bool IntoRegister> Value nestedForLoopWhileOp(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.loadQubit(reg.value, iv);
    b.h(q);
  });
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.loadQubit(reg.value, iv);
    b.scfWhile(
        [&] {
          auto measureResult = b.measure(q, 0, false);
          return measureResult;
        },
        [&] { b.h(q); });
  });
  return measureAndRecord(b, reg.qubits, IntoRegister, 1);
}

template <bool IntoRegister>
Value nestedForLoopCtrlOpWithSeparateQubit(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control = b.allocQubit();
  b.h(control);
  b.scfFor(0, 3, 1, [&](Value iv) {
    auto q0 = b.loadQubit(reg.value, iv);
    b.h(q0);
    b.cx(control, q0);
  });
  return measureAndRecord(b, {control}, IntoRegister);
}

template <bool IntoRegister>
Value nestedForLoopCtrlOpWithExtractedQubit(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.h(reg[0]);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto q0 = b.loadQubit(reg.value, iv);
    b.h(q0);
    b.cx(reg[0], q0);
  });
  return measureAndRecord(b, {reg[0]}, IntoRegister);
}

template <bool IntoRegister> Value ctrlTwo(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcx({q[0], q[1]}, q[2]);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndRecord(b, q.qubits, IntoRegister);
}

// Instantiate the templates for IntoRegister = false
template Value emptyQIR<false>(QIRProgramBuilder& builder);
template Value allocQubit<false>(QIRProgramBuilder& b);
template Value alloc1QubitRegister<false>(QIRProgramBuilder& b);
template Value allocQubitRegister<false>(QIRProgramBuilder& b);
template Value alloc3QubitRegister<false>(QIRProgramBuilder& b);
template Value allocMultipleQubitRegisters<false>(QIRProgramBuilder& b);
template Value allocMultipleQubitRegistersWithOps<false>(QIRProgramBuilder& b);
template Value allocLargeRegister<false>(QIRProgramBuilder& b);
template Value resetQubitWithoutOp<false>(QIRProgramBuilder& b);
template Value resetMultipleQubitsWithoutOp<false>(QIRProgramBuilder& b);
template Value repeatedResetWithoutOp<false>(QIRProgramBuilder& b);
template Value resetQubitAfterSingleOp<false>(QIRProgramBuilder& b);
template Value resetMultipleQubitsAfterSingleOp<false>(QIRProgramBuilder& b);
template Value repeatedResetAfterSingleOp<false>(QIRProgramBuilder& b);
template Value globalPhase<false>(QIRProgramBuilder& b);
template Value identity<false>(QIRProgramBuilder& b);
template Value singleControlledIdentity<false>(QIRProgramBuilder& b);
template Value twoQubitsOneIdentity<false>(QIRProgramBuilder& b);
template Value threeQubitsOneIdentity<false>(QIRProgramBuilder& b);
template Value multipleControlledIdentity<false>(QIRProgramBuilder& b);
template Value x<false>(QIRProgramBuilder& b);
template Value singleControlledX<false>(QIRProgramBuilder& b);
template Value multipleControlledX<false>(QIRProgramBuilder& b);
template Value y<false>(QIRProgramBuilder& b);
template Value singleControlledY<false>(QIRProgramBuilder& b);
template Value multipleControlledY<false>(QIRProgramBuilder& b);
template Value z<false>(QIRProgramBuilder& b);
template Value singleControlledZ<false>(QIRProgramBuilder& b);
template Value multipleControlledZ<false>(QIRProgramBuilder& b);
template Value h<false>(QIRProgramBuilder& b);
template Value singleControlledH<false>(QIRProgramBuilder& b);
template Value multipleControlledH<false>(QIRProgramBuilder& b);
template Value s<false>(QIRProgramBuilder& b);
template Value singleControlledS<false>(QIRProgramBuilder& b);
template Value multipleControlledS<false>(QIRProgramBuilder& b);
template Value sdg<false>(QIRProgramBuilder& b);
template Value singleControlledSdg<false>(QIRProgramBuilder& b);
template Value multipleControlledSdg<false>(QIRProgramBuilder& b);
template Value t_<false>(QIRProgramBuilder& b);
template Value singleControlledT<false>(QIRProgramBuilder& b);
template Value multipleControlledT<false>(QIRProgramBuilder& b);
template Value tdg<false>(QIRProgramBuilder& b);
template Value singleControlledTdg<false>(QIRProgramBuilder& b);
template Value multipleControlledTdg<false>(QIRProgramBuilder& b);
template Value sx<false>(QIRProgramBuilder& b);
template Value singleControlledSx<false>(QIRProgramBuilder& b);
template Value multipleControlledSx<false>(QIRProgramBuilder& b);
template Value sxdg<false>(QIRProgramBuilder& b);
template Value singleControlledSxdg<false>(QIRProgramBuilder& b);
template Value multipleControlledSxdg<false>(QIRProgramBuilder& b);
template Value rx<false>(QIRProgramBuilder& b);
template Value singleControlledRx<false>(QIRProgramBuilder& b);
template Value multipleControlledRx<false>(QIRProgramBuilder& b);
template Value ry<false>(QIRProgramBuilder& b);
template Value singleControlledRy<false>(QIRProgramBuilder& b);
template Value multipleControlledRy<false>(QIRProgramBuilder& b);
template Value rz<false>(QIRProgramBuilder& b);
template Value singleControlledRz<false>(QIRProgramBuilder& b);
template Value multipleControlledRz<false>(QIRProgramBuilder& b);
template Value p<false>(QIRProgramBuilder& b);
template Value singleControlledP<false>(QIRProgramBuilder& b);
template Value multipleControlledP<false>(QIRProgramBuilder& b);
template Value r<false>(QIRProgramBuilder& b);
template Value singleControlledR<false>(QIRProgramBuilder& b);
template Value multipleControlledR<false>(QIRProgramBuilder& b);
template Value u2<false>(QIRProgramBuilder& b);
template Value singleControlledU2<false>(QIRProgramBuilder& b);
template Value multipleControlledU2<false>(QIRProgramBuilder& b);
template Value u<false>(QIRProgramBuilder& b);
template Value singleControlledU<false>(QIRProgramBuilder& b);
template Value multipleControlledU<false>(QIRProgramBuilder& b);
template Value swap<false>(QIRProgramBuilder& b);
template Value singleControlledSwap<false>(QIRProgramBuilder& b);
template Value multipleControlledSwap<false>(QIRProgramBuilder& b);
template Value iswap<false>(QIRProgramBuilder& b);
template Value singleControlledIswap<false>(QIRProgramBuilder& b);
template Value multipleControlledIswap<false>(QIRProgramBuilder& b);
template Value dcx<false>(QIRProgramBuilder& b);
template Value singleControlledDcx<false>(QIRProgramBuilder& b);
template Value multipleControlledDcx<false>(QIRProgramBuilder& b);
template Value ecr<false>(QIRProgramBuilder& b);
template Value singleControlledEcr<false>(QIRProgramBuilder& b);
template Value multipleControlledEcr<false>(QIRProgramBuilder& b);
template Value rxx<false>(QIRProgramBuilder& b);
template Value singleControlledRxx<false>(QIRProgramBuilder& b);
template Value multipleControlledRxx<false>(QIRProgramBuilder& b);
template Value tripleControlledRxx<false>(QIRProgramBuilder& b);
template Value ryy<false>(QIRProgramBuilder& b);
template Value singleControlledRyy<false>(QIRProgramBuilder& b);
template Value multipleControlledRyy<false>(QIRProgramBuilder& b);
template Value rzx<false>(QIRProgramBuilder& b);
template Value singleControlledRzx<false>(QIRProgramBuilder& b);
template Value multipleControlledRzx<false>(QIRProgramBuilder& b);
template Value rzz<false>(QIRProgramBuilder& b);
template Value singleControlledRzz<false>(QIRProgramBuilder& b);
template Value multipleControlledRzz<false>(QIRProgramBuilder& b);
template Value xxPlusYY<false>(QIRProgramBuilder& b);
template Value singleControlledXxPlusYY<false>(QIRProgramBuilder& b);
template Value multipleControlledXxPlusYY<false>(QIRProgramBuilder& b);
template Value xxMinusYY<false>(QIRProgramBuilder& b);
template Value singleControlledXxMinusYY<false>(QIRProgramBuilder& b);
template Value multipleControlledXxMinusYY<false>(QIRProgramBuilder& b);
template Value rccx<false>(QIRProgramBuilder& b);
template Value singleControlledRccx<false>(QIRProgramBuilder& b);
template Value multipleControlledRccx<false>(QIRProgramBuilder& b);
template Value nestedIfOpForLoop<false>(QIRProgramBuilder& b);
template Value simpleWhileReset<false>(QIRProgramBuilder& b);
template Value simpleDoWhileReset<false>(QIRProgramBuilder& b);
template Value simpleForLoop<false>(QIRProgramBuilder& b);
template Value nestedForLoopIfOp<false>(QIRProgramBuilder& b);
template Value nestedForLoopWhileOp<false>(QIRProgramBuilder& b);
template Value
nestedForLoopCtrlOpWithSeparateQubit<false>(QIRProgramBuilder& b);
template Value
nestedForLoopCtrlOpWithExtractedQubit<false>(QIRProgramBuilder& b);
template Value ctrlTwo<false>(QIRProgramBuilder& b);

// Instantiate the templates for IntoRegister = true
template Value emptyQIR<true>(QIRProgramBuilder& builder);
template Value allocQubit<true>(QIRProgramBuilder& b);
template Value alloc1QubitRegister<true>(QIRProgramBuilder& b);
template Value allocQubitRegister<true>(QIRProgramBuilder& b);
template Value alloc3QubitRegister<true>(QIRProgramBuilder& b);
template Value allocMultipleQubitRegisters<true>(QIRProgramBuilder& b);
template Value allocMultipleQubitRegistersWithOps<true>(QIRProgramBuilder& b);
template Value allocLargeRegister<true>(QIRProgramBuilder& b);
template Value resetQubitWithoutOp<true>(QIRProgramBuilder& b);
template Value resetMultipleQubitsWithoutOp<true>(QIRProgramBuilder& b);
template Value repeatedResetWithoutOp<true>(QIRProgramBuilder& b);
template Value resetQubitAfterSingleOp<true>(QIRProgramBuilder& b);
template Value resetMultipleQubitsAfterSingleOp<true>(QIRProgramBuilder& b);
template Value repeatedResetAfterSingleOp<true>(QIRProgramBuilder& b);
template Value globalPhase<true>(QIRProgramBuilder& b);
template Value identity<true>(QIRProgramBuilder& b);
template Value singleControlledIdentity<true>(QIRProgramBuilder& b);
template Value twoQubitsOneIdentity<true>(QIRProgramBuilder& b);
template Value threeQubitsOneIdentity<true>(QIRProgramBuilder& b);
template Value multipleControlledIdentity<true>(QIRProgramBuilder& b);
template Value x<true>(QIRProgramBuilder& b);
template Value singleControlledX<true>(QIRProgramBuilder& b);
template Value multipleControlledX<true>(QIRProgramBuilder& b);
template Value y<true>(QIRProgramBuilder& b);
template Value singleControlledY<true>(QIRProgramBuilder& b);
template Value multipleControlledY<true>(QIRProgramBuilder& b);
template Value z<true>(QIRProgramBuilder& b);
template Value singleControlledZ<true>(QIRProgramBuilder& b);
template Value multipleControlledZ<true>(QIRProgramBuilder& b);
template Value h<true>(QIRProgramBuilder& b);
template Value singleControlledH<true>(QIRProgramBuilder& b);
template Value multipleControlledH<true>(QIRProgramBuilder& b);
template Value s<true>(QIRProgramBuilder& b);
template Value singleControlledS<true>(QIRProgramBuilder& b);
template Value multipleControlledS<true>(QIRProgramBuilder& b);
template Value sdg<true>(QIRProgramBuilder& b);
template Value singleControlledSdg<true>(QIRProgramBuilder& b);
template Value multipleControlledSdg<true>(QIRProgramBuilder& b);
template Value t_<true>(QIRProgramBuilder& b);
template Value singleControlledT<true>(QIRProgramBuilder& b);
template Value multipleControlledT<true>(QIRProgramBuilder& b);
template Value tdg<true>(QIRProgramBuilder& b);
template Value singleControlledTdg<true>(QIRProgramBuilder& b);
template Value multipleControlledTdg<true>(QIRProgramBuilder& b);
template Value sx<true>(QIRProgramBuilder& b);
template Value singleControlledSx<true>(QIRProgramBuilder& b);
template Value multipleControlledSx<true>(QIRProgramBuilder& b);
template Value sxdg<true>(QIRProgramBuilder& b);
template Value singleControlledSxdg<true>(QIRProgramBuilder& b);
template Value multipleControlledSxdg<true>(QIRProgramBuilder& b);
template Value rx<true>(QIRProgramBuilder& b);
template Value singleControlledRx<true>(QIRProgramBuilder& b);
template Value multipleControlledRx<true>(QIRProgramBuilder& b);
template Value ry<true>(QIRProgramBuilder& b);
template Value singleControlledRy<true>(QIRProgramBuilder& b);
template Value multipleControlledRy<true>(QIRProgramBuilder& b);
template Value rz<true>(QIRProgramBuilder& b);
template Value singleControlledRz<true>(QIRProgramBuilder& b);
template Value multipleControlledRz<true>(QIRProgramBuilder& b);
template Value p<true>(QIRProgramBuilder& b);
template Value singleControlledP<true>(QIRProgramBuilder& b);
template Value multipleControlledP<true>(QIRProgramBuilder& b);
template Value r<true>(QIRProgramBuilder& b);
template Value singleControlledR<true>(QIRProgramBuilder& b);
template Value multipleControlledR<true>(QIRProgramBuilder& b);
template Value u2<true>(QIRProgramBuilder& b);
template Value singleControlledU2<true>(QIRProgramBuilder& b);
template Value multipleControlledU2<true>(QIRProgramBuilder& b);
template Value u<true>(QIRProgramBuilder& b);
template Value singleControlledU<true>(QIRProgramBuilder& b);
template Value multipleControlledU<true>(QIRProgramBuilder& b);
template Value swap<true>(QIRProgramBuilder& b);
template Value singleControlledSwap<true>(QIRProgramBuilder& b);
template Value multipleControlledSwap<true>(QIRProgramBuilder& b);
template Value iswap<true>(QIRProgramBuilder& b);
template Value singleControlledIswap<true>(QIRProgramBuilder& b);
template Value multipleControlledIswap<true>(QIRProgramBuilder& b);
template Value dcx<true>(QIRProgramBuilder& b);
template Value singleControlledDcx<true>(QIRProgramBuilder& b);
template Value multipleControlledDcx<true>(QIRProgramBuilder& b);
template Value ecr<true>(QIRProgramBuilder& b);
template Value singleControlledEcr<true>(QIRProgramBuilder& b);
template Value multipleControlledEcr<true>(QIRProgramBuilder& b);
template Value rxx<true>(QIRProgramBuilder& b);
template Value singleControlledRxx<true>(QIRProgramBuilder& b);
template Value multipleControlledRxx<true>(QIRProgramBuilder& b);
template Value tripleControlledRxx<true>(QIRProgramBuilder& b);
template Value ryy<true>(QIRProgramBuilder& b);
template Value singleControlledRyy<true>(QIRProgramBuilder& b);
template Value multipleControlledRyy<true>(QIRProgramBuilder& b);
template Value rzx<true>(QIRProgramBuilder& b);
template Value singleControlledRzx<true>(QIRProgramBuilder& b);
template Value multipleControlledRzx<true>(QIRProgramBuilder& b);
template Value rzz<true>(QIRProgramBuilder& b);
template Value singleControlledRzz<true>(QIRProgramBuilder& b);
template Value multipleControlledRzz<true>(QIRProgramBuilder& b);
template Value xxPlusYY<true>(QIRProgramBuilder& b);
template Value singleControlledXxPlusYY<true>(QIRProgramBuilder& b);
template Value multipleControlledXxPlusYY<true>(QIRProgramBuilder& b);
template Value xxMinusYY<true>(QIRProgramBuilder& b);
template Value singleControlledXxMinusYY<true>(QIRProgramBuilder& b);
template Value multipleControlledXxMinusYY<true>(QIRProgramBuilder& b);
template Value rccx<true>(QIRProgramBuilder& b);
template Value singleControlledRccx<true>(QIRProgramBuilder& b);
template Value multipleControlledRccx<true>(QIRProgramBuilder& b);
template Value nestedIfOpForLoop<true>(QIRProgramBuilder& b);
template Value simpleWhileReset<true>(QIRProgramBuilder& b);
template Value simpleDoWhileReset<true>(QIRProgramBuilder& b);
template Value simpleForLoop<true>(QIRProgramBuilder& b);
template Value nestedForLoopIfOp<true>(QIRProgramBuilder& b);
template Value nestedForLoopWhileOp<true>(QIRProgramBuilder& b);
template Value nestedForLoopCtrlOpWithSeparateQubit<true>(QIRProgramBuilder& b);
template Value
nestedForLoopCtrlOpWithExtractedQubit<true>(QIRProgramBuilder& b);
template Value ctrlTwo<true>(QIRProgramBuilder& b);
} // namespace mlir::qir
