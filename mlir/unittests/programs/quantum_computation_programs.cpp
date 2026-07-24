/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "quantum_computation_programs.h"

#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace qc {

void allocQubit(QuantumComputation& comp) {
  auto qr = comp.addQubitRegister(1, "q");
  comp.measureAll(true, false);
}

void allocQubitRegister(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.measureAll(true, false);
}

void allocMultipleQubitRegisters(QuantumComputation& comp) {
  comp.addQubitRegister(2, "reg0");
  comp.addQubitRegister(3, "reg1");
  comp.measureAll(true, false);
}

void allocLargeRegister(QuantumComputation& comp) {
  comp.addQubitRegister(100, "q");
  comp.addClassicalRegister(2, "meas");
  comp.measure(0, 0);
  comp.measure(99, 1);
}

void singleMeasurementToSingleBit(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.addClassicalRegister(1, "c");
  comp.measure(0, 0);
}

void repeatedMeasurementToSameBit(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.addClassicalRegister(1, "c");
  comp.measure(0, 0);
  comp.measure(0, 0);
  comp.measure(0, 0);
}

void repeatedMeasurementToDifferentBits(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.addClassicalRegister(3, "c");
  comp.measure(0, 0);
  comp.measure(0, 1);
  comp.measure(0, 2);
}

void multipleClassicalRegistersAndMeasurements(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  const auto& c0 = comp.addClassicalRegister(1, "c0");
  const auto& c1 = comp.addClassicalRegister(2, "c1");
  comp.measure(0, c0[0]);
  comp.measure(1, c1[0]);
  comp.measure(2, c1[1]);
}

void resetQubitAfterSingleOp(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.addClassicalRegister(2, "c");
  comp.h(0);
  comp.measure(0, 0);
  comp.reset(0);
  comp.measure(0, 1);
}

void resetMultipleQubitsAfterSingleOp(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.addClassicalRegister(4, "c");
  comp.h(0);
  comp.measure(0, 0);
  comp.reset(0);
  comp.measure(0, 1);
  comp.h(1);
  comp.measure(1, 2);
  comp.reset(1);
  comp.measure(1, 3);
}

void repeatedResetAfterSingleOp(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.addClassicalRegister(2, "c");
  comp.h(0);
  comp.measure(0, 0);
  comp.reset(0);
  comp.reset(0);
  comp.reset(0);
  comp.measure(0, 1);
}

void globalPhase(QuantumComputation& comp) { comp.gphase(0.123); }

void identity(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.i(0);
  comp.measureAll(true, false);
}

void singleControlledIdentity(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ci(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledIdentity(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mci({0, 1}, 2);
  comp.measureAll(true, false);
}

void x(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.x(0);
  comp.measureAll(true, false);
}

void singleControlledX(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cx(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledX(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcx({0, 1}, 2);
  comp.measureAll(true, false);
}

void y(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.y(0);
  comp.measureAll(true, false);
}

void singleControlledY(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cy(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledY(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcy({0, 1}, 2);
  comp.measureAll(true, false);
}

void z(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.z(0);
  comp.measureAll(true, false);
}

void singleControlledZ(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cz(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledZ(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcz({0, 1}, 2);
  comp.measureAll(true, false);
}

void h(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.h(0);
  comp.measureAll(true, false);
}

void singleControlledH(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ch(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledH(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mch({0, 1}, 2);
  comp.measureAll(true, false);
}

void s(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.s(0);
  comp.measureAll(true, false);
}

void singleControlledS(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cs(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledS(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcs({0, 1}, 2);
  comp.measureAll(true, false);
}

void sdg(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.sdg(0);
  comp.measureAll(true, false);
}

void singleControlledSdg(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.csdg(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledSdg(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcsdg({0, 1}, 2);
  comp.measureAll(true, false);
}

void t_(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.t(0);
  comp.measureAll(true, false);
}

void singleControlledT(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ct(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledT(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mct({0, 1}, 2);
  comp.measureAll(true, false);
}

void tdg(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.tdg(0);
  comp.measureAll(true, false);
}

void singleControlledTdg(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ctdg(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledTdg(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mctdg({0, 1}, 2);
  comp.measureAll(true, false);
}

void sx(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.sx(0);
  comp.measureAll(true, false);
}

void singleControlledSx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.csx(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledSx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcsx({0, 1}, 2);
  comp.measureAll(true, false);
}

void sxdg(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.sxdg(0);
  comp.measureAll(true, false);
}

void singleControlledSxdg(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.csxdg(0, 1);
  comp.measureAll(true, false);
}

void multipleControlledSxdg(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcsxdg({0, 1}, 2);
  comp.measureAll(true, false);
}

void rx(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.rx(0.123, 0);
  comp.measureAll(true, false);
}

void singleControlledRx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.crx(0.123, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledRx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcrx(0.123, {0, 1}, 2);
  comp.measureAll(true, false);
}

void ry(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.ry(0.456, 0);
  comp.measureAll(true, false);
}

void singleControlledRy(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cry(0.456, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledRy(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcry(0.456, {0, 1}, 2);
  comp.measureAll(true, false);
}

void rz(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.rz(0.789, 0);
  comp.measureAll(true, false);
}

void singleControlledRz(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.crz(0.789, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledRz(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcrz(0.789, {0, 1}, 2);
  comp.measureAll(true, false);
}

void p(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.p(0.123, 0);
  comp.measureAll(true, false);
}

void singleControlledP(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cp(0.123, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledP(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcp(0.123, {0, 1}, 2);
  comp.measureAll(true, false);
}

void r(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.r(0.123, 0.456, 0);
  comp.measureAll(true, false);
}

void singleControlledR(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cr(0.123, 0.456, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledR(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcr(0.123, 0.456, {0, 1}, 2);
  comp.measureAll(true, false);
}

void u2(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.u2(0.234, 0.567, 0);
  comp.measureAll(true, false);
}

void singleControlledU2(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cu2(0.234, 0.567, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledU2(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcu2(0.234, 0.567, {0, 1}, 2);
  comp.measureAll(true, false);
}

void u(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.u(0.1, 0.2, 0.3, 0);
  comp.measureAll(true, false);
}

void singleControlledU(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cu(0.1, 0.2, 0.3, 0, 1);
  comp.measureAll(true, false);
}

void multipleControlledU(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcu(0.1, 0.2, 0.3, {0, 1}, 2);
  comp.measureAll(true, false);
}

void swap(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.swap(0, 1);
  comp.measureAll(true, false);
}

void singleControlledSwap(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cswap(0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledSwap(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcswap({0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void iswap(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.iswap(0, 1);
  comp.measureAll(true, false);
}

void singleControlledIswap(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.ciswap(0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledIswap(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mciswap({0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void inverseIswap(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.iswapdg(0, 1);
  comp.measureAll(true, false);
}

void inverseMultipleControlledIswap(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mciswapdg({0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void dcx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.dcx(0, 1);
  comp.measureAll(true, false);
}

void singleControlledDcx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cdcx(0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledDcx(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcdcx({0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void ecr(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ecr(0, 1);
  comp.measureAll(true, false);
}

void singleControlledEcr(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cecr(0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledEcr(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcecr({0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void rxx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.rxx(0.123, 0, 1);
  comp.measureAll(true, false);
}

void singleControlledRxx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.crxx(0.123, 0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledRxx(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcrxx(0.123, {0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void tripleControlledRxx(QuantumComputation& comp) {
  comp.addQubitRegister(5, "q");
  comp.mcrxx(0.123, {0, 1, 2}, 3, 4);
  comp.measureAll(true, false);
}

void ryy(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ryy(0.123, 0, 1);
  comp.measureAll(true, false);
}

void singleControlledRyy(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cryy(0.123, 0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledRyy(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcryy(0.123, {0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void rzx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.rzx(0.123, 0, 1);
  comp.measureAll(true, false);
}

void singleControlledRzx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.crzx(0.123, 0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledRzx(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcrzx(0.123, {0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void rzz(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.rzz(0.123, 0, 1);
  comp.measureAll(true, false);
}

void singleControlledRzz(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.crzz(0.123, 0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledRzz(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcrzz(0.123, {0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void xxPlusYY(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.xx_plus_yy(0.123, 0.456, 0, 1);
  comp.measureAll(true, false);
}

void singleControlledXxPlusYY(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cxx_plus_yy(0.123, 0.456, 0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledXxPlusYY(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcxx_plus_yy(0.123, 0.456, {0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void xxMinusYY(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.xx_minus_yy(0.123, 0.456, 0, 1);
  comp.measureAll(true, false);
}

void singleControlledXxMinusYY(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cxx_minus_yy(0.123, 0.456, 0, 1, 2);
  comp.measureAll(true, false);
}

void multipleControlledXxMinusYY(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcxx_minus_yy(0.123, 0.456, {0, 1}, 2, 3);
  comp.measureAll(true, false);
}

void rccx(QuantumComputation& comp) {
  comp.addQubitRegister(3);
  comp.rccx(0, 1, 2);
  comp.measureAll(true, false);
}

void singleControlledRccx(QuantumComputation& comp) {
  comp.addQubitRegister(4);
  comp.crccx(0, 1, 2, 3);
  comp.measureAll(true, false);
}

void multipleControlledRccx(QuantumComputation& comp) {
  comp.addQubitRegister(5);
  comp.mcrccx({0, 1}, 2, 3, 4);
  comp.measureAll(true, false);
}

void barrier(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.barrier(0);
  comp.measureAll(true, false);
}

void barrierTwoQubits(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.barrier({0, 1});
  comp.measureAll(true, false);
}

void barrierMultipleQubits(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.barrier({0, 1, 2});
  comp.measureAll(true, false);
}

void ctrlTwo(QuantumComputation& comp) {
  const auto& q = comp.addQubitRegister(4, "q");
  CompoundOperation compound;
  compound.emplace_back<StandardOperation>(2, X);
  compound.emplace_back<StandardOperation>(Targets{2, 3}, RXX,
                                           std::vector{0.123});
  compound.addControl(0);
  compound.addControl(1);
  comp.emplace_back<CompoundOperation>(std::move(compound));
  comp.measureAll(true, false);
}

void ctrlTwoMixed(QuantumComputation& comp) {
  const auto& q = comp.addQubitRegister(4, "q");
  CompoundOperation compound;
  compound.emplace_back<StandardOperation>(2, 3, X);
  compound.emplace_back<StandardOperation>(Targets{2, 3}, RXX,
                                           std::vector{0.123});
  compound.addControl(0);
  compound.addControl(1);
  comp.emplace_back<CompoundOperation>(std::move(compound));
  comp.measureAll(true, false);
}

void simpleIf(QuantumComputation& comp) {
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(1, "c");
  comp.h(q[0]);
  comp.measure(q[0], c[0]);
  comp.if_(X, q[0], c[0]);
  const auto& c2 = comp.addClassicalRegister(1, "meas");
  comp.measure(q[0], c2[0]);
}

void ifElse(QuantumComputation& comp) {
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(1, "c");
  comp.h(q[0]);
  comp.measure(q[0], c[0]);
  comp.ifElse(std::make_unique<StandardOperation>(q[0], X),
              std::make_unique<StandardOperation>(q[0], Z), c[0]);
  comp.measureAll(true, false);
}

void ifTwoQubits(QuantumComputation& comp) {
  const auto& q = comp.addQubitRegister(2, "q");
  const auto& c = comp.addClassicalRegister(1, "c");
  comp.h(q[0]);
  comp.measure(q[0], c[0]);
  CompoundOperation compound;
  compound.emplace_back<StandardOperation>(0, X);
  compound.emplace_back<StandardOperation>(1, X);
  IfElseOperation ifElse(
      std::make_unique<CompoundOperation>(std::move(compound)), nullptr, c[0]);
  comp.emplace_back<IfElseOperation>(std::move(ifElse));
  comp.measureAll(true, false);
}

void ifWithMeasurement(QuantumComputation& comp) {
  const auto& q = comp.addQubitRegister(1, "q");
  const auto& c = comp.addClassicalRegister(1, "c");
  const auto& meas = comp.addClassicalRegister(1, "meas");
  comp.h(q[0]);
  comp.measure(q[0], c[0]);
  // The `then` branch measures the qubit into a separate register.
  comp.emplace_back<IfElseOperation>(
      std::make_unique<NonUnitaryOperation>(q[0], meas[0]), nullptr, c[0]);
}

} // namespace qc
