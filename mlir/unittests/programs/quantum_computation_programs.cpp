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

namespace qc {

void allocQubit(QuantumComputation& comp) { comp.addQubitRegister(1, "q"); }

void allocQubitRegister(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
}

void allocMultipleQubitRegisters(QuantumComputation& comp) {
  comp.addQubitRegister(2, "reg0");
  comp.addQubitRegister(3, "reg1");
}

void allocLargeRegister(QuantumComputation& comp) {
  comp.addQubitRegister(100, "q");
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
  comp.addClassicalRegister(1, "c0");
  comp.addClassicalRegister(2, "c1");
  comp.measure(0, 0);
  comp.measure(1, 0);
  comp.measure(2, 1);
}

void resetQubitWithoutOp(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.reset(0);
}

void resetMultipleQubitsWithoutOp(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.reset(0);
  comp.reset(1);
}

void repeatedResetWithoutOp(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.reset(0);
  comp.reset(0);
  comp.reset(0);
}

void resetQubitAfterSingleOp(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.h(0);
  comp.reset(0);
}

void resetMultipleQubitsAfterSingleOp(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.h(0);
  comp.reset(0);
  comp.h(1);
  comp.reset(1);
}

void repeatedResetAfterSingleOp(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.h(0);
  comp.reset(0);
  comp.reset(0);
  comp.reset(0);
}

void globalPhase(QuantumComputation& comp) { comp.gphase(0.123); }

void identity(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.i(0);
}

void singleControlledIdentity(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ci(1, 0);
}

void multipleControlledIdentity(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mci({2, 1}, 0);
}

void x(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.x(0);
}

void singleControlledX(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cx(0, 1);
}

void multipleControlledX(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcx({0, 1}, 2);
}

void y(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.y(0);
}

void singleControlledY(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cy(0, 1);
}

void multipleControlledY(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcy({0, 1}, 2);
}

void z(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.z(0);
}

void singleControlledZ(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cz(0, 1);
}

void multipleControlledZ(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcz({0, 1}, 2);
}

void h(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.h(0);
}

void singleControlledH(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ch(0, 1);
}

void multipleControlledH(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mch({0, 1}, 2);
}

void s(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.s(0);
}

void singleControlledS(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cs(0, 1);
}

void multipleControlledS(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcs({0, 1}, 2);
}

void sdg(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.sdg(0);
}

void singleControlledSdg(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.csdg(0, 1);
}

void multipleControlledSdg(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcsdg({0, 1}, 2);
}

void t_(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.t(0);
}

void singleControlledT(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ct(0, 1);
}

void multipleControlledT(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mct({0, 1}, 2);
}

void tdg(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.tdg(0);
}

void singleControlledTdg(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ctdg(0, 1);
}

void multipleControlledTdg(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mctdg({0, 1}, 2);
}

void sx(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.sx(0);
}

void singleControlledSx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.csx(0, 1);
}

void multipleControlledSx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcsx({0, 1}, 2);
}

void sxdg(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.sxdg(0);
}

void singleControlledSxdg(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.csxdg(0, 1);
}

void multipleControlledSxdg(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcsxdg({0, 1}, 2);
}

void rx(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.rx(0.123, 0);
}

void singleControlledRx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.crx(0.123, 0, 1);
}

void multipleControlledRx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcrx(0.123, {0, 1}, 2);
}

void ry(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.ry(0.456, 0);
}

void singleControlledRy(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cry(0.456, 0, 1);
}

void multipleControlledRy(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcry(0.456, {0, 1}, 2);
}

void rz(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.rz(0.789, 0);
}

void singleControlledRz(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.crz(0.789, 0, 1);
}

void multipleControlledRz(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcrz(0.789, {0, 1}, 2);
}

void p(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.p(0.123, 0);
}

void singleControlledP(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cp(0.123, 0, 1);
}

void multipleControlledP(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcp(0.123, {0, 1}, 2);
}

void r(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.r(0.123, 0.456, 0);
}

void singleControlledR(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cr(0.123, 0.456, 0, 1);
}

void multipleControlledR(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcr(0.123, 0.456, {0, 1}, 2);
}

void u2(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.u2(0.234, 0.567, 0);
}

void singleControlledU2(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cu2(0.234, 0.567, 0, 1);
}

void multipleControlledU2(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcu2(0.234, 0.567, {0, 1}, 2);
}

void u(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.u(0.1, 0.2, 0.3, 0);
}

void singleControlledU(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.cu(0.1, 0.2, 0.3, 0, 1);
}

void multipleControlledU(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.mcu(0.1, 0.2, 0.3, {0, 1}, 2);
}

void swap(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.swap(0, 1);
}

void singleControlledSwap(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cswap(0, 1, 2);
}

void multipleControlledSwap(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcswap({0, 1}, 2, 3);
}

void iswap(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.iswap(0, 1);
}

void singleControlledIswap(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.ciswap(0, 1, 2);
}

void multipleControlledIswap(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mciswap({0, 1}, 2, 3);
}

void dcx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.dcx(0, 1);
}

void singleControlledDcx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cdcx(0, 1, 2);
}

void multipleControlledDcx(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcdcx({0, 1}, 2, 3);
}

void ecr(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ecr(0, 1);
}

void singleControlledEcr(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cecr(0, 1, 2);
}

void multipleControlledEcr(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcecr({0, 1}, 2, 3);
}

void rxx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.rxx(0.123, 0, 1);
}

void singleControlledRxx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.crxx(0.123, 0, 1, 2);
}

void multipleControlledRxx(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcrxx(0.123, {0, 1}, 2, 3);
}

void tripleControlledRxx(QuantumComputation& comp) {
  comp.addQubitRegister(5, "q");
  comp.mcrxx(0.123, {0, 1, 2}, 3, 4);
}

void ryy(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.ryy(0.123, 0, 1);
}

void singleControlledRyy(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cryy(0.123, 0, 1, 2);
}

void multipleControlledRyy(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcryy(0.123, {0, 1}, 2, 3);
}

void rzx(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.rzx(0.123, 0, 1);
}

void singleControlledRzx(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.crzx(0.123, 0, 1, 2);
}

void multipleControlledRzx(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcrzx(0.123, {0, 1}, 2, 3);
}

void rzz(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.rzz(0.123, 0, 1);
}

void singleControlledRzz(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.crzz(0.123, 0, 1, 2);
}

void multipleControlledRzz(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcrzz(0.123, {0, 1}, 2, 3);
}

void xxPlusYY(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.xx_plus_yy(0.123, 0.456, 0, 1);
}

void singleControlledXxPlusYY(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cxx_plus_yy(0.123, 0.456, 0, 1, 2);
}

void multipleControlledXxPlusYY(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcxx_plus_yy(0.123, 0.456, {0, 1}, 2, 3);
}

void xxMinusYY(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.xx_minus_yy(0.123, 0.456, 0, 1);
}

void singleControlledXxMinusYY(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.cxx_minus_yy(0.123, 0.456, 0, 1, 2);
}

void multipleControlledXxMinusYY(QuantumComputation& comp) {
  comp.addQubitRegister(4, "q");
  comp.mcxx_minus_yy(0.123, 0.456, {0, 1}, 2, 3);
}

void barrier(QuantumComputation& comp) {
  comp.addQubitRegister(1, "q");
  comp.barrier(0);
}

void barrierTwoQubits(QuantumComputation& comp) {
  comp.addQubitRegister(2, "q");
  comp.barrier({0, 1});
}

void barrierMultipleQubits(QuantumComputation& comp) {
  comp.addQubitRegister(3, "q");
  comp.barrier({0, 1, 2});
}

} // namespace qc
