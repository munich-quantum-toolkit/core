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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <numbers>

namespace mlir::qco {

void emptyQCO([[maybe_unused]] QCOProgramBuilder& builder) {}

void allocQubit(QCOProgramBuilder& b) { b.allocQubit(); }

void allocQubitRegister(QCOProgramBuilder& b) { b.allocQubitRegister(2); }

void allocMultipleQubitRegisters(QCOProgramBuilder& b) {
  b.allocQubitRegister(2, "reg0");
  b.allocQubitRegister(3, "reg1");
}

void allocLargeRegister(QCOProgramBuilder& b) { b.allocQubitRegister(100); }

void staticQubits(QCOProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
}

void allocDeallocPair(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
}

void singleMeasurementToSingleBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  q[0] = b.measure(q[0], c[0]);
}

void repeatedMeasurementToSameBit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  q[0] = b.measure(q[0], c[0]);
  q[0] = b.measure(q[0], c[0]);
  q[0] = b.measure(q[0], c[0]);
}

void repeatedMeasurementToDifferentBits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  q[0] = b.measure(q[0], c[0]);
  q[0] = b.measure(q[0], c[1]);
  q[0] = b.measure(q[0], c[2]);
}

void multipleClassicalRegistersAndMeasurements(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  b.measure(q[0], c0[0]);
  b.measure(q[1], c1[0]);
  b.measure(q[2], c1[1]);
}

void resetQubitWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
}

void resetMultipleQubitsWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.reset(q[0]);
  q[1] = b.reset(q[1]);
}

void repeatedResetWithoutOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
}

void resetQubitAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.reset(q);
}

void resetMultipleQubitsAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[1] = b.h(q[1]);
  q[1] = b.reset(q[1]);
}

void repeatedResetAfterSingleOp(QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
}

void globalPhase(QCOProgramBuilder& b) { b.gphase(0.123); }

void singleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
}

void multipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
}

void inverseGlobalPhase(QCOProgramBuilder& b) {
  b.inv({}, [&](mlir::ValueRange /*qubits*/) {
    b.gphase(-0.123);
    return llvm::SmallVector<mlir::Value>{};
  });
}

void inverseMultipleControlledGlobalPhase(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    llvm::SmallVector<mlir::Value> controls{qubits[0], qubits[1], qubits[2]};
    auto controlsOut = b.mcgphase(-0.123, controls);
    return llvm::SmallVector<mlir::Value>(controlsOut.begin(),
                                          controlsOut.end());
  });
}

void identity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
}

void singleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[1], q[0]);
}

void multipleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[2], q[1]}, q[0]);
}

void nestedControlledIdentity(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.id(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
}

void inverseIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.id(qubits[0])};
  });
}

void inverseMultipleControlledIdentity(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcid({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void x(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

void singleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
}

void multipleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
}

void nestedControlledX(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.x(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
}

void inverseX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.x(qubits[0])};
  });
}

void inverseMultipleControlledX(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcx({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void y(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
}

void singleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
}

void multipleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
}

void nestedControlledY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.y(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
}

void inverseY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.y(qubits[0])};
  });
}

void inverseMultipleControlledY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcy({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void z(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
}

void singleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
}

void multipleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
}

void nestedControlledZ(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.z(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
}

void inverseZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.z(qubits[0])};
  });
}

void inverseMultipleControlledZ(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcz({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void h(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
}

void singleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
}

void multipleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
}

void nestedControlledH(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.h(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
}

void inverseH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.h(qubits[0])};
  });
}

void inverseMultipleControlledH(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mch({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void s(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
}

void singleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
}

void multipleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
}

void nestedControlledS(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.s(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
}

void inverseS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.s(qubits[0])};
  });
}

void inverseMultipleControlledS(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcs({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void sdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
}

void singleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
}

void multipleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
}

void nestedControlledSdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.sdg(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
}

void inverseSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.sdg(qubits[0])};
  });
}

void inverseMultipleControlledSdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void t_(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
}

void singleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
}

void multipleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
}

void nestedControlledT(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.t(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
}

void inverseT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.t(qubits[0])};
  });
}

void inverseMultipleControlledT(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mct({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void tdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
}

void singleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
}

void multipleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
}

void nestedControlledTdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.tdg(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
}

void inverseTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.tdg(qubits[0])};
  });
}

void inverseMultipleControlledTdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mctdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void sx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
}

void singleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
}

void multipleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
}

void nestedControlledSx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.sx(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
}

void inverseSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.sx(qubits[0])};
  });
}

void inverseMultipleControlledSx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsx({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void sxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
}

void singleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
}

void multipleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
}

void nestedControlledSxdg(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.sxdg(innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
}

void inverseSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.sxdg(qubits[0])};
  });
}

void inverseMultipleControlledSxdg(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsxdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void rx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
}

void singleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
}

void multipleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
}

void nestedControlledRx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.rx(0.123, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
}

void inverseRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.rx(-0.123, qubits[0])};
  });
}

void inverseMultipleControlledRx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcrx(-0.123, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void ry(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
}

void singleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
}

void multipleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
}

void nestedControlledRy(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.ry(0.456, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
}

void inverseRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.ry(-0.456, qubits[0])};
  });
}

void inverseMultipleControlledRy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcry(-0.456, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void rz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
}

void singleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
}

void multipleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
}

void nestedControlledRz(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.rz(0.789, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
}

void inverseRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.rz(-0.789, qubits[0])};
  });
}

void inverseMultipleControlledRz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcrz(-0.789, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void p(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
}

void singleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
}

void multipleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
}

void nestedControlledP(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{b.p(0.123, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
}

void inverseP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.p(-0.123, qubits[0])};
  });
}

void inverseMultipleControlledP(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcp(-0.123, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void r(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
}

void singleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
}

void multipleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
}

void nestedControlledR(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{
              b.r(0.123, 0.456, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
}

void inverseR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.r(-0.123, 0.456, qubits[0])};
  });
}

void inverseMultipleControlledR(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcr(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void u2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
}

void singleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
}

void multipleControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
}

void nestedControlledU2(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{
              b.u2(0.234, 0.567, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledU2(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
}

void inverseU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{
        b.u2(-0.567 + pi, -0.234 - pi, qubits[0])};
  });
}

void inverseMultipleControlledU2(QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcu2(-0.567 + pi, -0.234 - pi, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void u(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
}

void singleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
}

void multipleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
}

void nestedControlledU(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  b.ctrl({reg[0]}, {reg[1], reg[2]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1]}, [&](mlir::ValueRange innerTargets) {
          return llvm::SmallVector<mlir::Value>{
              b.u(0.1, 0.2, 0.3, innerTargets[0])};
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
}

void inverseU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.u(-0.1, -0.3, -0.2, qubits[0])};
  });
}

void inverseMultipleControlledU(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcu(-0.1, -0.3, -0.2, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

void swap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
}

void singleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
}

void multipleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledSwap(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.swap(innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
}

void inverseSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledSwap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void iswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
}

void singleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
}

void multipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledIswap(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.iswap(innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
}

void inverseIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.iswap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledIswap(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mciswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void dcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
}

void singleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
}

void multipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledDcx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.dcx(innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
}

void inverseDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[1], q[0]}, [&](mlir::ValueRange qubits) {
    auto res = b.dcx(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledDcx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[3], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcdcx({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void ecr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
}

void singleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
}

void multipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
}

void nestedControlledEcr(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.ecr(innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
}

void inverseEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledEcr(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcecr({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void rxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
}

void singleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
}

void multipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRxx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.rxx(0.123, innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
}

void inverseRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.rxx(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrxx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void tripleControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
}

void fourControlledRxx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.mcrxx(0.123, {q[0], q[1], q[2], q[3]}, q[4], q[5]);
}

void ryy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
}

void singleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
}

void multipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRyy(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.ryy(0.123, innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
}

void inverseRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ryy(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledRyy(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcryy(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void rzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
}

void singleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
}

void multipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRzx(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.rzx(0.123, innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
}

void inverseRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.rzx(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledRzx(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrzx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void rzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
}

void singleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
}

void multipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledRzz(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.rzz(0.123, innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
}

void inverseRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.rzz(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledRzz(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrzz(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void xxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
}

void singleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

void multipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledXxPlusYY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.xx_plus_yy(0.123, 0.456, innerTargets[0],
                                         innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
}

void inverseXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledXxPlusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.mcxx_plus_yy(
        -0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void xxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
}

void singleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

void multipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

void nestedControlledXxMinusYY(QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.ctrl({reg[0]}, {reg[1], reg[2], reg[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto res = b.xx_minus_yy(0.123, 0.456, innerTargets[0],
                                          innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{res.first, res.second};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void trivialControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
}

void inverseXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

void inverseMultipleControlledXxMinusYY(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.mcxx_minus_yy(
        -0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

void barrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.barrier(q[0]);
}

void barrierTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier({q[0], q[1]});
}

void barrierMultipleQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.barrier({q[0], q[1], q[2]});
}

void singleControlledBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({q[1]}, {q[0]}, [&](mlir::ValueRange targets) {
    return llvm::SmallVector<mlir::Value>{b.barrier(targets[0])};
  });
}

void inverseBarrier(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.barrier(qubits[0])};
  });
}

void trivialCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, {q[0], q[1]}, [&](mlir::ValueRange targets) {
    auto [q0, q1] = b.rxx(0.123, targets[0], targets[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
}

void nestedCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0]}, {q[1], q[2], q[3]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0]}, {targets[1], targets[2]},
               [&](mlir::ValueRange innerTargets) {
                 auto [q0, q1] = b.rxx(0.123, innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{q0, q1};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void tripleNestedCtrl(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.ctrl({q[0]}, {q[1], q[2], q[3], q[4]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] = b.ctrl(
        {targets[0]}, {targets[1], targets[2], targets[3]},
        [&](mlir::ValueRange innerTargets) {
          const auto& [innerInnerControlsOut, innerInnerTargetsOut] =
              b.ctrl({innerTargets[0]}, {innerTargets[1], innerTargets[2]},
                     [&](mlir::ValueRange innerInnerTargets) {
                       auto [q0, q1] = b.rxx(0.123, innerInnerTargets[0],
                                             innerInnerTargets[1]);
                       return llvm::SmallVector<mlir::Value>{q0, q1};
                     });
          return llvm::to_vector(llvm::concat<mlir::Value>(
              innerInnerControlsOut, innerInnerTargetsOut));
        });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void doubleNestedCtrlTwoQubits(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(6);
  b.ctrl({q[0], q[1]}, {q[2], q[3], q[4], q[5]}, [&](mlir::ValueRange targets) {
    const auto& [innerControlsOut, innerTargetsOut] =
        b.ctrl({targets[0], targets[1]}, {targets[2], targets[3]},
               [&](mlir::ValueRange innerTargets) {
                 auto [q0, q1] = b.rxx(0.123, innerTargets[0], innerTargets[1]);
                 return llvm::SmallVector<mlir::Value>{q0, q1};
               });
    return llvm::to_vector(
        llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
  });
}

void ctrlInvSandwich(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.ctrl({q[0]}, {q[1], q[2], q[3]}, [&](mlir::ValueRange targets) {
    auto inner = b.inv(
        {targets[0], targets[1], targets[2]},
        [&](mlir::ValueRange innerTargets) {
          auto [innerControlsOut, innerTargetsOut] =
              b.ctrl({innerTargets[0]}, {innerTargets[1], innerTargets[2]},
                     [&](mlir::ValueRange innerInnerTargets) {
                       auto [q0, q1] = b.rxx(-0.123, innerInnerTargets[0],
                                             innerInnerTargets[1]);
                       return llvm::SmallVector<mlir::Value>{q0, q1};
                     });
          return llvm::to_vector(
              llvm::concat<mlir::Value>(innerControlsOut, innerTargetsOut));
        });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void nestedInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto inner =
        b.inv({qubits[0], qubits[1]}, [&](mlir::ValueRange innerQubits) {
          auto [q0, q1] = b.rxx(0.123, innerQubits[0], innerQubits[1]);
          return llvm::SmallVector<mlir::Value>{q0, q1};
        });
    return llvm::SmallVector<mlir::Value>{inner};
  });
}

void tripleNestedInv(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto inner1 =
        b.inv({qubits[0], qubits[1]}, [&](mlir::ValueRange innerQubits) {
          auto inner2 = b.inv({innerQubits[0], innerQubits[1]},
                              [&](mlir::ValueRange innerInnerQubbits) {
                                auto [q0, q1] =
                                    b.rxx(-0.123, innerInnerQubbits[0],
                                          innerInnerQubbits[1]);
                                return llvm::SmallVector<mlir::Value>{q0, q1};
                              });
          return llvm::SmallVector<mlir::Value>{inner2};
        });
    return llvm::SmallVector<mlir::Value>{inner1};
  });
}

void invCtrlSandwich(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.ctrl(
        {qubits[0]}, {qubits[1], qubits[2]}, [&](mlir::ValueRange targets) {
          auto inner = b.inv(
              {targets[0], targets[1]}, [&](mlir::ValueRange innerQubits) {
                auto [q0, q1] = b.rxx(0.123, innerQubits[0], innerQubits[1]);
                return llvm::SmallVector<mlir::Value>{q0, q1};
              });
          return llvm::SmallVector<mlir::Value>{inner};
        });
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targetsOut));
  });
}

} // namespace mlir::qco
