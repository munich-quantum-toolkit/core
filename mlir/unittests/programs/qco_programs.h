/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include <llvm/ADT/STLExtras.h>
#include <numbers>

// --- Qubit Management ----------------------------------------------------- //

/// Allocates a single qubit.
inline void allocQubit(mlir::qco::QCOProgramBuilder& b) { b.allocQubit(); }

/// Allocates a qubit register of size `2`.
inline void allocQubitRegister(mlir::qco::QCOProgramBuilder& b) {
  b.allocQubitRegister(2);
}

/// Allocates two qubit registers of size `2` and `3`.
inline void allocMultipleQubitRegisters(mlir::qco::QCOProgramBuilder& b) {
  b.allocQubitRegister(2, "reg0");
  b.allocQubitRegister(3, "reg1");
}

/// Allocates a large qubit register.
inline void allocLargeRegister(mlir::qco::QCOProgramBuilder& b) {
  b.allocQubitRegister(100);
}

/// Allocates two inline qubits.
inline void staticQubits(mlir::qco::QCOProgramBuilder& b) {
  b.staticQubit(0);
  b.staticQubit(1);
}

/// Allocates and explicitly deallocates a single qubit.
inline void allocDeallocPair(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  b.dealloc(q);
}

// --- MeasureOp ------------------------------------------------------------ //

/// Measures a single qubit into a single classical bit.
inline void singleMeasurementToSingleBit(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  q[0] = b.measure(q[0], c[0]);
}

/// Repeatedly measures a single qubit into the same classical bit.
inline void repeatedMeasurementToSameBit(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(1);
  q[0] = b.measure(q[0], c[0]);
  q[0] = b.measure(q[0], c[0]);
  q[0] = b.measure(q[0], c[0]);
}

/// Repeatedly measures a single qubit into different classical bits.
inline void
repeatedMeasurementToDifferentBits(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto& c = b.allocClassicalBitRegister(3);
  q[0] = b.measure(q[0], c[0]);
  q[0] = b.measure(q[0], c[1]);
  q[0] = b.measure(q[0], c[2]);
}

/// Measures multiple qubits into multiple classical bits.
inline void
multipleClassicalRegistersAndMeasurements(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  b.measure(q[0], c0[0]);
  b.measure(q[1], c1[0]);
  b.measure(q[2], c1[1]);
}

// --- ResetOp -------------------------------------------------------------- //

/// Resets a single qubit without any operations being applied.
inline void resetQubitWithoutOp(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
}

/// Resets multiple qubits without any operations being applied.
inline void resetMultipleQubitsWithoutOp(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.reset(q[0]);
  q[1] = b.reset(q[1]);
}

/// Repeatedly resets a single qubit without any operations being applied.
inline void repeatedResetWithoutOp(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
}

/// Resets a single qubit after a single operation.
inline void resetQubitAfterSingleOp(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.reset(q);
}

/// Resets multiple qubits after a single operation.
inline void resetMultipleQubitsAfterSingleOp(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  q[0] = b.h(q[0]);
  q[0] = b.reset(q[0]);
  q[1] = b.h(q[1]);
  q[1] = b.reset(q[1]);
}

/// Repeatedly resets a single qubit after a single operation.
inline void repeatedResetAfterSingleOp(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubit();
  q = b.h(q);
  q = b.reset(q);
  q = b.reset(q);
  q = b.reset(q);
}

// --- GPhaseOp ------------------------------------------------------------- //

/// Creates a circuit with just a global phase.
inline void globalPhase(mlir::qco::QCOProgramBuilder& b) { b.gphase(0.123); }

/// Creates a controlled global phase gate with a single control qubit.
inline void singleControlledGlobalPhase(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.cgphase(0.123, q[0]);
}

/// Creates a multi-controlled global phase gate with multiple control qubits.
inline void multipleControlledGlobalPhase(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcgphase(0.123, {q[0], q[1], q[2]});
}

/// Creates a circuit with an inverse modifier applied to a global phase gate.
inline void inverseGlobalPhase(mlir::qco::QCOProgramBuilder& b) {
  b.inv({}, [&](mlir::ValueRange /*qubits*/) {
    b.gphase(-0.123);
    return llvm::SmallVector<mlir::Value>{};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled global
/// phase gate.
inline void
inverseMultipleControlledGlobalPhase(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    llvm::SmallVector<mlir::Value> controls{qubits[0], qubits[1], qubits[2]};
    auto controlsOut = b.mcgphase(-0.123, controls);
    return llvm::SmallVector<mlir::Value>(controlsOut.begin(),
                                          controlsOut.end());
  });
}

// --- IdOp ----------------------------------------------------------------- //

/// Creates a circuit with just an identity gate.
inline void identity(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
}

/// Creates a controlled identity gate with a single control qubit.
inline void singleControlledIdentity(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[1], q[0]);
}

/// Creates a multi-controlled identity gate with multiple control qubits.
inline void multipleControlledIdentity(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[2], q[1]}, q[0]);
}

/// Creates a circuit with a nested controlled identity gate.
inline void nestedControlledIdentity(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled identity gate.
inline void trivialControlledIdentity(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcid({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an identity gate.
inline void inverseIdentity(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.id(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled identity
/// gate.
inline void inverseMultipleControlledIdentity(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcid({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- XOp ------------------------------------------------------------------ //

/// Creates a circuit with just an X gate.
inline void x(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
}

/// Creates a circuit with a single controlled X gate.
inline void singleControlledX(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled X gate.
inline void multipleControlledX(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled X gate.
inline void nestedControlledX(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled X gate.
inline void trivialControlledX(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcx({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an X gate.
inline void inverseX(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.x(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled X gate.
inline void inverseMultipleControlledX(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcx({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- YOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Y gate.
inline void y(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
}

/// Creates a circuit with a single controlled Y gate.
inline void singleControlledY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Y gate.
inline void multipleControlledY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Y gate.
inline void nestedControlledY(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled Y gate.
inline void trivialControlledY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcy({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a Y gate.
inline void inverseY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.y(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled Y gate.
inline void inverseMultipleControlledY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcy({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- ZOp ------------------------------------------------------------------ //

/// Creates a circuit with just a Z gate.
inline void z(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
}

/// Creates a circuit with a single controlled Z gate.
inline void singleControlledZ(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Z gate.
inline void multipleControlledZ(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Z gate.
inline void nestedControlledZ(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled Z gate.
inline void trivialControlledZ(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcz({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a Z gate.
inline void inverseZ(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.z(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled Z gate.
inline void inverseMultipleControlledZ(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcz({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- HOp ------------------------------------------------------------------ //

/// Creates a circuit with just an H gate.
inline void h(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
}

/// Creates a circuit with a single controlled H gate.
inline void singleControlledH(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled H gate.
inline void multipleControlledH(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled H gate.
inline void nestedControlledH(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled H gate.
inline void trivialControlledH(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mch({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an H gate.
inline void inverseH(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.h(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled H gate.
inline void inverseMultipleControlledH(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mch({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- SOp ------------------------------------------------------------------ //

/// Creates a circuit with just an S gate.
inline void s(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
}

/// Creates a circuit with a single controlled S gate.
inline void singleControlledS(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled S gate.
inline void multipleControlledS(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled S gate.
inline void nestedControlledS(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled S gate.
inline void trivialControlledS(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcs({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an S gate.
inline void inverseS(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.s(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled S gate.
inline void inverseMultipleControlledS(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcs({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- SdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just an Sdg gate.
inline void sdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
}

/// Creates a circuit with a single controlled Sdg gate.
inline void singleControlledSdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Sdg gate.
inline void multipleControlledSdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Sdg gate.
inline void nestedControlledSdg(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled Sdg gate.
inline void trivialControlledSdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsdg({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an Sdg gate.
inline void inverseSdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.sdg(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled Sdg gate.
inline void inverseMultipleControlledSdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- TOp ------------------------------------------------------------------ //

/// Creates a circuit with just a T gate.
inline void t_(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
}

/// Creates a circuit with a single controlled T gate.
inline void singleControlledT(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled T gate.
inline void multipleControlledT(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled T gate.
inline void nestedControlledT(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled T gate.
inline void trivialControlledT(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mct({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a T gate.
inline void inverseT(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.t(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled T gate.
inline void inverseMultipleControlledT(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mct({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- TdgOp ---------------------------------------------------------------- //

/// Creates a circuit with just a Tdg gate.
inline void tdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
}

/// Creates a circuit with a single controlled Tdg gate.
inline void singleControlledTdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled Tdg gate.
inline void multipleControlledTdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled Tdg gate.
inline void nestedControlledTdg(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled Tdg gate.
inline void trivialControlledTdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mctdg({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a Tdg gate.
inline void inverseTdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.tdg(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled Tdg gate.
inline void inverseMultipleControlledTdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mctdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- SXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an SX gate.
inline void sx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
}

/// Creates a circuit with a single controlled SX gate.
inline void singleControlledSx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled SX gate.
inline void multipleControlledSx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled SX gate.
inline void nestedControlledSx(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled SX gate.
inline void trivialControlledSx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsx({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an SX gate.
inline void inverseSx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.sx(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled SX gate.
inline void inverseMultipleControlledSx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsx({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- SXdgOp --------------------------------------------------------------- //

/// Creates a circuit with just an SXdg gate.
inline void sxdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
}

/// Creates a circuit with a single controlled SXdg gate.
inline void singleControlledSxdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
}

/// Creates a circuit with a multi-controlled SXdg gate.
inline void multipleControlledSxdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled SXdg gate.
inline void nestedControlledSxdg(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled SXdg gate.
inline void trivialControlledSxdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcsxdg({}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an SXdg gate.
inline void inverseSxdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.sxdg(qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled SXdg
/// gate.
inline void inverseMultipleControlledSxdg(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcsxdg({qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- RXOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RX gate.
inline void rx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
}

/// Creates a circuit with a single controlled RX gate.
inline void singleControlledRx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled RX gate.
inline void multipleControlledRx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled RX gate.
inline void nestedControlledRx(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RX gate.
inline void trivialControlledRx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrx(0.123, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an RX gate.
inline void inverseRx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.rx(-0.123, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RX gate.
inline void inverseMultipleControlledRx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcrx(-0.123, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- RYOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RY gate.
inline void ry(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
}

/// Creates a circuit with a single controlled RY gate.
inline void singleControlledRy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled RY gate.
inline void multipleControlledRy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled RY gate.
inline void nestedControlledRy(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RY gate.
inline void trivialControlledRy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcry(0.456, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an RY gate.
inline void inverseRy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.ry(-0.456, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RY gate.
inline void inverseMultipleControlledRy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcry(-0.456, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- RZOp ----------------------------------------------------------------- //

/// Creates a circuit with just an RZ gate.
inline void rz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
}

/// Creates a circuit with a single controlled RZ gate.
inline void singleControlledRz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled RZ gate.
inline void multipleControlledRz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled RZ gate.
inline void nestedControlledRz(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RZ gate.
inline void trivialControlledRz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcrz(0.789, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an RZ gate.
inline void inverseRz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.rz(-0.789, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RZ gate.
inline void inverseMultipleControlledRz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcrz(-0.789, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- POp ------------------------------------------------------------------ //

/// Creates a circuit with just a P gate.
inline void p(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
}

/// Creates a circuit with a single controlled P gate.
inline void singleControlledP(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled P gate.
inline void multipleControlledP(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled P gate.
inline void nestedControlledP(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled P gate.
inline void trivialControlledP(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcp(0.123, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a P gate.
inline void inverseP(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.p(-0.123, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled P gate.
inline void inverseMultipleControlledP(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcp(-0.123, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- ROp ------------------------------------------------------------------ //

/// Creates a circuit with just an R gate.
inline void r(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
}

/// Creates a circuit with a single controlled R gate.
inline void singleControlledR(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled R gate.
inline void multipleControlledR(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled R gate.
inline void nestedControlledR(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled R gate.
inline void trivialControlledR(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcr(0.123, 0.456, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to an R gate.
inline void inverseR(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.r(-0.123, 0.456, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled R gate.
inline void inverseMultipleControlledR(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcr(-0.123, 0.456, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- U2Op ----------------------------------------------------------------- //

/// Creates a circuit with just a U2 gate.
inline void u2(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
}

/// Creates a circuit with a single controlled U2 gate.
inline void singleControlledU2(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled U2 gate.
inline void multipleControlledU2(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled U2 gate.
inline void nestedControlledU2(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled U2 gate.
inline void trivialControlledU2(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu2(0.234, 0.567, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a U2 gate.
inline void inverseU2(mlir::qco::QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{
        b.u2(-0.567 + pi, -0.234 - pi, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled U2 gate.
inline void inverseMultipleControlledU2(mlir::qco::QCOProgramBuilder& b) {
  constexpr double pi = std::numbers::pi;
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcu2(-0.567 + pi, -0.234 - pi, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- UOp ------------------------------------------------------------------ //

/// Creates a circuit with just a U gate.
inline void u(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
}

/// Creates a circuit with a single controlled U gate.
inline void singleControlledU(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
}

/// Creates a circuit with a multi-controlled U gate.
inline void multipleControlledU(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
}

/// Creates a circuit with a nested controlled U gate.
inline void nestedControlledU(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled U gate.
inline void trivialControlledU(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.mcu(0.1, 0.2, 0.3, {}, q[0]);
}

/// Creates a circuit with an inverse modifier applied to a U gate.
inline void inverseU(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.u(-0.1, -0.3, -0.2, qubits[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled U gate.
inline void inverseMultipleControlledU(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.inv({q[0], q[1], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetOut] =
        b.mcu(-0.1, -0.3, -0.2, {qubits[0], qubits[1]}, qubits[2]);
    return llvm::to_vector(
        llvm::concat<mlir::Value>(controlsOut, mlir::ValueRange{targetOut}));
  });
}

// --- SWAPOp --------------------------------------------------------------- //

/// Creates a circuit with just a SWAP gate.
inline void swap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
}

/// Creates a circuit with a single controlled SWAP gate.
inline void singleControlledSwap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled SWAP gate.
inline void multipleControlledSwap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled SWAP gate.
inline void nestedControlledSwap(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled SWAP gate.
inline void trivialControlledSwap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcswap({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to a SWAP gate.
inline void inverseSwap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.swap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled SWAP
/// gate.
inline void inverseMultipleControlledSwap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- iSWAPOp -------------------------------------------------------------- //

/// Creates a circuit with just an iSWAP gate.
inline void iswap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
}

/// Creates a circuit with a single controlled iSWAP gate.
inline void singleControlledIswap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled iSWAP gate.
inline void multipleControlledIswap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled iSWAP gate.
inline void nestedControlledIswap(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled iSWAP gate.
inline void trivialControlledIswap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mciswap({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an iSWAP gate.
inline void inverseIswap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.iswap(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled iSWAP
/// gate.
inline void inverseMultipleControlledIswap(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mciswap({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- DCXOp ---------------------------------------------------------------- //

/// Creates a circuit with just a DCX gate.
inline void dcx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
}

/// Creates a circuit with a single controlled DCX gate.
inline void singleControlledDcx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled DCX gate.
inline void multipleControlledDcx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled DCX gate.
inline void nestedControlledDcx(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled DCX gate.
inline void trivialControlledDcx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcdcx({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to a DCX gate.
inline void inverseDcx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[1], q[0]}, [&](mlir::ValueRange qubits) {
    auto res = b.dcx(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled DCX gate.
inline void inverseMultipleControlledDcx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[3], q[2]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcdcx({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- ECROp ---------------------------------------------------------------- //

/// Creates a circuit with just an ECR gate.
inline void ecr(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
}

/// Creates a circuit with a single controlled ECR gate.
inline void singleControlledEcr(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled ECR gate.
inline void multipleControlledEcr(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled ECR gate.
inline void nestedControlledEcr(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled ECR gate.
inline void trivialControlledEcr(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcecr({}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an ECR gate.
inline void inverseEcr(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ecr(qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled ECR gate.
inline void inverseMultipleControlledEcr(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcecr({qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- RXXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RXX gate.
inline void rxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RXX gate.
inline void singleControlledRxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RXX gate.
inline void multipleControlledRxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RXX gate.
inline void nestedControlledRxx(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RXX gate.
inline void trivialControlledRxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrxx(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RXX gate.
inline void inverseRxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.rxx(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RXX gate.
inline void inverseMultipleControlledRxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrxx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

/// Creates a circuit with a triple-controlled RXX gate.
inline void tripleControlledRxx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
}

// --- RYYOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RYY gate.
inline void ryy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RYY gate.
inline void singleControlledRyy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RYY gate.
inline void multipleControlledRyy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RYY gate.
inline void nestedControlledRyy(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RYY gate.
inline void trivialControlledRyy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcryy(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RYY gate.
inline void inverseRyy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.ryy(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RYY gate.
inline void inverseMultipleControlledRyy(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcryy(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- RZXOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZX gate.
inline void rzx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RZX gate.
inline void singleControlledRzx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RZX gate.
inline void multipleControlledRzx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RZX gate.
inline void nestedControlledRzx(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RZX gate.
inline void trivialControlledRzx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzx(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RZX gate.
inline void inverseRzx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.rzx(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RZX gate.
inline void inverseMultipleControlledRzx(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrzx(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- RZZOp ---------------------------------------------------------------- //

/// Creates a circuit with just an RZZ gate.
inline void rzz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
}

/// Creates a circuit with a single controlled RZZ gate.
inline void singleControlledRzz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled RZZ gate.
inline void multipleControlledRzz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled RZZ gate.
inline void nestedControlledRzz(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled RZZ gate.
inline void trivialControlledRzz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcrzz(0.123, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an RZZ gate.
inline void inverseRzz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.rzz(-0.123, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled RZZ gate.
inline void inverseMultipleControlledRzz(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] =
        b.mcrzz(-0.123, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- XXPlusYYOp ----------------------------------------------------------- //

/// Creates a circuit with just an XXPlusYY gate.
inline void xxPlusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
}

/// Creates a circuit with a single controlled XXPlusYY gate.
inline void singleControlledXxPlusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled XXPlusYY gate.
inline void multipleControlledXxPlusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled XXPlusYY gate.
inline void nestedControlledXxPlusYY(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled XXPlusYY gate.
inline void trivialControlledXxPlusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_plus_yy(0.123, 0.456, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an XXPlusYY gate.
inline void inverseXxPlusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.xx_plus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled XXPlusYY
/// gate.
inline void inverseMultipleControlledXxPlusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.mcxx_plus_yy(
        -0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- XXMinusYYOp ---------------------------------------------------------- //

/// Creates a circuit with just an XXMinusYY gate.
inline void xxMinusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
}

/// Creates a circuit with a single controlled XXMinusYY gate.
inline void singleControlledXxMinusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
}

/// Creates a circuit with a multi-controlled XXMinusYY gate.
inline void multipleControlledXxMinusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
}

/// Creates a circuit with a nested controlled XXMinusYY gate.
inline void nestedControlledXxMinusYY(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with a trivial controlled XXMinusYY gate.
inline void trivialControlledXxMinusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.mcxx_minus_yy(0.123, 0.456, {}, q[0], q[1]);
}

/// Creates a circuit with an inverse modifier applied to an XXMinusYY gate.
inline void inverseXxMinusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.inv({q[0], q[1]}, [&](mlir::ValueRange qubits) {
    auto res = b.xx_minus_yy(-0.123, 0.456, qubits[0], qubits[1]);
    return llvm::SmallVector<mlir::Value>{res.first, res.second};
  });
}

/// Creates a circuit with an inverse modifier applied to a controlled XXMinusYY
/// gate.
inline void
inverseMultipleControlledXxMinusYY(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.inv({q[0], q[1], q[2], q[3]}, [&](mlir::ValueRange qubits) {
    const auto& [controlsOut, targetsOut] = b.mcxx_minus_yy(
        -0.123, 0.456, {qubits[0], qubits[1]}, qubits[2], qubits[3]);
    llvm::SmallVector<mlir::Value, 2> targets{targetsOut.first,
                                              targetsOut.second};
    return llvm::to_vector(llvm::concat<mlir::Value>(controlsOut, targets));
  });
}

// --- BarrierOp ------------------------------------------------------------ //

/// Creates a circuit with a barrier.
inline void barrier(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.barrier(q[0]);
}

/// Creates a circuit with a barrier on two qubits.
inline void barrierTwoQubits(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.barrier({q[0], q[1]});
}

/// Creates a circuit with a barrier on multiple qubits.
inline void barrierMultipleQubits(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.barrier({q[0], q[1], q[2]});
}

/// Creates a circuit with a single controlled barrier.
inline void singleControlledBarrier(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({q[1]}, {q[0]}, [&](mlir::ValueRange targets) {
    return llvm::SmallVector<mlir::Value>{b.barrier(targets[0])};
  });
}

/// Creates a circuit with an inverse modifier applied to a barrier.
inline void inverseBarrier(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.inv({q[0]}, [&](mlir::ValueRange qubits) {
    return llvm::SmallVector<mlir::Value>{b.barrier(qubits[0])};
  });
}

// --- CtrlOp --------------------------------------------------------------- //

/// Creates a circuit with a trivial ctrl modifier.
inline void trivialCtrl(mlir::qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctrl({}, {q[0], q[1]}, [&](mlir::ValueRange targets) {
    auto [q0, q1] = b.rxx(0.123, targets[0], targets[1]);
    return llvm::SmallVector<mlir::Value>{q0, q1};
  });
}

/// Creates a circuit with nested ctrl modifiers.
inline void nestedCtrl(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with triple nested ctrl modifiers.
inline void tripleNestedCtrl(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with control modifiers interleaved by an inverse modifier.
inline void ctrlInvSandwich(mlir::qco::QCOProgramBuilder& b) {
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

// --- InvOp ---------------------------------------------------------------- //

/// Creates a circuit with nested inverse modifiers.
inline void nestedInv(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with triple nested inverse modifiers.
inline void tripleNestedInv(mlir::qco::QCOProgramBuilder& b) {
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

/// Creates a circuit with inverse modifiers interleaved by a control modifier.
inline void invCtrlSandwich(mlir::qco::QCOProgramBuilder& b) {
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
