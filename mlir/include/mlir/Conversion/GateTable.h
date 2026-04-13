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

/**
 * @file GateTable.h
 * @brief Central gate registry for QC / QCO / QIR conversions.
 *
 * @details
 * This header defines a single source of truth for the set of supported gates
 * shared by QCO↔QC conversion and QC→QIR lowering in `mlir/lib/Conversion/`.
 *
 * Each entry specifies:
 * - a canonical gate key (identifier),
 * - number of targets,
 * - number of parameters,
 * - the corresponding QCO and QC op types,
 * - QIR lowering kind and function-name selector (for QC→QIR only).
 */

#include <cstddef>
#include <cstdint>

namespace mlir::mqt::gates {

/** @brief QIR lowering kind for a gate entry. */
enum class QIRKind : std::uint8_t {
  Unitary,     //!< Lower to a QIR runtime call (name depends on num ctrls).
  Unsupported, //!< Not supported in QC-to-QIR lowering.
};

} // namespace mlir::mqt::gates

/**
 * @brief Central gate table.
 *
 * Columns:
 * - KEY: canonical identifier
 * - TARGETS: number of target qubits
 * - PARAMS: number of floating parameters
 * - QCO_OP: QCO op type
 * - QC_OP: QC op type
 * - QIR_KIND: QIRKind
 * - QIR_FN: function-name selector for QC-to-QIR lowering (unitary only)
 */
#define MQT_GATE_TABLE(ENTRY)                                                  \
  ENTRY(Id, 1, 0, ::mlir::qco::IdOp, ::mlir::qc::IdOp,                         \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameI)         \
  ENTRY(X, 1, 0, ::mlir::qco::XOp, ::mlir::qc::XOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameX)         \
  ENTRY(Y, 1, 0, ::mlir::qco::YOp, ::mlir::qc::YOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameY)         \
  ENTRY(Z, 1, 0, ::mlir::qco::ZOp, ::mlir::qc::ZOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameZ)         \
  ENTRY(H, 1, 0, ::mlir::qco::HOp, ::mlir::qc::HOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameH)         \
  ENTRY(S, 1, 0, ::mlir::qco::SOp, ::mlir::qc::SOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameS)         \
  ENTRY(Sdg, 1, 0, ::mlir::qco::SdgOp, ::mlir::qc::SdgOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameSDG)       \
  ENTRY(T, 1, 0, ::mlir::qco::TOp, ::mlir::qc::TOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameT)         \
  ENTRY(Tdg, 1, 0, ::mlir::qco::TdgOp, ::mlir::qc::TdgOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameTDG)       \
  ENTRY(SX, 1, 0, ::mlir::qco::SXOp, ::mlir::qc::SXOp,                         \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameSX)        \
  ENTRY(SXdg, 1, 0, ::mlir::qco::SXdgOp, ::mlir::qc::SXdgOp,                   \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameSXDG)      \
  ENTRY(RX, 1, 1, ::mlir::qco::RXOp, ::mlir::qc::RXOp,                         \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRX)        \
  ENTRY(RY, 1, 1, ::mlir::qco::RYOp, ::mlir::qc::RYOp,                         \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRY)        \
  ENTRY(RZ, 1, 1, ::mlir::qco::RZOp, ::mlir::qc::RZOp,                         \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRZ)        \
  ENTRY(P, 1, 1, ::mlir::qco::POp, ::mlir::qc::POp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameP)         \
  ENTRY(R, 1, 2, ::mlir::qco::ROp, ::mlir::qc::ROp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameR)         \
  ENTRY(U2, 1, 2, ::mlir::qco::U2Op, ::mlir::qc::U2Op,                         \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameU2)        \
  ENTRY(U, 1, 3, ::mlir::qco::UOp, ::mlir::qc::UOp,                            \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameU)         \
  ENTRY(SWAP, 2, 0, ::mlir::qco::SWAPOp, ::mlir::qc::SWAPOp,                   \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameSWAP)      \
  ENTRY(iSWAP, 2, 0, ::mlir::qco::iSWAPOp, ::mlir::qc::iSWAPOp,                \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameISWAP)     \
  ENTRY(DCX, 2, 0, ::mlir::qco::DCXOp, ::mlir::qc::DCXOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameDCX)       \
  ENTRY(ECR, 2, 0, ::mlir::qco::ECROp, ::mlir::qc::ECROp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameECR)       \
  ENTRY(RXX, 2, 1, ::mlir::qco::RXXOp, ::mlir::qc::RXXOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRXX)       \
  ENTRY(RYY, 2, 1, ::mlir::qco::RYYOp, ::mlir::qc::RYYOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRYY)       \
  ENTRY(RZX, 2, 1, ::mlir::qco::RZXOp, ::mlir::qc::RZXOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRZX)       \
  ENTRY(RZZ, 2, 1, ::mlir::qco::RZZOp, ::mlir::qc::RZZOp,                      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRZZ)       \
  ENTRY(XXPlusYY, 2, 2, ::mlir::qco::XXPlusYYOp, ::mlir::qc::XXPlusYYOp,       \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameXXPLUSYY)  \
  ENTRY(XXMinusYY, 2, 2, ::mlir::qco::XXMinusYYOp, ::mlir::qc::XXMinusYYOp,    \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameXXMINUSYY)
