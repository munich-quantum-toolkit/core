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
 * - QIR function-name selector.
 */

/**
 * @brief Central gate table.
 *
 * Columns:
 * - KEY: canonical identifier
 * - TARGETS: number of target qubits
 * - PARAMS: number of floating parameters
 * - QCO_OP: QCO op type
 * - QC_OP: QC op type
 * - QIR_FN: function-name selector for QC-to-QIR lowering
 */
#define MQT_GATE_TABLE(ENTRY)                                                  \
  ENTRY(Id, 1, 0, ::mlir::qco::IdOp, ::mlir::qc::IdOp,                         \
        ::mlir::qir::getFnNameI)                                               \
  ENTRY(X, 1, 0, ::mlir::qco::XOp, ::mlir::qc::XOp, ::mlir::qir::getFnNameX)   \
  ENTRY(Y, 1, 0, ::mlir::qco::YOp, ::mlir::qc::YOp, ::mlir::qir::getFnNameY)   \
  ENTRY(Z, 1, 0, ::mlir::qco::ZOp, ::mlir::qc::ZOp, ::mlir::qir::getFnNameZ)   \
  ENTRY(H, 1, 0, ::mlir::qco::HOp, ::mlir::qc::HOp, ::mlir::qir::getFnNameH)   \
  ENTRY(S, 1, 0, ::mlir::qco::SOp, ::mlir::qc::SOp, ::mlir::qir::getFnNameS)   \
  ENTRY(Sdg, 1, 0, ::mlir::qco::SdgOp, ::mlir::qc::SdgOp,                      \
        ::mlir::qir::getFnNameSDG)                                             \
  ENTRY(T, 1, 0, ::mlir::qco::TOp, ::mlir::qc::TOp, ::mlir::qir::getFnNameT)   \
  ENTRY(Tdg, 1, 0, ::mlir::qco::TdgOp, ::mlir::qc::TdgOp,                      \
        ::mlir::qir::getFnNameTDG)                                             \
  ENTRY(SX, 1, 0, ::mlir::qco::SXOp, ::mlir::qc::SXOp,                         \
        ::mlir::qir::getFnNameSX)                                              \
  ENTRY(SXdg, 1, 0, ::mlir::qco::SXdgOp, ::mlir::qc::SXdgOp,                   \
        ::mlir::qir::getFnNameSXDG)                                            \
  ENTRY(RX, 1, 1, ::mlir::qco::RXOp, ::mlir::qc::RXOp,                         \
        ::mlir::qir::getFnNameRX)                                              \
  ENTRY(RY, 1, 1, ::mlir::qco::RYOp, ::mlir::qc::RYOp,                         \
        ::mlir::qir::getFnNameRY)                                              \
  ENTRY(RZ, 1, 1, ::mlir::qco::RZOp, ::mlir::qc::RZOp,                         \
        ::mlir::qir::getFnNameRZ)                                              \
  ENTRY(P, 1, 1, ::mlir::qco::POp, ::mlir::qc::POp, ::mlir::qir::getFnNameP)   \
  ENTRY(R, 1, 2, ::mlir::qco::ROp, ::mlir::qc::ROp, ::mlir::qir::getFnNameR)   \
  ENTRY(U2, 1, 2, ::mlir::qco::U2Op, ::mlir::qc::U2Op,                         \
        ::mlir::qir::getFnNameU2)                                              \
  ENTRY(U, 1, 3, ::mlir::qco::UOp, ::mlir::qc::UOp, ::mlir::qir::getFnNameU)   \
  ENTRY(SWAP, 2, 0, ::mlir::qco::SWAPOp, ::mlir::qc::SWAPOp,                   \
        ::mlir::qir::getFnNameSWAP)                                            \
  ENTRY(iSWAP, 2, 0, ::mlir::qco::iSWAPOp, ::mlir::qc::iSWAPOp,                \
        ::mlir::qir::getFnNameISWAP)                                           \
  ENTRY(DCX, 2, 0, ::mlir::qco::DCXOp, ::mlir::qc::DCXOp,                      \
        ::mlir::qir::getFnNameDCX)                                             \
  ENTRY(ECR, 2, 0, ::mlir::qco::ECROp, ::mlir::qc::ECROp,                      \
        ::mlir::qir::getFnNameECR)                                             \
  ENTRY(RXX, 2, 1, ::mlir::qco::RXXOp, ::mlir::qc::RXXOp,                      \
        ::mlir::qir::getFnNameRXX)                                             \
  ENTRY(RYY, 2, 1, ::mlir::qco::RYYOp, ::mlir::qc::RYYOp,                      \
        ::mlir::qir::getFnNameRYY)                                             \
  ENTRY(RZX, 2, 1, ::mlir::qco::RZXOp, ::mlir::qc::RZXOp,                      \
        ::mlir::qir::getFnNameRZX)                                             \
  ENTRY(RZZ, 2, 1, ::mlir::qco::RZZOp, ::mlir::qc::RZZOp,                      \
        ::mlir::qir::getFnNameRZZ)                                             \
  ENTRY(XXPlusYY, 2, 2, ::mlir::qco::XXPlusYYOp, ::mlir::qc::XXPlusYYOp,       \
        ::mlir::qir::getFnNameXXPLUSYY)                                        \
  ENTRY(XXMinusYY, 2, 2, ::mlir::qco::XXMinusYYOp, ::mlir::qc::XXMinusYYOp,    \
        ::mlir::qir::getFnNameXXMINUSYY)
