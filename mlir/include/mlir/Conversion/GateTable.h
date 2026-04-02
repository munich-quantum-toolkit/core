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
 * @brief Central gate registry for MLIR conversions.
 *
 * @details
 * This header defines a single source of truth for the set of supported gates
 * across the conversion passes in `mlir/lib/Conversion/`.
 *
 * The registry is provided as an X-macro table so it can be consumed in
 * multiple translation units without introducing ODR issues.
 *
 * Each entry specifies:
 * - a canonical gate key (identifier),
 * - number of targets,
 * - number of parameters,
 * - the corresponding QCO/QC op types,
 * - Jeff lowering kind,
 * - QIR function-name selector.
 *
 * Conversions can consume this table to automatically derive target/parameter
 * counts and stay consistent across backends.
 */

#include <cstddef>
#include <cstdint>

namespace mlir::mqt::gates {

/** @brief Jeff lowering kind for a gate entry. */
enum class JeffKind : std::uint8_t {
  Native,       //!< Dedicated Jeff op exists.
  Custom,       //!< Lower to jeff.custom with a name string.
  PPR,          //!< Lower to jeff.ppr with Pauli-gate encoding.
  SpecialU2ToU, //!< Lower qco.u2 via jeff.u with injected theta=pi/2.
};

/** @brief QIR lowering kind for a gate entry. */
enum class QIRKind : std::uint8_t {
  Unitary,     //!< Lower to a QIR runtime call (name depends on num ctrls).
  Unsupported, //!< Not supported in QC-to-QIR lowering.
};

/**
 * @brief Pauli gate encoding used for jeff.ppr.
 *
 * @details
 * Encoding matches existing QCO<->Jeff conversion logic:
 * 1=X, 2=Y, 3=Z.
 */
struct PPRPaulis {
  std::int32_t p0 = 0;
  std::int32_t p1 = 0;
};

// Helper macros to keep `MQT_GATE_TABLE` macro-friendly.
#define MQT_PPR(P0, P1)                                                        \
  ::mlir::mqt::gates::PPRPaulis { (P0), (P1) }
#define MQT_PPR_NONE MQT_PPR(0, 0)

/**
 * @brief Central gate table.
 *
 * @details
 * The table is intentionally limited to the gates currently supported by all
 * relevant conversions in this repository.
 *
 * Columns:
 * - KEY: canonical identifier
 * - TARGETS: number of target qubits
 * - PARAMS: number of floating parameters
 * - QCO_OP: QCO op type
 * - QC_OP: QC op type
 * - JEFF_KIND: JeffKind
 * - JEFF_OP: Jeff op type (only for Native/SpecialU2ToU; otherwise void)
 * - JEFF_BASE_ADJOINT: whether the base gate is adjoint (e.g. Sdg, Tdg, SXdg)
 * - JEFF_CUSTOM_NAME: identifier token for custom op name (only for Custom)
 * - JEFF_PPR: PPRPaulis (only for PPR; otherwise {0,0})
 * - QIR_KIND: QIRKind
 * - QIR_FN: function-name selector for the QC-to-QIR lowering (unitary only)
 *
 * @details Jeff adjoint encoding for custom gates
 * Some gates are represented in Jeff as `jeff.custom "<name>"` and the
 * adjointness is carried in the Jeff op attribute `is_adjoint`. For such gates,
 * `JEFF_BASE_ADJOINT` indicates whether the *unmodified* QCO/QC gate already
 * corresponds to the adjoint of the canonical Jeff base gate. A prominent
 * example is `SXdg`, which shares the Jeff custom name `"sx"` but is inherently
 * adjoint compared to `SX`. Conversions must avoid double-encoding adjointness
 * (i.e., they must not flip `is_adjoint` twice).
 */
#define MQT_GATE_TABLE(ENTRY)                                                  \
  ENTRY(Id, 1, 0, ::mlir::qco::IdOp, ::mlir::qc::IdOp,                         \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::IOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameI)                                               \
  ENTRY(X, 1, 0, ::mlir::qco::XOp, ::mlir::qc::XOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::XOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameX)                                               \
  ENTRY(Y, 1, 0, ::mlir::qco::YOp, ::mlir::qc::YOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::YOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameY)                                               \
  ENTRY(Z, 1, 0, ::mlir::qco::ZOp, ::mlir::qc::ZOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::ZOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameZ)                                               \
  ENTRY(H, 1, 0, ::mlir::qco::HOp, ::mlir::qc::HOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::HOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameH)                                               \
  ENTRY(S, 1, 0, ::mlir::qco::SOp, ::mlir::qc::SOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::SOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameS)                                               \
  ENTRY(Sdg, 1, 0, ::mlir::qco::SdgOp, ::mlir::qc::SdgOp,                      \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::SOp, true, _,      \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameSDG)                                             \
  ENTRY(T, 1, 0, ::mlir::qco::TOp, ::mlir::qc::TOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::TOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameT)                                               \
  ENTRY(Tdg, 1, 0, ::mlir::qco::TdgOp, ::mlir::qc::TdgOp,                      \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::TOp, true, _,      \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameTDG)                                             \
  ENTRY(P, 1, 1, ::mlir::qco::POp, ::mlir::qc::POp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::R1Op, false, _,    \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameP)                                               \
  ENTRY(RX, 1, 1, ::mlir::qco::RXOp, ::mlir::qc::RXOp,                         \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::RxOp, false, _,    \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameRX)                                              \
  ENTRY(RY, 1, 1, ::mlir::qco::RYOp, ::mlir::qc::RYOp,                         \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::RyOp, false, _,    \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameRY)                                              \
  ENTRY(RZ, 1, 1, ::mlir::qco::RZOp, ::mlir::qc::RZOp,                         \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::RzOp, false, _,    \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameRZ)                                              \
  ENTRY(R, 1, 2, ::mlir::qco::ROp, ::mlir::qc::ROp,                            \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, r, MQT_PPR_NONE,    \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameR)         \
  ENTRY(U2, 1, 2, ::mlir::qco::U2Op, ::mlir::qc::U2Op,                         \
        ::mlir::mqt::gates::JeffKind::SpecialU2ToU, ::mlir::jeff::UOp, false,  \
        _, MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                 \
        ::mlir::qir::getFnNameU2)                                              \
  ENTRY(U, 1, 3, ::mlir::qco::UOp, ::mlir::qc::UOp,                            \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::UOp, false, _,     \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameU)                                               \
  ENTRY(SX, 1, 0, ::mlir::qco::SXOp, ::mlir::qc::SXOp,                         \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, sx, MQT_PPR_NONE,   \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameSX)        \
  ENTRY(SXdg, 1, 0, ::mlir::qco::SXdgOp, ::mlir::qc::SXdgOp,                   \
        ::mlir::mqt::gates::JeffKind::Custom, void, true, sx, MQT_PPR_NONE,    \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameSXDG)      \
  ENTRY(SWAP, 2, 0, ::mlir::qco::SWAPOp, ::mlir::qc::SWAPOp,                   \
        ::mlir::mqt::gates::JeffKind::Native, ::mlir::jeff::SwapOp, false, _,  \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameSWAP)                                            \
  ENTRY(iSWAP, 2, 0, ::mlir::qco::iSWAPOp, ::mlir::qc::iSWAPOp,                \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, iswap,              \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameISWAP)                                           \
  ENTRY(DCX, 2, 0, ::mlir::qco::DCXOp, ::mlir::qc::DCXOp,                      \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, dcx, MQT_PPR_NONE,  \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameDCX)       \
  ENTRY(ECR, 2, 0, ::mlir::qco::ECROp, ::mlir::qc::ECROp,                      \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, ecr, MQT_PPR_NONE,  \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameECR)       \
  ENTRY(RXX, 2, 1, ::mlir::qco::RXXOp, ::mlir::qc::RXXOp,                      \
        ::mlir::mqt::gates::JeffKind::PPR, void, false, _, MQT_PPR(1, 1),      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRXX)       \
  ENTRY(RYY, 2, 1, ::mlir::qco::RYYOp, ::mlir::qc::RYYOp,                      \
        ::mlir::mqt::gates::JeffKind::PPR, void, false, _, MQT_PPR(2, 2),      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRYY)       \
  ENTRY(RZX, 2, 1, ::mlir::qco::RZXOp, ::mlir::qc::RZXOp,                      \
        ::mlir::mqt::gates::JeffKind::PPR, void, false, _, MQT_PPR(3, 1),      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRZX)       \
  ENTRY(RZZ, 2, 1, ::mlir::qco::RZZOp, ::mlir::qc::RZZOp,                      \
        ::mlir::mqt::gates::JeffKind::PPR, void, false, _, MQT_PPR(3, 3),      \
        ::mlir::mqt::gates::QIRKind::Unitary, ::mlir::qir::getFnNameRZZ)       \
  ENTRY(XXPlusYY, 2, 2, ::mlir::qco::XXPlusYYOp, ::mlir::qc::XXPlusYYOp,       \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, xx_plus_yy,         \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameXXPLUSYY)                                        \
  ENTRY(XXMinusYY, 2, 2, ::mlir::qco::XXMinusYYOp, ::mlir::qc::XXMinusYYOp,    \
        ::mlir::mqt::gates::JeffKind::Custom, void, false, xx_minus_yy,        \
        MQT_PPR_NONE, ::mlir::mqt::gates::QIRKind::Unitary,                    \
        ::mlir::qir::getFnNameXXMINUSYY)

} // namespace mlir::mqt::gates
