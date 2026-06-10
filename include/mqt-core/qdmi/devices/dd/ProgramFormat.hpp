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

#include "mqt_ddsim_qdmi/constants.h"

namespace qdmi::dd {

/**
 * @brief Whether @p fmt is a text-based QDMI program format.
 * @details QDMI program formats fall into two byte-shape categories:
 * - text formats (QASM, QIR Base/Adaptive String) are shipped with a trailing
 *   '\0' counted in the buffer size.
 * - binary formats (QIR Base/Adaptive Module bitcode) are shipped as exact byte
 *   counts since '\0' may appear inside the payload.
 */
inline bool isTextProgramFormat(QDMI_Program_Format fmt) {
  return fmt == QDMI_PROGRAM_FORMAT_QASM2 || fmt == QDMI_PROGRAM_FORMAT_QASM3 ||
         fmt == QDMI_PROGRAM_FORMAT_QIRBASESTRING ||
         fmt == QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
}

} // namespace qdmi::dd
