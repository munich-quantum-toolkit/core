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

#include <qdmi/constants.h>

#include <algorithm>
#include <ranges>

[[nodiscard]] constexpr bool
operator==(const QDMI_Program_Format& lhs,
           const QDMI_Program_Format& rhs) noexcept {
  return lhs.version == rhs.version && lhs.encoding == rhs.encoding &&
         std::ranges::equal(lhs.id, rhs.id) &&
         std::ranges::equal(lhs.profile, rhs.profile);
}

namespace qdmi {

inline constexpr QDMI_Program_Format OPENQASM2{
    QDMI_MAKE_VERSION(2, 0, 0), QDMI_PROGRAM_ENCODING_TEXT, "openqasm", ""};
inline constexpr QDMI_Program_Format OPENQASM3{
    QDMI_MAKE_VERSION(3, 0, 0), QDMI_PROGRAM_ENCODING_TEXT, "openqasm", ""};
inline constexpr QDMI_Program_Format QIR21_BASE_TEXT{
    QDMI_MAKE_VERSION(2, 1, 0), QDMI_PROGRAM_ENCODING_TEXT, "qir", "base"};
inline constexpr QDMI_Program_Format QIR21_BASE_BINARY{
    QDMI_MAKE_VERSION(2, 1, 0), QDMI_PROGRAM_ENCODING_BINARY, "qir", "base"};
inline constexpr QDMI_Program_Format QIR21_ADAPTIVE_TEXT{
    QDMI_MAKE_VERSION(2, 1, 0), QDMI_PROGRAM_ENCODING_TEXT, "qir", "adaptive"};
inline constexpr QDMI_Program_Format QIR21_ADAPTIVE_BINARY{
    QDMI_MAKE_VERSION(2, 1, 0), QDMI_PROGRAM_ENCODING_BINARY, "qir",
    "adaptive"};

[[nodiscard]] constexpr bool equal(const QDMI_Program_Format& lhs,
                                   const QDMI_Program_Format& rhs) noexcept {
  return lhs == rhs;
}

[[nodiscard]] constexpr bool
isBinaryProgramFormat(const QDMI_Program_Format& format) noexcept {
  return format.encoding == QDMI_PROGRAM_ENCODING_BINARY;
}

} // namespace qdmi
