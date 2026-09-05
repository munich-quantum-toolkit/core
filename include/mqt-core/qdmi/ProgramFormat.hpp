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
#include <iterator>

namespace qdmi {

namespace detail {
template <size_t N>
[[nodiscard]] constexpr bool
isCanonicalFixedString(const char (&value)[N]) noexcept {
  const auto terminator = std::ranges::find(value, '\0');
  return terminator != std::end(value) &&
         std::ranges::all_of(terminator, std::end(value),
                             [](const char byte) { return byte == '\0'; });
}
} // namespace detail

inline constexpr QDMI_Program_Format OPENQASM2{
    .version = QDMI_MAKE_VERSION(2, 0, 0),
    .encoding = QDMI_PROGRAM_ENCODING_TEXT,
    .id = "openqasm",
    .profile = ""};
inline constexpr QDMI_Program_Format OPENQASM3{
    .version = QDMI_MAKE_VERSION(3, 0, 0),
    .encoding = QDMI_PROGRAM_ENCODING_TEXT,
    .id = "openqasm",
    .profile = ""};
inline constexpr QDMI_Program_Format QIR21_BASE_TEXT{
    .version = QDMI_MAKE_VERSION(2, 1, 0),
    .encoding = QDMI_PROGRAM_ENCODING_TEXT,
    .id = "qir",
    .profile = "base"};
inline constexpr QDMI_Program_Format QIR21_BASE_BINARY{
    .version = QDMI_MAKE_VERSION(2, 1, 0),
    .encoding = QDMI_PROGRAM_ENCODING_BINARY,
    .id = "qir",
    .profile = "base"};
inline constexpr QDMI_Program_Format QIR21_ADAPTIVE_TEXT{
    .version = QDMI_MAKE_VERSION(2, 1, 0),
    .encoding = QDMI_PROGRAM_ENCODING_TEXT,
    .id = "qir",
    .profile = "adaptive"};
inline constexpr QDMI_Program_Format QIR21_ADAPTIVE_BINARY{
    .version = QDMI_MAKE_VERSION(2, 1, 0),
    .encoding = QDMI_PROGRAM_ENCODING_BINARY,
    .id = "qir",
    .profile = "adaptive"};

[[nodiscard]] inline bool equal(const QDMI_Program_Format& lhs,
                                const QDMI_Program_Format& rhs) noexcept {
  return QDMI_program_format_equal(&lhs, &rhs) != 0;
}

[[nodiscard]] constexpr bool
isBinaryProgramFormat(const QDMI_Program_Format& format) noexcept {
  return format.encoding == QDMI_PROGRAM_ENCODING_BINARY;
}

[[nodiscard]] constexpr bool
isValidProgramFormat(const QDMI_Program_Format& format) noexcept {
  return format.version != 0U && format.id[0] != '\0' &&
         (format.encoding == QDMI_PROGRAM_ENCODING_TEXT ||
          format.encoding == QDMI_PROGRAM_ENCODING_BINARY) &&
         detail::isCanonicalFixedString(format.id) &&
         detail::isCanonicalFixedString(format.profile);
}

[[nodiscard]] constexpr bool
isValidProgramFeature(const QDMI_Program_Feature& feature) noexcept {
  return feature.id[0] != '\0' && detail::isCanonicalFixedString(feature.id) &&
         detail::isCanonicalFixedString(feature.constraint_id) &&
         (feature.constraint_id[0] != '\0' || feature.constraint_value == 0U);
}

} // namespace qdmi
