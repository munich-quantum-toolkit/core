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

#include <string_view>

namespace qir {

inline constexpr std::string_view ENTRY_POINT_ATTR = "entry_point";
inline constexpr std::string_view OUTPUT_LABELING_SCHEMA_ATTR =
    "output_labeling_schema";
inline constexpr std::string_view QIR_PROFILES_ATTR = "qir_profiles";
inline constexpr std::string_view IRREVERSIBLE_ATTR = "irreversible";

inline constexpr std::string_view BASE_PROFILE = "base_profile";
inline constexpr std::string_view ADAPTIVE_PROFILE = "adaptive_profile";
inline constexpr std::string_view LABELED_SCHEMA = "labeled";
inline constexpr std::string_view ORDERED_SCHEMA = "ordered";

inline constexpr std::string_view QIS_PREFIX = "__quantum__qis__";

} // namespace qir
