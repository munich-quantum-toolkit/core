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

namespace qir {

inline constexpr char ENTRY_POINT_ATTR[] = "entry_point";
inline constexpr char OUTPUT_LABELING_SCHEMA_ATTR[] = "output_labeling_schema";
inline constexpr char QIR_PROFILES_ATTR[] = "qir_profiles";
inline constexpr char IRREVERSIBLE_ATTR[] = "irreversible";

inline constexpr char BASE_PROFILE[] = "base_profile";
inline constexpr char ADAPTIVE_PROFILE[] = "adaptive_profile";
inline constexpr char LABELED_SCHEMA[] = "labeled";
inline constexpr char ORDERED_SCHEMA[] = "ordered";

inline constexpr char QIS_PREFIX[] = "__quantum__qis__";

} // namespace qir
