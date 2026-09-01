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

#include <string>
#include <string_view>

namespace mqt::bench::detail {

/// Return the lowercase SHA-256 digest used for stable benchmark case IDs.
[[nodiscard]] std::string sha256Hex(std::string_view input);

} // namespace mqt::bench::detail
