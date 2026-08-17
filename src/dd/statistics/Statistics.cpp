/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/Statistics.hpp"

#include <string>

namespace dd {

/// The base class carries no statistics, which JSON renders as a null value.
std::string Statistics::toString() const { return "null"; }

} // namespace dd
