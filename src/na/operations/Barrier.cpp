/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/Barrier.hpp"

#include <string>

namespace na {
auto Barrier::toString() const -> std::string { return "// barrier"; }
} // namespace na
