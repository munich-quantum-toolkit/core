/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 MQSC GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"

namespace qc {
class QuantumComputation;

[[nodiscard]] auto createGHZState(Qubit nq) -> QuantumComputation;
} // namespace qc
