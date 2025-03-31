/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>

namespace qc {
[[nodiscard]] auto createRandomCliffordCircuit(Qubit nq, std::size_t depth = 1,
                                               std::size_t seed = 0)
    -> QuantumComputation;
} // namespace qc
