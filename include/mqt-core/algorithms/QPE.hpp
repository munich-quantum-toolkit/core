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
[[nodiscard]] auto createQPE(Qubit nq, bool exact = true, std::size_t seed = 0)
    -> QuantumComputation;

[[nodiscard]] auto createQPE(fp lambda, Qubit precision) -> QuantumComputation;

[[nodiscard]] auto createIterativeQPE(Qubit nq, bool exact = true,
                                      std::size_t seed = 0)
    -> QuantumComputation;

[[nodiscard]] auto createIterativeQPE(fp lambda, Qubit precision)
    -> QuantumComputation;
} // namespace qc
