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

#include <bitset>
#include <cstddef>

namespace qc {

using GroverBitString = std::bitset<128>;
auto appendGroverInitialization(QuantumComputation& qc) -> void;
auto appendGroverOracle(QuantumComputation& qc,
                        const GroverBitString& targetValue) -> void;
auto appendGroverDiffusion(QuantumComputation& qc) -> void;

[[nodiscard]] auto computeNumberOfIterations(Qubit nq) -> std::size_t;

[[nodiscard]] auto createGrover(Qubit nq, const GroverBitString& targetValue)
    -> QuantumComputation;
[[nodiscard]] auto createGrover(Qubit nq, std::size_t seed = 0)
    -> QuantumComputation;
} // namespace qc
