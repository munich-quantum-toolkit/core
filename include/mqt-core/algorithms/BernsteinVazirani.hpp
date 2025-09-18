/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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

using BVBitString = std::bitset<4096>;

[[nodiscard]] auto createBernsteinVazirani(const BVBitString& hiddenString)
    -> QuantumComputation;
[[nodiscard]] auto createBernsteinVazirani(Qubit nq, std::size_t seed = 0)
    -> QuantumComputation;
[[nodiscard]] auto createBernsteinVazirani(const BVBitString& hiddenString,
                                           Qubit nq) -> QuantumComputation;

[[nodiscard]] auto
createIterativeBernsteinVazirani(const BVBitString& hiddenString)
    -> QuantumComputation;
[[nodiscard]] auto createIterativeBernsteinVazirani(Qubit nq,
                                                    std::size_t seed = 0)
    -> QuantumComputation;
[[nodiscard]] auto
createIterativeBernsteinVazirani(const BVBitString& hiddenString, Qubit nq)
    -> QuantumComputation;
} // namespace qc
