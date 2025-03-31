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

namespace qc {
[[nodiscard]] auto createQFT(Qubit nq, bool includeMeasurements = true)
    -> QuantumComputation;

[[nodiscard]] auto createIterativeQFT(Qubit nq) -> QuantumComputation;
} // namespace qc
