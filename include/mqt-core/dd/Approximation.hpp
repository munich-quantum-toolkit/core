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

#include "dd/Node.hpp"
#include "dd/Package.hpp"

namespace dd {

enum ApproximationStrategy { None, FidelityDriven, MemoryDriven };

VectorDD approximate(const VectorDD& state, double fidelity, Package& dd);
} // namespace dd
