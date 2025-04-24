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

/**
 * @brief Approximate the input state to a given final fidelity.
 *
 * @details Traverses the decision diagram layer by layer in a breadth-first
 * manner and eliminates edges greedily until the budget is exhausted.
 *
 * @param state The DD to approximate.
 * @param fidelity The desired final fidelity after approximation.
 * @param dd The DD package to use for the simulation
 */
VectorDD approximate(VectorDD& state, double fidelity, Package& dd);

} // namespace dd
