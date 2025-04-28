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

#include <utility>

namespace dd {

/**
 * @brief Approximate the @p state based on fidelity. The fidelity of the
 * approximated state will be at least @p fidelity.
 *
 * @details Traverses the decision diagram layer by layer in a breadth-first
 * manner (iterative deepening algorithm) and eliminates edges greedily until
 * the budget (1 - @p fidelity) is exhausted.
 *
 * @param state The DD to approximate.
 * @param fidelity The desired minimum fidelity after approximation.
 * @param dd The DD package to use for the approximation.
 */
std::pair<VectorDD, double> approximate(const VectorDD& state, double fidelity,
                                        Package& dd);

} // namespace dd
