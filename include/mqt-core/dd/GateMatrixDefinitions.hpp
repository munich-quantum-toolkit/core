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

#include "dd/DDDefinitions.hpp"
#include "ir/operations/OpType.hpp"

#include <vector>

namespace dd {

/// Single-qubit gate matrix for collapsing a qubit to the |0> state
constexpr GateMatrix MEAS_ZERO_MAT{1, 0, 0, 0};
/// Single-qubit gate matrix for collapsing a qubit to the |1> state
constexpr GateMatrix MEAS_ONE_MAT{0, 0, 0, 1};

/**
 * @brief Converts a given quantum operation to a single-qubit gate matrix
 * @param t The quantum operation to convert
 * @param params The parameters of the quantum operation
 * @return The single-qubit gate matrix representation of the quantum operation
 */
GateMatrix opToSingleQubitGateMatrix(qc::OpType t,
                                     const std::vector<fp>& params = {});

/**
 * @brief Converts a given quantum operation to a two-qubit gate matrix
 * @param t The quantum operation to convert
 * @param params The parameters of the quantum operation
 * @return The two-qubit gate matrix representation of the quantum operation
 */
TwoQubitGateMatrix opToTwoQubitGateMatrix(qc::OpType t,
                                          const std::vector<fp>& params = {});

} // namespace dd
