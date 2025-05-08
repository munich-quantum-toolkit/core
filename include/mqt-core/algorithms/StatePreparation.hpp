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

#include "ir/QuantumComputation.hpp"

#include <complex>
#include <vector>

namespace qc {
/**
 * @brief Prepares a generic quantum state from a list of normalized
 *        complex amplitudes
 *
 * Adapted implementation of IBM Qiskit's State Preparation:
 * https://github.com/Qiskit/qiskit/blob/e9ccd3f374fd5424214361d47febacfa5919e1e3/qiskit/circuit/library/data_preparation/state_preparation.py
 * based on the following paper:
 *
 *      V. V. Shende, S. S. Bullock and I. L. Markov,
 *      "Synthesis of quantum-logic circuits", in IEEE Transactions on
 *      Computer-Aided Design of Integrated Circuits and Systems,
 *      vol. 25, no. 6, pp. 1000-1010, June 2006,
 *      doi: 10.1109/TCAD.2005.855930.
 *
 * @param amplitudes State (vector) to prepare.
 * Must be normalized and have a size that is a power of two.
 * @param eps Precision wanted for computations, default 1e-10
 * @return Quantum computation that prepares the state
 * @throws invalid_argument If @p amplitudes is not normalized or its length is
 * not a power of two.
 **/
[[nodiscard]] auto createStatePreparationCircuit(
    const std::vector<std::complex<double>>& amplitudes, double eps = 1e-10)
    -> QuantumComputation;
} // namespace qc
