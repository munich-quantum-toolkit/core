/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines functions for classically simulating quantum circuits.
 */

#pragma once

#include "dd/Approximation.hpp"
#include "dd/Package_fwd.hpp"

#include <cstddef>
#include <map>
#include <string>

namespace qc {
class QuantumComputation;
}

namespace dd {

/**
 * @brief Simulate a purely-quantum @ref qc::QuantumComputation on a given input
 * state using decision diagrams.
 *
 * @details This method classically simulates the quantum computation @p qc on
 * the input state @p in by sequentially applying the operations in the circuit
 * to the initial state via decision diagram multiplication.
 *
 * This simple simulation method can only handle circuits that do not contain
 * any classical control operations or measurements.
 * Its main purpose is to construct a representation of the statevector after
 * simulating the quantum computation for the given input state.
 * For more elaborate simulation methods that can handle classical control and
 * mid-circuit measurements, see @ref sample(const QuantumComputation&,
 * std::size_t, std::size_t).
 *
 * @tparam stgy The approximation strategy applied.
 * @param qc The quantum computation to simulate
 * @param in The input state to simulate. Represented as a vector DD.
 * @param dd The DD package to use for the simulation
 * @param approx The object describing the approximation strategy.
 * @return A vector DD representing the output state of the simulation
 */
template <const ApproximationStrategy stgy = None>
VectorDD simulate(const qc::QuantumComputation& qc, const VectorDD& in,
                  Package& dd,
                  Approximation<stgy> approx = Approximation<stgy>{});

/**
 * @brief Sample from the output distribution of a quantum computation
 *
 * @details This method classically simulates the quantum computation @p qc
 * starting from the all-zero state and samples @p shots times from the output
 * distribution.
 * The seed for the random number generator can be set using @p seed.
 *
 * For a circuit without mid-circuit measurements, this function will construct
 * a representation of the final statevector similar to @ref simulate and then
 * repeatedly sample from the resulting decision diagram, without actually
 * collapsing the state. For a fixed number of qubits, each sample can be drawn
 * in constant time, which is a significant of the decision diagram structure.
 *
 * For a circuit with mid-circuit measurements, this function will separately
 * execute the circuit for each sample, probabilistically collapsing the state
 * after each measurement.
 *
 * @param qc The quantum computation to simulate
 * @param shots The number of shots to sample
 * @param seed The seed for the random number generator
 * @return A histogram of the measurement results
 */
std::map<std::string, std::size_t> sample(const qc::QuantumComputation& qc,
                                          std::size_t shots = 1024U,
                                          std::size_t seed = 0U);

/**
 * @brief Sample from the output distribution of a quantum computation
 *
 * @details This is a more general version of @ref sample that allows for
 * choosing the input state to simulate as well as the DD package to use for the
 * simulation.
 *
 * @param qc The quantum computation to simulate
 * @param in The input state to simulate. Represented as a vector DD.
 * @param dd The DD package to use for the simulation
 * @param shots The number of shots to sample
 * @param seed The seed for the random number generator
 * @return A histogram of the measurement results
 */
std::map<std::string, std::size_t> sample(const qc::QuantumComputation& qc,
                                          const VectorDD& in, Package& dd,
                                          std::size_t shots,
                                          std::size_t seed = 0U);
} // namespace dd
