/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

#include "dd/Package_fwd.hpp"

#include <cstddef>
#include <map>
#include <random>
#include <string>

namespace qc {
class QuantumComputation;
}

namespace dd {

/** @brief Result of sampling a quantum computation. */
struct SamplingResult {
  std::map<std::string, std::size_t> counts{};
  VectorDD state{};
  std::size_t executions = 0U;
};

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
 * mid-circuit measurements, see @ref dd::sample.
 *
 * @param qc The quantum computation to simulate
 * @param in The input state to simulate. Represented as a vector DD.
 * @param dd The DD package to use for the simulation
 * @return A vector DD representing the output state of the simulation
 */
VectorDD simulate(const qc::QuantumComputation& qc, const VectorDD& in,
                  Package& dd);

/**
 * @brief Execute and sample a quantum computation using a caller-owned state,
 * package, and random-number generator.
 *
 * @details This is the low-level execution primitive underlying the sampling
 * convenience functions. Static circuits are executed once and sampled
 * without collapsing the retained state. Dynamic circuits are executed once
 * per shot and retain the final state from the last execution.
 *
 * The registered root reference owned by @p in is transferred to this
 * function. The returned @ref SamplingResult::state owns exactly one
 * registered root reference in @p dd and must eventually be passed to
 * @ref Package::decRef. For a dynamic circuit with zero shots, the input state
 * is returned unchanged and no execution is reported.
 *
 * @param qc The quantum computation to execute
 * @param in The input state whose registered root reference is transferred
 * @param dd The DD package to use for execution
 * @param shots The number of samples to draw
 * @param rng The random-number generator to use
 * @return Counts, the retained final state, and the number of executions
 */
[[nodiscard]] SamplingResult sample(const qc::QuantumComputation& qc,
                                    VectorDD in, Package& dd, std::size_t shots,
                                    std::mt19937_64& rng);

/**
 * @brief Sample from the output distribution of a quantum computation
 *
 * @details This method classically simulates the quantum computation @p qc
 * starting from the all-zero state and samples @p shots times from the output
 * distribution.
 * The seed for the random number generator can be set using @p seed.
 *
 * For a circuit without mid-circuit measurements, this function will construct
 * a representation of the final statevector similar to @ref dd::simulate and
 * then repeatedly sample from the resulting decision diagram, without actually
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
 * @details This is a more general version of @ref dd::sample that allows for
 * choosing the input state to simulate as well as the DD package to use for the
 * simulation. The registered root reference owned by @p in is transferred to
 * this function.
 *
 * @param qc The quantum computation to simulate
 * @param in The input state whose registered root reference is transferred.
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
