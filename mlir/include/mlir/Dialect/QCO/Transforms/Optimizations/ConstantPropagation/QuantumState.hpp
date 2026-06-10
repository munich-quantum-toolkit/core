/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUANTUMSTATEORTOP_H
#define MQT_CORE_QUANTUMSTATEORTOP_H
#include <mlir/IR/Operation.h>

#include <complex>
#include <cstddef>
#include <span>
#include <unordered_map>

namespace mlir::qco {

/**
 * @brief This class represents a quantum state.
 *
 * This class holds n qubits in different basis states with their corresponding
 * complex amplitude. It holds the information about which global qubit number
 * corresponds to which local qubit number.
 */

class QuantumState {
  std::size_t nQubits;
  std::size_t maxNonzeroAmplitudes;
  std::unordered_map<unsigned int, unsigned int> globalToLocalQubitNumber;
  std::unordered_map<unsigned int, std::complex<double>> amplitudeMap;

public:
  QuantumState(std::span<unsigned int> globalQubitNumber,
               std::size_t maxNonzeroAmplitudes);

  ~QuantumState();

  void print(std::ostream& os) const;

  std::string toString() const;

  bool operator==(const QuantumState& that) const;

  /**
   * @brief This method unifies two QuantumState.
   *
   * This method unifies the current QuantumState with the given one and returns
   * a new QuantumState, if the new state has no more than maxNonzeroAmplitude
   * nonzero amplitudes. Otherwise, throws a domain_error.
   *
   * @param that The QuantumState to unify this with.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes of this.
   */
  QuantumState unify(const QuantumState& that);

  /**
   * @brief This method applies a gate to the qubits.
   *
   * This method changes the amplitudes of a QuantumState according to the
   * applied gate. Returns the current QuantumState if it has no more than
   * maxNonZeroAmplitude nonzero amplitudes. Otherwise, throws a domain_error.
   *
   * @param gate The gate to be applied.
   * @param targets A span of the global indices of the target qubits.
   * @param posCtrls A span of the global indices of the ctrl qubits.
   * @param params The parameter applied to the gate.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes.
   */
  void propagateGate(Operation* gate, const std::span<unsigned int>& targets,
                     const std::span<unsigned int>& posCtrls = {},
                     const std::span<double>& params = {});

  /**
   * @brief This method applies a measurement to the qubits.
   *
   * This method applies a measurement to the qubits. It returns the
   * QuantumState in case the measurement was 0 and in case it was 1, alongside
   * the respective probabilities.
   *
   * @param target The global index of the qubit to be measured.
   * @return A map of the measurement result (zero and/or one) pointing to the
   * probability for the result and the QuantumStates after measurement.
   */
  std::map<unsigned int, std::pair<double, std::shared_ptr<QuantumState>>>
  measureQubit(unsigned int target);

  /**
   * @brief This method resets a qubit.
   *
   * This method resets a qubit. This is done by assuming that a measurement is
   * applied and measurements in the one-state are set to the zero state. In
   * order to not get a mixed state, the method returns one to two
   * QuantumStates, one in case the measurement was 0 and one in case it was 1,
   * alongside the respective probabilities. In both cases, the target qubit
   * will be zero (as a reset is performed).
   *
   * @param target The global index of the qubit to be measured.
   * @return A set of one to two QuantumStates and their corresponding
   * probabilities.
   */
  std::set<std::pair<double, std::shared_ptr<QuantumState>>>
  resetQubit(unsigned int target);

  /**
   * @brief This method checks if only amplitudes with a given qubit = 1 are
   * nonzero.
   *
   * @param q The global index of the qubit.
   * @return A set of one to two QuantumStates and their corresponding
   * probabilities.
   */
  [[nodiscard("QuantumState::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(size_t q) const;

  /**
   * @brief This method checks if only amplitudes with a given qubit = 0 are
   * nonzero.
   *
   * @param q The global index of the qubit.
   * @return A set of one to two QuantumStates and their corresponding
   * probabilities.
   */
  [[nodiscard("QuantumState::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(size_t q) const;

  /**
   * @brief Returns whether the given qubits have for a given value always a
   * zero amplitude.
   *
   * This method receives a number of global qubit indices and checks whether
   * they have for a given value always a zero amplitude.
   *
   * @param qubits The qubits which are being checked.
   * @param value The value for which is tested whether there is a nonzero
   * amplitude.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard("QuantumState::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                         unsigned int value) const;
};

} // namespace mlir::qco

#endif // MQT_CORE_QUANTUMSTATEORTOP_H
