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

#include <array>
#include <complex>
#include <cstddef>
#include <ostream>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::qco {

/**
 * @brief Result of a measurement or reset.
 *
 * The result contains 0, 1 or 2 QuantumStates, together with their respective
 * results.
 */
struct MeasurementResult {
  // The pair is (probability, resulting state).
  std::array<std::pair<double, std::shared_ptr<class QuantumState>>, 2>
      states{};
  // How many entries in `states` are actually filled (0, 1 or 2).
  std::size_t size = 0;

  [[nodiscard]] constexpr std::size_t count() const noexcept { return size; }

  // Range‑for friendliness
  [[nodiscard]] constexpr auto begin() const noexcept { return states.begin(); }
  [[nodiscard]] constexpr auto end() const noexcept {
    return states.begin() + size;
  }
};

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

  std::string qubitStringToBinary(unsigned int q) const {
    std::string str;
    for (int i = static_cast<int>(nQubits) - 1; i >= 0; i--) {
      if (const auto currentDigit = static_cast<unsigned int>(pow(2, i));
          q & currentDigit) {
        str += "1";
        q -= currentDigit;
      } else {
        str += "0";
      }
    }
    return str;
  }

  /**
   * @brief This method receives a two qubit gate mapping and a bitmask for
   * targets and ctrls.
   *
   * This method receives a two qubit gate mapping and a bitmask for target and
   * ctrl qubits. The gate is applied to the valid qubit states. It returns the
   * map which would be the qubit state after gate application.
   *
   * @param gateMapping The mapping representing the gate
   * @param bitmaskForQubitTargets The bitmask of the qubit targets. I.e. 011,
   * if zeroth and first qubit are targets.
   * @param bitmaskForCtrls The bitmask of the positively controlling qubits.
   * @return The qubit state after the gate has been applied.
   */
  std::unordered_map<unsigned int, std::complex<double>>
  getNewMappingForTwoQubitGate(
      std::unordered_map<unsigned int,
                         std::unordered_map<unsigned int, std::complex<double>>>
          gateMapping,
      std::unordered_map<unsigned int, unsigned int> bitmaskForQubitTargets,
      const unsigned int bitmaskForCtrls) {
    std::unordered_map<unsigned int, std::complex<double>> newValues;

    for (const auto& [key, value] : amplitudeMap) {
      if ((bitmaskForCtrls & key) != bitmaskForCtrls) {
        newValues[key] += value;
        continue;
      }

      unsigned int mapFrom = 0;
      std::vector<unsigned int> keysForNewValue(4);

      if ((key & bitmaskForQubitTargets[3]) == bitmaskForQubitTargets[3]) {
        mapFrom = 3;
        keysForNewValue[3] = key;
        keysForNewValue[2] = key - bitmaskForQubitTargets[1];
        keysForNewValue[1] = key - bitmaskForQubitTargets[2];
        keysForNewValue[0] = key - bitmaskForQubitTargets[3];
      } else if ((key & bitmaskForQubitTargets[2]) ==
                 bitmaskForQubitTargets[2]) {
        mapFrom = 2;
        keysForNewValue[3] = key + bitmaskForQubitTargets[1];
        keysForNewValue[2] = key;
        keysForNewValue[1] = key ^ bitmaskForQubitTargets[3];
        keysForNewValue[0] = key - bitmaskForQubitTargets[2];
      } else if ((key & bitmaskForQubitTargets[1]) ==
                 bitmaskForQubitTargets[1]) {
        mapFrom = 1;
        keysForNewValue[3] = key + bitmaskForQubitTargets[2];
        keysForNewValue[2] = key ^ bitmaskForQubitTargets[3];
        keysForNewValue[1] = key;
        keysForNewValue[0] = key - bitmaskForQubitTargets[1];
      } else {
        keysForNewValue[3] = key + bitmaskForQubitTargets[3];
        keysForNewValue[2] = key + bitmaskForQubitTargets[2];
        keysForNewValue[1] = key + bitmaskForQubitTargets[1];
        keysForNewValue[0] = key;
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (int i = 0; i < 4; i++) {
        if (auto valueToI = mapForThisQubit[i]; abs(valueToI) > 1e-4) {
          newValues[keysForNewValue[i]] += valueToI * value;
        }
      }
    }

    return newValues;
  }

  /**
   * @brief This method receives a single qubit gate mapping and a bitmask for
   * target and ctrls.
   *
   * This method receives a single qubit gate mapping and a bitmask for target
   * and ctrl qubits. The gate is applied to the valid qubit states. It returns
   * the map which would be the qubit state after gate application.
   *
   * @param gateMapping The mapping representing the gate
   * @param bitmaskForQubitTargets The bitmask of the qubit targets. I.e. 011,
   * if zeroth and first qubit are targets.
   * @param bitmaskForCtrls The bitmask of the positively controlling qubits.
   * @return The qubit state after the gate has been applied.
   */
  std::unordered_map<unsigned int, std::complex<double>>
  getNewMappingForSingleQubitGate(
      std::unordered_map<unsigned int,
                         std::unordered_map<unsigned int, std::complex<double>>>
          gateMapping,
      std::unordered_map<unsigned int, unsigned int> bitmaskForQubitTargets,
      const unsigned int bitmaskForCtrls) {
    std::unordered_map<unsigned int, std::complex<double>> newValues;

    for (const auto& [key, value] : amplitudeMap) {
      if ((bitmaskForCtrls & key) != bitmaskForCtrls) {
        newValues[key] += value;
        continue;
      }

      unsigned int mapFrom = 0;
      std::vector<unsigned int> keysForNewValue(2);

      if ((key & bitmaskForQubitTargets[1]) == bitmaskForQubitTargets[1]) {
        mapFrom = 1;
        keysForNewValue[1] = key;
        keysForNewValue[0] = key - bitmaskForQubitTargets[1];
      } else {
        keysForNewValue[1] = key + bitmaskForQubitTargets[1];
        keysForNewValue[0] = key;
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (int i = 0; i < 2; i++) {
        if (auto valueToI = mapForThisQubit[i]; abs(valueToI) > 1e-4) {
          newValues[keysForNewValue[i]] += valueToI * value;
        }
      }
    }

    return newValues;
  }

  /**
   * @brief This method applies a measurement or reset to the qubits.
   *
   * This method applies a measurement or reset to the qubits. It returns the
   * QuantumState in case the measurement was 0 and in case it was 1, alongside
   * the respective probabilities. If a reset was applied, the qubit that was
   * measured is set to 0.
   *
   * @param target The global index of the qubit to be measured.
   * @return MeasurementResult, containing the probability for the result and
   * the QuantumStates after measurement.
   */
  MeasurementResult measureOrResetQubit(const unsigned int target,
                                        const bool reset) {
    const auto qubitMask = static_cast<unsigned int>(pow(2, target) + 0.1);

    double probabilityZero = 0.0;
    double probabilityOne = 0.0;
    std::unordered_map<unsigned int, std::complex<double>> newValuesZeroRes;
    std::unordered_map<unsigned int, std::complex<double>> newValuesOneRes;

    for (const auto& [key, value] : amplitudeMap) {
      if ((qubitMask & key) == 0) {
        probabilityZero += norm(value);
        newValuesZeroRes.insert({key, value});
      } else {
        if (reset) {
          const unsigned int newKey = key ^ qubitMask;
        }
        probabilityOne += norm(value);
        newValuesOneRes.insert({key, value});
      }
    }

    if (std::abs(1.0 - probabilityZero - probabilityOne) > 1e-4) {
      throw std::domain_error(
          "Probabilities of 0 and 1 do not add up to one after measurement.");
    }
    auto globalKeysView = std::views::keys(globalToLocalQubitNumber);
    std::vector<unsigned int> globalKeys{globalKeysView.begin(),
                                         globalKeysView.end()};
    auto stateZero =
        std::make_shared<QuantumState>(globalKeys, maxNonzeroAmplitudes);
    stateZero->amplitudeMap = newValuesZeroRes;
    stateZero->normalize();
    auto stateOne =
        std::make_shared<QuantumState>(globalKeys, maxNonzeroAmplitudes);
    stateOne->amplitudeMap = newValuesOneRes;
    stateOne->normalize();

    MeasurementResult res = {};
    if (probabilityZero > 1e-4) {
      ++res.size;
      res.states[0] = {probabilityZero, stateZero};
    }

    if (probabilityOne > 1e-4) {
      ++res.size;
      res.states[1] = {probabilityOne, stateOne};
    }

    return res;
  }

public:
  QuantumState(std::span<unsigned int> globalQubitNumber,
               std::size_t maxNonzeroAmplitudes);

  ~QuantumState() = default;

  void print(std::ostream& os) const;

  [[nodiscard("QuantumState::toString called but ignored")]] std::string
  toString() const;

  [[nodiscard("QuantumState::== called but ignored")]] bool
  operator==(const QuantumState& that) const;

  /**
   * @brief This method normalizes the amplitudes of a state.
   */
  void normalize();

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
  [[nodiscard("QuantumState::unify called but ignored")]] QuantumState
  unify(const QuantumState& that);

  /**
   * @brief This method applies a gate to the qubits.
   *
   * This method changes the amplitudes of a QuantumState according to the
   * applied gate. Returns the current QuantumState if it has no more than
   * maxNonZeroAmplitude nonzero amplitudes. Otherwise, throws a domain_error.
   *
   * @param gate The gate to be applied.
   * @param targets A span of the global indices of the target qubits.
   * @param ctrls A span of the global indices of the ctrl qubits.
   * @param params The parameter applied to the gate.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes.
   */
  void propagateGate(Operation* gate, std::span<unsigned int> targets,
                     std::span<unsigned int> ctrls = {},
                     std::span<double> params = {});

  /**
   * @brief This method applies a measurement to the qubits.
   *
   * This method applies a measurement to the qubits. It returns the
   * QuantumState in case the measurement was 0 and in case it was 1, alongside
   * the respective probabilities.
   *
   * @param target The global index of the qubit to be measured.
   * @return MeasurementResult, containing the probability for the result and
   * the QuantumStates after measurement.
   */
  [[nodiscard(
      "QuantumState::measureQubit called but ignored")]] MeasurementResult
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
   * @return MeasurementResult, containing the probability for the result and
   * the QuantumStates after measurement.
   */
  [[nodiscard("QuantumState::resetQubit called but ignored")]] MeasurementResult
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
  isQubitAlwaysOne(unsigned int q) const;

  /**
   * @brief This method checks if only amplitudes with a given qubit = 0 are
   * nonzero.
   *
   * @param q The global index of the qubit.
   * @return A set of one to two QuantumStates and their corresponding
   * probabilities.
   */
  [[nodiscard("QuantumState::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(unsigned int q) const;

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
  hasAlwaysZeroAmplitude(std::span<unsigned int> qubits,
                         unsigned int value) const;
};

} // namespace mlir::qco

#endif // MQT_CORE_QUANTUMSTATEORTOP_H
