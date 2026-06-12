/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUANTUMSTATE_H
#define MQT_CORE_QUANTUMSTATE_H
#include <mlir/IR/Operation.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <ostream>
#include <ranges>
#include <span>
#include <stdexcept>
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
    std::string result;
    result.reserve(nQubits);

    for (std::size_t i = nQubits; i > 0; --i) {
      result.push_back((q >> (i - 1) & 1U) != 0 ? '1' : '0');
    }
    return result;
  }

  /** @brief Puts a zero or one at bit position n in a bitstring.
   *
   * @param bitstring Bitstring that gets altered
   * @param position Position of bit that needs to be changed
   * @param one True if bit should be set to one
   * @return The altered bitstring
   */
  static unsigned int setBitN(const unsigned int bitstring,
                              const unsigned int position, const bool one) {
    const unsigned int mask = 1U << position;
    if (one) {
      return bitstring | mask;
    }
    return bitstring & ~mask;
  }

  /**
   * @brief This method receives a gate mapping and a bitmask ctrls and the
   * positions of the one (or two) qubit targets.
   *
   * This method receives a two qubit gate mapping and a bitmask for ctrl qubits
   * and a span with the first target qubit position at first position and the
   * potential second target qubit at second position. The gate is applied to
   * the valid qubit states. It returns the map which would be the qubit state
   * after gate application.
   *
   * @param gateMapping The mapping representing the gate
   * @param positionOfTargetQubits The position of the qubit targets.
   * @param bitmaskForCtrls The bitmask of the positively controlling qubits.
   * @return The qubit state after the gate has been applied.
   */
  std::unordered_map<unsigned int, std::complex<double>>
  getNewMappingFromQubitGate(
      std::unordered_map<unsigned int,
                         std::unordered_map<unsigned int, std::complex<double>>>
          gateMapping,
      std::span<unsigned int> positionOfTargetQubits,
      const unsigned int bitmaskForCtrls) {
    std::unordered_map<unsigned int, std::complex<double>> newValues;
    const auto numberOfTargetValues = 1U << positionOfTargetQubits.size();

    for (const auto& [key, value] : amplitudeMap) {
      if ((bitmaskForCtrls & key) != bitmaskForCtrls) {
        newValues[key] += value;
        continue;
      }

      unsigned int mapFrom = 0;

      std::vector<unsigned int> keysForNewValue(numberOfTargetValues);

      // Find the target keys from the current keys
      for (unsigned int i = 0; i < numberOfTargetValues; ++i) {
        if (i % 2 == 1) {
          keysForNewValue[i] = setBitN(key, positionOfTargetQubits[0], true);
        } else {
          keysForNewValue[i] = setBitN(key, positionOfTargetQubits[0], false);
        }
        if (numberOfTargetValues > 2) {
          if (i > 1) {
            keysForNewValue[i] =
                setBitN(keysForNewValue[i], positionOfTargetQubits[1], true);
          } else {
            keysForNewValue[i] =
                setBitN(keysForNewValue[i], positionOfTargetQubits[1], false);
          }
        }
        if (keysForNewValue[i] == key) {
          mapFrom = i;
        }
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (unsigned int i = 0; i < numberOfTargetValues; i++) {
        if (auto valueToI = mapForThisQubit[i]; abs(valueToI) > 1e-4) {
          newValues[keysForNewValue.at(i)] += valueToI * value;
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
   * @param reset True if target should be resetted in addition to measured.
   * @return MeasurementResult, containing the probability for the result and
   * the QuantumStates after measurement.
   */
  MeasurementResult measureOrResetQubit(const unsigned int target,
                                        const bool reset) {
    const auto qubitMask = 1U << target;

    double probabilityZero = 0.0;
    double probabilityOne = 0.0;
    std::unordered_map<unsigned int, std::complex<double>> newValuesZeroRes;
    std::unordered_map<unsigned int, std::complex<double>> newValuesOneRes;

    for (const auto& [key, value] : amplitudeMap) {
      unsigned int newKey = key;
      const bool isZero = (qubitMask & key) == 0;
      const double probability = norm(value);
      if (isZero) {
        probabilityZero += probability;
        newValuesZeroRes[newKey] = value;
      } else {
        if (reset) {
          newKey = key ^ qubitMask;
        }
        probabilityOne += probability;
        newValuesOneRes[newKey] = value;
      }
    }

    if (std::abs(1.0 - probabilityZero - probabilityOne) > 1e-4) {
      throw std::domain_error(
          "Probabilities of 0 and 1 do not add up to one after measurement.");
    }
    auto globalKeysView = std::views::keys(globalToLocalQubitNumber);
    std::vector<unsigned int> globalKeys{globalKeysView.begin(),
                                         globalKeysView.end()};
    MeasurementResult res = {};
    if (probabilityZero > 1e-4) {
      ++res.size;
      auto stateZero =
          std::make_shared<QuantumState>(globalKeys, maxNonzeroAmplitudes);
      stateZero->amplitudeMap = std::move(newValuesZeroRes);
      stateZero->normalize();
      res.states[0] = {probabilityZero, stateZero};
    }
    if (probabilityOne > 1e-4) {
      ++res.size;
      auto stateOne =
          std::make_shared<QuantumState>(globalKeys, maxNonzeroAmplitudes);
      stateOne->amplitudeMap = std::move(newValuesOneRes);
      stateOne->normalize();
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

#endif // MQT_CORE_QUANTUMSTATE_H
