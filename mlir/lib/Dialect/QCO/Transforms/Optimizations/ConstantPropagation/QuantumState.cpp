/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUANTUMSTATEORTOP
#define MQT_CORE_QUANTUMSTATEORTOP
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/GateToMap.h"

#include <mlir/IR/Operation.h>

#include <format>
#include <iterator>
#include <ranges>
#include <vector>

namespace mlir::qco {

QuantumState::QuantumState(const std::span<unsigned int> globalQubitNumber,
                           const std::size_t maxNonzeroAmplitudes)
    : maxNonzeroAmplitudes(maxNonzeroAmplitudes) {
  nQubits = globalQubitNumber.size();
  std::ranges::sort(globalQubitNumber);
  unsigned int localQ = 0;
  for (auto globalQ : globalQubitNumber) {
    globalToLocalQubitNumber.insert({globalQ, localQ});
    ++localQ;
  }
  amplitudeMap = std::unordered_map<unsigned int, std::complex<double>>();
  amplitudeMap.insert({0, std::complex<double>(1.0, 0.0)});
}

void QuantumState::print(std::ostream& os) const { os << this->toString(); }

std::string QuantumState::toString() const {
  std::string str;
  bool first = true;
  for (auto ordered =
           std::map(this->amplitudeMap.begin(), this->amplitudeMap.end());
       auto const& [key, val] : ordered) {
    if (!first) {
      str += ", ";
    }
    first = false;
    std::string cn = std::format("{:.2f}", val.real());
    if (val.imag() > 1e-4) {
      cn += " + i" + std::format("{:.2f}", val.imag());
    } else if (val.imag() < -1e-4) {
      cn += " - i" + std::format("{:.2f}", -val.imag());
    }
    str += "|" + qubitStringToBinary(key) + "> -> " + cn;
  }

  return str;
}

bool QuantumState::operator==(const QuantumState& that) const {
  if (this->nQubits != that.nQubits ||
      this->maxNonzeroAmplitudes != that.maxNonzeroAmplitudes ||
      this->globalToLocalQubitNumber != that.globalToLocalQubitNumber) {
    return false;
  }

  return std::ranges::all_of(
      this->amplitudeMap,
      [&](const std::pair<unsigned int, std::complex<double>>& p) {
        auto [key, val] = p;
        return that.amplitudeMap.contains(key) &&
               abs(val - that.amplitudeMap.at(key)) < 1e-4;
      });
}

void QuantumState::normalize() {
  double denominator = 0.0;
  for (const auto& value : amplitudeMap | std::views::values) {
    denominator += norm(value);
  }
  for (const auto& key : amplitudeMap | std::views::keys) {
    amplitudeMap[key] /= std::sqrt(denominator);
  }
}

QuantumState QuantumState::unify(const QuantumState& that) {
  // Check if future state would be too large
  if (amplitudeMap.size() * that.amplitudeMap.size() > maxNonzeroAmplitudes) {
    throw std::domain_error("Number of nonzero amplitudes too high. State "
                            "needs to be treated as TOP.");
  }

  std::unordered_map<unsigned int, unsigned int> newGlobalToLocalMapping;
  std::unordered_map<unsigned int, std::complex<double>> newAmplitudes;

  // Create the new global indices
  const auto globalIndicesThis = std::views::keys(globalToLocalQubitNumber);
  const auto globalIndicesThat =
      std::views::keys(that.globalToLocalQubitNumber);
  std::vector<unsigned int> combinedGlobalIndices;
  combinedGlobalIndices.reserve(globalToLocalQubitNumber.size() +
                                that.globalToLocalQubitNumber.size());
  std::ranges::copy(globalIndicesThis,
                    std::back_inserter(combinedGlobalIndices));
  std::ranges::copy(globalIndicesThat,
                    std::back_inserter(combinedGlobalIndices));
  std::ranges::sort(combinedGlobalIndices);

  for (unsigned int i = 0; i < combinedGlobalIndices.size(); ++i) {
    newGlobalToLocalMapping[combinedGlobalIndices[i]] = i;
  }

  // Create mappings from the old to the new indices
  std::unordered_map<unsigned int, unsigned int> oldToNewIndicesThis;
  std::unordered_map<unsigned int, unsigned int> oldToNewIndicesThat;
  for (const auto& keyThis : globalToLocalQubitNumber | std::views::keys) {
    oldToNewIndicesThis[globalToLocalQubitNumber.at(keyThis)] =
        newGlobalToLocalMapping.at(keyThis);
  }
  for (const auto& keyThat : that.globalToLocalQubitNumber | std::views::keys) {
    oldToNewIndicesThat[that.globalToLocalQubitNumber.at(keyThat)] =
        newGlobalToLocalMapping.at(keyThat);
  }

  // Create new amplitude map
  for (const auto& [keyThis, valThis] : amplitudeMap) {
    for (const auto& [keyThat, valThat] : that.amplitudeMap) {
      unsigned int currentQubitState = 0;

      for (const auto& indicesOfThis : oldToNewIndicesThis | std::views::keys) {
        unsigned int bitOfQubitState = pow(2, indicesOfThis);
        if ((keyThis & bitOfQubitState) == bitOfQubitState) {
          currentQubitState += pow(2, oldToNewIndicesThis.at(indicesOfThis));
        }
      }

      for (const auto& indicesOfThat : oldToNewIndicesThat | std::views::keys) {
        unsigned int bitOfQubitState = pow(2, indicesOfThat);
        if ((keyThat & bitOfQubitState) == bitOfQubitState) {
          currentQubitState += pow(2, oldToNewIndicesThat.at(indicesOfThat));
        }
      }
      newAmplitudes[currentQubitState] = valThis * valThat;
    }
  }
  auto newState = QuantumState(combinedGlobalIndices, maxNonzeroAmplitudes);
  newState.amplitudeMap = newAmplitudes;
  newState.globalToLocalQubitNumber = newGlobalToLocalMapping;

  return newState;
}

void QuantumState::propagateGate(Operation* gate,
                                 std::span<unsigned int> targets,
                                 std::span<unsigned int> ctrls,
                                 std::span<double> params) {
  const auto gateMapping = getQubitMappingOfGates(gate, params);

  unsigned int ctrlMask = 0;
  for (unsigned int const posCtrl : ctrls) {
    ctrlMask += static_cast<unsigned int>(
        pow(2, globalToLocalQubitNumber.at(posCtrl)) + 0.1);
  }
  std::vector<unsigned int> localTargets;
  for (unsigned int q : targets) {
    localTargets.push_back(globalToLocalQubitNumber.at(q));
  }

  std::unordered_map<unsigned int, std::complex<double>> newValues =
      getNewMappingFromQubitGate(gateMapping, localTargets, ctrlMask);

  amplitudeMap.clear();
  for (const auto& [key, value] : newValues) {
    if (norm(value) > 1e-4) {
      amplitudeMap[key] = value;
    }
  }
  if (amplitudeMap.size() > maxNonzeroAmplitudes) {
    throw std::domain_error("Number of nonzero amplitudes too high. State "
                            "needs to be treated as TOP.");
  }
}

MeasurementResult QuantumState::measureQubit(const unsigned int target) {
  return measureOrResetQubit(target, false);
}

MeasurementResult QuantumState::resetQubit(const unsigned int target) {
  return measureOrResetQubit(target, true);
}

bool QuantumState::isQubitAlwaysOne(unsigned int q) const {
  const auto mask =
      static_cast<unsigned int>(pow(2, static_cast<double>(q)) + 0.1);
  return std::ranges::all_of(
      amplitudeMap | std::views::keys,
      [mask](auto qubits) { return (qubits & mask) == mask; });
}

bool QuantumState::isQubitAlwaysZero(unsigned int q) const {
  const auto mask =
      static_cast<unsigned int>(pow(2, static_cast<double>(q)) + 0.1);
  return std::ranges::all_of(
      amplitudeMap | std::views::keys,
      [mask](auto qubits) { return (qubits & mask) == 0; });
}
bool QuantumState::hasAlwaysZeroAmplitude(const std::span<unsigned int> qubits,
                                          const unsigned int value) const {
  unsigned int localValue = 0;
  unsigned int mask = 0;
  for (unsigned int i = 0; i < qubits.size(); ++i) {
    const unsigned int currentPower =
        static_cast<unsigned int>(pow(2, i) + 0.1);
    const unsigned int qubitPower =
        static_cast<unsigned int>(pow(2, qubits[i]) + 0.1);
    mask += qubitPower;
    if ((value & currentPower) != 0) {
      localValue += qubitPower;
    }
  }
  return std::ranges::all_of(
      amplitudeMap | std::views::keys,
      [localValue, mask](auto qbit) { return (qbit & mask) != localValue; });
}

} // namespace mlir::qco

#endif // MQT_CORE_QUANTUMSTATEORTOP
