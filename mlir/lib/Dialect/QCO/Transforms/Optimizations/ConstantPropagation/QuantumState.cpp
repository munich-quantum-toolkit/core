/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUANTUMSTATE
#define MQT_CORE_QUANTUMSTATE
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/GateToMap.h"

#include <mlir/IR/Operation.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iterator>
#include <map>
#include <ostream>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::qco {

QuantumState::QuantumState(const std::span<unsigned int> globalQubitNumber,
                           const std::size_t maxNonzeroAmplitudes)
    : nQubits(globalQubitNumber.size()),
      maxNonzeroAmplitudes(maxNonzeroAmplitudes) {
  std::ranges::sort(globalQubitNumber);
  unsigned int localQ = 0;
  for (auto globalQ : globalQubitNumber) {
    globalToLocalQubitNumber[globalQ] = localQ;
    ++localQ;
  }
  amplitudeMap[0] = std::complex(1.0, 0.0);
}

void QuantumState::print(std::ostream& os) const { os << this->toString(); }

std::string QuantumState::toString() const {
  if (nQubits == 0) {
    return "";
  }

  std::ostringstream oss;
  bool first = true;
  for (auto ordered =
           std::map(this->amplitudeMap.begin(), this->amplitudeMap.end());
       auto const& [key, val] : ordered) {
    if (!first) {
      oss << ", ";
    }
    first = false;

    oss << "|" << qubitStringToBinary(key) << "> -> " << std::fixed
        << std::setprecision(2) << val.real();

    if (std::abs(val.imag()) > 1e-4) {
      oss << (val.imag() > 0 ? " + i" : " - i");
      oss << std::fixed << std::setprecision(2) << std::abs(val.imag());
    }
  }

  return oss.str();
}

bool QuantumState::operator==(const QuantumState& that) const {
  if (this->nQubits != that.nQubits ||
      this->maxNonzeroAmplitudes != that.maxNonzeroAmplitudes ||
      this->globalToLocalQubitNumber != that.globalToLocalQubitNumber) {
    return false;
  }

  if (amplitudeMap.size() != that.amplitudeMap.size()) {
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
  const double invDenominator = 1 / std::sqrt(denominator);
  for (const auto& key : amplitudeMap | std::views::keys) {
    amplitudeMap[key] *= invDenominator;
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
        unsigned int bitOfQubitState = 1U << indicesOfThis;
        if ((keyThis & bitOfQubitState) == bitOfQubitState) {
          currentQubitState += 1U << oldToNewIndicesThis.at(indicesOfThis);
        }
      }

      for (const auto& indicesOfThat : oldToNewIndicesThat | std::views::keys) {
        unsigned int bitOfQubitState = 1U << indicesOfThat;
        if ((keyThat & bitOfQubitState) == bitOfQubitState) {
          currentQubitState += 1U << oldToNewIndicesThat.at(indicesOfThat);
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

void QuantumState::changeGlobalIndex(const unsigned int target,
                                     const unsigned int newIndex) {
  const auto localIndex = globalToLocalQubitNumber.at(target);
  globalToLocalQubitNumber.erase(target);
  globalToLocalQubitNumber[newIndex] = localIndex;
}

void QuantumState::propagateGate(Operation* gate,
                                 const std::span<unsigned int> targets,
                                 const std::span<unsigned int> ctrls,
                                 const std::span<double> params) {
  const auto gateMapping = getQubitMappingOfGates(gate, params);

  unsigned int ctrlMask = 0;
  for (unsigned int const ctrl : ctrls) {
    ctrlMask |= 1U << globalToLocalQubitNumber.at(ctrl);
  }
  std::vector<unsigned int> localTargets;
  localTargets.reserve(targets.size());
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

bool QuantumState::isQubitAlwaysOne(const unsigned int q) const {
  const auto localIndex = globalToLocalQubitNumber.at(q);
  const auto mask = 1U << localIndex;
  return std::ranges::all_of(
      amplitudeMap | std::views::keys,
      [mask](auto qubits) { return (qubits & mask) == mask; });
}

bool QuantumState::isQubitAlwaysZero(const unsigned int q) const {
  const auto localIndex = globalToLocalQubitNumber.at(q);
  const auto mask = 1U << localIndex;
  return std::ranges::all_of(
      amplitudeMap | std::views::keys,
      [mask](auto qubits) { return (qubits & mask) == 0; });
}
bool QuantumState::hasAlwaysZeroAmplitude(
    const std::unordered_map<unsigned int, bool>& qubitValues) const {
  unsigned int localValue = 0;
  unsigned int mask = 0;
  for (const auto& [qubitIndex, qubitOne] : qubitValues) {
    mask |= 1U << globalToLocalQubitNumber.at(qubitIndex);
    if (qubitOne) {
      localValue |= 1U << globalToLocalQubitNumber.at(qubitIndex);
    }
  }
  return std::ranges::all_of(
      amplitudeMap | std::views::keys,
      [localValue, mask](auto qbit) { return (qbit & mask) != localValue; });
}

} // namespace mlir::qco

#endif // MQT_CORE_QUANTUMSTATE
