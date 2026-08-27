/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Simulation.hpp"

#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dd {
namespace {

using MeasurementAssignment = std::pair<qc::Qubit, std::size_t>;

struct CircuitAnalysis {
  bool isDynamic = false;
  bool hasMeasurements = false;
  std::vector<MeasurementAssignment> terminalMeasurements{};
};

class VectorRootGuard {
public:
  VectorRootGuard(Package& package, VectorDD state)
      : package(package), state(state) {}

  VectorRootGuard(const VectorRootGuard&) = delete;
  VectorRootGuard& operator=(const VectorRootGuard&) = delete;
  VectorRootGuard(VectorRootGuard&&) = delete;
  VectorRootGuard& operator=(VectorRootGuard&&) = delete;

  ~VectorRootGuard() { package.decRef(state); }

  [[nodiscard]] VectorDD& get() noexcept { return state; }

  [[nodiscard]] VectorDD release() noexcept {
    const auto released = state;
    state = VectorDD::zero();
    return released;
  }

private:
  Package& package;
  VectorDD state;
};

[[nodiscard]] CircuitAnalysis
analyzeCircuit(const qc::QuantumComputation& circuit) {
  auto analysis = CircuitAnalysis{};
  auto measurementSeen = false;

  for (const auto& operation : circuit) {
    if (operation->isIfElseOperation() || operation->getType() == qc::Reset) {
      analysis.isDynamic = true;
    }

    if (const auto* measurement =
            dynamic_cast<const qc::NonUnitaryOperation*>(operation.get());
        measurement != nullptr && measurement->getType() == qc::Measure) {
      analysis.hasMeasurements = true;
      measurementSeen = true;

      const auto& qubits = measurement->getTargets();
      const auto& bits = measurement->getClassics();
      if (qubits.size() != bits.size()) {
        throw std::invalid_argument(
            "Measurement targets and classical bits must have equal sizes.");
      }
      for (std::size_t i = 0U; i < qubits.size(); ++i) {
        const auto qubit = qubits.at(i);
        const auto bit = bits.at(i);
        if (circuit.initialLayout.apply(qubit) >= circuit.getNqubits()) {
          throw std::out_of_range("Measurement qubit is out of range.");
        }
        if (bit >= circuit.getNcbits()) {
          throw std::out_of_range("Measurement bit is out of range.");
        }
        analysis.terminalMeasurements.emplace_back(qubit, bit);
      }
      continue;
    }

    if (measurementSeen &&
        (operation->isUnitary() || operation->isIfElseOperation())) {
      analysis.isDynamic = true;
    }
  }

  return analysis;
}

[[nodiscard]] bool
isExecutableVirtually(const qc::Operation& operation) noexcept {
  switch (operation.getType()) {
  case qc::I:
  case qc::Barrier:
    return true;
  case qc::SWAP:
    return !operation.isControlled();
  default:
    return false;
  }
}

void applyVirtualOperation(const qc::Operation& operation,
                           qc::Permutation& permutation) noexcept {
  if (operation.getType() == qc::SWAP) {
    const auto& targets = operation.getTargets();
    std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
  }
}

void finalizeState(const qc::QuantumComputation& circuit, VectorDD& state,
                   qc::Permutation& permutation, Package& package) {
  changePermutation(state, permutation, circuit.outputPermutation, package);
  state = package.reduceGarbage(state, circuit.getGarbage());
  if (circuit.hasGlobalPhase()) {
    state = applyGlobalPhase(state, circuit.getGlobalPhase(), package);
  }
}

[[nodiscard]] std::map<std::string, std::size_t>
sampleState(VectorDD& state, const std::size_t shots, Package& package,
            std::mt19937_64& rng) {
  std::map<std::string, std::size_t> counts{};
  for (std::size_t shot = 0U; shot < shots; ++shot) {
    ++counts[package.measureAll(state, false, rng)];
  }
  return counts;
}

[[nodiscard]] std::string formatExplicitMeasurement(
    const std::string& measuredQubits,
    const std::vector<MeasurementAssignment>& measurements,
    const qc::Permutation& permutation, const std::size_t numClassicalBits) {
  std::string result(numClassicalBits, '0');
  for (const auto& [qubit, bit] : measurements) {
    result.at(numClassicalBits - 1U - bit) =
        measuredQubits.at(measuredQubits.size() - 1U - permutation.at(qubit));
  }
  return result;
}

[[nodiscard]] std::string
formatImplicitMeasurement(const std::string& measuredQubits,
                          const qc::QuantumComputation& circuit) {
  const auto numQubits = circuit.getNqubits();
  if (measuredQubits.size() > numQubits) {
    throw std::invalid_argument(
        "Measured state contains more qubits than the circuit.");
  }
  return std::string(numQubits - measuredQubits.size(), '0') + measuredQubits;
}

void executeStateOperation(const qc::Operation& operation, VectorDD& state,
                           Package& package, qc::Permutation& permutation,
                           const std::vector<bool>& measurements) {
  if (operation.isUnitary()) {
    if (isExecutableVirtually(operation)) {
      applyVirtualOperation(operation, permutation);
    } else {
      state = applyUnitaryOperation(operation, state, package, permutation);
    }
    return;
  }

  if (operation.isIfElseOperation()) {
    const auto& ifElse = dynamic_cast<const qc::IfElseOperation&>(operation);
    state =
        applyIfElseOperation(ifElse, state, package, measurements, permutation);
    return;
  }

  qc::unreachable();
}

} // namespace

SamplingResult sample(const qc::QuantumComputation& circuit, VectorDD in,
                      Package& package, const std::size_t shots,
                      std::mt19937_64& rng) {
  auto inputRoot = VectorRootGuard(package, in);
  const auto analysis = analyzeCircuit(circuit);

  if (!analysis.isDynamic) {
    auto permutation = circuit.initialLayout;
    auto& state = inputRoot.get();

    for (const auto& operation : circuit) {
      if (operation->isUnitary()) {
        executeStateOperation(*operation, state, package, permutation, {});
      }
    }

    std::map<std::string, std::size_t> counts{};
    if (analysis.hasMeasurements) {
      for (const auto& [rawResult, count] :
           sampleState(state, shots, package, rng)) {
        const auto result =
            formatExplicitMeasurement(rawResult, analysis.terminalMeasurements,
                                      permutation, circuit.getNcbits());
        counts[result] += count;
      }
      finalizeState(circuit, state, permutation, package);
    } else {
      finalizeState(circuit, state, permutation, package);
      for (const auto& [rawResult, count] :
           sampleState(state, shots, package, rng)) {
        counts[formatImplicitMeasurement(rawResult, circuit)] += count;
      }
    }

    return {.counts = std::move(counts),
            .state = inputRoot.release(),
            .executions = 1U};
  }

  if (shots == 0U) {
    return {.counts = {}, .state = inputRoot.release(), .executions = 0U};
  }

  std::map<std::string, std::size_t> counts{};
  auto finalState = VectorDD{};
  for (std::size_t shot = 0U; shot < shots; ++shot) {
    auto measurements = std::vector<bool>(circuit.getNcbits(), false);
    auto permutation = circuit.initialLayout;
    package.incRef(inputRoot.get());
    auto stateRoot = VectorRootGuard(package, inputRoot.get());
    auto& state = stateRoot.get();

    for (const auto& operation : circuit) {
      if (operation->isUnitary() || operation->isIfElseOperation()) {
        executeStateOperation(*operation, state, package, permutation,
                              measurements);
      } else if (operation->getType() == qc::Measure) {
        const auto& measurement =
            dynamic_cast<const qc::NonUnitaryOperation&>(*operation);
        state = applyMeasurement(measurement, state, package, rng, measurements,
                                 permutation);
      } else if (operation->getType() == qc::Reset) {
        const auto& reset =
            dynamic_cast<const qc::NonUnitaryOperation&>(*operation);
        state = applyReset(reset, state, package, rng, permutation);
      } else {
        qc::unreachable();
      }
    }

    std::string result{};
    if (analysis.hasMeasurements) {
      result.assign(circuit.getNcbits(), '0');
      for (std::size_t bit = 0U; bit < measurements.size(); ++bit) {
        if (measurements.at(bit)) {
          result.at(measurements.size() - 1U - bit) = '1';
        }
      }
      if (shot + 1U == shots) {
        finalizeState(circuit, state, permutation, package);
      }
    } else {
      finalizeState(circuit, state, permutation, package);
      result = formatImplicitMeasurement(package.measureAll(state, false, rng),
                                         circuit);
    }
    ++counts[result];

    if (shot + 1U == shots) {
      finalState = stateRoot.release();
    }
  }

  return {
      .counts = std::move(counts), .state = finalState, .executions = shots};
}

std::map<std::string, std::size_t> sample(const qc::QuantumComputation& circuit,
                                          const VectorDD& in, Package& package,
                                          const std::size_t shots,
                                          const std::size_t seed) {
  std::mt19937_64 rng{};
  if (seed != 0U) {
    rng.seed(seed);
  } else {
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device randomDevice;
    std::ranges::generate(randomData,
                          [&randomDevice]() { return randomDevice(); });
    std::seed_seq seeds(std::begin(randomData), std::end(randomData));
    rng.seed(seeds);
  }

  auto result = sample(circuit, in, package, shots, rng);
  package.decRef(result.state);
  return std::move(result.counts);
}

VectorDD simulate(const qc::QuantumComputation& circuit, const VectorDD& in,
                  Package& package) {
  auto permutation = circuit.initialLayout;
  auto out = in;
  for (const auto& operation : circuit) {
    if (isExecutableVirtually(*operation)) {
      applyVirtualOperation(*operation, permutation);
    } else {
      out = applyUnitaryOperation(*operation, out, package, permutation);
    }
  }

  finalizeState(circuit, out, permutation, package);
  return out;
}

std::map<std::string, std::size_t> sample(const qc::QuantumComputation& circuit,
                                          const std::size_t shots,
                                          const std::size_t seed) {
  const auto nqubits = circuit.getNqubits();
  const auto package = std::make_unique<Package>(nqubits);
  return sample(circuit, makeZeroState(nqubits, *package), *package, shots,
                seed);
}
} // namespace dd
