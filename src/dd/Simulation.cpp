/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Simulation.hpp"

#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace dd {
template <class Config>
std::map<std::string, std::size_t>
sample(const QuantumComputation& qc, const VectorDD& in, Package<Config>& dd,
       const std::size_t shots, const std::size_t seed) {
  auto isDynamicCircuit = false;
  auto hasMeasurements = false;
  auto measurementsLast = true;

  std::mt19937_64 mt{};
  if (seed != 0U) {
    mt.seed(seed);
  } else {
    // create and properly seed rng
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::generate(std::begin(randomData), std::end(randomData),
                  [&rd]() { return rd(); });
    std::seed_seq seeds(std::begin(randomData), std::end(randomData));
    mt.seed(seeds);
  }

  std::map<qc::Qubit, std::size_t> measurementMap{};

  // rudimentary check whether circuit is dynamic
  for (const auto& op : qc) {
    // if it contains any dynamic circuit primitives, it certainly is dynamic
    if (op->isClassicControlledOperation() || op->getType() == qc::Reset) {
      isDynamicCircuit = true;
      break;
    }

    // once a measurement is encountered we store the corresponding mapping
    // (qubit -> bit)
    if (const auto* measure = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
        measure != nullptr && measure->getType() == qc::Measure) {
      hasMeasurements = true;

      const auto& quantum = measure->getTargets();
      const auto& classic = measure->getClassics();

      for (std::size_t i = 0; i < quantum.size(); ++i) {
        measurementMap[quantum.at(i)] = classic.at(i);
      }
    }

    // if an operation happens after a measurement, the resulting circuit can
    // only be simulated in single shots
    if (hasMeasurements &&
        (op->isUnitary() || op->isClassicControlledOperation())) {
      measurementsLast = false;
    }
  }

  if (!measurementsLast) {
    isDynamicCircuit = true;
  }

  if (!isDynamicCircuit) {
    // if all gates are unitary (besides measurements at the end), we just
    // simulate once and measure all qubits repeatedly
    auto permutation = qc.initialLayout;
    auto e = in;

    for (const auto& op : qc) {
      // simply skip any non-unitary
      if (!op->isUnitary()) {
        continue;
      }

      // SWAP gates can be executed virtually by changing the permutation
      if (op->getType() == OpType::SWAP && !op->isControlled()) {
        const auto& targets = op->getTargets();
        std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
        continue;
      }

      e = applyUnitaryOperation(*op, e, dd, permutation);
    }

    // correct permutation if necessary
    changePermutation(e, permutation, qc.outputPermutation, dd);
    e = dd.reduceGarbage(e, qc.getGarbage());

    // measure all qubits
    std::map<std::string, std::size_t> counts{};
    for (std::size_t i = 0U; i < shots; ++i) {
      // measure all returns a string of the form "q(n-1) ... q(0)"
      auto measurement = dd.measureAll(e, false, mt);
      counts.operator[](measurement) += 1U;
    }
    // reduce reference count of measured state
    dd.decRef(e);

    std::map<std::string, std::size_t> actualCounts{};
    const auto numBits =
        qc.getClassicalRegisters().empty() ? qc.getNqubits() : qc.getNcbits();
    for (const auto& [bitstring, count] : counts) {
      std::string measurement(numBits, '0');
      if (hasMeasurements) {
        // if the circuit contains measurements, we only want to return the
        // measured bits
        for (const auto& [qubit, bit] : measurementMap) {
          // measurement map specifies that the circuit `qubit` is measured into
          // a certain `bit`
          measurement[numBits - 1U - bit] =
              bitstring[bitstring.size() - 1U - qc.outputPermutation.at(qubit)];
        }
      } else {
        // otherwise, we consider the output permutation for determining where
        // to measure the qubits to
        for (const auto& [qubit, bit] : qc.outputPermutation) {
          measurement[numBits - 1U - bit] =
              bitstring[bitstring.size() - 1U - qubit];
        }
      }
      actualCounts[measurement] += count;
    }
    return actualCounts;
  }

  std::map<std::string, std::size_t> counts{};

  for (std::size_t i = 0U; i < shots; i++) {
    std::vector<bool> measurements(qc.getNcbits(), false);

    auto permutation = qc.initialLayout;
    auto e = in;
    dd.incRef(e);
    for (const auto& op : qc) {
      if (op->isUnitary()) {
        // SWAP gates can be executed virtually by changing the permutation
        if (op->getType() == OpType::SWAP && !op->isControlled()) {
          const auto& targets = op->getTargets();
          std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
          continue;
        }

        e = applyUnitaryOperation(*op, e, dd, permutation);
        continue;
      }

      if (op->getType() == Measure) {
        const auto& measure = dynamic_cast<const NonUnitaryOperation&>(*op);
        e = applyMeasurement(measure, e, dd, mt, measurements, permutation);
        continue;
      }

      if (op->getType() == Reset) {
        const auto& reset = dynamic_cast<const NonUnitaryOperation&>(*op);
        e = applyReset(reset, e, dd, mt, permutation);
        continue;
      }

      if (op->isClassicControlledOperation()) {
        const auto& classic =
            dynamic_cast<const ClassicControlledOperation&>(*op);
        e = applyClassicControlledOperation(classic, e, dd, measurements,
                                            permutation);
        continue;
      }

      qc::unreachable();
    }

    // reduce reference count of measured state
    dd.decRef(e);

    std::string shot(qc.getNcbits(), '0');
    for (size_t bit = 0U; bit < qc.getNcbits(); ++bit) {
      if (measurements[bit]) {
        shot[qc.getNcbits() - bit - 1U] = '1';
      }
    }
    counts[shot]++;
  }
  return counts;
}

std::map<std::string, std::size_t> sample(const QuantumComputation& qc,
                                          const std::size_t shots,
                                          const std::size_t seed) {
  const auto nqubits = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  return sample(qc, dd->makeZeroState(nqubits), *dd, shots, seed);
}

template std::map<std::string, std::size_t>
sample<DDPackageConfig>(const QuantumComputation& qc, const VectorDD& in,
                        Package<>& dd, std::size_t shots, std::size_t seed);
} // namespace dd
