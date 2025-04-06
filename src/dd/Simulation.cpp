/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Simulation.hpp"

#include "dd/Approximation.hpp"
#include "dd/Complex.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace dd {
/**
 * @brief Returns `true` if the operation is virtually swappable.
 */
bool isVirtuallySwappable(const qc::Operation& op) noexcept {
  return op.getType() == qc::SWAP && !op.isControlled();
}

/**
 * @brief Virtually SWAP by permuting the layout.
 */
void virtualSwap(const qc::Operation& op, qc::Permutation& perm) noexcept {
  const auto& targets = op.getTargets();
  std::swap(perm.at(targets[0U]), perm.at(targets[1U]));
}

/**
 * @brief Returns `true` if the circuit has a global phase.
 */
bool hasGlobalPhase(const fp& phase) noexcept { return std::abs(phase) > 0; }

/**
 * @brief Apply global phase to `out.w`. Decreases reference count of old `w`
 * value.
 */
void applyGlobalPhase(const fp& phase, VectorDD& out, Package& dd) {
  const Complex oldW = out.w; // create a temporary copy for reference counting

  out.w = dd.cn.lookup(out.w * ComplexValue{std::polar(1.0, phase)});

  // adjust reference counts
  dd.cn.incRef(out.w);
  dd.cn.decRef(oldW);
}

std::map<std::string, std::size_t> sample(const qc::QuantumComputation& qc,
                                          const VectorDD& in, Package& dd,
                                          const std::size_t shots,
                                          const std::size_t seed) {
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
      if (op->getType() == qc::OpType::SWAP && !op->isControlled()) {
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
        if (op->getType() == qc::OpType::SWAP && !op->isControlled()) {
          const auto& targets = op->getTargets();
          std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
          continue;
        }

        e = applyUnitaryOperation(*op, e, dd, permutation);
        continue;
      }

      if (op->getType() == qc::OpType::Measure) {
        const auto& measure = dynamic_cast<const qc::NonUnitaryOperation&>(*op);
        e = applyMeasurement(measure, e, dd, mt, measurements, permutation);
        continue;
      }

      if (op->getType() == qc::OpType::Reset) {
        const auto& reset = dynamic_cast<const qc::NonUnitaryOperation&>(*op);
        e = applyReset(reset, e, dd, mt, permutation);
        continue;
      }

      if (op->isClassicControlledOperation()) {
        const auto& classic =
            dynamic_cast<const qc::ClassicControlledOperation&>(*op);
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

template <const ApproximationStrategy stgy>
VectorDD simulate(const qc::QuantumComputation& qc, const VectorDD& in,
                  Package& dd, Approximation<stgy> approx) {
  qc::Permutation permutation = qc.initialLayout;
  dd::VectorDD out = in;
  for (const auto& op : qc) {
    if (isVirtuallySwappable(*op)) {
      virtualSwap(*op, permutation);
    } else {
      out = applyUnitaryOperation(*op, out, dd, permutation);

      // TODO: this applies approximation after each operation.
      if constexpr (stgy != None) {
        applyApproximation(out, approx, dd);
      }
    }
  }

  changePermutation(out, permutation, qc.outputPermutation, dd);
  out = dd.reduceGarbage(out, qc.getGarbage());

  // properly account for the global phase of the circuit
  if (fp phase = qc.getGlobalPhase(); hasGlobalPhase(phase)) {
    applyGlobalPhase(phase, out, dd);
  }

  return out;
}
template VectorDD simulate(const qc::QuantumComputation& qc, const VectorDD& in,
                           Package& dd, Approximation<None> approx);
template VectorDD simulate(const qc::QuantumComputation& qc, const VectorDD& in,
                           Package& dd, Approximation<FidelityDriven> approx);
template VectorDD simulate(const qc::QuantumComputation& qc, const VectorDD& in,
                           Package& dd, Approximation<MemoryDriven> approx);

std::map<std::string, std::size_t> sample(const qc::QuantumComputation& qc,
                                          const std::size_t shots,
                                          const std::size_t seed) {
  const auto nqubits = qc.getNqubits();
  const auto dd = std::make_unique<Package>(nqubits);
  return sample(qc, dd->makeZeroState(nqubits), *dd, shots, seed);
}
} // namespace dd
