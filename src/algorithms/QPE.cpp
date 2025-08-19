/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/QPE.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/OpType.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qc {

namespace {
// generate a random n-bit number and convert it to an appropriate phase
[[nodiscard]] auto createExactPhase(const Qubit nq, std::mt19937_64& mt) -> fp {
  const std::uint64_t max = 1ULL << nq;
  auto distribution = std::uniform_int_distribution<std::uint64_t>(0, max - 1);
  std::uint64_t theta = 0;
  while (theta == 0) {
    theta = distribution(mt);
  }
  fp lambda = 0.;
  for (std::size_t i = 0; i < nq; ++i) {
    if ((theta & (1ULL << (nq - i - 1))) != 0) {
      lambda += 1. / static_cast<double>(1ULL << i);
    }
  }
  return lambda;
}

// generate a random n+1-bit number (that has its last bit set) and convert it
// to an appropriate phase
[[nodiscard]] auto createInexactPhase(const Qubit nq, std::mt19937_64& mt)
    -> fp {
  const std::uint64_t max = 1ULL << (nq + 1);
  auto distribution = std::uniform_int_distribution<std::uint64_t>(0, max - 1);
  std::uint64_t theta = 0;
  while ((theta & 1) == 0) {
    theta = distribution(mt);
  }
  fp lambda = 0.;
  for (std::size_t i = 0; i <= nq; ++i) {
    if ((theta & (1ULL << (nq - i))) != 0) {
      lambda += 1. / static_cast<double>(1ULL << i);
    }
  }
  return lambda;
}

[[nodiscard]] auto getName(const bool iterative, const Qubit nq,
                           const fp lambda) -> std::string {
  std::stringstream ss;
  ss << (iterative ? "iterative_" : "") << "qpe_";
  ss << nq << "_";
  ss.precision(std::numeric_limits<fp>::digits10);
  ss << lambda;
  return ss.str();
}

auto constructQPECircuit(QuantumComputation& qc, const fp lambda,
                         const Qubit nq) -> void {
  qc.setName(getName(false, nq, lambda));
  qc.addQubitRegister(1, "psi");
  qc.addQubitRegister(nq, "q");
  qc.addClassicalRegister(nq, "c");
  // store lambda in global phase
  qc.gphase(lambda);

  // prepare eigenvalue
  qc.x(0);

  // set up initial layout
  qc.initialLayout[0] = nq;
  for (Qubit i = 1; i <= nq; ++i) {
    qc.initialLayout[i] = i - 1;
  }
  qc.setLogicalQubitGarbage(nq);
  qc.outputPermutation.erase(0);

  // Hadamard Layer
  for (Qubit i = 1; i <= nq; ++i) {
    qc.h(i);
  }

  for (Qubit i = 0; i < nq; ++i) {
    // normalize angle
    const auto angle =
        std::remainder(static_cast<double>(1ULL << (nq - 1 - i)) * lambda, 2.0);

    // controlled phase rotation
    qc.cp(angle * PI, 1 + i, 0);

    // inverse QFT
    for (Qubit j = 1; j < 1 + i; j++) {
      const auto iQFTLambda = -PI / static_cast<double>(2ULL << (i - j));
      if (j == i) {
        qc.csdg(i, 1 + i);
      } else if (j == (i - 1)) {
        qc.ctdg(i - 1, 1 + i);
      } else {
        qc.cp(iQFTLambda, j, 1 + i);
      }
    }
    qc.h(1 + i);
  }

  // measure results
  for (Qubit i = 0; i < nq; i++) {
    qc.measure(i + 1, i);
    qc.outputPermutation[i + 1] = i;
  }
}
} // namespace

auto createQPE(const Qubit nq, const bool exact, const std::size_t seed)
    -> QuantumComputation {
  auto qc = QuantumComputation(0, 0, seed);
  const auto lambda = exact ? createExactPhase(nq, qc.getGenerator())
                            : createInexactPhase(nq, qc.getGenerator());
  constructQPECircuit(qc, lambda, nq);
  return qc;
}

auto createQPE(const fp lambda, const Qubit precision) -> QuantumComputation {
  auto qc = QuantumComputation();

  constructQPECircuit(qc, lambda, precision);
  return qc;
}

namespace {
auto constructIterativeQPECircuit(QuantumComputation& qc, const fp lambda,
                                  const Qubit nq) -> void {
  qc.setName(getName(true, nq, lambda));
  qc.addQubitRegister(1, "psi");
  qc.addQubitRegister(1, "q");
  qc.addClassicalRegister(nq, "c");
  // store lambda in global phase
  qc.gphase(lambda);

  // prepare eigenvalue
  qc.x(0);

  // set up initial layout
  qc.initialLayout[0] = 1;
  qc.initialLayout[1] = 0;
  qc.setLogicalQubitGarbage(1);
  qc.outputPermutation.erase(0);
  qc.outputPermutation[1] = 0;

  for (Qubit i = 0; i < nq; i++) {
    // Hadamard
    qc.h(1);

    // normalize angle
    const auto angle =
        std::remainder(static_cast<double>(1ULL << (nq - 1 - i)) * lambda, 2.0);

    // controlled phase rotation
    qc.cp(angle * PI, 1, 0);

    // hybrid quantum-classical inverse QFT
    for (std::size_t j = 0; j < i; j++) {
      auto iQFTLambda = -PI / static_cast<double>(1ULL << (i - j));
      const auto params = std::vector<fp>{iQFTLambda};
      qc.if_(P, 1, j, true, Eq, params);
    }
    qc.h(1);

    // measure result
    qc.measure(1, i);

    // reset qubit if not finished
    if (i < nq - 1) {
      qc.reset(1);
    }
  }
}
} // namespace

auto createIterativeQPE(const Qubit nq, const bool exact,
                        const std::size_t seed) -> QuantumComputation {
  auto qc = QuantumComputation(0, 0, seed);
  const auto lambda = exact ? createExactPhase(nq, qc.getGenerator())
                            : createInexactPhase(nq, qc.getGenerator());
  constructIterativeQPECircuit(qc, lambda, nq);
  return qc;
}

auto createIterativeQPE(const fp lambda, const Qubit precision)
    -> QuantumComputation {
  auto qc = QuantumComputation();
  constructIterativeQPECircuit(qc, lambda, precision);
  return qc;
}
} // namespace qc
