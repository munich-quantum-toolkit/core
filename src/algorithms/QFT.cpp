/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/QFT.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/OpType.hpp"

#include <cmath>
#include <string>

namespace qc {

auto createQFT(const Qubit nq, const bool includeMeasurements)
    -> QuantumComputation {
  auto qc = QuantumComputation(nq, nq);
  qc.setName("qft_" + std::to_string(nq));
  if (nq == 0) {
    return qc;
  }

  // apply quantum Fourier transform
  for (Qubit i = 0; i < nq; ++i) {
    // apply controlled rotations
    for (Qubit j = i; j > 0; --j) {
      const auto d = i - j;
      if (j == 1) {
        qc.cs(i, d);
      } else if (j == 2) {
        qc.ct(i, d);
      } else {
        const auto powerOfTwo = std::pow(2., j);
        const auto lambda = PI / powerOfTwo;
        qc.cp(lambda, i, d);
      }
    }

    // apply Hadamard
    qc.h(i);
  }

  if (includeMeasurements) {
    // measure qubits in reverse order
    for (Qubit i = 0; i < nq; ++i) {
      qc.measure(i, nq - 1 - i);
      qc.outputPermutation[i] = nq - 1 - i;
    }
  } else {
    for (Qubit i = 0; i < nq / 2; ++i) {
      qc.swap(i, nq - 1 - i);
    }
    for (Qubit i = 0; i < nq; ++i) {
      qc.outputPermutation[i] = nq - 1 - i;
    }
  }

  return qc;
}

auto createIterativeQFT(const Qubit nq) -> QuantumComputation {
  auto qc = QuantumComputation(0, nq);
  qc.setName("iterative_qft_" + std::to_string(nq));
  if (nq == 0) {
    return qc;
  }
  qc.addQubitRegister(1U);

  for (Qubit i = 0; i < nq; ++i) {
    // apply classically controlled phase rotations
    for (Qubit j = 1; j <= i; ++j) {
      const auto d = nq - j;
      if (j == i) {
        qc.classicControlled(S, 0, d, 1U);
      } else if (j == i - 1) {
        qc.classicControlled(T, 0, d, 1U);
      } else {
        const auto powerOfTwo = std::pow(2., i - j + 1);
        const auto lambda = PI / powerOfTwo;
        qc.classicControlled(P, 0, d, 1U, Eq, {lambda});
      }
    }

    // apply Hadamard
    qc.h(0);

    // measure result
    qc.measure(0, nq - 1 - i);

    // reset qubit if not finished
    if (i < nq - 1) {
      qc.reset(0);
    }
  }

  return qc;
}
} // namespace qc
