/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/WState.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <cmath>
#include <string>

namespace qc {
namespace {
void fGate(QuantumComputation& qc, const Qubit i, const Qubit j, const Qubit k,
           const Qubit n) {
  const auto theta = std::acos(std::sqrt(1.0 / static_cast<double>(k - n + 1)));
  qc.ry(-theta, j);
  qc.cz(i, j);
  qc.ry(theta, j);
}
} // namespace

auto createWState(const Qubit nq) -> QuantumComputation {
  auto qc = QuantumComputation(nq, nq);
  qc.setName("wstate_" + std::to_string(nq));

  qc.x(nq - 1);

  for (Qubit m = 1; m < nq; m++) {
    fGate(qc, nq - m, nq - m - 1, nq, m);
  }

  for (Qubit k = nq - 1; k > 0; k--) {
    qc.cx(k - 1, k);
  }

  return qc;
}
} // namespace qc
