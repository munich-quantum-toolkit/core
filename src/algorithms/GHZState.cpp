/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/GHZState.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <string>

namespace qc {
auto createGHZState(const Qubit nq) -> QuantumComputation {
  auto qc = QuantumComputation(nq, nq);
  qc.setName("ghz_" + std::to_string(nq));

  const auto top = nq - 1;
  qc.h(top);
  for (Qubit i = 0; i < top; i++) {
    qc.cx(top, i);
  }
  return qc;
}
} // namespace qc
