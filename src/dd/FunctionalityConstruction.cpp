/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/FunctionalityConstruction.hpp"

#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"

#include <utility>

namespace dd {
MatrixDD buildFunctionality(const qc::QuantumComputation& qc, Package& dd) {
  if (qc.getNqubits() == 0U) {
    return MatrixDD::one();
  }

  auto permutation = qc.initialLayout;
  auto e = dd.createInitialMatrix(qc.getAncillary());

  for (const auto& op : qc) {
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
  e = dd.reduceAncillae(e, qc.getAncillary());
  e = dd.reduceGarbage(e, qc.getGarbage());

  return e;
}

} // namespace dd
