/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DensityDDContainer.hpp"

#include "dd/GateMatrixDefinitions.hpp"

namespace dd {

dEdge DensityDDContainer::makeZeroDensityOperator(const std::size_t n) {
  auto f = dEdge::one();
  for (std::size_t p = 0; p < n; p++) {
    f = makeDDNode(static_cast<Qubit>(p),
                   std::array{f, dEdge::zero(), dEdge::zero(), dEdge::zero()});
  }
  incRef(f);
  return f;
}

} // namespace dd
