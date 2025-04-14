/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DesicionDiagramContainer.hpp"

namespace dd {

class DensityDDContainer : public DDContainer<dNode> {
public:
  using DDContainer<dNode>::DDContainer;

  /**
 * @brief Construct the all-zero density operator
          \f$|0...0\rangle\langle0...0|\f$
 * @param n The number of qubits
 * @return A decision diagram for the all-zero density operator
 */
  dEdge makeZeroDensityOperator(std::size_t n);

  char measureOneCollapsing(dEdge& e, Qubit index, std::mt19937_64& mt);
};

} // namespace dd
