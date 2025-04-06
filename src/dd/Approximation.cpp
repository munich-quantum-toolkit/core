/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Approximation.hpp"

#include <algorithm>

template <>
void dd::applyApproximation<dd::FidelityDriven>(
    VectorDD& v, Approximation<FidelityDriven>& approx) {
  NodeContributions::Vector contributions = NodeContributions{}(v);

  const auto it =
      std::find_if(contributions.begin(), contributions.end(),
                   [f = approx.fidelity]([[maybe_unused]] const auto& p) {
                     return f < (1. - p.contribution);
                   });

  if (it != contributions.end()) {
    std::cout << it->contribution << '\n';
  }
};

template <>
void dd::applyApproximation<dd::MemoryDriven>(
    VectorDD& v, Approximation<MemoryDriven>& approx) {
  Approximation<FidelityDriven> approxFidelity(approx.fidelity);
  if (v.size() > approx.maxNodes) {
    applyApproximation(v, approxFidelity);
    approx.increaseMaxNodes();
  }
};
