/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <cstddef>

namespace dd {

enum ApproximationStrategy { None, FidelityDriven, MemoryDriven };

template <const ApproximationStrategy stgy> struct Approximation {};

template <> struct Approximation<None> {};

template <> struct Approximation<FidelityDriven> {
  constexpr explicit Approximation(double fidelity) noexcept
      : fidelity(fidelity) {}
  double fidelity;
};

template <>
struct Approximation<MemoryDriven> : public Approximation<FidelityDriven> {
  constexpr Approximation(std::size_t threshold, double fidelity) noexcept
      : Approximation<FidelityDriven>(fidelity), threshold(threshold) {}

  std::size_t threshold;
};

VectorDD approximate(const VectorDD& state, double fidelity, Package& dd);
} // namespace dd
