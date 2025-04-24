/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing a barrier in the NAComputation.
 * @details The barrier is mainly used to represent barriers stemming from the
 * input circuit preventing gates to commute.
 */

#pragma once

#include "na/operations/Op.hpp"

namespace na {
/// @brief Represents a barrier in the NAComputation.
/// @details A barrier always spans all atoms.
class Barrier : public Op {
public:
  /// Creates a barrier.
  Barrier() {}

  /// Returns a string representation of the operation.
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
