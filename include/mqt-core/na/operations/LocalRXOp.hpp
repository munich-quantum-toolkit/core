/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local RZ operations.
 */

#pragma once

#include "Definitions.hpp"
#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <vector>

namespace na {
/// Represents a local RZ operation in the NAComputation.
class LocalRXOp final : public LocalOp {
public:
  /// Creates a new RZ operation with the given atoms and angle.
  /// @param atom The atoms the operation is applied to.
  /// @param angle The angle of the operation.
  LocalRXOp(const std::vector<const Atom*>& atom, const qc::fp angle)
      : LocalOp(atom, {angle}) {
    name_ = "rx";
  }

  /// Creates a new RZ operation with the given atom and angle.
  /// @param atom The atom the operation is applied to.
  /// @param angle The angle of the operation.
  LocalRXOp(const Atom& atom, const qc::fp angle) : LocalRXOp({&atom}, angle) {}
};
} // namespace na
