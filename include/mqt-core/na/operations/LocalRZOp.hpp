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

#include "ir/Definitions.hpp"
#include "na/entities/Atom.hpp"
#include "na/operations/LocalOp.hpp"

#include <string>
#include <utility>
#include <vector>

namespace na {
/// Represents a local RZ operation in the NAComputation.
class LocalRZOp final : public LocalOp {
public:
  /// Creates a new RZ operation with the given atoms and angle.
  /// @param atoms The atoms the operation is applied to.
  /// @param angle The angle of the operation.
  LocalRZOp(std::vector<const Atom*> atoms, const qc::fp angle)
      : LocalOp(std::move(atoms), {angle}) {
    name_ = "rz";
  }

  /// Creates a new RZ operation with the given atom and angle.
  /// @param atom The atom the operation is applied to.
  /// @param angle The angle of the operation.
  LocalRZOp(const Atom& atom, const qc::fp angle) : LocalRZOp({&atom}, angle) {}
};
} // namespace na
