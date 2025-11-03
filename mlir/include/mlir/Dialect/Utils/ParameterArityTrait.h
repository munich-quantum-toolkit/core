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

#include <cstddef>
#include <mlir/IR/OpDefinition.h>

namespace mlir::utils {

template <size_t n> class ParameterArityTrait {
public:
  template <typename ConcreteType>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    size_t getNumParams() { return n; }
  };
};

} // namespace mlir::utils
