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

#include "mlir/Dialect/Utils/ParameterDescriptor.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <string>

// Suppress warnings about ambiguous reversed operators in MLIR
// (see https://github.com/llvm/llvm-project/issues/45853)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#endif
#include <mlir/Interfaces/InferTypeOpInterface.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#define DIALECT_NAME_FLUX "flux"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Flux/IR/FluxOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::flux {

template <size_t n> class TargetArityTrait {
public:
  template <typename ConcreteType>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    size_t getNumQubits() { return n; }
    size_t getNumTargets() { return n; }
    size_t getNumControls() { return 0; }
    size_t getNumPosControls() { return 0; }
    size_t getNumNegControls() { return 0; }

    Value getPosControl(size_t i) {
      llvm_unreachable("Operation does not have controls");
    }

    Value getNegControl(size_t i) {
      llvm_unreachable("Operation does not have controls");
    }
  };
};

template <size_t n> class ParameterArityTrait {
public:
  template <typename ConcreteType>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    size_t getNumParams() { return n; }
  };
};

} // namespace mlir::flux

#include "mlir/Dialect/Flux/IR/FluxInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOps.h.inc" // IWYU pragma: export
