/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MLIR_DIALECT_FLUX_IR_FLUXOPS_H
#define MLIR_DIALECT_FLUX_IR_FLUXOPS_H

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Flux/IR/FluxTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOps.h.inc"

#endif // MLIR_DIALECT_FLUX_IR_FLUXOPS_H
