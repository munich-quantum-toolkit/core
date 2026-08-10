/*
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompat.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>

#define GET_OP_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/CCCompatOps.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOps.h.inc" // IWYU pragma: export
