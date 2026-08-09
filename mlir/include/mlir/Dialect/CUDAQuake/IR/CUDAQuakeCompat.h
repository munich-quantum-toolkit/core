/*
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

#include "mlir/Dialect/CUDAQuake/IR/CCCompatOpsDialect.h.inc" // IWYU pragma: export
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOpsDialect.h.inc" // IWYU pragma: export

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/CCCompatOpsTypes.h.inc" // IWYU pragma: export

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOpsTypes.h.inc" // IWYU pragma: export
