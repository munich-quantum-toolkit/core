/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

class FloatAttr;
class InFlightDiagnostic;

namespace mqt::detail {

/// Verify that an optional fidelity is finite, is f64, and lies in [0, 1].
[[nodiscard]] LogicalResult
verifyFidelity(const function_ref<InFlightDiagnostic()>& emitError,
               FloatAttr fidelity, StringRef description);

} // namespace mqt::detail

} // namespace mlir
