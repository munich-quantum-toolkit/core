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

#include "mqt/Dialect/QIR/Utils/QIRProfile.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class TargetEnvironment;

#define GEN_PASS_DECL_QCTOQIR
#include "mqt/Conversion/QCToQIR/QCToQIR.h.inc"

/// Lower with an existing environment without materializing its target in IR.
std::unique_ptr<Pass> createQCToQIR(const QCToQIROptions& options,
                                    const TargetEnvironment& environment);

#define GEN_PASS_REGISTRATION
#include "mqt/Conversion/QCToQIR/QCToQIR.h.inc"
} // namespace mlir
