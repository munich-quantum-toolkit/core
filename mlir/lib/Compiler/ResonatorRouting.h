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

#include "mqt/Compiler/Target.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <memory>

namespace mlir {

class Pass;

/// Visit virtual CZ routes through sites excluded from program-qubit placement.
///
/// The callback receives the moving qubit, stationary qubit, resonator, and
/// whether the native CZ lists the resonator first.
void forEachVirtualCZ(
    const CompilerTarget& target, ArrayRef<CompilerTarget::SiteId> sites,
    function_ref<void(CompilerTarget::SiteId, CompilerTarget::SiteId,
                      CompilerTarget::SiteId, bool)>
        visit);

/// Lower static virtual QC CZs using the attached execution target.
///
/// Insert closed MOVE/CZ/MOVE sequences and schedule independent terminal
/// readout after them. Without a target, leave the module unchanged. Diagnose
/// invalid target environments. Run after QCO-to-QC conversion.
/// Resonators must be idle at each virtual CZ; native MOVE sequences must not
/// leave a resonator occupied across a virtual CZ. The pass does not check this
/// precondition.
[[nodiscard]] std::unique_ptr<Pass> createLowerResonatorCZPass();

} // namespace mlir
