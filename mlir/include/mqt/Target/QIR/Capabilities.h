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

#include <array>
#include <utility>

namespace mlir {
class ModuleOp;
struct ProgramCapability;

namespace qir {

/// Optional QIR module flags and their payload capability IDs.
inline constexpr std::array OPTIONAL_CAPABILITIES{
    std::pair{"dynamic_qubit_management", "qir.dynamic-qubit-management"},
    std::pair{"dynamic_result_management", "qir.dynamic-result-management"},
    std::pair{"arrays", "qir.arrays"},
    std::pair{"ir_functions", "qir.ir-functions"},
    std::pair{"multiple_return_points", "qir.multiple-return-points"},
    std::pair{"int_computations", "qir.int-computations"},
    std::pair{"float_computations", "qir.float-computations"},
    std::pair{"multiple_target_branching", "multiway-branching"},
};

/// Check emitted QIR requirements against the selected payload capabilities.
///
/// Required loop capabilities must be unconstrained because lowered QIR does
/// not retain a proof of their bounds.
LogicalResult validateCapabilities(ModuleOp moduleOp,
                                   ArrayRef<ProgramCapability> capabilities);

} // namespace qir
} // namespace mlir
