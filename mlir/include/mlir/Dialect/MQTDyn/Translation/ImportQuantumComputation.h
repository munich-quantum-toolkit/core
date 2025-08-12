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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace qc {
class QuantumComputation;
}

mlir::OwningOpRef<mlir::ModuleOp>
translateQuantumComputationToMLIR(mlir::MLIRContext& context,
                                  qc::QuantumComputation& qc);
