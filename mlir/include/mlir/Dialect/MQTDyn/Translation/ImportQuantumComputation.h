/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef IMPORT_QUANTUM_COMPUTATION
#define IMPORT_QUANTUM_COMPUTATION

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace qc {
class QuantumComputation;
}

mlir::OwningOpRef<mlir::ModuleOp>
translateQuantumComputationToMLIR(mlir::MLIRContext& context,
                                  qc::QuantumComputation& qc);

#endif // IMPORT_QUANTUM_COMPUTATION
