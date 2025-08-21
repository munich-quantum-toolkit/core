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

#include <mlir/Bytecode/BytecodeOpInterface.h>

// Suppress warnings about ambiguous reversed operators in MLIR
// (see https://github.com/llvm/llvm-project/issues/45853)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#endif
#include "mlir/Interfaces/InferTypeOpInterface.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#define DIALECT_NAME_MQTOPT "mqtopt"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Common/IR/CommonTraits.h"         // IWYU pragma: export
#include "mlir/Dialect/MQTOpt/IR/MQTOptInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.h.inc" // IWYU pragma: export

namespace mqt::ir::opt {
mlir::ParseResult
parseOptOutputTypes(mlir::OpAsmParser& parser,
                    llvm::SmallVectorImpl<mlir::Type>& out_qubits,
                    llvm::SmallVectorImpl<mlir::Type>& pos_ctrl_out_qubits,
                    llvm::SmallVectorImpl<mlir::Type>& neg_ctrl_out_qubits);

void printOptOutputTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op,
                         mlir::TypeRange out_qubits,
                         mlir::TypeRange pos_ctrl_out_qubits,
                         mlir::TypeRange neg_ctrl_out_qubits);
} // namespace mqt::ir::opt
