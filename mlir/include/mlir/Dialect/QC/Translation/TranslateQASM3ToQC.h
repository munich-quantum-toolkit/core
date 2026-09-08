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

#include "mlir/Target/OpenQASM/Frontend.h"

#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;

namespace qc {

/// Controls source acceptance and the size of the emitted QC program.
struct QASM3ImportOptions {
  oq3::frontend::GatePolicy gatePolicy =
      oq3::frontend::GatePolicy::MQTCompatibility;
  /// Maximum number of inserted operations, excluding the module itself.
  size_t maxOperations = 10'000'000;
};

/// @brief Translate an OpenQASM 3 program to a QC program.
///
/// Frontend and lowering failures are reported through the diagnostic engine of
/// @p context and result in a null return value.
///
/// @param sourceMgr Source manager containing the OpenQASM program.
/// @param context MLIRContext to create the module in.
/// @param options Frontend policy and emission resource limit.
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(llvm::SourceMgr& sourceMgr, MLIRContext* context,
                   const QASM3ImportOptions& options = {});

/// @brief Translate an OpenQASM 3 program to a QC program.
///
/// Frontend and lowering failures are reported through the diagnostic engine of
/// @p context and result in a null return value.
///
/// @param source String containing the OpenQASM program.
/// @param context MLIRContext to create the module in.
/// @param options Frontend policy and emission resource limit.
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(StringRef source, MLIRContext* context,
                   const QASM3ImportOptions& options = {});

} // namespace qc

} // namespace mlir
