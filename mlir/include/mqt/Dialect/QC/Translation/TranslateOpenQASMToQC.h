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

#include "mqt/Target/OpenQASM/Frontend.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/SourceMgr.h"

#include <cstddef>

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;

namespace qc {

/// Controls source acceptance and the size of the emitted QC program.
struct OpenQASMImportOptions {
  openqasm::frontend::GatePolicy gatePolicy =
      openqasm::frontend::GatePolicy::MQTCompatibility;
  /// Maximum number of inserted operations, excluding the module itself.
  size_t maxOperations = 10'000'000;
};

/// Translate supported OpenQASM to QC.
///
/// Accepts versionless input and versions 2.0, 3.0, and 3.1.
///
/// Frontend and emission failures are reported through the diagnostic engine of
/// @p context and result in a null return value.
///
/// @param sourceMgr Source manager containing the OpenQASM program.
/// @param context MLIRContext to create the module in.
/// @param options Frontend policy and emission resource limit.
[[nodiscard]] OwningOpRef<ModuleOp>
translateOpenQASMToQC(llvm::SourceMgr& sourceMgr, MLIRContext* context,
                      const OpenQASMImportOptions& options = {});

/// Translate supported OpenQASM to QC.
///
/// Accepts versionless input and versions 2.0, 3.0, and 3.1.
///
/// Frontend and emission failures are reported through the diagnostic engine of
/// @p context and result in a null return value.
///
/// @param source String containing the OpenQASM program.
/// @param context MLIRContext to create the module in.
/// @param options Frontend policy and emission resource limit.
[[nodiscard]] OwningOpRef<ModuleOp>
translateOpenQASMToQC(StringRef source, MLIRContext* context,
                      const OpenQASMImportOptions& options = {});

} // namespace qc

} // namespace mlir
