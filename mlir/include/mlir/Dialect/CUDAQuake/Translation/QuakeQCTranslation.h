/*
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>

#include <string>

namespace mlir::cudaq_compat {

struct QuakeExportOptions {
  std::string entryPointName = "mqt_kernel";
  bool ignoreGlobalPhase = false;
};

/// Translate a specialized CUDA-Q reference-form Quake module to QC.
[[nodiscard]] FailureOr<OwningOpRef<ModuleOp>>
translateQuakeToQC(ModuleOp input);

/// Translate a QC module to conservative CUDA-Q reference-form Quake.
[[nodiscard]] FailureOr<OwningOpRef<ModuleOp>>
translateQCToQuake(ModuleOp input, const QuakeExportOptions& options = {});

} // namespace mlir::cudaq_compat
