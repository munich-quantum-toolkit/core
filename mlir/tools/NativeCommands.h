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

#include "llvm/Support/Error.h"

#include <string>

namespace mlir {
class QCProgram;
class TargetEnvironment;
} // namespace mlir

/// Optional file serializer; receives compiled QC without execution
/// preparation.
using MQTQPYWriter = llvm::Expected<std::string> (*)(
    mlir::QCProgram, const mlir::TargetEnvironment*);

int mqtCoreRunCompiler(int argc, char** argv, MQTQPYWriter qpyWriter = nullptr);
int mqtCoreRunBench(int argc, char** argv);
int mqtCoreRunProgram(int argc, char** argv);
int mqtCoreRunWorker(const char* socketPath);
