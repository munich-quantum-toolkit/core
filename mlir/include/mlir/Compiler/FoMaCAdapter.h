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

#include "mlir/Compiler/Target.h"

#include <llvm/Support/Error.h>

#include <string_view>

namespace fomac {
class Device;
} // namespace fomac

namespace mlir {

/**
 * @brief Snapshot a circuit-model QDMI device as an MLIR compiler target.
 *
 * @details The returned target owns all queried metadata and remains valid
 * after the originating device and session have been destroyed. Neutral-atom
 * zone models and site-dependent operation support are not supported by the
 * circuit-model compiler pipeline.
 */
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDevice(const fomac::Device& device);

/**
 * @brief Open a registered QDMI device and snapshot it as a compiler target.
 *
 * @details This is the exception boundary for consumers that are built without
 * C++ exception support. Any exception raised while opening or querying the
 * FoMaC device is converted into an LLVM error.
 */
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDeviceId(std::string_view deviceId);

} // namespace mlir
