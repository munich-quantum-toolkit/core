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

#include <string>
#include <string_view>
#include <vector>

namespace qdmi {
class Device;
} // namespace qdmi

namespace mlir {

/**
 * @brief Snapshot a circuit-model QDMI device as an MLIR compiler target.
 *
 * @details The returned target owns all queried metadata and remains valid
 * after the originating device and session have been destroyed. Neutral-atom
 * zone models are not supported. Explicit QDMI site lists must cover every
 * site for one-qubit operations, every undirected topology edge for two-qubit
 * operations, and every ordered tuple of distinct sites for higher arities.
 * Each supported ordered placement carries optional calibration data.
 */
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDevice(const qdmi::Device& device);

/**
 * @brief Open a Client-visible QDMI device and snapshot it as a compiler
 * target.
 *
 * @details This adapter contains exceptions from the QDMI C++ API and returns
 * them as LLVM errors. The returned target owns all queried metadata.
 */
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDeviceId(std::string_view deviceId);

/**
 * @brief List the stable IDs visible to a fresh QDMI Client session.
 *
 * @details This adapter contains exceptions from QDMI Client discovery and
 * returns them as LLVM errors.
 */
[[nodiscard]] llvm::Expected<std::vector<std::string>>
registeredQDMIDeviceIds();

} // namespace mlir
