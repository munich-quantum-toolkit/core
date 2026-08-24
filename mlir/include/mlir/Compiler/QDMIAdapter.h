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
#include "mlir/Compiler/TargetEnvironment.h"

#include <llvm/Support/Error.h>
#include <qdmi/constants.h>

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
 * @brief Open a registered QDMI device and snapshot it as a compiler target.
 *
 * @details This adapter contains exceptions from the QDMI C++ API and returns
 * them as LLVM errors. The returned target owns all queried metadata.
 */
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDeviceId(std::string_view deviceId);

/**
 * @brief Snapshot a QDMI device and one accepted payload as a target
 * environment.
 *
 * @details The adapter preserves the exact program format, groups feature
 * records with the same ID and value, and adds the normative baseline of a
 * standard payload. Unknown optional feature metadata remains unknown.
 */
[[nodiscard]] llvm::Expected<TargetEnvironment>
targetEnvironmentFromDevice(const qdmi::Device& device,
                            const QDMI_Program_Format& format);

/**
 * @brief Open a registered QDMI device and snapshot one accepted payload.
 *
 * @details This adapter contains exceptions from the QDMI C++ API and returns
 * them as LLVM errors. The returned environment owns all queried metadata.
 */
[[nodiscard]] llvm::Expected<TargetEnvironment>
targetEnvironmentFromDeviceId(std::string_view deviceId,
                              const QDMI_Program_Format& format);

/**
 * @brief List the stable IDs of registered QDMI devices.
 *
 * @details This adapter contains exceptions from QDMI registry discovery and
 * returns them as LLVM errors.
 */
[[nodiscard]] llvm::Expected<std::vector<std::string>>
registeredQDMIDeviceIds();

} // namespace mlir
