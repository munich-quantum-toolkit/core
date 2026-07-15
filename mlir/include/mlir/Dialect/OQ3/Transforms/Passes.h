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

#include <memory>

namespace mlir {
class Pass;
namespace oq3 {

/**
 * @brief Options controlling conversion of typed OpenQASM IR to QC.
 */
struct OpenQASMLoweringOptions {
  /// Whether the selected target can diagnose a zero step at runtime.
  bool supportsRuntimeAssertions = false;
};

/**
 * @brief Create the pass that lowers supported OQ3 operations to QC.
 * @param options Target capability options.
 * @return The newly created lowering pass.
 */
std::unique_ptr<Pass>
createLowerOQ3ToQCPass(OpenQASMLoweringOptions options = {});

} // namespace oq3
} // namespace mlir
