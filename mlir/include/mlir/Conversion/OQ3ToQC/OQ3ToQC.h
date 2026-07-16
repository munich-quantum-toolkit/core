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
 * @brief Create the pass that converts supported OQ3 operations to QC.
 * @return The newly created lowering pass.
 */
std::unique_ptr<Pass> createOQ3ToQCPass();

} // namespace oq3
} // namespace mlir
