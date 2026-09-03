/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Operations.hpp
 * @brief Arithmetic, tensor, and measurement operations on decision diagrams.
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"

#include <vector>

namespace dd {

/**
 * @brief Get the decision diagram representation of an operation based on its
 * constituent parts.
 *
 * @note This function is only intended for internal use and should not be
 * called directly.
 *
 * @param dd The DD package to use
 * @param type The operation type
 * @param params The operation parameters
 * @param controls The operation controls
 * @param targets The operation targets
 * @return The decision diagram representation of the operation
 */
MatrixDD getGateDD(Package& dd, GateType type, const std::vector<fp>& params,
                   const Controls& controls, const Targets& targets);

/**
 * @brief Apply global phase to a given DD.
 *
 * @param in The input DD
 * @param phase The phase to apply
 * @param dd The DD package to use
 * @return The output DD
 */
VectorDD applyGlobalPhase(VectorDD& in, const fp& phase, Package& dd);

} // namespace dd
