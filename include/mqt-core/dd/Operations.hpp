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
 * @brief Arithmetic operations on decision diagrams.
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"

namespace dd {

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
