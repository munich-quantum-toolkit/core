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

#include <llvm/IR/Module.h>

#include <cstddef>

namespace qir {

/**
 * @brief Strips QIR measurement and result-management calls from @p m
 *        in place.
 * @details Erases calls to the QIR measurement intrinsics, to the
 * result-recording intrinsic, and to the result reference-count update
 * intrinsic (whose Result operands would otherwise reference the null
 * pointers left by the stripped measurements).
 * Intended for QIR Base Profile programs only: in Adaptive Profile programs,
 * measurement results feed classical control flow, so removing them silently
 * changes observable behavior.
 *
 * The typical use is to prepare a Base Profile module for state-vector
 * extraction: after this transform the JIT'd @c main can be run once and
 * the resulting state remains in the @ref qir::Runtime DD instead of being
 * collapsed by measurement.
 *
 * @param m Module to rewrite in place.
 * @return Number of instructions erased.
 */
std::size_t stripMeasurementsAndRecording(llvm::Module& m);

} // namespace qir
