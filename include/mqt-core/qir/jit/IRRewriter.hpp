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

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>

#include <array>
#include <cstddef>

namespace qir {

/// The set of call targets that @ref stripMeasurementRelatedCalls erases.
inline constexpr std::array<llvm::StringRef, 5> STRIP_TARGETS = {
    "__quantum__qis__mz__body",
    "__quantum__qis__m__body",
    "__quantum__qis__measure__body",
    "__quantum__rt__result_record_output",
    "__quantum__rt__result_update_reference_count",
};

/**
 * @brief Strips QIR measurement-related calls from @p m in place.
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
std::size_t stripMeasurementRelatedCalls(llvm::Module& m);

} // namespace qir
