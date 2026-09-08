/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file IRRewriter.hpp
 * @brief QIR JIT IR-rewriting utilities.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
class Function;
}

namespace qir {

/**
 * @brief Prepares a QIR entry point for state extraction.
 * @details Truncates @p entryPoint immediately before its first call to a
 * function carrying the QIR @c irreversible attribute, then removes the
 * unreachable measurement and output region. This uses the semantic boundary
 * defined by the QIR Base Profile instead of relying on a fixed list of
 * measurement and output function names.
 *
 * The transform requires an entry point marked with @c base_profile. Adaptive
 * Profile measurements may feed classical control flow and cannot be removed
 * without changing the program's meaning.
 *
 * @param entryPoint QIR entry point to rewrite in place.
 * @return Whether an irreversible boundary was found and truncated.
 * @throws std::invalid_argument if the entry point is not Base Profile, does
 * not use the QIR 2.x @c i64() signature, or has non-terminal irreversible
 * operations or calls to defined or indirect callees.
 */
bool prepareForStateExtraction(llvm::Function& entryPoint);

/// Return logical qubit IDs in recorded-result order when measurements can be
/// deferred for sampling. Only an acyclic unconditional Base or Adaptive path
/// with constant gate arguments and scalar result records is supported. Unknown
/// calls, result-dependent computation, resets and memory accesses return
/// std::nullopt, leaving ordinary per-shot execution available.
std::optional<std::vector<uintptr_t>>
getStaticSamplingOutputs(const llvm::Function& entryPoint);

} // namespace qir
