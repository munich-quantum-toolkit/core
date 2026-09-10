/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file IRRewriter.hpp
/// QIR JIT IR-rewriting utilities.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
class Function;
}

namespace qir {

/// Prepares a QIR entry point for state extraction.
///
/// Truncates @p entryPoint immediately before its first call to a
/// function carrying the QIR @c irreversible attribute, then removes the
/// unreachable measurement and output region. This uses the semantic boundary
/// defined by the QIR Base Profile instead of relying on a fixed list of
/// measurement and output function names.
///
/// Base Profile entry points use this terminal-region transform. Adaptive
/// Profile entry points are validated for runtime measurement deferral instead:
/// direct helpers and classical control flow are supported, but resets,
/// measurement-dependent computation and unknown external effects are rejected.
/// Reads used only by boolean output records are allowed. The runtime must
/// reject operations on measured wires when executing a validated Adaptive
/// entry.
///
/// @param entryPoint QIR entry point to rewrite in place.
/// @return Whether an irreversible boundary was found and truncated.
/// @throws std::invalid_argument for an unsupported profile, signature or
/// effects. Base Profile extraction additionally rejects non-terminal
/// irreversible regions and defined helpers; neither profile supports indirect
/// calls.
bool prepareForStateExtraction(llvm::Function& entryPoint);

/// Return logical qubit IDs in recorded-result order when measurements can be
/// deferred for sampling. Only an acyclic unconditional Base or Adaptive path
/// with constant gate arguments and scalar result records is supported. Unknown
/// calls, result-dependent computation, resets and memory accesses return
/// std::nullopt, leaving ordinary per-shot execution available.
std::optional<std::vector<uintptr_t>>
getStaticSamplingOutputs(const llvm::Function& entryPoint);

} // namespace qir
