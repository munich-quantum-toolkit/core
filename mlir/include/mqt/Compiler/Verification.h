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

#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <map>
#include <stop_token>

namespace mlir {
class QCOProgram;

/// Concrete attributes indexed by an entry function's argument position.
using EntryArgumentBindings = std::map<size_t, Attribute>;

/// Verify borrowed QCO programs at their marked entry functions.
///
/// Neither program is copied, compiled, or consumed. Both owners must remain
/// valid and must not be mutated during the call. A successful result concerns
/// only these concrete classical bindings and the supplied quantum contract.
/// Invalid owners, missing entries, and invalid binding indices return failure;
/// unsupported semantics and unfinished checks have explicit result verdicts.
[[nodiscard]] FailureOr<qco::VerificationResult>
verify(const QCOProgram& left, const QCOProgram& right,
       const qco::VerificationContract& contract = {},
       const EntryArgumentBindings& leftBindings = {},
       const EntryArgumentBindings& rightBindings = {},
       const qco::VerificationOptions& options = {},
       const std::stop_token& stop = {});
} // namespace mlir
