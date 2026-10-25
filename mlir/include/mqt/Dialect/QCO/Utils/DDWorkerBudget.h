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

#include <chrono>
#include <cstddef>
#include <stop_token>

namespace mlir::qco {

/// Acquire a shared DD worker grant; zero means cancellation or deadline.
///
/// Wait for at least one slot, then take currently available slots up to the
/// request. Never wait for additional slots while holding a partial grant.
/// The caller must release a successful grant, including on exceptions.
[[nodiscard]] size_t
acquireDDWorkers(size_t requested, const std::stop_token& stop = {},
                 std::chrono::steady_clock::time_point deadline =
                     std::chrono::steady_clock::time_point::max());

/// Return a grant acquired from the common DD worker budget.
void releaseDDWorkers(size_t count);

} // namespace mlir::qco
