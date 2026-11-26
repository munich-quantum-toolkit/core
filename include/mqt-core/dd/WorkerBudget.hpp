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

#include "llvm/Support/Threading.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <semaphore>
#include <string_view>
#include <system_error>

namespace dd {
/// Counts the DD simulation workers that run at once.
///
/// The limit is the number of hardware cores, lowered by a positive
/// `MQT_CORE_DD_WORKER_BUDGET`.
class WorkerBudget {
public:
  WorkerBudget()
      : limit_(readLimit()), slots_(static_cast<ptrdiff_t>(limit_)) {}

  [[nodiscard]] size_t limit() const { return limit_; }

  /// Wait for a first worker unless some are `held`, then add idle workers up
  /// to `requested`. Return the number of workers now held.
  [[nodiscard]] size_t reserve(size_t requested, size_t held = 0) {
    if (held == 0) {
      slots_.acquire();
      held = 1;
    }
    while (held < requested && slots_.try_acquire()) {
      ++held;
    }
    return held;
  }

  /// Return `count` held workers to the budget.
  void release(size_t count) { slots_.release(static_cast<ptrdiff_t>(count)); }

private:
  [[nodiscard]] static size_t readLimit() {
    const size_t hardware = std::max(
        1U, llvm::heavyweight_hardware_concurrency().compute_thread_count());
    if (const char* raw = std::getenv("MQT_CORE_DD_WORKER_BUDGET")) {
      const std::string_view value(raw);
      size_t parsed = 0;
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), parsed);
      if (error == std::errc{} && end == value.data() + value.size() &&
          parsed > 0) {
        return std::min(parsed, hardware);
      }
    }
    return hardware;
  }

  size_t limit_;
  std::counting_semaphore<> slots_;
};
} // namespace dd
