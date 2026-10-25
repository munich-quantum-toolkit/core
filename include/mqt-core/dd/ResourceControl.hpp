/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file ResourceControl.hpp
/// Cooperative stopping and allocation limits for decision-diagram packages.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <stop_token>

namespace dd {

/// Why a controlled DD computation stopped without a result.
enum class ResourceLimit : uint8_t { Cancelled, Deadline, Memory };

/// Stop unwinding a resource-limited DD computation.
///
/// Destroy the affected package after catching this exception. Interrupted
/// arithmetic may have incomplete temporary roots or compute-table entries.
class ResourceLimitExceeded : public std::runtime_error {
public:
  explicit ResourceLimitExceeded(ResourceLimit reason);
  [[nodiscard]] ResourceLimit reason() const noexcept { return reason_; }

private:
  ResourceLimit reason_;
};

/// Share a byte budget and cancellation request across DD packages.
///
/// Counts bytes requested from this memory resource, including simultaneously
/// live replacement storage. This excludes allocator overhead, stack storage,
/// and containers not using this resource; it is not a process RSS limit.
/// The control must outlive all packages and containers that use it.
class ResourceControl final : public std::pmr::memory_resource {
public:
  struct Options {
    std::stop_token stop;
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max();
    size_t maxBytes = std::numeric_limits<size_t>::max();
  };

  explicit ResourceControl(Options options);

  /// Throw when cancellation or the deadline has been reached.
  void checkpoint() const;
  [[nodiscard]] size_t allocatedBytes() const noexcept {
    return allocated_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] size_t peakBytes() const noexcept {
    return peak_.load(std::memory_order_relaxed);
  }

private:
  void* do_allocate(size_t bytes, size_t alignment) override;
  void do_deallocate(void* pointer, size_t bytes, size_t alignment) override;
  [[nodiscard]] bool
  do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  Options options_;
  std::atomic<size_t> allocated_{0};
  std::atomic<size_t> peak_{0};
};

} // namespace dd
