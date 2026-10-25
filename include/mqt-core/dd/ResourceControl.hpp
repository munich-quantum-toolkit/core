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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <stop_token>

namespace dd {

/// Why a controlled DD computation stopped without a result.
enum class ResourceLimit : uint8_t { Cancelled, Memory };

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
/// Counts bytes requested through this control, including simultaneously
/// live replacement storage. This excludes allocator overhead, stack storage,
/// and containers not using this control; it is not a process RSS limit.
/// The control must outlive all packages and containers that use it.
class ResourceControl final {
public:
  struct Options {
    std::stop_token stop;
    size_t maxBytes = std::numeric_limits<size_t>::max();
  };

  explicit ResourceControl(Options options);

  /// Throw when cancellation has been requested.
  void checkpoint() const;
  [[nodiscard]] size_t allocatedBytes() const noexcept {
    return allocated_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] size_t peakBytes() const noexcept {
    return peak_.load(std::memory_order_relaxed);
  }

  /// Allocate aligned storage, checking and reserving the byte budget first.
  [[nodiscard]] void* allocate(size_t bytes,
                               size_t alignment = alignof(std::max_align_t));
  /// Release storage with its original size and alignment, even after stopping.
  void deallocate(void* pointer, size_t bytes,
                  size_t alignment = alignof(std::max_align_t)) noexcept;

private:
  Options options_;
  std::atomic<size_t> allocated_{0};
  std::atomic<size_t> peak_{0};
};

namespace detail {
// Keep DD storage compatible with deployment targets without std::pmr.
template <class T> class ResourceAllocator {
public:
  using value_type = T;

  ResourceAllocator() = default;
  explicit ResourceAllocator(ResourceControl* control) : control_(control) {}
  template <class U>
  ResourceAllocator(const ResourceAllocator<U>& other)
      : control_(other.control()) {}

  [[nodiscard]] ResourceControl* control() const noexcept { return control_; }
  [[nodiscard]] T* allocate(const size_t count) {
    if (control_ == nullptr) {
      return std::allocator<T>{}.allocate(count);
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(control_->allocate(count * sizeof(T), alignof(T)));
  }
  void deallocate(T* pointer, const size_t count) noexcept {
    if (control_ == nullptr) {
      std::allocator<T>{}.deallocate(pointer, count);
    } else {
      control_->deallocate(pointer, count * sizeof(T), alignof(T));
    }
  }
  template <class U>
  bool operator==(const ResourceAllocator<U>& other) const noexcept {
    return control_ == other.control();
  }

private:
  ResourceControl* control_ = nullptr;
};
} // namespace detail
} // namespace dd
