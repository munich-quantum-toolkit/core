/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ResourceControl.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

namespace dd {
namespace {
const char* resourceLimitMessage(const ResourceLimit reason) {
  switch (reason) {
  case ResourceLimit::Cancelled:
    return "DD computation cancelled.";
  case ResourceLimit::Deadline:
    return "DD computation deadline exceeded.";
  case ResourceLimit::Memory:
    return "DD allocation budget exceeded.";
  }
  return "DD resource limit exceeded.";
}
} // namespace

ResourceLimitExceeded::ResourceLimitExceeded(const ResourceLimit reason)
    : std::runtime_error(resourceLimitMessage(reason)), reason_(reason) {}

ResourceControl::ResourceControl(Options options)
    : options_(std::move(options)) {}

void ResourceControl::checkpoint() const {
  if (options_.stop.stop_requested()) {
    throw ResourceLimitExceeded(ResourceLimit::Cancelled);
  }
  if (options_.deadline != std::chrono::steady_clock::time_point::max() &&
      std::chrono::steady_clock::now() >= options_.deadline) {
    throw ResourceLimitExceeded(ResourceLimit::Deadline);
  }
}

void* ResourceControl::allocate(const size_t bytes, const size_t alignment) {
  checkpoint();
  auto allocated = allocated_.load(std::memory_order_relaxed);
  for (;;) {
    if (bytes > options_.maxBytes - allocated) {
      throw ResourceLimitExceeded(ResourceLimit::Memory);
    }
    if (allocated_.compare_exchange_weak(allocated, allocated + bytes,
                                         std::memory_order_relaxed)) {
      break;
    }
  }
  void* pointer = nullptr;
  try {
    pointer = ::operator new(bytes, std::align_val_t{alignment});
  } catch (...) {
    allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    throw;
  }
  auto peak = peak_.load(std::memory_order_relaxed);
  while (peak < allocated + bytes &&
         !peak_.compare_exchange_weak(peak, allocated + bytes,
                                      std::memory_order_relaxed)) {
  }
  return pointer;
}

void ResourceControl::deallocate(void* pointer, const size_t bytes,
                                 const size_t alignment) noexcept {
  ::operator delete(pointer, std::align_val_t{alignment});
  allocated_.fetch_sub(bytes, std::memory_order_relaxed);
}
} // namespace dd
