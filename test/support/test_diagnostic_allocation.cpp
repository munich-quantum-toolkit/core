/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/DiagnosticFormatting.hpp"
#include "support/Diagnostics.hpp"

#include "gtest/gtest.h"

#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <string>

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool failAllocations = false;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local size_t failedAllocations = 0;
} // namespace

void* operator new(const size_t size) {
  if (failAllocations) {
    ++failedAllocations;
    throw std::bad_alloc{};
  }
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  if (auto* memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(memory);
}

void operator delete(void* memory, size_t /*size*/) noexcept {
  ::operator delete(memory);
}

TEST(DiagnosticFormatting, ReportsPersistentAllocationFailure) {
  const std::string argument(4096, 'x');
  bool handled = false;
  const mqt::ScopedDiagnosticHandler handler([&](const mqt::Diagnostic&) {
    handled = true;
    return mlir::success();
  });
  testing::internal::CaptureStderr();
  failedAllocations = 0;
  failAllocations = true;
  mqt::diagnostics::error(
      "Allocation failed while reporting this diagnostic: {}", argument);
  failAllocations = false;
  EXPECT_GT(failedAllocations, 0U);
  EXPECT_FALSE(handled);
  EXPECT_EQ(testing::internal::GetCapturedStderr(),
            "[mqt-core] [error] Allocation failed while reporting this "
            "diagnostic: {}\n");
}
