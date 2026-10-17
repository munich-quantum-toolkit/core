/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "gtest/gtest.h"
#include "qdmi/client.h"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool failNextAllocation = false;
} // namespace

void* operator new(const size_t size) {
  if (std::exchange(failNextAllocation, false)) {
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

TEST(QDMIDriverAllocation, RecoversAfterSessionAllocationFailure) {
  QDMI_Session session = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&session), QDMI_SUCCESS);
  QDMI_session_free(session);
  session = nullptr;

  failNextAllocation = true;
  const auto status = QDMI_session_alloc(&session);
  failNextAllocation = false;
  EXPECT_EQ(status, QDMI_ERROR_OUTOFMEM);
  EXPECT_EQ(session, nullptr);

  ASSERT_EQ(QDMI_session_alloc(&session), QDMI_SUCCESS);
  QDMI_session_free(session);
}
