/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt_sc_qdmi/device.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool failNextAllocation = false;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local size_t allocationCount = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local void* trackedJob = nullptr;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local size_t jobDeallocations = 0;
} // namespace

void* operator new(const size_t size) {
  ++allocationCount;
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
  if (memory == trackedJob) {
    ++jobDeallocations;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(memory);
}

void operator delete(void* memory, size_t /*size*/) noexcept {
  ::operator delete(memory);
}

TEST(ScDeviceAllocation, CheckedSessionAndJobAllocation) {
  MQT_SC_QDMI_Device_Session session = nullptr;
  failNextAllocation = true;
  const auto allocationStatus = MQT_SC_QDMI_device_session_alloc(&session);
  failNextAllocation = false;
  EXPECT_EQ(allocationStatus, QDMI_ERROR_OUTOFMEM);
  ASSERT_EQ(session, nullptr);
  ASSERT_EQ(MQT_SC_QDMI_device_session_alloc(&session), QDMI_SUCCESS);
  ASSERT_EQ(MQT_SC_QDMI_device_session_init(session), QDMI_SUCCESS);

  MQT_SC_QDMI_Device_Job job = nullptr;
  failNextAllocation = true;
  const auto status =
      MQT_SC_QDMI_device_session_create_device_job(session, &job);
  failNextAllocation = false;
  EXPECT_EQ(status, QDMI_ERROR_OUTOFMEM);
  EXPECT_EQ(job, nullptr);

  allocationCount = 0;
  const auto retryStatus =
      MQT_SC_QDMI_device_session_create_device_job(session, &job);
  const auto allocations = allocationCount;
  EXPECT_EQ(retryStatus, QDMI_SUCCESS);
  EXPECT_EQ(allocations, 1);
  ASSERT_NE(job, nullptr);
  trackedJob = job;
  jobDeallocations = 0;
  MQT_SC_QDMI_device_job_free(job);
  EXPECT_EQ(jobDeallocations, 1);
  trackedJob = nullptr;

  ASSERT_EQ(MQT_SC_QDMI_device_session_create_device_job(session, &job),
            QDMI_SUCCESS);
  trackedJob = job;
  jobDeallocations = 0;
  MQT_SC_QDMI_device_session_free(session);
  EXPECT_EQ(jobDeallocations, 1);
  trackedJob = nullptr;
}
