/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Driver.hpp"

#include "gtest/gtest.h"
#include "qdmi/client.h"
#include "qdmi/device.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool failNextAllocation = false;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local size_t allocationCount = 0;
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

  allocationCount = 0;
  const auto retryStatus = QDMI_session_alloc(&session);
  const auto retryAllocations = allocationCount;
  EXPECT_EQ(retryStatus, QDMI_SUCCESS);
  EXPECT_EQ(retryAllocations, 1);
  QDMI_session_free(session);
}

namespace {

// Provider callbacks deliberately allocate nothing: the measured allocation is
// the driver's job wrapper, including publication into its ownership list.
class AllocationDeviceAPI final : public qdmi::DeviceAPI {
  static inline AllocationDeviceAPI* active = nullptr;

public:
  size_t createdJobs = 0;
  size_t retrievedJobs = 0;
  size_t freedJobs = 0;

  AllocationDeviceAPI() {
    active = this;
    device_session_alloc = [](QDMI_Device_Session* session) -> int {
      *session = reinterpret_cast<QDMI_Device_Session>(active);
      return QDMI_SUCCESS;
    };
    device_session_init = [](QDMI_Device_Session) -> int {
      return QDMI_SUCCESS;
    };
    device_session_free = [](QDMI_Device_Session) {};
    device_session_query_device_property =
        [](QDMI_Device_Session, QDMI_Device_Property, size_t, void*,
           size_t*) -> int { return QDMI_ERROR_NOTSUPPORTED; };
    device_session_create_device_job = [](QDMI_Device_Session session,
                                          QDMI_Device_Job* job) -> int {
      auto* provider = reinterpret_cast<AllocationDeviceAPI*>(session);
      ++provider->createdJobs;
      *job = reinterpret_cast<QDMI_Device_Job>(provider);
      return QDMI_SUCCESS;
    };
    device_session_retrieve_device_job_by_id = [](QDMI_Device_Session session,
                                                  const char*,
                                                  QDMI_Device_Job* job) -> int {
      auto* provider = reinterpret_cast<AllocationDeviceAPI*>(session);
      ++provider->retrievedJobs;
      *job = reinterpret_cast<QDMI_Device_Job>(provider);
      return QDMI_SUCCESS;
    };
    device_job_free = [](QDMI_Device_Job job) {
      ++reinterpret_cast<AllocationDeviceAPI*>(job)->freedJobs;
    };
  }
};

class QDMIDriverJobAllocation : public testing::TestWithParam<bool> {};

TEST_P(QDMIDriverJobAllocation, FailsBeforeCallingProviderAndRecovers) {
  AllocationDeviceAPI provider;
  auto device = QDMI_Device_impl_d::create(provider);
  ASSERT_TRUE(!!device) << llvm::toString(device.takeError());
  const auto allocate = [&](QDMI_Job* job) {
    return GetParam() ? (*device)->retrieveJobById("existing-job", job)
                      : (*device)->createJob(job);
  };

  // A nonnull initial output also verifies that failure clears the handle.
  auto* job = reinterpret_cast<QDMI_Job>(&provider);
  failNextAllocation = true;
  const auto status = allocate(&job);
  failNextAllocation = false;
  EXPECT_EQ(status, QDMI_ERROR_OUTOFMEM);
  EXPECT_EQ(job, nullptr);
  EXPECT_EQ(provider.createdJobs, 0);
  EXPECT_EQ(provider.retrievedJobs, 0);
  EXPECT_EQ(provider.freedJobs, 0);

  allocationCount = 0;
  const auto retryStatus = allocate(&job);
  const auto retryAllocations = allocationCount;
  ASSERT_EQ(retryStatus, QDMI_SUCCESS);
  ASSERT_NE(job, nullptr);
  EXPECT_EQ(retryAllocations, 1);
  EXPECT_EQ(provider.createdJobs, GetParam() ? 0 : 1);
  EXPECT_EQ(provider.retrievedJobs, GetParam() ? 1 : 0);
  (*device)->freeJob(job);
  EXPECT_EQ(provider.freedJobs, 1);
  device->reset();
  EXPECT_EQ(provider.freedJobs, 1);
}

INSTANTIATE_TEST_SUITE_P(CreateAndRetrieve, QDMIDriverJobAllocation,
                         testing::Bool());
} // namespace
