/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "test_lifecycle_qdmi/device.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace {
struct LifecycleState {
  std::mutex mutex;
  std::condition_variable changed;
  bool blockFinalize = false;
  bool finalizeStarted = false;
  bool releaseFinalize = false;
  size_t initializeCount = 0;
  size_t finalizeCount = 0;
  size_t overlappingInitializeCount = 0;
};

[[nodiscard]] LifecycleState& lifecycleState() {
  static LifecycleState state;
  return state;
}
} // namespace

extern "C" {

TEST_LIFECYCLE_QDMI_EXPORT void testLifecyclePrepareBlockingFinalize() {
  auto& state = lifecycleState();
  const std::scoped_lock lock(state.mutex);
  state.blockFinalize = true;
  state.finalizeStarted = false;
  state.releaseFinalize = false;
  state.initializeCount = 0;
  state.finalizeCount = 0;
  state.overlappingInitializeCount = 0;
}

TEST_LIFECYCLE_QDMI_EXPORT bool
testLifecycleWaitForFinalize(const size_t timeoutMs) {
  auto& state = lifecycleState();
  std::unique_lock lock(state.mutex);
  return state.changed.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                                [&state] { return state.finalizeStarted; });
}

TEST_LIFECYCLE_QDMI_EXPORT bool
testLifecycleWaitForInitializations(const size_t expected,
                                    const size_t timeoutMs) {
  auto& state = lifecycleState();
  std::unique_lock lock(state.mutex);
  return state.changed.wait_for(
      lock, std::chrono::milliseconds(timeoutMs),
      [&state, expected] { return state.initializeCount >= expected; });
}

TEST_LIFECYCLE_QDMI_EXPORT void testLifecycleReleaseFinalize() {
  auto& state = lifecycleState();
  {
    const std::scoped_lock lock(state.mutex);
    state.releaseFinalize = true;
  }
  state.changed.notify_all();
}

TEST_LIFECYCLE_QDMI_EXPORT size_t testLifecycleInitializeCount() {
  auto& state = lifecycleState();
  const std::scoped_lock lock(state.mutex);
  return state.initializeCount;
}

TEST_LIFECYCLE_QDMI_EXPORT size_t testLifecycleFinalizeCount() {
  auto& state = lifecycleState();
  const std::scoped_lock lock(state.mutex);
  return state.finalizeCount;
}

TEST_LIFECYCLE_QDMI_EXPORT size_t testLifecycleOverlappingInitializeCount() {
  auto& state = lifecycleState();
  const std::scoped_lock lock(state.mutex);
  return state.overlappingInitializeCount;
}

TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_initialize() {
  auto& state = lifecycleState();
  {
    const std::scoped_lock lock(state.mutex);
    ++state.initializeCount;
    if (state.finalizeStarted) {
      ++state.overlappingInitializeCount;
    }
  }
  state.changed.notify_all();
  return QDMI_SUCCESS;
}

TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_finalize() {
  auto& state = lifecycleState();
  std::unique_lock lock(state.mutex);
  state.finalizeStarted = true;
  state.changed.notify_all();
  if (state.blockFinalize) {
    state.changed.wait(lock, [&state] { return state.releaseFinalize; });
  }
  state.finalizeStarted = false;
  ++state.finalizeCount;
  state.changed.notify_all();
  return QDMI_SUCCESS;
}

// The remaining functions only need to be resolvable for this lifecycle test.
// NOLINTBEGIN(readability-named-parameter)
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_alloc(TEST_LIFECYCLE_QDMI_Device_Session*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_init(TEST_LIFECYCLE_QDMI_Device_Session) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT void
TEST_LIFECYCLE_QDMI_device_session_free(TEST_LIFECYCLE_QDMI_Device_Session) {}
TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_session_set_parameter(
    TEST_LIFECYCLE_QDMI_Device_Session, QDMI_Device_Session_Parameter, size_t,
    const void*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_create_device_job(
    TEST_LIFECYCLE_QDMI_Device_Session, TEST_LIFECYCLE_QDMI_Device_Job*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT void
TEST_LIFECYCLE_QDMI_device_job_free(TEST_LIFECYCLE_QDMI_Device_Job) {}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_set_parameter(TEST_LIFECYCLE_QDMI_Device_Job,
                                             QDMI_Device_Job_Parameter, size_t,
                                             const void*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_query_property(TEST_LIFECYCLE_QDMI_Device_Job,
                                              QDMI_Device_Job_Property, size_t,
                                              void*, size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_submit(TEST_LIFECYCLE_QDMI_Device_Job) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_cancel(TEST_LIFECYCLE_QDMI_Device_Job) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_check(TEST_LIFECYCLE_QDMI_Device_Job,
                                     QDMI_Job_Status*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_wait(TEST_LIFECYCLE_QDMI_Device_Job, size_t) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_job_get_results(
    TEST_LIFECYCLE_QDMI_Device_Job, QDMI_Job_Result, size_t, void*, size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_query_device_property(
    TEST_LIFECYCLE_QDMI_Device_Session, QDMI_Device_Property, size_t, void*,
    size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_query_site_property(
    TEST_LIFECYCLE_QDMI_Device_Session, TEST_LIFECYCLE_QDMI_Site,
    QDMI_Site_Property, size_t, void*, size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_query_operation_property(
    TEST_LIFECYCLE_QDMI_Device_Session, TEST_LIFECYCLE_QDMI_Operation, size_t,
    const TEST_LIFECYCLE_QDMI_Site*, size_t, const double*,
    QDMI_Operation_Property, size_t, void*, size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTEND(readability-named-parameter)

} // extern "C"
