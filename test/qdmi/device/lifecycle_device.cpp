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
std::mutex stateMutex;
std::condition_variable stateChanged;
bool blockFinalize = false;
bool finalizeStarted = false;
bool releaseFinalize = false;
std::size_t initializeCount = 0;
std::size_t finalizeCount = 0;
std::size_t overlappingInitializeCount = 0;
} // namespace

extern "C" {

TEST_LIFECYCLE_QDMI_EXPORT void TEST_LIFECYCLE_prepare_blocking_finalize() {
  const std::scoped_lock lock(stateMutex);
  blockFinalize = true;
  finalizeStarted = false;
  releaseFinalize = false;
  initializeCount = 0;
  finalizeCount = 0;
  overlappingInitializeCount = 0;
}

TEST_LIFECYCLE_QDMI_EXPORT bool
TEST_LIFECYCLE_wait_for_finalize(const std::size_t timeoutMs) {
  std::unique_lock lock(stateMutex);
  return stateChanged.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                               [] { return finalizeStarted; });
}

TEST_LIFECYCLE_QDMI_EXPORT bool
TEST_LIFECYCLE_wait_for_initializations(const std::size_t expected,
                                        const std::size_t timeoutMs) {
  std::unique_lock lock(stateMutex);
  return stateChanged.wait_for(
      lock, std::chrono::milliseconds(timeoutMs),
      [expected] { return initializeCount >= expected; });
}

TEST_LIFECYCLE_QDMI_EXPORT void TEST_LIFECYCLE_release_finalize() {
  {
    const std::scoped_lock lock(stateMutex);
    releaseFinalize = true;
  }
  stateChanged.notify_all();
}

TEST_LIFECYCLE_QDMI_EXPORT std::size_t TEST_LIFECYCLE_initialize_count() {
  const std::scoped_lock lock(stateMutex);
  return initializeCount;
}

TEST_LIFECYCLE_QDMI_EXPORT std::size_t TEST_LIFECYCLE_finalize_count() {
  const std::scoped_lock lock(stateMutex);
  return finalizeCount;
}

TEST_LIFECYCLE_QDMI_EXPORT std::size_t
TEST_LIFECYCLE_overlapping_initialize_count() {
  const std::scoped_lock lock(stateMutex);
  return overlappingInitializeCount;
}

TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_initialize() {
  {
    const std::scoped_lock lock(stateMutex);
    ++initializeCount;
    if (finalizeStarted) {
      ++overlappingInitializeCount;
    }
  }
  stateChanged.notify_all();
  return QDMI_SUCCESS;
}

TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_finalize() {
  std::unique_lock lock(stateMutex);
  finalizeStarted = true;
  stateChanged.notify_all();
  if (blockFinalize) {
    stateChanged.wait(lock, [] { return releaseFinalize; });
  }
  finalizeStarted = false;
  ++finalizeCount;
  stateChanged.notify_all();
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
    TEST_LIFECYCLE_QDMI_Device_Session, QDMI_Device_Session_Parameter,
    std::size_t, const void*) {
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
                                             QDMI_Device_Job_Parameter,
                                             std::size_t, const void*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int TEST_LIFECYCLE_QDMI_device_job_query_property(
    TEST_LIFECYCLE_QDMI_Device_Job, QDMI_Device_Job_Property, std::size_t,
    void*, std::size_t*) {
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
TEST_LIFECYCLE_QDMI_device_job_wait(TEST_LIFECYCLE_QDMI_Device_Job,
                                    std::size_t) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_job_get_results(TEST_LIFECYCLE_QDMI_Device_Job,
                                           QDMI_Job_Result, std::size_t, void*,
                                           std::size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_query_device_property(
    TEST_LIFECYCLE_QDMI_Device_Session, QDMI_Device_Property, std::size_t,
    void*, std::size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_query_site_property(
    TEST_LIFECYCLE_QDMI_Device_Session, TEST_LIFECYCLE_QDMI_Site,
    QDMI_Site_Property, std::size_t, void*, std::size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
TEST_LIFECYCLE_QDMI_EXPORT int
TEST_LIFECYCLE_QDMI_device_session_query_operation_property(
    TEST_LIFECYCLE_QDMI_Device_Session, TEST_LIFECYCLE_QDMI_Operation,
    std::size_t, const TEST_LIFECYCLE_QDMI_Site*, std::size_t, const double*,
    QDMI_Operation_Property, std::size_t, void*, std::size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTEND(readability-named-parameter)

} // extern "C"
