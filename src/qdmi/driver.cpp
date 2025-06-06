/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief An example driver implementation in C++.
 */

#include "qdmi/driver.hpp"

#include "mqt_na_qdmi/device.h"

#include <cstdlib>
#include <qdmi/client.h>

namespace {
enum class SessionStatus : uint8_t {
  ALLOCATED,  ///< The session has been allocated but not initialized
  INITIALIZED ///< The session has been initialized and is ready for use
};
} // namespace

/**
 * @brief Definition of the QDMI Device.
 */
struct QDMI_Device_impl_d {
  PREFIXED(QDMI_Device_Session) session = nullptr;
  decltype(PREFIXED(QDMI_device_initialize))* deviceInitialize =
      &PREFIXED(QDMI_device_initialize);
  decltype(PREFIXED(QDMI_device_finalize))* deviceFinalize =
      &PREFIXED(QDMI_device_finalize);
  decltype(PREFIXED(QDMI_device_session_alloc))* deviceSessionAlloc =
      &PREFIXED(QDMI_device_session_alloc);
  decltype(PREFIXED(QDMI_device_session_init))* deviceSessionInit =
      &PREFIXED(QDMI_device_session_init);
  decltype(PREFIXED(QDMI_device_session_free))* deviceSessionFree =
      &PREFIXED(QDMI_device_session_free);
  decltype(PREFIXED(
      QDMI_device_session_set_parameter))* deviceSessionSetParameter =
      &PREFIXED(QDMI_device_session_set_parameter);
  decltype(PREFIXED(
      QDMI_device_session_create_device_job))* deviceSessionCreateDeviceJob =
      &PREFIXED(QDMI_device_session_create_device_job);
  decltype(PREFIXED(QDMI_device_job_free))* deviceJobFree =
      &PREFIXED(QDMI_device_job_free);
  decltype(PREFIXED(QDMI_device_job_set_parameter))* deviceJobSetParameter =
      &PREFIXED(QDMI_device_job_set_parameter);
  decltype(PREFIXED(QDMI_device_job_submit))* deviceJobSubmit =
      &PREFIXED(QDMI_device_job_submit);
  decltype(PREFIXED(QDMI_device_job_cancel))* deviceJobCancel =
      &PREFIXED(QDMI_device_job_cancel);
  decltype(PREFIXED(QDMI_device_job_check))* deviceJobCheck =
      &PREFIXED(QDMI_device_job_check);
  decltype(PREFIXED(QDMI_device_job_wait))* deviceJobWait =
      &PREFIXED(QDMI_device_job_wait);
  decltype(PREFIXED(QDMI_device_job_get_results))* deviceJobGetResults =
      &PREFIXED(QDMI_device_job_get_results);
  decltype(PREFIXED(QDMI_device_session_query_device_property))*
      deviceSessionQueryDeviceProperty =
          &PREFIXED(QDMI_device_session_query_device_property);
  decltype(PREFIXED(QDMI_device_session_query_site_property))*
      deviceSessionQuerySiteProperty =
          &PREFIXED(QDMI_device_session_query_site_property);
  decltype(PREFIXED(QDMI_device_session_query_operation_property))*
      deviceSessionQueryOperationProperty =
          &PREFIXED(QDMI_device_session_query_operation_property);
};

/**
 * @brief Definition of the QDMI Session.
 */
struct QDMI_Session_impl_d {
  SessionStatus status = SessionStatus::ALLOCATED;
};

/**
 * @brief Definition of the QDMI Job.
 */
struct QDMI_Job_impl_d {
  PREFIXED(QDMI_Device_Job) deviceJob = nullptr;
  QDMI_Device device = nullptr;
};

namespace {

[[nodiscard]] auto device() -> QDMI_Device_impl_d& {
  static QDMI_Device_impl_d device;
  return device;
}

} // namespace

auto initialize() -> void {
  device().deviceInitialize();
  device().deviceSessionAlloc(&device().session);
  device().deviceSessionInit(device().session);
}

auto finalize() -> void {
  device().deviceSessionFree(device().session);
  device().deviceFinalize();
}

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new QDMI_Session_impl_d();
  return QDMI_SUCCESS;
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

void QDMI_session_free(QDMI_Session session) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete session;
}

int QDMI_session_set_parameter(QDMI_Session session,
                               QDMI_Session_Parameter param, const size_t size,
                               const void* value) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_session_query_session_property(QDMI_Session session,
                                        QDMI_Session_Property prop, size_t size,
                                        void* value, size_t* sizeRet) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_SESSION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == QDMI_SESSION_PROPERTY_DEVICES) {
    if (value != nullptr) {
      if (size < sizeof(QDMI_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      *static_cast<QDMI_Device*>(value) = &device();
    }
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(QDMI_Device);
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_device_create_job(QDMI_Device dev, QDMI_Job* job) {
  if (dev == nullptr || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *job = new QDMI_Job_impl_d();
  (*job)->device = dev;
  return dev->deviceSessionCreateDeviceJob(dev->session, &(*job)->deviceJob);
}

void QDMI_job_free(QDMI_Job job) {
  if (job != nullptr) {
    job->device->deviceJobFree(job->deviceJob);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete job;
  }
}

int QDMI_job_set_parameter(QDMI_Job job, QDMI_Job_Parameter param,
                           const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->deviceJobSetParameter(
      job->deviceJob, static_cast<QDMI_Device_Job_Parameter>(param), size,
      value);
}

int QDMI_job_submit(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->deviceJobSubmit(job->deviceJob);
}

int QDMI_job_cancel(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->deviceJobCancel(job->deviceJob);
}

int QDMI_job_check(QDMI_Job job, QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->deviceJobCheck(job->deviceJob, status);
}

int QDMI_job_wait(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->deviceJobWait(job->deviceJob);
}

int QDMI_job_get_results(QDMI_Job job, QDMI_Job_Result result,
                         const size_t size, void* data, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->deviceJobGetResults(job->deviceJob, result, size, data,
                                          sizeRet);
}

int QDMI_device_query_device_property(QDMI_Device device,
                                      QDMI_Device_Property prop,
                                      const size_t size, void* value,
                                      size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->deviceSessionQueryDeviceProperty(device->session, prop, size,
                                                  value, sizeRet);
}

int QDMI_device_query_site_property(QDMI_Device device, QDMI_Site site,
                                    QDMI_Site_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->deviceSessionQuerySiteProperty(
      device->session, reinterpret_cast<PREFIXED(QDMI_Site)>(site), prop, size,
      value, sizeRet);
}

int QDMI_device_query_operation_property(
    QDMI_Device device, QDMI_Operation operation, const size_t numSites,
    const QDMI_Site* sites, const size_t numParams, const double* params,
    QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->deviceSessionQueryOperationProperty(
      device->session, reinterpret_cast<PREFIXED(QDMI_Operation)>(operation),
      numSites, reinterpret_cast<const PREFIXED(QDMI_Site)*>(sites), numParams,
      params, prop, size, value, sizeRet);
}
