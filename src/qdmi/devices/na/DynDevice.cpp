/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file This file is a thin wrapper around MQT's Neutral Atom QDMI Device with
 * another prefix.
 */

#include "mqt_na_dyn_qdmi/device.h"
#include "mqt_na_qdmi/device.h"

#include <cstddef>
#include <cstring>

int MQT_NA_DYN_QDMI_device_initialize() {
  return MQT_NA_QDMI_device_initialize();
}

int MQT_NA_DYN_QDMI_device_finalize() { return MQT_NA_QDMI_device_finalize(); }

int MQT_NA_DYN_QDMI_device_session_alloc(
    MQT_NA_DYN_QDMI_Device_Session* session) {
  return MQT_NA_QDMI_device_session_alloc(
      reinterpret_cast<MQT_NA_QDMI_Device_Session*>(session));
}

int MQT_NA_DYN_QDMI_device_session_init(
    MQT_NA_DYN_QDMI_Device_Session session) {
  return MQT_NA_QDMI_device_session_init(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session));
}

void MQT_NA_DYN_QDMI_device_session_free(
    MQT_NA_DYN_QDMI_Device_Session session) {
  MQT_NA_QDMI_device_session_free(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session));
}

int MQT_NA_DYN_QDMI_device_session_set_parameter(
    MQT_NA_DYN_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  return MQT_NA_QDMI_device_session_set_parameter(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session), param, size,
      value);
}

int MQT_NA_DYN_QDMI_device_session_create_device_job(
    MQT_NA_DYN_QDMI_Device_Session session, MQT_NA_DYN_QDMI_Device_Job* job) {
  return MQT_NA_QDMI_device_session_create_device_job(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session),
      reinterpret_cast<MQT_NA_QDMI_Device_Job*>(job));
}

void MQT_NA_DYN_QDMI_device_job_free(MQT_NA_DYN_QDMI_Device_Job job) {
  MQT_NA_QDMI_device_job_free(reinterpret_cast<MQT_NA_QDMI_Device_Job>(job));
}

int MQT_NA_DYN_QDMI_device_job_set_parameter(
    MQT_NA_DYN_QDMI_Device_Job job, const QDMI_Device_Job_Parameter param,
    const size_t size, const void* value) {
  return MQT_NA_QDMI_device_job_set_parameter(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job), param, size, value);
}

int MQT_NA_DYN_QDMI_device_job_query_property(
    MQT_NA_DYN_QDMI_Device_Job job, const QDMI_Device_Job_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  return MQT_NA_QDMI_device_job_query_property(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job), prop, size, value,
      sizeRet);
}

int MQT_NA_DYN_QDMI_device_job_submit(MQT_NA_DYN_QDMI_Device_Job job) {
  return MQT_NA_QDMI_device_job_submit(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job));
}

int MQT_NA_DYN_QDMI_device_job_cancel(MQT_NA_DYN_QDMI_Device_Job job) {
  return MQT_NA_QDMI_device_job_cancel(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job));
}

int MQT_NA_DYN_QDMI_device_job_check(MQT_NA_DYN_QDMI_Device_Job job,
                                     QDMI_Job_Status* status) {
  return MQT_NA_QDMI_device_job_check(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job), status);
}

int MQT_NA_DYN_QDMI_device_job_wait(MQT_NA_DYN_QDMI_Device_Job job,
                                    const size_t timeout) {
  return MQT_NA_QDMI_device_job_wait(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job), timeout);
}

int MQT_NA_DYN_QDMI_device_job_get_results(MQT_NA_DYN_QDMI_Device_Job job,
                                           QDMI_Job_Result result,
                                           const size_t size, void* data,
                                           size_t* sizeRet) {
  return MQT_NA_QDMI_device_job_get_results(
      reinterpret_cast<MQT_NA_QDMI_Device_Job>(job), result, size, data,
      sizeRet);
}

int MQT_NA_DYN_QDMI_device_session_query_device_property(
    MQT_NA_DYN_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  const auto result = MQT_NA_QDMI_device_session_query_device_property(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session), prop, size, value,
      sizeRet);
  // let the proper device implementation do the error handling and check for
  // the name property afterward
  if (result == QDMI_SUCCESS && prop == QDMI_DEVICE_PROPERTY_NAME) {
    if (value != nullptr) {
      if (size < 27) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      strncpy(static_cast<char*>(value), "MQT NA Dynamic QDMI Device", size);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      static_cast<char*>(value)[size - 1] = '\0';
    }
    if ((sizeRet) != nullptr) {
      *sizeRet = 27;
    }
  }
  return result;
}

int MQT_NA_DYN_QDMI_device_session_query_site_property(
    MQT_NA_DYN_QDMI_Device_Session session, MQT_NA_DYN_QDMI_Site site,
    const QDMI_Site_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  return MQT_NA_QDMI_device_session_query_site_property(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session),
      reinterpret_cast<MQT_NA_QDMI_Site>(site), prop, size, value, sizeRet);
}

int MQT_NA_DYN_QDMI_device_session_query_operation_property(
    MQT_NA_DYN_QDMI_Device_Session session, MQT_NA_DYN_QDMI_Operation operation,
    const size_t numSites, const MQT_NA_DYN_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  return MQT_NA_QDMI_device_session_query_operation_property(
      reinterpret_cast<MQT_NA_QDMI_Device_Session>(session),
      reinterpret_cast<MQT_NA_QDMI_Operation>(operation), numSites,
      reinterpret_cast<const MQT_NA_QDMI_Site*>(sites), numParams, params, prop,
      size, value, sizeRet);
}
