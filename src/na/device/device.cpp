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
 * @brief The MQT QDMI device implementation for neutral atom devices.
 */

#include "mqt_na_qdmi/device.h"

#include "device.pb.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Session structure.
 * @details This structure can, e.g., be used to store a token to access an API.
 */
struct MQT_NA_QDMI_Device_Session_impl_d {};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Job structure.
 * @details This structure can, e.g., be used to store the job id.
 */
struct MQT_NA_QDMI_Device_Job_impl_d {};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Site structure.
 * @details This structure can, e.g., be used to store the site id.
 */
struct MQT_NA_QDMI_Device_Site_impl_d {};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Operation structure.
 * @details This structure can, e.g., be used to store the operation id.
 */
struct MQT_NA_QDMI_Device_Operation_impl_d {};

int MQT_NA_QDMI_device_initialize() { return QDMI_ERROR_NOTIMPLEMENTED; }

int MQT_NA_QDMI_device_finalize() { return QDMI_ERROR_NOTIMPLEMENTED; }

int MQT_NA_QDMI_device_session_alloc(MQT_NA_QDMI_Device_Session* session) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_session_init(MQT_NA_QDMI_Device_Session session) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

void MQT_NA_QDMI_device_session_free(MQT_NA_QDMI_Device_Session session) {}

int MQT_NA_QDMI_device_session_set_parameter(
    MQT_NA_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_session_create_device_job(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Device_Job* job) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

void MQT_NA_QDMI_device_job_free(MQT_NA_QDMI_Device_Job job) {}

int MQT_NA_QDMI_device_job_set_parameter(MQT_NA_QDMI_Device_Job job,
                                         const QDMI_Device_Job_Parameter param,
                                         const size_t size, const void* value) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_job_query_property(MQT_NA_QDMI_Device_Job job,
                                          const QDMI_Device_Job_Property prop,
                                          const size_t size, void* value,
                                          size_t* size_ret) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_job_submit(MQT_NA_QDMI_Device_Job job) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_job_cancel(MQT_NA_QDMI_Device_Job job) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_job_check(MQT_NA_QDMI_Device_Job job,
                                 QDMI_Job_Status* status) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_job_wait(MQT_NA_QDMI_Device_Job job,
                                const size_t timeout) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_job_get_results(MQT_NA_QDMI_Device_Job job,
                                       QDMI_Job_Result result,
                                       const size_t size, void* data,
                                       size_t* size_ret) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_session_query_device_property(
    MQT_NA_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* size_ret) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_session_query_site_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Site site,
    const QDMI_Site_Property prop, const size_t size, void* value,
    size_t* size_ret) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}

int MQT_NA_QDMI_device_session_query_operation_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Operation operation,
    const size_t num_sites, const MQT_NA_QDMI_Site* sites,
    const size_t num_params, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* size_ret) {
  return QDMI_ERROR_NOTIMPLEMENTED;
}
