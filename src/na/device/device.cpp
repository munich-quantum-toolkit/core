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
#include "spdlog/spdlog.h"

#include <fstream>
#include <google/protobuf/util/json_util.h>

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

namespace {
/**
 * @brief Returns a reference to the device singleton.
 * @details This function initializes the device singleton on the first call
 * and returns a reference to it.
 * @return A reference to the device singleton.
 */
auto getDevice() -> const na::Device& {
  static na::Device device;
  static bool initialized = false;
  if (!initialized) {
    // Get the path to the JSON file from the environment variable
    const char* path = std::getenv("MQT_CORE_NA_QDMI_DEVICE_JSON_FILE");
    if (path == nullptr) {
      throw std::runtime_error(
          "Environment variable MQT_CORE_NA_QDMI_DEVICE_JSON_FILE is not set.");
    }
    // Read the device configuration from a JSON file
    std::ifstream ifs(path);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    const std::string json = buffer.str();
    ifs.close();
    // Parse the JSON string into the protobuf message
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const auto status =
        google::protobuf::util::JsonStringToMessage(json, &device);
    if (!status.ok()) {
      throw std::runtime_error("Failed to parse device JSON: " +
                               status.ToString());
    }
    // Set initialized to true to avoid re-initialization
    initialized = true;
  }
  return device;
}
} // namespace

int MQT_NA_QDMI_device_initialize() {
  try {
    getDevice();
  } catch (const std::runtime_error& e) {
    SPDLOG_ERROR(e);
    return QDMI_ERROR_FATAL;
  }
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_finalize() { return QDMI_SUCCESS; }

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
