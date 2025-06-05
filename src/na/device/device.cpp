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

#include <fstream>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>
#include <spdlog/spdlog.h>
#include <sstream>

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
 * @brief Populates all repeated fields of the message type in the given
 * Protobuf message with empty messages.
 * @param message The Protobuf message to populate.
 * @throws std::runtime_error if a repeated field has an unsupported type, i.e.,
 * not a message type.
 * @note This is a recursive auxiliary function used by @ref writeJsonSchema
 */
auto populateRepeatedFields(google::protobuf::Message* message) -> void {
  const google::protobuf::Descriptor* descriptor = message->GetDescriptor();
  const google::protobuf::Reflection* reflection = message->GetReflection();

  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* field = descriptor->field(i);
    if (field->is_repeated()) {
      if (field->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
        populateRepeatedFields(reflection->AddMessage(message, field));
      } else {
        std::stringstream ss;
        ss << "Unsupported repeated field type in device configuration: "
           << field->cpp_type();
        throw std::runtime_error(ss.str());
      }
    } else if (field->type() ==
               google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
      // Message fields must be explicitly initialized such that they appear in
      // the written JSON schema, primitive fields are automatically
      // initialized
      reflection->MutableMessage(message, field);
    }
  }
}

// todo: Should this function be exposed and how because it is not part of the
//  public QDMI API?
/**
 * @brief Writes a JSON schema with default values for the device configuration
 * to the specified path.
 * @param path The path to write the JSON schema to.
 * @throws std::runtime_error if the JSON conversion fails or the file cannot be
 * opened.
 */
auto writeJsonSchema(const std::string& path) -> void {
  // Create a default device configuration
  na::Device device;

  // Fill each repeated field with an empty message
  populateRepeatedFields(&device);

  // Set print options
  google::protobuf::util::JsonPrintOptions options;
  options.always_print_fields_with_no_presence = true;

  // Convert to JSON
  std::string json;
  const auto status =
      google::protobuf::util::MessageToJsonString(device, &json, options);
  if (!status.ok()) {
    std::stringstream ss;
    ss << "Failed to convert Protobuf message to JSON: " << status.ToString();
    throw std::runtime_error(ss.str());
  }

  // Write to file
  std::ofstream ofs(path);
  if (ofs.is_open()) {
    ofs << json;
    ofs.close();
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
    std::stringstream ss;
    ss << "JSON template written to " << path;
    SPDLOG_INFO(ss.str());
#endif
  } else {
    std::stringstream ss;
    ss << "Failed to open file for writing: " << path;
    throw std::runtime_error(ss.str());
  }
}

/**
 * @brief Returns a reference to the device singleton.
 * @details This function initializes the device singleton on the first call
 * and returns a reference to it.
 * @returns A reference to the device singleton.
 * @throws std::runtime_error if the environment variable
 * MQT_CORE_NA_QDMI_DEVICE_JSON_FILE is not set, the JSON file does not exist,
 * or the JSON file cannot be parsed.
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
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open JSON file: " +
                               std::string(path));
    }
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
      std::stringstream ss;
      ss << "Failed to parse JSON string into Protobuf message: "
         << status.ToString();
      throw std::runtime_error(ss.str());
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
    SPDLOG_ERROR(e.what());
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

int MQT_NA_QDMI_device_job_wait(MQT_NA_QDMI_Device_Job job) {
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
