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

namespace {
enum class DeviceSessionStatus : uint8_t { ALLOCATED, INITIALIZED };
} // namespace

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Session structure.
 * @details This structure can, e.g., be used to store a token to access an API.
 */
struct MQT_NA_QDMI_Device_Session_impl_d {
  DeviceSessionStatus status = DeviceSessionStatus::ALLOCATED;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Job structure.
 * @details This structure can, e.g., be used to store the job id.
 */
struct MQT_NA_QDMI_Device_Job_impl_d {};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Site structure.
 * @details This structure can, e.g., be used to store the site id.
 */
struct MQT_NA_QDMI_Site_impl_d {
  int64_t x = 0;
  int64_t y = 0;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Operation structure.
 * @details This structure can, e.g., be used to store the operation id.
 */
struct MQT_NA_QDMI_Operation_impl_d {
  std::string name;
};

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
//  public QDMI API? If it is exposed, it also should be moved outside the
//  anonymous namespace.
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

[[nodiscard]] auto initialized() -> bool& {
  static bool initialized = false;
  return initialized;
}

[[nodiscard]] auto name() -> std::string& {
  static std::string name;
  return name;
}

[[nodiscard]] auto sites()
    -> std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>>& {
  static std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>> sites;
  return sites;
}

[[nodiscard]] auto operations()
    -> std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>>& {
  static std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>> operations;
  return operations;
}

/**
 * @brief Parses the device configuration from a JSON file specified by the
 * environment variable MQT_CORE_NA_QDMI_DEVICE_JSON_FILE.
 * @returns The parsed device configuration as a Protobuf message.
 * @throws std::runtime_error if the environment variable is not set, the JSON
 * file does not exist, or the JSON file cannot be parsed.
 */
[[nodiscard]] auto parse() -> na::Device {
  // Get the path to the JSON file from the environment variable
  const char* path = std::getenv("MQT_CORE_NA_QDMI_DEVICE_JSON_FILE");
  if (path == nullptr) {
    throw std::runtime_error(
        "Environment variable MQT_CORE_NA_QDMI_DEVICE_JSON_FILE is not set.");
  }
  // Read the device configuration from a JSON file
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + std::string(path));
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  const std::string json = buffer.str();
  ifs.close();
  // Parse the JSON string into the protobuf message
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  na::Device device;
  const auto status =
      google::protobuf::util::JsonStringToMessage(json, &device);
  if (!status.ok()) {
    std::stringstream ss;
    ss << "Failed to parse JSON string into Protobuf message: "
       << status.ToString();
    throw std::runtime_error(ss.str());
  }
  // Validate device
  for (const auto& lattice : device.traps()) {
    if (lattice.lattice_vectors_size() > 2) {
      std::stringstream ss;
      ss << "Lattice vectors size " << lattice.lattice_vectors_size()
         << "exceeds 2 which means that specification of traps is not unique "
            "anymore in the 2D plane.";
      throw std::runtime_error(ss.str());
    }
  }
  return device;
}

/**
 * @brief Increments the indices in lexicographic order.
 * @details This function increments the first index that is less than its
 * limit, resets all previous indices to zero.
 * @param indices The vector of indices to increment.
 * @param limits The limits for each index.
 * @returns true if the increment was successful, false if all indices have
 * reached their limits.
 */
[[nodiscard]] auto increment(std::vector<size_t>& indices,
                             const std::vector<size_t>& limits) -> bool {
  size_t i = 0;
  for (; i < indices.size() && indices[i] < limits[i]; ++i) {
  }
  if (i == indices.size()) {
    return false;
  }
  for (size_t j = 0; j < i; ++j) {
    indices[j] = 0; // Reset all previous indices
  }
  indices[i]++; // Increment the next index
  return true;
}

/**
 * @brief Initializes the device with the configuration parsed from the JSON
 * file.
 * @details This function transfers all data from the parsed Protobuf
 * message to the device's internal structures, such as the name, sites, and
 * operations.
 * @throws std::runtime_error if the device has already been initialized.
 */
auto initialize() -> void {
  if (initialized()) {
    throw std::runtime_error("The device has already been initialized.");
  }
  const auto& device = parse();
  // Transfer all data from the protobuf message to the device
  name() = device.name();
  for (const auto& lattice : device.traps()) {
    const auto originX = lattice.lattice_origin().x();
    const auto originY = lattice.lattice_origin().y();
    std::vector limits(lattice.lattice_vectors_size(), 0UL);
    std::transform(lattice.lattice_vectors().begin(),
                   lattice.lattice_vectors().end(), limits.begin(),
                   [](const auto& vector) { return vector.repeat(); });
    std::vector indices(lattice.lattice_vectors_size(), 0UL);
    do {
      // For every sublattice offset, add a site for repetition indices
      for (const auto& offset : lattice.sublattice_offsets()) {
        auto& site =
            sites().emplace_back(std::make_unique<MQT_NA_QDMI_Site_impl_d>());
        site->x = originX + offset.x();
        site->y = originY + offset.y();
        for (size_t i = 0; i < lattice.lattice_vectors_size(); ++i) {
          const auto& vector = lattice.lattice_vectors(i).vector();
          site->x += indices[i] * vector.x();
          site->y += indices[i] * vector.y();
        }
      }
    } while (increment(indices, limits));
  }
  // Set initialized to true to avoid re-initialization
  initialized() = true;
}
} // namespace

int MQT_NA_QDMI_device_initialize() {
  try {
    initialize();
  } catch (const std::runtime_error& e) {
    SPDLOG_ERROR(e.what());
    return QDMI_ERROR_FATAL;
  }
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_finalize() { return QDMI_SUCCESS; }

int MQT_NA_QDMI_device_session_alloc(MQT_NA_QDMI_Device_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = new MQT_NA_QDMI_Device_Session_impl_d();
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_session_init(MQT_NA_QDMI_Device_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != DeviceSessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = DeviceSessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

void MQT_NA_QDMI_device_session_free(MQT_NA_QDMI_Device_Session session) {
  delete session;
}

int MQT_NA_QDMI_device_session_set_parameter(
    MQT_NA_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != DeviceSessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_create_device_job(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Device_Job* job) {
  if (session == nullptr || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != DeviceSessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

void MQT_NA_QDMI_device_job_free(MQT_NA_QDMI_Device_Job /* unused */) {}

int MQT_NA_QDMI_device_job_set_parameter(MQT_NA_QDMI_Device_Job job,
                                         const QDMI_Device_Job_Parameter param,
                                         const size_t size, const void* value) {
  if (job == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_JOB_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

int MQT_NA_QDMI_device_job_submit(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

int MQT_NA_QDMI_device_job_cancel(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

int MQT_NA_QDMI_device_job_check(MQT_NA_QDMI_Device_Job job,
                                 QDMI_Job_Status* status) {
  if (job == nullptr || status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

int MQT_NA_QDMI_device_job_wait(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

int MQT_NA_QDMI_device_job_get_results(MQT_NA_QDMI_Device_Job job,
                                       QDMI_Job_Result result,
                                       const size_t size, void* data,
                                       size_t* size_ret) {
  if (job == nullptr || (data != nullptr && size == 0) ||
      result >= QDMI_JOB_RESULT_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_PERMISSIONDENIED;
}

// NOLINTBEGIN(bugprone-macro-parentheses)
#define ADD_SINGLE_VALUE_PROPERTY(prop_name, prop_type, prop_value, prop,      \
                                  size, value, size_ret)                       \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < sizeof(prop_type)) {                                      \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        *static_cast<prop_type*>(value) = prop_value;                          \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = sizeof(prop_type);                                         \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  } /// [DOXYGEN MACRO END]

#define ADD_STRING_PROPERTY(prop_name, prop_value, prop, size, value,          \
                            size_ret)                                          \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < strlen(prop_value) + 1) {                                 \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        strncpy(static_cast<char*>(value), prop_value, size);                  \
        static_cast<char*>(value)[size - 1] = '\0';                            \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = strlen(prop_value) + 1;                                    \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  } /// [DOXYGEN MACRO END]

#define ADD_LIST_PROPERTY(prop_name, prop_type, prop_values, prop, size,       \
                          value, size_ret)                                     \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < (prop_values).size() * sizeof(prop_type)) {               \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        memcpy(static_cast<void*>(value),                                      \
               static_cast<const void*>((prop_values).data()),                 \
               (prop_values).size() * sizeof(prop_type));                      \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = (prop_values).size() * sizeof(prop_type);                  \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  } /// [DOXYGEN MACRO END]
// NOLINTEND(bugprone-macro-parentheses)

int MQT_NA_QDMI_device_session_query_device_property(
    MQT_NA_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* size_ret) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_DEVICE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != DeviceSessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_NAME, name().c_str(), prop, size,
                      value, size_ret)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_VERSION, MQT_CORE_VERSION, prop,
                      size, value, size_ret)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LIBRARYVERSION, QDMI_VERSION, prop,
                      size, value, size_ret)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_STATUS, QDMI_Device_Status,
                            QDMI_DEVICE_STATUS_OFFLINE, prop, size, value,
                            size_ret)
  // This device never needs calibration
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, size_t, 0,
                            prop, size, value, size_ret)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_NA_QDMI_Site, sites(), prop,
                    size, value, size_ret)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_NA_QDMI_Operation,
                    operations(), prop, size, value, size_ret)
  return QDMI_ERROR_NOTSUPPORTED;
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
