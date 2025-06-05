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
/// The status of the session.
enum class SessionStatus : uint8_t {
  ALLOCATED,   ///< The session has been allocated but not initialized
  INITIALIZED, ///< The session has been initialized and is ready for use
};

/// The type of an operation.
enum class OperationType : uint8_t {
  GLOBAL_SINGLE_QUBIT, ///< Global single-qubit operation
  GLOBAL_MULTI_QUBIT,  ///< Global multi-qubit operation
  LOCAL_SINGLE_QUBIT,  ///< Local single-qubit operation
  LOCAL_MULTI_QUBIT,   ///< Local multi-qubit operation
  SHUTTLING_LOAD,      ///< Shuttling load operation
  SHUTTLING_MOVE,      ///< Shuttling move operation
  SHUTTLING_STORE,     ///< Shuttling store operation
};

/**
 * @brief Checks if the operation type is a single-qubit operation.
 * @param type The operation type to check.
 * @return true if the operation type is a single-qubit operation, false
 * otherwise.
 */
[[nodiscard]] auto isSingleQubit(const OperationType type) -> bool {
  switch (type) {
  case OperationType::GLOBAL_SINGLE_QUBIT:
  case OperationType::LOCAL_SINGLE_QUBIT:
    return true;
  default:
    return false;
  }
}

/**
 * @brief Checks if the operation type is a local operation.
 * @param type The operation type to check.
 * @return true if the operation type is a local operation, false
 * otherwise.
 */
[[nodiscard]] auto isLocal(const OperationType type) -> bool {
  switch (type) {
  case OperationType::LOCAL_SINGLE_QUBIT:
  case OperationType::LOCAL_MULTI_QUBIT:
    return true;
  default:
    return false;
  }
}

/**
 * @brief Checks if the operation type is a shuttling operation.
 * @param type The operation type to check.
 * @return true if the operation type is a shuttling operation, false
 * otherwise.
 */
[[nodiscard]] auto isShuttling(const OperationType type) -> bool {
  switch (type) {
  case OperationType::SHUTTLING_LOAD:
  case OperationType::SHUTTLING_MOVE:
  case OperationType::SHUTTLING_STORE:
    return true;
  default:
    return false;
  }
}
} // namespace

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Session structure.
 */
struct MQT_NA_QDMI_Device_Session_impl_d {
  SessionStatus status = SessionStatus::ALLOCATED;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Job structure.
 */
struct MQT_NA_QDMI_Device_Job_impl_d {};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Site structure.
 */
struct MQT_NA_QDMI_Site_impl_d {
  size_t id; ///< Unique identifier of the site
  int64_t x; ///< X coordinate of the site in the lattice
  int64_t y; ///< Y coordinate of the site in the lattice
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Operation structure.
 */
struct MQT_NA_QDMI_Operation_impl_d {
  std::string name;     ///< Name of the operation
  OperationType type;   ///< Type of the operation
  size_t numParameters; ///< Number of parameters for the operation
  /**
   * @brief Number of qubits involved in the operation
   * @note This number is only valid if the operation is a multi-qubit
   * operation.
   */
  size_t numQubits;
  double duration; ///< Duration of the operation in microseconds
  double fidelity; ///< Fidelity of the operation
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
      populateRepeatedFields(reflection->MutableMessage(message, field));
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

/**
 * @brief Indicates whether the device has been initialized.
 * @details This function uses a static variable to track the initialization
 * state of the device. It is initialized to false and set to true after the
 * first successful initialization in @ref initialize.
 * @return A reference to a boolean indicating the initialization state.
 */
[[nodiscard]] auto initialized() -> bool& {
  static bool initialized = false;
  return initialized;
}

/**
 * @brief Provides access to the device name.
 * @returns a reference to a static variables that stores the
 * device name.
 */
[[nodiscard]] auto name() -> std::string& {
  static std::string name;
  return name;
}

/**
 * @brief Provides access to the lists of sites and operations.
 * @returns a reference to a static vector of unique pointers to
 * @ref MQT_NA_QDMI_Site_impl_d.
 */
[[nodiscard]] auto sites()
    -> std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>>& {
  static std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>> sites;
  return sites;
}

/**
 * @brief Provides access to the list of operations.
 * @returns a reference to a static vector of unique pointers to
 * @ref MQT_NA_QDMI_Operation_impl_d.
 */
[[nodiscard]] auto operations()
    -> std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>>& {
  static std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>> operations;
  return operations;
}

/// Collects decoherence times for the device.
struct DecoherenceTimes {
  double t1; ///< T1 time in microseconds
  double t2; ///< T2 time in microseconds
};

/**
 * @brief Provides access to the decoherence times of the device.
 * @details This function returns a reference to a static instance of
 * @ref DecoherenceTimes, which is used to store the decoherence times for the
 * device.
 * @returns A reference to a static instance of @ref DecoherenceTimes.
 */
[[nodiscard]] auto decoherence() -> DecoherenceTimes& {
  static DecoherenceTimes decoherenceTimes;
  return decoherenceTimes;
}

/**
 * @brief Provides access to the time factor for the device.
 * @details This function returns a reference to a static variable that stores
 * the time factor for the device. The time factor is used to convert time
 * values from the device's time unit to microseconds.
 * @returns A reference to a static double variable that stores the time factor.
 */
[[nodiscard]] auto timeFactor() -> double& {
  static double timeFactor;
  return timeFactor;
}

/**
 * @brief Provides access to the length factor for the device.
 * @details This function returns a reference to a static variable that stores
 * the length factor for the device. The length factor is used to convert length
 * values from the device's length unit to micrometers.
 * @returns A reference to a static double variable that stores the length
 * factor.
 */
[[nodiscard]] auto lengthFactor() -> double& {
  static double lengthFactor;
  return lengthFactor;
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
  for (; i < indices.size() && indices[i] == limits[i]; ++i) {
  }
  if (i == indices.size()) {
    // all indices are at their limits
    return false;
  }
  for (size_t j = 0; j < i; ++j) {
    indices[j] = 0; // Reset all previous indices
  }
  ++indices[i]; // Increment the next index
  return true;
}

/**
 * @brief Imports the name of the device from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 */
auto importName(const na::Device& device) -> void { name() = device.name(); }

/**
 * @brief Imports the sites from the Protobuf message into the device.
 * @param device The Protobuf message containing the device configuration.
 */
auto importSites(const na::Device& device) -> void {
  size_t count = 0;
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
        site->id = count++;
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
}

/**
 * @brief Imports the operations from the Protobuf message into the device.
 * @param device The Protobuf message containing the device configuration.
 */
auto importOperations(const na::Device& device) -> void {
  for (const auto& operation : device.global_single_qubit_operations()) {
    auto& op = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    op->name = operation.name();
    op->type = OperationType::GLOBAL_SINGLE_QUBIT;
    op->numParameters = operation.num_parameters();
    op->numQubits = 1;
    op->duration = operation.duration() * timeFactor();
    op->fidelity = operation.fidelity();
  }
  for (const auto& operation : device.global_multi_qubit_operations()) {
    auto& op = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    op->name = operation.name();
    op->type = OperationType::GLOBAL_MULTI_QUBIT;
    op->numParameters = operation.num_parameters();
    op->numQubits = operation.num_qubits();
    op->duration = operation.duration() * timeFactor();
    op->fidelity = operation.fidelity();
  }
  for (const auto& operation : device.local_single_qubit_operations()) {
    auto& op = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    op->name = operation.name();
    op->type = OperationType::LOCAL_SINGLE_QUBIT;
    op->numParameters = operation.num_parameters();
    op->numQubits = 1;
    op->duration = operation.duration() * timeFactor();
    op->fidelity = operation.fidelity();
  }
  for (const auto& operation : device.local_multi_qubit_operations()) {
    auto& op = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    op->name = operation.name();
    op->type = OperationType::LOCAL_MULTI_QUBIT;
    op->numParameters = operation.num_parameters();
    op->numQubits = operation.num_qubits();
    op->duration = operation.duration() * timeFactor();
    op->fidelity = operation.fidelity();
  }
  for (const auto& operation : device.shuttling_units()) {
    auto& load = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    load->name = operation.name();
    load->type = OperationType::SHUTTLING_LOAD;
    load->numParameters = operation.num_parameters();
    load->duration = operation.load_duration() * timeFactor();
    load->fidelity = operation.load_fidelity();
    auto& move = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    move->name = operation.name();
    move->type = OperationType::SHUTTLING_MOVE;
    move->numParameters = operation.num_parameters();
    auto& store = operations().emplace_back(
        std::make_unique<MQT_NA_QDMI_Operation_impl_d>());
    store->name = operation.name();
    store->type = OperationType::SHUTTLING_STORE;
    store->numParameters = operation.num_parameters();
    store->duration = operation.store_duration() * timeFactor();
    store->fidelity = operation.store_fidelity();
  }
}

/**
 * @brief Imports the decoherence times from the Protobuf message into the
 * device.
 * @param device The Protobuf message containing the device configuration.
 */
auto importDecoherenceTimes(const na::Device& device) -> void {
  decoherence().t1 = device.decoherence_times().t1() * timeFactor();
  decoherence().t2 = device.decoherence_times().t2() * timeFactor();
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
  // Initialize units
  timeFactor() = device.time_unit().value();
  if (device.time_unit().unit() == "ns") {
    timeFactor() *= 1e-3;
  } else if (device.time_unit().unit() != "us") {
    std::stringstream ss;
    ss << "Unsupported time unit: " << device.time_unit().unit();
    throw std::runtime_error(ss.str());
  }
  lengthFactor() = device.length_unit().value();
  if (device.length_unit().unit() == "nm") {
    lengthFactor() *= 1e-3;
  } else if (device.length_unit().unit() != "um") {
    std::stringstream ss;
    ss << "Unsupported length unit: " << device.length_unit().unit();
    throw std::runtime_error(ss.str());
  }
  // Transfer all data from the protobuf message to the device
  importName(device);
  importSites(device);
  importOperations(device);
  importDecoherenceTimes(device);
  // Set initialized to true to avoid re-initialization
  initialized() = true;
}
} // namespace

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

int MQT_NA_QDMI_device_initialize() {
  try {
    initialize();
  } catch (const std::runtime_error& e) {
    SPDLOG_ERROR(e.what());
    return QDMI_ERROR_FATAL;
  }
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_finalize() {
  initialized() = false;
  return QDMI_SUCCESS;
}

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
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = SessionStatus::INITIALIZED;
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
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_create_device_job(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Device_Job* job) {
  if (session == nullptr || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::INITIALIZED) {
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

int MQT_NA_QDMI_device_session_query_device_property(
    MQT_NA_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* size_ret) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_DEVICE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::INITIALIZED) {
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
  if (session == nullptr || site == nullptr ||
      (value != nullptr && size == 0) || prop >= QDMI_SITE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_ID, uint64_t, site->id, prop,
                            size, value, size_ret)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T1, double, decoherence().t1,
                            prop, size, value, size_ret)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T2, double, decoherence().t2,
                            prop, size, value, size_ret)
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_query_operation_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Operation operation,
    const size_t num_sites, const MQT_NA_QDMI_Site* sites,
    const size_t num_params, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* size_ret) {
  if (session == nullptr || operation == nullptr ||
      (sites != nullptr && num_sites == 0) ||
      (params != nullptr && num_params == 0) ||
      (value != nullptr && size == 0) || prop >= QDMI_OPERATION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, operation->name.c_str(),
                      prop, size, value, size_ret)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            operation->numParameters, prop, size, value,
                            size_ret)
  if (operation->type != OperationType::SHUTTLING_MOVE) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_DURATION, double,
                              operation->duration, prop, size, value, size_ret)
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double,
                              operation->fidelity, prop, size, value, size_ret)
  }
  if (!isShuttling(operation->type)) {
    if (isSingleQubit(operation->type)) {
      ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t, 1UL,
                                prop, size, value, size_ret)
    } else {
      ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                                operation->numQubits, prop, size, value,
                                size_ret)
    }
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
