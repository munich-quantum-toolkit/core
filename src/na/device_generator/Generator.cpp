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
 * @brief The MQT QDMI device generator for neutral atom devices.
 */

#include "na/device/Generator.hpp"

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
 * @brief Parses the device configuration from a JSON file specified by the
 * environment variable MQT_CORE_NA_QDMI_DEVICE_JSON_FILE.
 * @returns The parsed device configuration as a Protobuf message.
 * @throws std::runtime_error if the environment variable is not set, the JSON
 * file does not exist, or the JSON file cannot be parsed.
 */
[[nodiscard]] auto parse(const std::string& path) -> na::Device {
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
} // namespace

namespace na {
auto writeHeaderFile(const std::string& json, const std::string& path) -> void {
  const auto& device = parse(json);
  std::ofstream ofs(path);
  ofs << "#pragma once\n\n";
  ofs << "#define DEVICE_NAME \"" << device.name() << "\"\n";
}
} // namespace na
