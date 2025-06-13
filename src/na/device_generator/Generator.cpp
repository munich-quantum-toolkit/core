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

#include "na/device_generator/Generator.hpp"

#include "na/device_generator/device.pb.h"

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

namespace na {
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

[[nodiscard]] auto getTimeUnit(const Device& device) -> double {
  if (device.time_unit().unit() == "us") {
    return static_cast<double>(device.time_unit().value());
  }
  if (device.time_unit().unit() == "ns") {
    return static_cast<double>(device.time_unit().value()) * 1e-3;
  }
  std::stringstream ss;
  ss << "Unsupported time unit: " << device.time_unit().unit();
  throw std::runtime_error(ss.str());
}

[[nodiscard]] auto getLengthUnit(const Device& device) -> double {
  if (device.length_unit().unit() == "um") {
    return static_cast<double>(device.length_unit().value());
  }
  if (device.length_unit().unit() == "nm") {
    return static_cast<double>(device.length_unit().value()) * 1e-3;
  }
  std::stringstream ss;
  ss << "Unsupported length unit: " << device.length_unit().unit();
  throw std::runtime_error(ss.str());
}

/**
 * @brief Writes the name from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 */
auto writeName(const Device& device, std::ostream& os) -> void {
  os << "#define DEVICE_NAME \"" << device.name() << "\"\n";
}

/**
 * @brief Writes the sites from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 */
auto writeSites(const Device& device, std::ostream& os) -> void {
  size_t count = 0;
  os << "#define SITES std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>>{";
  for (const auto& lattice : device.traps()) {
    const auto originX = lattice.lattice_origin().x();
    const auto originY = lattice.lattice_origin().y();
    std::vector limits(static_cast<size_t>(lattice.lattice_vectors_size()),
                       0UL);
    std::transform(lattice.lattice_vectors().begin(),
                   lattice.lattice_vectors().end(), limits.begin(),
                   [](const auto& vector) { return vector.repeat(); });
    std::vector indices(static_cast<size_t>(lattice.lattice_vectors_size()),
                        0UL);
    do {
      // For every sublattice offset, add a site for repetition indices
      for (const auto& offset : lattice.sublattice_offsets()) {
        const auto id = count++;
        auto x = originX + offset.x();
        auto y = originY + offset.y();
        for (size_t i = 0;
             i < static_cast<size_t>(lattice.lattice_vectors_size()); ++i) {
          const auto& vector =
              lattice.lattice_vectors(static_cast<int32_t>(i)).vector();
          x += static_cast<int64_t>(indices[i]) * vector.x();
          y += static_cast<int64_t>(indices[i]) * vector.y();
        }
        if (id > 0) {
          os << ",\\\n";
        }
        os << "  std::make_unique<MQT_NA_QDMI_Site_impl_d>("
              "MQT_NA_QDMI_Site_impl_d{"
           << id << ", " << x << ", " << y << "})";
      }
    } while (increment(indices, limits));
  }
  os << "\\\n}\n";
}

/**
 * @brief Imports the operations from the Protobuf message into the device.
 * @param device The Protobuf message containing the device configuration.
 */
auto writeOperations(const Device& device, const double timeUnit,
                     std::ostream& os) -> void {
  os << "#define OPERATIONS "
        "std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>>{";
  bool first = true;
  for (const auto& operation : device.global_single_qubit_operations()) {
    if (first) {
      os << ",\\\n";
      first = false;
    }
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::GLOBAL_SINGLE_QUBIT, "
       << operation.num_parameters() << ", 1, "
       << static_cast<double>(operation.duration()) * timeUnit << ", "
       << operation.fidelity() << "})";
  }
  for (const auto& operation : device.global_multi_qubit_operations()) {
    if (first) {
      os << ",\\\n";
      first = false;
    }
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::GLOBAL_MULTI_QUBIT, "
       << operation.num_parameters() << ", " << operation.num_qubits() << ", "
       << static_cast<double>(operation.duration()) * timeUnit << ", "
       << operation.fidelity() << "})";
  }
  for (const auto& operation : device.local_single_qubit_operations()) {
    if (first) {
      os << ",\\\n";
      first = false;
    }
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::LOCAL_SINGLE_QUBIT, "
       << operation.num_parameters() << ", 1, "
       << static_cast<double>(operation.duration()) * timeUnit << ", "
       << operation.fidelity() << "})";
  }
  for (const auto& operation : device.local_multi_qubit_operations()) {
    if (first) {
      os << ",\\\n";
      first = false;
    }
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::LOCAL_MULTI_QUBIT, "
       << operation.num_parameters() << ", " << operation.num_qubits() << ", "
       << static_cast<double>(operation.duration()) * timeUnit << ", "
       << operation.fidelity() << "})";
  }
  for (const auto& operation : device.shuttling_units()) {
    if (first) {
      os << ",\\\n";
      first = false;
    }
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::SHUTTLING_LOAD, "
       << operation.num_parameters() << ", 0, "
       << static_cast<double>(operation.load_duration()) * timeUnit << ", "
       << operation.load_fidelity() << "})";
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::SHUTTLING_MOVE, "
       << operation.num_parameters() << ", 0, 0, 0})";
    os << "  std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{"
       << operation.name() << ", OperationType::SHUTTLING_STORE, "
       << operation.num_parameters() << ", 0, "
       << static_cast<double>(operation.store_duration()) * timeUnit << ", "
       << operation.store_fidelity() << "})";
  }
  os << "\\\n}\n";
}

/**
 * @brief Writes the decoherence times from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 */
auto writeDecoherenceTimes(const Device& device, const double timeUnit,
                           std::ostream& os) -> void {
  os << "#define T1 "
     << static_cast<double>(device.decoherence_times().t1()) * timeUnit << "\n";
  os << "#define T2 "
     << static_cast<double>(device.decoherence_times().t2()) * timeUnit << "\n";
}
} // namespace

auto writeJsonSchema(const std::string& path) -> void {
  // Create a default device configuration
  Device device;

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

[[nodiscard]] auto readJsonFile(const std::string& path) -> Device {
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
  Device device;
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

auto writeHeaderFile(const Device& device, const std::string& path) -> void {
  std::ofstream ofs(path);
  ofs << "#pragma once\n\n";
  const auto timeUnit = getTimeUnit(device);
  writeName(device, ofs);
  writeSites(device, ofs);
  writeOperations(device, timeUnit, ofs);
  writeDecoherenceTimes(device, timeUnit, ofs);
}
} // namespace na
