/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Configuration.cpp
/// Runtime configuration parsing for superconducting QDMI devices.

#include "qdmi/devices/sc/Configuration.hpp"

#include "qdmi/common/Common.hpp"

#include "JSON.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "qdmi/constants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <istream>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sc {
using qdmi::Error;
using qdmi::Result;
namespace {
using Json = nlohmann::json;

// Keep all schema diagnostics anchored to the selected configuration source.
[[nodiscard]] Error fail(const std::string_view source,
                         const std::string_view pointer,
                         const std::string_view message) {
  return Error{
      .status = QDMI_ERROR_INVALIDARGUMENT,
      .message = std::string(source) + ":" + std::string(pointer) + " " +
                 std::string(message),
  };
}

std::optional<Error> object(const Json& value, const std::string_view source,
                            const std::string_view pointer) {
  if (!value.is_object()) {
    return fail(source, pointer, "must be an object");
  }
  return std::nullopt;
}

std::optional<Error> keys(const Json& value,
                          const std::initializer_list<std::string_view> known,
                          const std::string_view source,
                          const std::string_view pointer) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(known, key) == known.end()) {
      return fail(source, pointer, "contains unknown key '" + key + "'");
    }
  }
  return std::nullopt;
}

template <class T>
[[nodiscard]] Result<T> required(const Json& value, const std::string& key,
                                 const std::string_view source,
                                 const std::string& pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    return fail(source, pointer + "/" + key, "is required");
  }
  if constexpr (std::is_same_v<T, uint64_t>) {
    if (!found->is_number_unsigned() &&
        (!found->is_number_integer() || found->get<int64_t>() < 0)) {
      return fail(source, pointer + "/" + key,
                  "must be a non-negative integer");
    }
  }
  if constexpr (std::is_same_v<T, std::string>) {
    if (!found->is_string()) {
      return fail(source, pointer + "/" + key, "has an invalid type");
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if (!found->is_number()) {
      return fail(source, pointer + "/" + key, "has an invalid type");
    }
  }
  return found->get<T>();
}

template <class T>
[[nodiscard]] Result<std::optional<T>>
optional(const Json& value, const std::string& key,
         const std::string_view source, const std::string& pointer) {
  if (!value.contains(key)) {
    return std::nullopt;
  }
  auto result = required<T>(value, key, source, pointer);
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  return std::get<0>(std::move(result));
}

[[nodiscard]] Result<std::vector<uint64_t>>
indices(const Json& value, const std::string_view source,
        const std::string& pointer) {
  if (!value.is_array()) {
    return fail(source, pointer, "must be an array of unsigned integers");
  }
  std::vector<uint64_t> result;
  result.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    const auto& item = value[i];
    if (!item.is_number_unsigned() &&
        (!item.is_number_integer() || item.get<int64_t>() < 0)) {
      return fail(source, pointer + "/" + std::to_string(i),
                  "must be a non-negative integer");
    }
    result.emplace_back(item.get<uint64_t>());
  }
  return result;
}

[[nodiscard]] Result<Device::QubitCalibration>
calibration(const Json& value, const std::string_view source,
            const std::string& pointer) {
  if (auto error = object(value, source, pointer)) {
    return std::move(*error);
  }
  if (auto error = keys(value, {"t1", "t2"}, source, pointer)) {
    return std::move(*error);
  }
  Device::QubitCalibration result;
  auto resultT1Result = optional<uint64_t>(value, "t1", source, pointer);
  if (auto* error = std::get_if<Error>(&resultT1Result)) {
    return std::move(*error);
  }
  result.t1 = std::get<0>(std::move(resultT1Result));
  auto resultT2Result = optional<uint64_t>(value, "t2", source, pointer);
  if (auto* error = std::get_if<Error>(&resultT2Result)) {
    return std::move(*error);
  }
  result.t2 = std::get<0>(std::move(resultT2Result));
  if ((result.t1 && *result.t1 == 0) || (result.t2 && *result.t2 == 0)) {
    return fail(source, pointer, "t1 and t2 must be positive when present");
  }
  return result;
}

std::optional<Error> validateFidelity(const std::optional<double>& fidelity,
                                      const std::string_view source,
                                      const std::string& pointer) {
  if (fidelity &&
      (!std::isfinite(*fidelity) || *fidelity < 0. || *fidelity > 1.)) {
    return fail(source, pointer, "must be finite and in [0, 1]");
  }
  return std::nullopt;
}

[[nodiscard]] Result<Device> parse(const Json& root,
                                   const std::string_view source) {
  if (auto error = object(root, source, "$")) {
    return std::move(*error);
  }
  if (auto error = keys(root,
                        {
                            "schema-version",
                            "name",
                            "numQubits",
                            "durationUnit",
                            "qubitProperties",
                            "couplings",
                            "operations",
                        },
                        source, "$")) {
    return std::move(*error);
  }
  for (const auto* const key : {
           "schema-version",
           "name",
           "numQubits",
           "durationUnit",
           "qubitProperties",
           "couplings",
           "operations",
       }) {
    if (!root.contains(key)) {
      return fail(source, "$/" + std::string(key), "is required");
    }
  }
  Device result;
  auto resultSchemaVersionResult =
      required<uint64_t>(root, "schema-version", source, "$");
  if (auto* error = std::get_if<Error>(&resultSchemaVersionResult)) {
    return std::move(*error);
  }
  result.schemaVersion = std::get<0>(std::move(resultSchemaVersionResult));
  if (result.schemaVersion != 1) {
    return fail(source, "$/schema-version", "must be 1");
  }
  auto resultNameResult = required<std::string>(root, "name", source, "$");
  if (auto* error = std::get_if<Error>(&resultNameResult)) {
    return std::move(*error);
  }
  result.name = std::get<0>(std::move(resultNameResult));
  if (result.name.empty() || result.name.find('\0') != std::string::npos) {
    return fail(source, "$/name", "must be non-empty and contain no NUL bytes");
  }
  auto resultNumQubitsResult =
      required<uint64_t>(root, "numQubits", source, "$");
  if (auto* error = std::get_if<Error>(&resultNumQubitsResult)) {
    return std::move(*error);
  }
  result.numQubits = std::get<0>(std::move(resultNumQubitsResult));
  if (result.numQubits == 0 ||
      result.numQubits > std::vector<void*>{}.max_size()) {
    return fail(source, "$/numQubits", "must be positive and representable");
  }

  const auto& unit = root["durationUnit"];
  if (auto error = object(unit, source, "$/durationUnit")) {
    return std::move(*error);
  }
  if (auto error =
          keys(unit, {"unit", "scaleFactor"}, source, "$/durationUnit")) {
    return std::move(*error);
  }
  auto resultDurationUnitUnitResult =
      required<std::string>(unit, "unit", source, "$/durationUnit");
  if (auto* error = std::get_if<Error>(&resultDurationUnitUnitResult)) {
    return std::move(*error);
  }
  result.durationUnit.unit =
      std::get<0>(std::move(resultDurationUnitUnitResult));
  auto resultDurationUnitScaleFactorResult =
      required<double>(unit, "scaleFactor", source, "$/durationUnit");
  if (auto* error = std::get_if<Error>(&resultDurationUnitScaleFactorResult)) {
    return std::move(*error);
  }
  result.durationUnit.scaleFactor =
      std::get<0>(std::move(resultDurationUnitScaleFactorResult));
  constexpr std::array supportedUnits{"s", "ms", "us", "ns"};
  if (std::ranges::find(supportedUnits, result.durationUnit.unit) ==
          supportedUnits.end() ||
      !std::isfinite(result.durationUnit.scaleFactor) ||
      result.durationUnit.scaleFactor <= 0.) {
    return fail(source, "$/durationUnit",
                "must use s, ms, us, or ns and a positive finite scaleFactor");
  }

  {
    const auto& properties = root["qubitProperties"];
    if (auto error = object(properties, source, "$/qubitProperties")) {
      return std::move(*error);
    }
    if (auto error = keys(properties, {"defaults", "overrides"}, source,
                          "$/qubitProperties")) {
      return std::move(*error);
    }
    for (const auto* const key : {"defaults", "overrides"}) {
      if (!properties.contains(key)) {
        return fail(source, "$/qubitProperties/" + std::string(key),
                    "is required");
      }
    }
    auto resultQubitPropertiesDefaultsResult = calibration(
        properties["defaults"], source, "$/qubitProperties/defaults");
    if (auto* error =
            std::get_if<Error>(&resultQubitPropertiesDefaultsResult)) {
      return std::move(*error);
    }
    result.qubitProperties.defaults =
        std::get<0>(std::move(resultQubitPropertiesDefaultsResult));
    {
      const auto& overrides = properties["overrides"];
      if (!overrides.is_array()) {
        return fail(source, "$/qubitProperties/overrides", "must be an array");
      }
      std::set<uint64_t> overridden;
      for (size_t i = 0; i < overrides.size(); ++i) {
        const auto pointer = "$/qubitProperties/overrides/" + std::to_string(i);
        const auto& value = overrides[i];
        if (auto error = object(value, source, pointer)) {
          return std::move(*error);
        }
        if (auto error =
                keys(value, {"qubit", "name", "t1", "t2"}, source, pointer)) {
          return std::move(*error);
        }
        Device::QubitOverride entry;
        auto entryQubitResult =
            required<uint64_t>(value, "qubit", source, pointer);
        if (auto* error = std::get_if<Error>(&entryQubitResult)) {
          return std::move(*error);
        }
        entry.qubit = std::get<0>(std::move(entryQubitResult));
        auto entryNameResult =
            optional<std::string>(value, "name", source, pointer);
        if (auto* error = std::get_if<Error>(&entryNameResult)) {
          return std::move(*error);
        }
        entry.name = std::get<0>(std::move(entryNameResult));
        auto entryT1Result = optional<uint64_t>(value, "t1", source, pointer);
        if (auto* error = std::get_if<Error>(&entryT1Result)) {
          return std::move(*error);
        }
        entry.t1 = std::get<0>(std::move(entryT1Result));
        auto entryT2Result = optional<uint64_t>(value, "t2", source, pointer);
        if (auto* error = std::get_if<Error>(&entryT2Result)) {
          return std::move(*error);
        }
        entry.t2 = std::get<0>(std::move(entryT2Result));
        if (entry.qubit >= result.numQubits ||
            (entry.name && (entry.name->empty() ||
                            entry.name->find('\0') != std::string::npos)) ||
            (entry.t1 && *entry.t1 == 0) || (entry.t2 && *entry.t2 == 0) ||
            (!entry.name && !entry.t1 && !entry.t2) ||
            !overridden.emplace(entry.qubit).second) {
          return fail(
              source, pointer,
              "must select one unique valid qubit and override name, t1, or "
              "t2 with valid values");
        }
        result.qubitProperties.overrides.emplace_back(entry);
      }
    }
  }

  const auto& couplings = root["couplings"];
  if (!couplings.is_array()) {
    return fail(source, "$/couplings", "must be an array");
  }
  std::set<std::pair<uint64_t, uint64_t>> uniqueCouplings;
  for (size_t i = 0; i < couplings.size(); ++i) {
    const auto pointer = "$/couplings/" + std::to_string(i);
    if (!couplings[i].is_array() || couplings[i].size() != 2) {
      return fail(source, pointer, "must contain exactly two qubit indices");
    }
    std::pair<uint64_t, uint64_t> coupling;
    auto parsedResult = indices(couplings[i], source, pointer);
    if (auto* error = std::get_if<Error>(&parsedResult)) {
      return std::move(*error);
    }
    auto& parsed = std::get<0>(parsedResult);
    coupling = {parsed[0], parsed[1]};
    if (coupling.first >= result.numQubits ||
        coupling.second >= result.numQubits ||
        coupling.first == coupling.second ||
        !uniqueCouplings.emplace(coupling).second) {
      return fail(source, pointer,
                  "must be a unique, non-self tuple of valid qubits");
    }
    result.couplings.emplace_back(coupling);
  }

  const auto& operations = root["operations"];
  if (!operations.is_array()) {
    return fail(source, "$/operations", "must be an array");
  }
  std::set<std::string> names;
  for (size_t i = 0; i < operations.size(); ++i) {
    const auto pointer = "$/operations/" + std::to_string(i);
    const auto& value = operations[i];
    if (auto error = object(value, source, pointer)) {
      return std::move(*error);
    }
    if (auto error = keys(value,
                          {
                              "name",
                              "numParameters",
                              "numQubits",
                              "sites",
                              "duration",
                              "fidelity",
                              "siteOverrides",
                          },
                          source, pointer)) {
      return std::move(*error);
    }
    Device::Operation operation;
    auto operationNameResult =
        required<std::string>(value, "name", source, pointer);
    if (auto* error = std::get_if<Error>(&operationNameResult)) {
      return std::move(*error);
    }
    operation.name = std::get<0>(std::move(operationNameResult));
    auto operationNumParametersResult =
        required<uint64_t>(value, "numParameters", source, pointer);
    if (auto* error = std::get_if<Error>(&operationNumParametersResult)) {
      return std::move(*error);
    }
    operation.numParameters =
        std::get<0>(std::move(operationNumParametersResult));
    auto operationNumQubitsResult =
        required<uint64_t>(value, "numQubits", source, pointer);
    if (auto* error = std::get_if<Error>(&operationNumQubitsResult)) {
      return std::move(*error);
    }
    operation.numQubits = std::get<0>(std::move(operationNumQubitsResult));
    auto operationDurationResult =
        optional<uint64_t>(value, "duration", source, pointer);
    if (auto* error = std::get_if<Error>(&operationDurationResult)) {
      return std::move(*error);
    }
    operation.duration = std::get<0>(std::move(operationDurationResult));
    auto operationFidelityResult =
        optional<double>(value, "fidelity", source, pointer);
    if (auto* error = std::get_if<Error>(&operationFidelityResult)) {
      return std::move(*error);
    }
    operation.fidelity = std::get<0>(std::move(operationFidelityResult));
    if (auto error = validateFidelity(operation.fidelity, source,
                                      pointer + "/fidelity")) {
      return std::move(*error);
    }
    if (operation.name.empty() ||
        operation.name.find('\0') != std::string::npos ||
        operation.numQubits == 0 || operation.numQubits > result.numQubits ||
        operation.numParameters > std::numeric_limits<size_t>::max() ||
        !names.emplace(operation.name).second) {
      return fail(source, pointer,
                  "must have a unique non-empty name without NUL bytes and "
                  "representable counts");
    }
    std::set<std::vector<uint64_t>> uniqueSites;
    if (const auto sites = value.find("sites"); sites != value.end()) {
      if (!sites->is_array()) {
        return fail(source, pointer + "/sites", "must be an array");
      }
      operation.sites.emplace();
      for (size_t j = 0; j < sites->size(); ++j) {
        auto tupleResult = indices((*sites)[j], source,
                                   pointer + "/sites/" + std::to_string(j));
        if (auto* error = std::get_if<Error>(&tupleResult)) {
          return std::move(*error);
        }
        auto& tuple = std::get<0>(tupleResult);
        const auto supportedByTopology =
            operation.numQubits != 2 ||
            (tuple.size() == 2 &&
             uniqueCouplings.contains(std::pair{tuple[0], tuple[1]}));
        if (const std::set<uint64_t> tupleSites(tuple.begin(), tuple.end());
            tuple.size() != operation.numQubits ||
            tupleSites.size() != tuple.size() ||
            std::ranges::any_of(
                tuple,
                [&](const auto qubit) { return qubit >= result.numQubits; }) ||
            !supportedByTopology || !uniqueSites.emplace(tuple).second) {
          return fail(source, pointer + "/sites/" + std::to_string(j),
                      "must be a unique tuple matching the operation arity and "
                      "device connectivity");
        }
        operation.sites->emplace_back(std::move(tuple));
      }
    }
    if (!operation.sites && operation.numQubits > 2) {
      return fail(source, pointer + "/sites",
                  "is required for operations with arity greater than two");
    }
    if (const auto overrides = value.find("siteOverrides");
        overrides != value.end()) {
      if (!overrides->is_array()) {
        return fail(source, pointer + "/siteOverrides", "must be an array");
      }
      std::set<std::vector<uint64_t>> overriddenSites;
      for (size_t j = 0; j < overrides->size(); ++j) {
        const auto overridePointer =
            pointer + "/siteOverrides/" + std::to_string(j);
        const auto& overrideJson = (*overrides)[j];
        if (auto error = object(overrideJson, source, overridePointer)) {
          return std::move(*error);
        }
        if (auto error = keys(overrideJson, {"sites", "duration", "fidelity"},
                              source, overridePointer)) {
          return std::move(*error);
        }
        Device::SiteOverride override;
        const auto siteValues = overrideJson.find("sites");
        if (siteValues == overrideJson.end()) {
          return fail(source, overridePointer + "/sites", "is required");
        }
        auto overrideSitesResult =
            indices(*siteValues, source, overridePointer + "/sites");
        if (auto* error = std::get_if<Error>(&overrideSitesResult)) {
          return std::move(*error);
        }
        override.sites = std::get<0>(std::move(overrideSitesResult));
        auto overrideDurationResult = optional<uint64_t>(
            overrideJson, "duration", source, overridePointer);
        if (auto* error = std::get_if<Error>(&overrideDurationResult)) {
          return std::move(*error);
        }
        override.duration = std::get<0>(std::move(overrideDurationResult));
        auto overrideFidelityResult =
            optional<double>(overrideJson, "fidelity", source, overridePointer);
        if (auto* error = std::get_if<Error>(&overrideFidelityResult)) {
          return std::move(*error);
        }
        override.fidelity = std::get<0>(std::move(overrideFidelityResult));
        if (auto error = validateFidelity(override.fidelity, source,
                                          overridePointer + "/fidelity")) {
          return std::move(*error);
        }
        const std::set<uint64_t> tupleSites(override.sites.begin(),
                                            override.sites.end());
        auto supported = false;
        if (override.sites.size() == operation.numQubits) {
          if (operation.sites) {
            supported = uniqueSites.contains(override.sites);
          } else if (operation.numQubits == 1) {
            supported = true;
          } else if (operation.numQubits == 2) {
            supported = uniqueCouplings.contains(
                std::pair{override.sites[0], override.sites[1]});
          }
        }
        if (override.sites.size() != operation.numQubits ||
            tupleSites.size() != override.sites.size() ||
            std::ranges::any_of(
                override.sites,
                [&](const auto qubit) { return qubit >= result.numQubits; }) ||
            !supported || !overriddenSites.emplace(override.sites).second ||
            (!override.duration && !override.fidelity)) {
          return fail(
              source, overridePointer,
              "must be one unique supported tuple and override a valid value");
        }
        operation.siteOverrides.emplace_back(std::move(override));
      }
    }
    result.operations.emplace_back(std::move(operation));
  }
  return result;
}
} // namespace

Result<Device> readJSON(const std::string_view json,
                        const std::string_view source) {
  auto result = mqt::detail::parseJSON(json);
  if (auto const* error = std::get_if<std::string>(&result)) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = std::string(source) + ": invalid JSON: " + *error,
    };
  }
  return parse(std::get<0>(result), source);
}

Result<Device> readJSON(std::istream& stream, const std::string_view source) {
  const std::string json{std::istreambuf_iterator<char>(stream), {}};
  if (stream.bad()) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Failed to read JSON: " + std::string(source),
    };
  }
  return readJSON(json, source);
}

Result<Device> readJSON(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    return Error{
        .status = QDMI_ERROR_NOTFOUND,
        .message = "Failed to open JSON file: " + path,
    };
  }
  return readJSON(input, path);
}

} // namespace sc
