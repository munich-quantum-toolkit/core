/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/devices/sc/Configuration.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace sc {
namespace {
using Json = nlohmann::json;

// Keep all schema diagnostics anchored to the selected configuration source.
[[noreturn]] void fail(const std::string_view source,
                       const std::string_view pointer,
                       const std::string_view message) {
  throw std::invalid_argument(std::string(source) + ":" + std::string(pointer) +
                              " " + std::string(message));
}

void object(const Json& value, const std::string_view source,
            const std::string_view pointer) {
  if (!value.is_object()) {
    fail(source, pointer, "must be an object");
  }
}

void keys(const Json& value,
          const std::initializer_list<std::string_view> known,
          const std::string_view source, const std::string_view pointer) {
  const std::set<std::string_view> allowed(known);
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (!allowed.contains(key)) {
      fail(source, pointer, "contains unknown key '" + key + "'");
    }
  }
}

template <class T>
[[nodiscard]] T required(const Json& value, const std::string& key,
                         const std::string_view source,
                         const std::string& pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    fail(source, pointer + "/" + key, "is required");
  }
  if constexpr (std::is_same_v<T, uint64_t>) {
    if (!found->is_number_unsigned() &&
        (!found->is_number_integer() || found->get<int64_t>() < 0)) {
      fail(source, pointer + "/" + key, "must be a non-negative integer");
    }
  }
  try {
    return found->get<T>();
  } catch (const Json::exception&) {
    fail(source, pointer + "/" + key, "has an invalid type");
  }
}

template <class T>
[[nodiscard]] std::optional<T>
optional(const Json& value, const std::string& key,
         const std::string_view source, const std::string& pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    return std::nullopt;
  }
  if constexpr (std::is_same_v<T, uint64_t>) {
    if (!found->is_number_unsigned() &&
        (!found->is_number_integer() || found->get<int64_t>() < 0)) {
      fail(source, pointer + "/" + key, "must be a non-negative integer");
    }
  }
  try {
    return found->get<T>();
  } catch (const Json::exception&) {
    fail(source, pointer + "/" + key, "has an invalid type");
  }
}

[[nodiscard]] std::vector<uint64_t> indices(const Json& value,
                                            const std::string_view source,
                                            const std::string& pointer) {
  if (!value.is_array()) {
    fail(source, pointer, "must be an array of unsigned integers");
  }
  std::vector<uint64_t> result;
  result.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    const auto& item = value[i];
    if (!item.is_number_unsigned() &&
        (!item.is_number_integer() || item.get<int64_t>() < 0)) {
      fail(source, pointer + "/" + std::to_string(i),
           "must be a non-negative integer");
    }
    result.emplace_back(item.get<uint64_t>());
  }
  return result;
}

[[nodiscard]] Device::QubitCalibration
calibration(const Json& value, const std::string_view source,
            const std::string& pointer) {
  object(value, source, pointer);
  keys(value, {"t1", "t2"}, source, pointer);
  auto result = Device::QubitCalibration{
      .t1 = optional<uint64_t>(value, "t1", source, pointer),
      .t2 = optional<uint64_t>(value, "t2", source, pointer)};
  if ((result.t1 && *result.t1 == 0) || (result.t2 && *result.t2 == 0)) {
    fail(source, pointer, "t1 and t2 must be positive when present");
  }
  return result;
}

void validateFidelity(const std::optional<double>& fidelity,
                      const std::string_view source,
                      const std::string& pointer) {
  if (fidelity &&
      (!std::isfinite(*fidelity) || *fidelity < 0. || *fidelity > 1.)) {
    fail(source, pointer, "must be finite and in [0, 1]");
  }
}

[[nodiscard]] Device parse(const Json& root, const std::string_view source) {
  object(root, source, "$");
  keys(root,
       {"schema-version", "name", "numQubits", "durationUnit",
        "qubitProperties", "couplings", "operations"},
       source, "$");
  for (const auto* const key :
       {"schema-version", "name", "numQubits", "durationUnit",
        "qubitProperties", "couplings", "operations"}) {
    if (!root.contains(key)) {
      fail(source, "$/" + std::string(key), "is required");
    }
  }
  Device result;
  result.schemaVersion =
      required<uint64_t>(root, "schema-version", source, "$");
  if (result.schemaVersion != 1) {
    fail(source, "$/schema-version", "must be 1");
  }
  result.name = required<std::string>(root, "name", source, "$");
  if (result.name.empty()) {
    fail(source, "$/name", "must not be empty");
  }
  result.numQubits = required<uint64_t>(root, "numQubits", source, "$");
  if (result.numQubits == 0 ||
      result.numQubits > std::vector<void*>{}.max_size()) {
    fail(source, "$/numQubits", "must be positive and representable");
  }

  const auto& unit = root.at("durationUnit");
  object(unit, source, "$/durationUnit");
  keys(unit, {"unit", "scaleFactor"}, source, "$/durationUnit");
  result.durationUnit.unit =
      required<std::string>(unit, "unit", source, "$/durationUnit");
  result.durationUnit.scaleFactor =
      required<double>(unit, "scaleFactor", source, "$/durationUnit");
  constexpr std::array supportedUnits{"s", "ms", "us", "ns"};
  if (std::ranges::find(supportedUnits, result.durationUnit.unit) ==
          supportedUnits.end() ||
      !std::isfinite(result.durationUnit.scaleFactor) ||
      result.durationUnit.scaleFactor <= 0.) {
    fail(source, "$/durationUnit",
         "must use s, ms, us, or ns and a positive finite scaleFactor");
  }

  {
    const auto& properties = root.at("qubitProperties");
    object(properties, source, "$/qubitProperties");
    keys(properties, {"defaults", "overrides"}, source, "$/qubitProperties");
    for (const auto* const key : {"defaults", "overrides"}) {
      if (!properties.contains(key)) {
        fail(source, "$/qubitProperties/" + std::string(key), "is required");
      }
    }
    result.qubitProperties.defaults = calibration(
        properties.at("defaults"), source, "$/qubitProperties/defaults");
    {
      const auto& overrides = properties.at("overrides");
      if (!overrides.is_array()) {
        fail(source, "$/qubitProperties/overrides", "must be an array");
      }
      std::set<uint64_t> overridden;
      for (size_t i = 0; i < overrides.size(); ++i) {
        const auto pointer = "$/qubitProperties/overrides/" + std::to_string(i);
        const auto& value = overrides[i];
        object(value, source, pointer);
        keys(value, {"qubit", "t1", "t2"}, source, pointer);
        Device::QubitOverride entry;
        entry.qubit = required<uint64_t>(value, "qubit", source, pointer);
        entry.t1 = optional<uint64_t>(value, "t1", source, pointer);
        entry.t2 = optional<uint64_t>(value, "t2", source, pointer);
        if (entry.qubit >= result.numQubits || (entry.t1 && *entry.t1 == 0) ||
            (entry.t2 && *entry.t2 == 0) || (!entry.t1 && !entry.t2) ||
            !overridden.emplace(entry.qubit).second) {
          fail(source, pointer,
               "must select one unique valid qubit and override t1 or t2");
        }
        result.qubitProperties.overrides.emplace_back(entry);
      }
    }
  }

  const auto& couplings = root.at("couplings");
  if (!couplings.is_array()) {
    fail(source, "$/couplings", "must be an array");
  }
  std::set<std::pair<uint64_t, uint64_t>> uniqueCouplings;
  for (size_t i = 0; i < couplings.size(); ++i) {
    const auto pointer = "$/couplings/" + std::to_string(i);
    if (!couplings[i].is_array() || couplings[i].size() != 2) {
      fail(source, pointer, "must contain exactly two qubit indices");
    }
    std::pair<uint64_t, uint64_t> coupling;
    const auto parsed = indices(couplings[i], source, pointer);
    coupling = {parsed[0], parsed[1]};
    if (coupling.first >= result.numQubits ||
        coupling.second >= result.numQubits ||
        coupling.first == coupling.second ||
        !uniqueCouplings.emplace(coupling).second) {
      fail(source, pointer, "must be a unique, non-self tuple of valid qubits");
    }
    result.couplings.emplace_back(coupling);
  }

  const auto& operations = root.at("operations");
  if (!operations.is_array()) {
    fail(source, "$/operations", "must be an array");
  }
  std::set<std::string> names;
  for (size_t i = 0; i < operations.size(); ++i) {
    const auto pointer = "$/operations/" + std::to_string(i);
    const auto& value = operations[i];
    object(value, source, pointer);
    keys(value,
         {"name", "numParameters", "numQubits", "sites", "duration", "fidelity",
          "siteOverrides"},
         source, pointer);
    Device::Operation operation;
    operation.name = required<std::string>(value, "name", source, pointer);
    operation.numParameters =
        required<uint64_t>(value, "numParameters", source, pointer);
    operation.numQubits =
        required<uint64_t>(value, "numQubits", source, pointer);
    operation.duration = optional<uint64_t>(value, "duration", source, pointer);
    operation.fidelity = optional<double>(value, "fidelity", source, pointer);
    validateFidelity(operation.fidelity, source, pointer + "/fidelity");
    if (operation.name.empty() || operation.numQubits == 0 ||
        operation.numQubits > result.numQubits ||
        operation.numParameters > std::numeric_limits<size_t>::max() ||
        !names.emplace(operation.name).second) {
      fail(source, pointer,
           "must have a unique non-empty name and representable counts");
    }
    if (const auto sites = value.find("sites"); sites != value.end()) {
      if (!sites->is_array()) {
        fail(source, pointer + "/sites", "must be an array");
      }
      operation.sites.emplace();
      std::set<std::vector<uint64_t>> uniqueSites;
      for (size_t j = 0; j < sites->size(); ++j) {
        auto tuple = indices((*sites)[j], source,
                             pointer + "/sites/" + std::to_string(j));
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
          fail(source, pointer + "/sites/" + std::to_string(j),
               "must be a unique tuple matching the operation arity and "
               "device connectivity");
        }
        operation.sites->emplace_back(std::move(tuple));
      }
    }
    if (!operation.sites && operation.numQubits > 2) {
      fail(source, pointer + "/sites",
           "is required for operations with arity greater than two");
    }
    if (const auto overrides = value.find("siteOverrides");
        overrides != value.end()) {
      if (!overrides->is_array()) {
        fail(source, pointer + "/siteOverrides", "must be an array");
      }
      std::set<std::vector<uint64_t>> overriddenSites;
      for (size_t j = 0; j < overrides->size(); ++j) {
        const auto overridePointer =
            pointer + "/siteOverrides/" + std::to_string(j);
        const auto& overrideJson = (*overrides)[j];
        object(overrideJson, source, overridePointer);
        keys(overrideJson, {"sites", "duration", "fidelity"}, source,
             overridePointer);
        Device::SiteOverride override;
        const auto siteValues = overrideJson.find("sites");
        if (siteValues == overrideJson.end()) {
          fail(source, overridePointer + "/sites", "is required");
        }
        override.sites =
            indices(*siteValues, source, overridePointer + "/sites");
        override.duration = optional<uint64_t>(overrideJson, "duration", source,
                                               overridePointer);
        override.fidelity =
            optional<double>(overrideJson, "fidelity", source, overridePointer);
        validateFidelity(override.fidelity, source,
                         overridePointer + "/fidelity");
        const std::set<uint64_t> tupleSites(override.sites.begin(),
                                            override.sites.end());
        auto supported = false;
        if (override.sites.size() == operation.numQubits) {
          if (operation.sites) {
            supported = std::ranges::find(*operation.sites, override.sites) !=
                        operation.sites->end();
          } else if (operation.numQubits == 1) {
            supported = true;
          } else if (operation.numQubits == 2) {
            supported = std::ranges::find(
                            result.couplings,
                            std::pair{override.sites[0], override.sites[1]}) !=
                        result.couplings.end();
          }
        }
        if (override.sites.size() != operation.numQubits ||
            tupleSites.size() != override.sites.size() ||
            std::ranges::any_of(
                override.sites,
                [&](const auto qubit) { return qubit >= result.numQubits; }) ||
            !supported || !overriddenSites.emplace(override.sites).second ||
            (!override.duration && !override.fidelity)) {
          fail(source, overridePointer,
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

Device readJSON(const std::string_view json, const std::string_view source) {
  try {
    return parse(Json::parse(json), source);
  } catch (const Json::exception& error) {
    throw std::invalid_argument(std::string(source) +
                                ": invalid JSON: " + error.what());
  }
}

Device readJSON(std::istream& stream, const std::string_view source) {
  const std::string json{std::istreambuf_iterator<char>(stream),
                         std::istreambuf_iterator<char>()};
  return readJSON(json, source);
}

Device readJSON(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open JSON file: " + path);
  }
  return readJSON(input, path);
}

} // namespace sc
