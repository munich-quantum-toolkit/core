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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sc {
namespace {
using llvm::json::ObjectMapper;
using llvm::json::Path;
using llvm::json::Value;

bool keys(const Value& value, std::initializer_list<llvm::StringRef> known,
          Path path) {
  const auto* object = value.getAsObject();
  if (object == nullptr) {
    path.report("must be an object");
    return false;
  }
  return std::ranges::all_of(*object, [&](const auto& field) {
    const llvm::StringRef key = field.first;
    if (std::ranges::find(known, key) != known.end()) {
      return true;
    }
    path.field(key).report("unknown key");
    return false;
  });
}

template <class T>
bool optional(const Value& value, llvm::StringRef key, std::optional<T>& out,
              Path path) {
  if (const auto* field = value.getAsObject()->get(key)) {
    T parsed{};
    if (!llvm::json::fromJSON(*field, parsed, path.field(key))) {
      return false;
    }
    out = std::move(parsed);
  }
  return true;
}
} // namespace

// ObjectMapper needs these conversions in sc for argument-dependent lookup.
// NOLINTBEGIN(misc-use-anonymous-namespace)
static bool fromJSON(const Value& value, Device::Unit& out, Path path) {
  ObjectMapper object(value, path);
  return object && keys(value, {"unit", "scaleFactor"}, path) &&
         object.map("unit", out.unit) &&
         object.map("scaleFactor", out.scaleFactor);
}

static bool fromJSON(const Value& value, Device::QubitCalibration& out,
                     Path path) {
  return keys(value, {"t1", "t2"}, path) &&
         optional(value, "t1", out.t1, path) &&
         optional(value, "t2", out.t2, path);
}

static bool fromJSON(const Value& value, Device::QubitOverride& out,
                     Path path) {
  ObjectMapper object(value, path);
  return object && keys(value, {"qubit", "name", "t1", "t2"}, path) &&
         object.map("qubit", out.qubit) &&
         optional(value, "name", out.name, path) &&
         optional(value, "t1", out.t1, path) &&
         optional(value, "t2", out.t2, path);
}

static bool fromJSON(const Value& value, Device::QubitProperties& out,
                     Path path) {
  ObjectMapper object(value, path);
  return object && keys(value, {"defaults", "overrides"}, path) &&
         object.map("defaults", out.defaults) &&
         object.map("overrides", out.overrides);
}

static bool fromJSON(const Value& value, Device::SiteOverride& out, Path path) {
  ObjectMapper object(value, path);
  return object && keys(value, {"sites", "duration", "fidelity"}, path) &&
         object.map("sites", out.sites) &&
         optional(value, "duration", out.duration, path) &&
         optional(value, "fidelity", out.fidelity, path);
}

static bool fromJSON(const Value& value, Device::Operation& out, Path path) {
  ObjectMapper object(value, path);
  return object &&
         keys(value,
              {
                  "name",
                  "numParameters",
                  "numQubits",
                  "sites",
                  "duration",
                  "fidelity",
                  "siteOverrides",
              },
              path) &&
         object.map("name", out.name) &&
         object.map("numParameters", out.numParameters) &&
         object.map("numQubits", out.numQubits) &&
         optional(value, "sites", out.sites, path) &&
         optional(value, "duration", out.duration, path) &&
         optional(value, "fidelity", out.fidelity, path) &&
         object.mapOptional("siteOverrides", out.siteOverrides);
}

// NOLINTEND(misc-use-anonymous-namespace)
namespace {
llvm::Error fail(std::string_view source, std::string_view pointer,
                 std::string_view message) {
  return llvm::createStringError(llvm::StringRef(source) + ":" +
                                 llvm::StringRef(pointer) + " " +
                                 llvm::StringRef(message));
}

bool positiveCalibration(const Device::QubitCalibration& value) {
  return (!value.t1 || *value.t1 > 0) && (!value.t2 || *value.t2 > 0);
}

bool validFidelity(const std::optional<double>& value) {
  return !value || (std::isfinite(*value) && *value >= 0. && *value <= 1.);
}

llvm::Expected<Device> parse(const Value& root, std::string_view source) {
  Path::Root diagnostic(source);
  ObjectMapper object(root, diagnostic);
  Device result;
  std::vector<std::vector<uint64_t>> couplings;
  if (!object ||
      !keys(root,
            {
                "schema-version",
                "name",
                "numQubits",
                "durationUnit",
                "qubitProperties",
                "couplings",
                "operations",
            },
            diagnostic) ||
      !object.map("schema-version", result.schemaVersion) ||
      !object.map("name", result.name) ||
      !object.map("numQubits", result.numQubits) ||
      !object.map("durationUnit", result.durationUnit) ||
      !object.map("qubitProperties", result.qubitProperties) ||
      !object.map("couplings", couplings) ||
      !object.map("operations", result.operations)) {
    return diagnostic.getError();
  }
  if (result.schemaVersion != 1) {
    return fail(source, "$/schema-version", "must be 1");
  }
  if (result.name.empty() || result.name.find('\0') != std::string::npos) {
    return fail(source, "$/name", "must be non-empty and contain no NUL bytes");
  }
  if (result.numQubits == 0 ||
      result.numQubits > std::vector<void*>{}.max_size()) {
    return fail(source, "$/numQubits", "must be positive and representable");
  }
  constexpr std::array units{"s", "ms", "us", "ns"};
  if (std::ranges::find(units, result.durationUnit.unit) == units.end() ||
      !std::isfinite(result.durationUnit.scaleFactor) ||
      result.durationUnit.scaleFactor <= 0.) {
    return fail(source, "$/durationUnit",
                "must use s, ms, us, or ns and a positive finite scaleFactor");
  }
  if (!positiveCalibration(result.qubitProperties.defaults)) {
    return fail(source, "$/qubitProperties/defaults",
                "t1 and t2 must be positive when present");
  }
  llvm::DenseSet<uint64_t> overridden;
  for (size_t i = 0; i < result.qubitProperties.overrides.size(); ++i) {
    const auto& entry = result.qubitProperties.overrides[i];
    if (entry.qubit >= result.numQubits ||
        !overridden.insert(entry.qubit).second ||
        (!entry.name && !entry.t1 && !entry.t2) ||
        !positiveCalibration(entry) ||
        (entry.name && (entry.name->empty() ||
                        entry.name->find('\0') != std::string::npos))) {
      return fail(source, "$/qubitProperties/overrides/" + std::to_string(i),
                  "must select one unique valid qubit and override name, t1, "
                  "or t2 with valid values");
    }
  }
  llvm::DenseSet<std::pair<uint64_t, uint64_t>> uniqueCouplings;
  for (size_t i = 0; i < couplings.size(); ++i) {
    const auto& coupling = couplings[i];
    const auto pointer = "$/couplings/" + std::to_string(i);
    if (coupling.size() != 2) {
      return fail(source, pointer, "must contain exactly two qubit indices");
    }
    if (coupling[0] >= result.numQubits || coupling[1] >= result.numQubits ||
        coupling[0] == coupling[1] ||
        !uniqueCouplings.insert({coupling[0], coupling[1]}).second) {
      return fail(source, pointer,
                  "must be a unique, non-self tuple of valid qubits");
    }
    result.couplings.emplace_back(coupling[0], coupling[1]);
  }
  llvm::StringSet<> names;
  for (size_t i = 0; i < result.operations.size(); ++i) {
    const auto pointer = "$/operations/" + std::to_string(i);
    const auto& operation = result.operations[i];
    if (!validFidelity(operation.fidelity)) {
      return fail(source, pointer + "/fidelity",
                  "must be finite and in [0, 1]");
    }
    if (operation.name.empty() ||
        operation.name.find('\0') != std::string::npos ||
        operation.numQubits == 0 || operation.numQubits > result.numQubits ||
        operation.numParameters > std::numeric_limits<size_t>::max() ||
        !names.insert(operation.name).second) {
      return fail(source, pointer,
                  "must have a unique non-empty name without NUL bytes and "
                  "representable counts");
    }
    const auto validTuple = [&](const std::vector<uint64_t>& tuple) {
      return tuple.size() == operation.numQubits &&
             std::ranges::all_of(
                 tuple,
                 [&](uint64_t qubit) { return qubit < result.numQubits; }) &&
             llvm::SmallDenseSet<uint64_t, 4>(tuple.begin(), tuple.end())
                     .size() == tuple.size();
    };
    llvm::DenseSet<llvm::ArrayRef<uint64_t>> uniqueSites;
    if (operation.sites) {
      for (size_t j = 0; j < operation.sites->size(); ++j) {
        const auto& tuple = (*operation.sites)[j];
        if (!validTuple(tuple) ||
            (operation.numQubits == 2 &&
             !uniqueCouplings.contains({tuple[0], tuple[1]})) ||
            !uniqueSites.insert(tuple).second) {
          return fail(source, pointer + "/sites/" + std::to_string(j),
                      "must be a unique tuple matching the operation arity and "
                      "device connectivity");
        }
      }
    } else if (operation.numQubits > 2) {
      return fail(source, pointer + "/sites",
                  "is required for operations with arity greater than two");
    }
    llvm::DenseSet<llvm::ArrayRef<uint64_t>> overriddenSites;
    for (size_t j = 0; j < operation.siteOverrides.size(); ++j) {
      const auto& siteOverride = operation.siteOverrides[j];
      const auto overridePointer =
          pointer + "/siteOverrides/" + std::to_string(j);
      if (!validFidelity(siteOverride.fidelity)) {
        return fail(source, overridePointer + "/fidelity",
                    "must be finite and in [0, 1]");
      }
      const auto supported =
          validTuple(siteOverride.sites) &&
          (operation.sites
               ? uniqueSites.contains(siteOverride.sites)
               : operation.numQubits == 1 ||
                     uniqueCouplings.contains(
                         {siteOverride.sites[0], siteOverride.sites[1]}));
      if (!supported || !overriddenSites.insert(siteOverride.sites).second ||
          (!siteOverride.duration && !siteOverride.fidelity)) {
        return fail(source, overridePointer,
                    "must be one unique supported tuple and override a "
                    "valid value");
      }
    }
  }
  return result;
}
} // namespace

llvm::Expected<Device> readJSON(std::string_view json,
                                std::string_view source) {
  auto value = llvm::json::parse(json);
  if (!value) {
    return llvm::createStringError(llvm::StringRef(source) + ": " +
                                   llvm::toString(value.takeError()));
  }
  return parse(*value, source);
}

llvm::Expected<Device> readJSON(const std::string& path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return llvm::createStringError(buffer.getError(), "failed to read '%s'",
                                   path.c_str());
  }
  return readJSON((*buffer)->getBuffer(), path);
}
} // namespace sc
