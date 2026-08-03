/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Configuration.hpp
 * @brief Superconducting QDMI device configuration.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sc {

/// Strict schema-version-1 description of a superconducting device.
struct Device {
  struct Unit {
    std::string unit;
    double scaleFactor = 1.;
  };

  struct QubitCalibration {
    std::optional<uint64_t> t1;
    std::optional<uint64_t> t2;
  };

  struct QubitOverride : QubitCalibration {
    uint64_t qubit = 0;
    std::optional<std::string> name;
  };

  struct QubitProperties {
    QubitCalibration defaults;
    std::vector<QubitOverride> overrides;
  };

  struct SiteOverride {
    std::vector<uint64_t> sites;
    std::optional<uint64_t> duration;
    std::optional<double> fidelity;
  };

  struct Operation {
    std::string name;
    uint64_t numParameters = 0;
    uint64_t numQubits = 0;
    std::optional<std::vector<std::vector<uint64_t>>> sites;
    std::optional<uint64_t> duration;
    std::optional<double> fidelity;
    std::vector<SiteOverride> siteOverrides;
  };

  uint64_t schemaVersion = 1;
  std::string name;
  uint64_t numQubits = 0;
  Unit durationUnit;
  QubitProperties qubitProperties;
  std::vector<std::pair<uint64_t, uint64_t>> couplings;
  std::vector<Operation> operations;
};

/// Parse and validate a device description and report errors with @p source.
[[nodiscard]] Device readJSON(std::string_view json, std::string_view source);
[[nodiscard]] Device readJSON(std::istream& stream,
                              std::string_view source = "input");
[[nodiscard]] Device readJSON(const std::string& path);

} // namespace sc
