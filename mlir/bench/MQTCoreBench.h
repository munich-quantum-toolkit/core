/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cstdint>
#include <string>

struct BenchmarkOptions {
  enum class Command : uint8_t { None, List, Describe, Generate, Evaluate };
  Command command;
  const std::string& benchmarkId;
  const std::string& instanceSpecificationPath;
  const std::string& outputFormat;
  const std::string& outputDirectory;
  const std::string& manifestInputPath;
  const std::string& countsInputPath;
};

int runMQTCoreBench(const BenchmarkOptions& options);
