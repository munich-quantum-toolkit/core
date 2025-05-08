/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Statement_fwd.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qasm3 {
struct GateInfo {
  size_t nControls;
  size_t nTargets;
  size_t nParameters;
  qc::OpType type;
};

struct Gate {
  virtual ~Gate() = default;

  virtual size_t getNControls() = 0;
  virtual size_t getNTargets() = 0;
  virtual size_t getNParameters() = 0;
};

struct StandardGate final : Gate {
  GateInfo info;

  explicit StandardGate(const GateInfo& gateInfo) : info(gateInfo) {}

  size_t getNControls() override { return info.nControls; }

  size_t getNTargets() override { return info.nTargets; }
  size_t getNParameters() override { return info.nParameters; }
};

struct CompoundGate final : Gate {
  std::vector<std::string> parameterNames;
  std::vector<std::string> targetNames;
  std::vector<std::shared_ptr<QuantumStatement>> body;

  explicit CompoundGate(
      std::vector<std::string> parameters, std::vector<std::string> targets,
      std::vector<std::shared_ptr<QuantumStatement>> bodyStatements)
      : parameterNames(std::move(parameters)), targetNames(std::move(targets)),
        body(std::move(bodyStatements)) {}

  size_t getNControls() override { return 0; }

  size_t getNTargets() override { return targetNames.size(); }
  size_t getNParameters() override { return parameterNames.size(); }
};
} // namespace qasm3
