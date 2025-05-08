/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/GlobalOp.hpp"

#include "ir/Definitions.hpp"

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

namespace na {
namespace {
auto printParams(const std::vector<qc::fp>& params, std::ostringstream& os)
    -> void {
  if (!params.empty()) {
    for (const auto& p : params) {
      os << p << " ";
    }
  }
}
} // namespace

auto GlobalOp::toString() const -> std::string {
  std::ostringstream ss;
  ss << std::setprecision(5) << std::fixed;
  ss << "@+ " << name_ << " ";
  if (zones_.size() == 1) {
    printParams(params_, ss);
    ss << *zones_.front();
    return ss.str();
  }
  ss << "[\n";
  for (const auto& atom : zones_) {
    ss << "    ";
    printParams(params_, ss);
    ss << *atom << "\n";
  }
  ss << "]";
  return ss.str();
}
} // namespace na
