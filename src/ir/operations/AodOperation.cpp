/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/AodOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"
#include "ir/operations/OpType.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace na {
AodOperation::AodOperation(qc::OpType s, std::vector<qc::Qubit> qubits,
                           const std::vector<uint32_t>& dirs,
                           const std::vector<qc::fp>& start,
                           const std::vector<qc::fp>& end)
    : AodOperation(s, std::move(qubits), convertToDimension(dirs), start, end) {
}

std::string SingleOperation::toQASMString() const {
  std::stringstream ss;
  ss << static_cast<std::size_t>(dir) << ", " << start << ", " << end << "; ";
  return ss.str();
}

std::vector<Dimension>
AodOperation::convertToDimension(const std::vector<uint32_t>& dirs) {
  std::vector<Dimension> dirsEnum(dirs.size());
  for (size_t i = 0; i < dirs.size(); ++i) {
    dirsEnum[i] = static_cast<Dimension>(dirs[i]);
  }
  return dirsEnum;
}

AodOperation::AodOperation(const qc::OpType s, std::vector<qc::Qubit> qubits,
                           const std::vector<Dimension>& dirs,
                           const std::vector<qc::fp>& start,
                           const std::vector<qc::fp>& end) {
  assert(dirs.size() == start.size() && start.size() == end.size());
  type = s;
  targets = std::move(qubits);
  name = toString(type);

  for (size_t i = 0; i < dirs.size(); ++i) {
    operations.emplace_back(dirs[i], start[i], end[i]);
  }
}

AodOperation::AodOperation(const std::string& typeName,
                           std::vector<qc::Qubit> qubits,
                           const std::vector<uint32_t>& dirs,
                           const std::vector<qc::fp>& start,
                           const std::vector<qc::fp>& end)
    : AodOperation(qc::opTypeFromString(typeName), std::move(qubits),
                   convertToDimension(dirs), start, end) {}

AodOperation::AodOperation(
    const qc::OpType s, std::vector<qc::Qubit> qubits,
    const std::vector<std::tuple<Dimension, qc::fp, qc::fp>>& ops) {
  type = s;
  targets = std::move(qubits);
  name = toString(type);

  for (const auto& [dir, index, param] : ops) {
    operations.emplace_back(dir, index, param);
  }
}

AodOperation::AodOperation(qc::OpType s, std::vector<qc::Qubit> t,
                           std::vector<SingleOperation> ops)
    : operations(std::move(ops)) {
  type = s;
  targets = std::move(t);
  name = toString(type);
}

std::vector<qc::fp> AodOperation::getEnds(const Dimension dir) const {
  std::vector<qc::fp> ends;
  for (const auto& op : operations) {
    if (op.dir == dir) {
      ends.emplace_back(op.end);
    }
  }
  return ends;
}

std::vector<qc::fp> AodOperation::getStarts(const Dimension dir) const {
  std::vector<qc::fp> starts;
  for (const auto& op : operations) {
    if (op.dir == dir) {
      starts.emplace_back(op.start);
    }
  }
  return starts;
}

qc::fp AodOperation::getMaxDistance(const Dimension dir) const {
  const auto distances = getDistances(dir);
  if (distances.empty()) {
    return 0;
  }
  return *std::ranges::max_element(distances);
}

std::vector<qc::fp> AodOperation::getDistances(const Dimension dir) const {
  std::vector<qc::fp> params;
  for (const auto& op : operations) {
    if (op.dir == dir) {
      params.emplace_back(std::abs(op.end - op.start));
    }
  }
  return params;
}

void AodOperation::dumpOpenQASM(
    std::ostream& of, const qc::QubitIndexToRegisterMap& qubitMap,
    [[maybe_unused]] const qc::BitIndexToRegisterMap& bitMap,
    const size_t indent, bool /*openQASM3*/) const {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<qc::fp>::digits10);

  // Add indentation and operation name
  oss << std::string(indent * OUTPUT_INDENT_SIZE, ' ') << name;

  // Write AOD operations
  oss << " (";
  for (const auto& op : operations) {
    oss << op.toQASMString();
  }

  // Remove the last "; " if present
  std::string content = oss.str();
  const std::string semicolon = "; ";
  if (content.size() >= semicolon.size() &&
      content.compare(content.size() - semicolon.size(), semicolon.size(),
                      semicolon) == 0) {
    content.erase(content.size() - semicolon.size());
  }

  content += ")";

  // Write qubit start
  for (const auto& qubit : targets) {
    content += " " + qubitMap.at(qubit).second + ",";
  }

  // Remove the last comma if present
  if (!content.empty() && content.back() == ',') {
    content.pop_back();
  }

  content += ";\n";

  // Write final content to the provided ostream
  of << content;
}

void AodOperation::invert() {
  if (type == qc::OpType::AodMove) {
    for (auto& op : operations) {
      std::swap(op.start, op.end);
    }
  } else if (type == qc::OpType::AodActivate) {
    type = qc::OpType::AodDeactivate;
  } else if (type == qc::OpType::AodDeactivate) {
    type = qc::OpType::AodActivate;
  }
}
} // namespace na
