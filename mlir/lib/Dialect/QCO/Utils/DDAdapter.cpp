/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/DDAdapter.h"

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mlir::qco {

static auto addIdentityWires(dd::Package& package, dd::mCachedEdge child,
                             std::size_t firstWire, std::size_t endWire)
    -> dd::mCachedEdge {
  for (auto wire = firstWire; wire < endWire; ++wire) {
    child = package.makeDDNode<dd::mNode, dd::CachedEdge>(
        static_cast<dd::Qubit>(wire),
        {child, dd::mCachedEdge::zero(), dd::mCachedEdge::zero(), child});
  }
  return child;
}

static auto buildEmbeddedLocalDD(dd::Package& package,
                                 const DynamicMatrix& local,
                                 llvm::ArrayRef<dd::Qubit> wires,
                                 std::size_t maxWire, std::size_t row,
                                 std::size_t col) -> dd::mCachedEdge {
  std::optional<std::pair<dd::Qubit, std::size_t>> highestOperand;
  for (std::size_t operand = 0; operand < wires.size(); ++operand) {
    const auto wire = wires[operand];
    if (wire < maxWire && (!highestOperand || wire > highestOperand->first)) {
      highestOperand.emplace(wire, operand);
    }
  }

  if (!highestOperand) {
    auto terminal = dd::mCachedEdge::terminal(
        local(static_cast<int64_t>(row), static_cast<int64_t>(col)));
    return addIdentityWires(package, terminal, 0, maxWire);
  }

  const auto [wire, operand] = *highestOperand;
  const std::size_t operandMask = std::size_t{1}
                                  << (wires.size() - 1 - operand);
  const auto edge00 =
      buildEmbeddedLocalDD(package, local, wires, wire, row, col);
  const auto edge01 =
      buildEmbeddedLocalDD(package, local, wires, wire, row, col | operandMask);
  const auto edge10 =
      buildEmbeddedLocalDD(package, local, wires, wire, row | operandMask, col);
  const auto edge11 = buildEmbeddedLocalDD(
      package, local, wires, wire, row | operandMask, col | operandMask);
  auto root = package.makeDDNode<dd::mNode, dd::CachedEdge>(
      wire, {edge00, edge01, edge10, edge11});
  return addIdentityWires(package, root, static_cast<std::size_t>(wire) + 1,
                          maxWire);
}

static auto
makeEmbeddedLocalDD(dd::Package& package, const DynamicMatrix& local,
                    std::size_t numQubits, llvm::ArrayRef<dd::Qubit> wires)
    -> dd::MatrixDD {
  const auto root =
      buildEmbeddedLocalDD(package, local, wires, numQubits, 0, 0);
  return {.p = root.p, .w = package.cn.lookup(root.w)};
}

auto makeGateDD(dd::Package& package, const DynamicMatrix& matrix,
                std::size_t numQubits, llvm::ArrayRef<dd::Qubit> targets,
                const dd::Controls& controls) -> dd::MatrixDD {
  if (targets.size() >= std::numeric_limits<int64_t>::digits ||
      matrix.rows() != (int64_t{1} << targets.size())) {
    throw std::invalid_argument(
        "Unitary matrix dimension does not match its target count");
  }

  if (targets.size() == 1) {
    const dd::GateMatrix converted{matrix(0, 0), matrix(0, 1), matrix(1, 0),
                                   matrix(1, 1)};
    return package.makeGateDD(converted, controls, targets[0]);
  }

  if (targets.size() == 2) {
    dd::TwoQubitGateMatrix converted{};
    for (std::size_t row = 0; row < converted.size(); ++row) {
      for (std::size_t col = 0; col < converted[row].size(); ++col) {
        converted[row][col] =
            matrix(static_cast<int64_t>(row), static_cast<int64_t>(col));
      }
    }
    return package.makeTwoQubitGateDD(converted, controls, targets[0],
                                      targets[1]);
  }

  if (targets.size() == 3) {
    dd::ThreeQubitGateMatrix converted{};
    for (std::size_t row = 0; row < converted.size(); ++row) {
      for (std::size_t col = 0; col < converted[row].size(); ++col) {
        converted[row][col] =
            matrix(static_cast<int64_t>(row), static_cast<int64_t>(col));
      }
    }
    return package.makeThreeQubitGateDD(converted, controls, targets[0],
                                        targets[1], targets[2]);
  }

  if (!controls.empty()) {
    throw std::invalid_argument(
        "Sparse controls are only supported for up to three target qubits");
  }
  return makeEmbeddedLocalDD(package, matrix, numQubits, targets);
}

} // namespace mlir::qco
