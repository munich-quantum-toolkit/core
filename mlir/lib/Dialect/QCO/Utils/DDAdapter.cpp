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
#include <llvm/ADT/SmallVector.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace mlir::qco {

static auto addIdentityWires(dd::Package& package, dd::mCachedEdge child,
                             size_t firstWire, size_t endWire)
    -> dd::mCachedEdge {
  for (auto wire = firstWire; wire < endWire; ++wire) {
    child = package.makeDDNode<dd::mNode, dd::CachedEdge>(
        static_cast<dd::Qubit>(wire),
        {child, dd::mCachedEdge::zero(), dd::mCachedEdge::zero(), child});
  }
  return child;
}

namespace {
struct EmbeddedOperand {
  dd::Qubit wire;
  size_t mask;
};
} // namespace

static auto buildEmbeddedLocalDD(dd::Package& package,
                                 const std::span<const Complex> local,
                                 const size_t dimension,
                                 const llvm::ArrayRef<EmbeddedOperand> operands,
                                 const size_t operandIndex,
                                 const size_t maxWire, const size_t row,
                                 const size_t col) -> dd::mCachedEdge {
  if (operandIndex == operands.size()) {
    auto terminal = dd::mCachedEdge::terminal(local[(row * dimension) + col]);
    return addIdentityWires(package, terminal, 0, maxWire);
  }

  const auto [wire, mask] = operands[operandIndex];
  const auto edge00 = buildEmbeddedLocalDD(package, local, dimension, operands,
                                           operandIndex + 1, wire, row, col);
  const auto edge01 =
      buildEmbeddedLocalDD(package, local, dimension, operands,
                           operandIndex + 1, wire, row, col | mask);
  const auto edge10 =
      buildEmbeddedLocalDD(package, local, dimension, operands,
                           operandIndex + 1, wire, row | mask, col);
  const auto edge11 =
      buildEmbeddedLocalDD(package, local, dimension, operands,
                           operandIndex + 1, wire, row | mask, col | mask);
  auto root = package.makeDDNode<dd::mNode, dd::CachedEdge>(
      wire, {edge00, edge01, edge10, edge11});
  return addIdentityWires(package, root, static_cast<size_t>(wire) + 1,
                          maxWire);
}

static auto makeEmbeddedLocalDD(dd::Package& package,
                                const std::span<const Complex> local,
                                const size_t dimension, const size_t numQubits,
                                const llvm::ArrayRef<dd::Qubit> wires)
    -> dd::MatrixDD {
  llvm::SmallVector<EmbeddedOperand, 8> operands;
  operands.reserve(wires.size());
  for (size_t operand = 0; operand < wires.size(); ++operand) {
    operands.push_back({.wire = wires[operand],
                        .mask = size_t{1} << (wires.size() - 1 - operand)});
  }
  std::ranges::sort(operands, std::greater{}, &EmbeddedOperand::wire);

  const auto root = buildEmbeddedLocalDD(package, local, dimension, operands, 0,
                                         numQubits, 0, 0);
  return {.p = root.p, .w = package.cn.lookup(root.w)};
}

auto makeGateDD(dd::Package& package, const std::span<const Complex> matrix,
                const size_t numQubits, const llvm::ArrayRef<dd::Qubit> targets,
                const dd::Controls& controls) -> dd::MatrixDD {
  if (targets.size() >= std::numeric_limits<size_t>::digits) {
    throw std::invalid_argument(
        "Unitary matrix dimension does not match its target count");
  }
  const size_t dimension = size_t{1} << targets.size();
  if (dimension > std::numeric_limits<size_t>::max() / dimension ||
      matrix.size() != dimension * dimension) {
    throw std::invalid_argument(
        "Unitary matrix dimension does not match its target count");
  }

  if (targets.size() == 1) {
    return package.makeGateDD(
        std::span<const Complex, dd::NEDGE>{matrix.data(), dd::NEDGE}, controls,
        targets[0]);
  }

  if (targets.size() == 2) {
    constexpr size_t matrixSize = static_cast<size_t>(dd::NEDGE) * dd::NEDGE;
    return package.makeTwoQubitGateDD(
        std::span<const Complex, matrixSize>{matrix.data(), matrixSize},
        controls, targets[0], targets[1]);
  }

  if (targets.size() == 3) {
    constexpr size_t matrixSize =
        static_cast<size_t>(dd::THREE_QUBIT_GATE_DIM) *
        dd::THREE_QUBIT_GATE_DIM;
    return package.makeThreeQubitGateDD(
        std::span<const Complex, matrixSize>{matrix.data(), matrixSize},
        controls, targets[0], targets[1], targets[2]);
  }

  if (!controls.empty()) {
    throw std::invalid_argument(
        "Sparse controls are only supported for up to three target qubits");
  }
  return makeEmbeddedLocalDD(package, matrix, dimension, numQubits, targets);
}

} // namespace mlir::qco
