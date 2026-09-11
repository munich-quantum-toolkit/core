/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"

#include "dd/CachedEdge.hpp"
#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "dd/UniqueTable.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dd {
namespace {
constexpr GateMatrix MEAS_ZERO_MAT{1, 0, 0, 0};
constexpr GateMatrix MEAS_ONE_MAT{0, 0, 0, 1};

void checkMeasurementQubit(const vEdge& state, const Qubit index) {
  if (state.isTerminal() || index > state.p->v) {
    throw std::invalid_argument("Measurement qubit is outside the state.");
  }
}
} // namespace

Package::Package(const std::size_t nq, const DDPackageConfig& config)
    : nqubits(nq), config_(config) {
  resize(nq);
}

void Package::resize(const std::size_t nq) {
  if (nq > MAX_POSSIBLE_QUBITS) {
    throw std::invalid_argument("Requested too many qubits from package. "
                                "Qubit datatype only allows up to " +
                                std::to_string(MAX_POSSIBLE_QUBITS) +
                                " qubits, while " + std::to_string(nq) +
                                " were requested. Please recompile the "
                                "package with a wider Qubit type!");
  }
  nqubits = nq;
  vUniqueTable.resize(nqubits);
  mUniqueTable.resize(nqubits);
}

void Package::reset() {
  clearUniqueTables();
  resetMemoryManagers();
  clearComputeTables();
  roots.reset();
}

void Package::resetMemoryManagers(const bool resizeToTotal) {
  vMemoryManager.reset(resizeToTotal);
  mMemoryManager.reset(resizeToTotal);
  cMemoryManager.reset(resizeToTotal);
}

void Package::clearUniqueTables() {
  vUniqueTable.clear();
  mUniqueTable.clear();
  cUniqueTable.clear();
}

bool Package::garbageCollect(bool force) {
  using flags = std::tuple<bool, bool, bool>;

  // return immediately if no table needs collection
  if (!force && !vUniqueTable.possiblyNeedsCollection() &&
      !mUniqueTable.possiblyNeedsCollection() &&
      !cUniqueTable.possiblyNeedsCollection()) {
    return false;
  }

  const auto sweep = [this, &force]() -> flags {
    const bool invC = cUniqueTable.garbageCollect(force) > 0;
    force |= invC;
    const bool invV = vUniqueTable.garbageCollect(force) > 0;
    const bool invM = mUniqueTable.garbageCollect(force) > 0;
    return {invC, invV, invM};
  };

  const auto [invC, invV, invM] = roots.execute<flags>(sweep);

  // invalidate all compute tables involving vectors if any vector node has
  // been collected
  if (invV) {
    conjugateVector.clear();
    vectorAddMagnitudes.clear();
    vectorAdd.clear();
    vectorInnerProduct.clear();
    vectorKronecker.clear();
    matrixVectorMultiplication.clear();
  }
  // invalidate all compute tables involving matrices if any matrix node has
  // been collected
  if (invM) {
    matrixAddMagnitudes.clear();
    matrixAdd.clear();
    conjugateMatrixTranspose.clear();
    matrixKronecker.clear();
    matrixTrace.clear();
    matrixVectorMultiplication.clear();
    matrixMatrixMultiplication.clear();
  }
  // invalidate all compute tables where any component of the entry contains
  // numbers from the complex table if any complex numbers were collected
  if (invC) {
    matrixVectorMultiplication.clear();
    matrixMatrixMultiplication.clear();
    conjugateMatrixTranspose.clear();
    vectorInnerProduct.clear();
    vectorKronecker.clear();
    matrixKronecker.clear();
    matrixTrace.clear();
  }
  return invC || invV || invM;
}

Package::ActiveCounts Package::computeActiveCounts() {
  const auto count = [this]() -> ActiveCounts {
    return {
        .vector = vUniqueTable.countMarkedEntries(),
        .matrix = mUniqueTable.countMarkedEntries(),
        .reals = cUniqueTable.countMarkedEntries(),
    };
  };
  return roots.execute<ActiveCounts>(count);
}

namespace {

[[noreturn]] void throwGateQubitOutOfRange(const std::size_t nqubits) {
  if (nqubits == 0U) {
    throw std::runtime_error(
        "Cannot construct a gate in a package with zero qubits.");
  }
  throw std::runtime_error{
      "Requested gate acting on qubit(s) with index larger than " +
      std::to_string(nqubits - 1U) +
      " while the package configuration only supports up to " +
      std::to_string(nqubits) +
      " qubits. Please allocate a larger package instance."};
}

[[noreturn]] void throwGateQubitsNotDistinct() {
  throw std::runtime_error{
      "Requested gate has duplicate or overlapping control/target qubits."};
}

void ensureGateQubitsInRange(const std::size_t nqubits,
                             const Controls& controls,
                             const std::span<const Qubit> targets) {
  if (nqubits == 0U ||
      std::ranges::any_of(controls,
                          [nqubits](const auto& c) {
                            return static_cast<std::size_t>(c.qubit) >= nqubits;
                          }) ||
      std::ranges::any_of(targets, [nqubits](const Qubit target) {
        return static_cast<std::size_t>(target) >= nqubits;
      })) {
    throwGateQubitOutOfRange(nqubits);
  }

  if (std::ranges::adjacent_find(controls, {}, &Control::qubit) !=
      controls.end()) {
    throwGateQubitsNotDistinct();
  }

  for (size_t i = 0; i < targets.size(); ++i) {
    if (std::ranges::find(targets.first(i), targets[i]) !=
        targets.first(i).end()) {
      throwGateQubitsNotDistinct();
    }
  }

  if (std::ranges::any_of(controls, [&targets](const auto& control) {
        return std::ranges::find(targets, control.qubit) != targets.end();
      })) {
    throwGateQubitsNotDistinct();
  }
}

template <std::size_t Dim>
void fillTerminalMatrix(
    std::array<std::array<mCachedEdge, Dim>, Dim>& em,
    const std::array<std::array<std::complex<fp>, Dim>, Dim>& mat) {
  for (std::size_t row = 0; row < Dim; ++row) {
    for (std::size_t col = 0; col < Dim; ++col) {
      em[row][col] = mCachedEdge::terminal(mat[row][col]);
    }
  }
}

template <std::size_t Dim>
void fillTerminalMatrix(
    std::array<std::array<mCachedEdge, Dim>, Dim>& em,
    const std::span<const std::complex<fp>, Dim * Dim> mat) {
  for (std::size_t row = 0; row < Dim; ++row) {
    for (std::size_t col = 0; col < Dim; ++col) {
      em[row][col] = mCachedEdge::terminal(mat[(row * Dim) + col]);
    }
  }
}

void fillTerminalVector(std::array<mCachedEdge, NEDGE>& em,
                        const GateMatrix& mat) {
  for (std::size_t i = 0; i < NEDGE; ++i) {
    em[i] = mCachedEdge::terminal(mat[i]);
  }
}

void fillTerminalVector(std::array<mCachedEdge, NEDGE>& em,
                        const std::span<const std::complex<fp>, NEDGE> mat) {
  for (std::size_t i = 0; i < NEDGE; ++i) {
    em[i] = mCachedEdge::terminal(mat[i]);
  }
}

[[nodiscard]] mCachedEdge makeControlledNode(Package& dd,
                                             const Qubit controlQubit,
                                             const Control::Type type,
                                             const mCachedEdge& gate,
                                             const bool identity) {
  std::array<mCachedEdge, NEDGE> edges{
      mCachedEdge::zero(),
      mCachedEdge::zero(),
      mCachedEdge::zero(),
      mCachedEdge::zero(),
  };
  const auto idEdge = identity ? mCachedEdge::one() : mCachedEdge::zero();
  if (type == Control::Type::Neg) {
    edges[0] = gate;
    edges[3] = idEdge;
  } else {
    edges[0] = idEdge;
    edges[3] = gate;
  }
  return dd.makeDDNode(controlQubit, edges);
}

/// Diagonal entries of a flattened 2×2 edge block (indices 0 and 3).
[[nodiscard]] constexpr bool isTwoByTwoDiagonal(const std::size_t i) {
  return i == 0U || i == 3U;
}

template <std::size_t Dim>
void wrapControlsUntil(Package& dd, Controls::const_iterator& it,
                       const Controls::const_iterator end, const Qubit bound,
                       std::array<std::array<mCachedEdge, Dim>, Dim>& em) {
  for (; it != end && it->qubit < bound; ++it) {
    for (std::size_t row = 0; row < Dim; ++row) {
      for (std::size_t col = 0; col < Dim; ++col) {
        em[row][col] = makeControlledNode(dd, it->qubit, it->type, em[row][col],
                                          row == col);
      }
    }
  }
}

void wrapControlsUntil(Package& dd, Controls::const_iterator& it,
                       const Controls::const_iterator end, const Qubit bound,
                       std::array<mCachedEdge, NEDGE>& em) {
  for (; it != end && it->qubit < bound; ++it) {
    for (std::size_t i = 0; i < NEDGE; ++i) {
      em[i] = makeControlledNode(dd, it->qubit, it->type, em[i],
                                 isTwoByTwoDiagonal(i));
    }
  }
}

void wrapControlsAbove(Package& dd, Controls::const_iterator& it,
                       const Controls::const_iterator end, mCachedEdge& e) {
  for (; it != end; ++it) {
    e = makeControlledNode(dd, it->qubit, it->type, e, true);
  }
}

[[nodiscard]] mEdge toMatrixDD(Package& dd, const mCachedEdge& e) {
  return {.p = e.p, .w = dd.cn.lookup(e.w)};
}

template <typename Matrix>
[[nodiscard]] mEdge buildSingleQubitGateDD(Package& dd, const Matrix& mat,
                                           const Controls& controls,
                                           const Qubit target) {
  const std::array targets{target};
  ensureGateQubitsInRange(dd.qubits(), controls, targets);

  std::array<mCachedEdge, NEDGE> em{};
  fillTerminalVector(em, mat);

  if (controls.empty()) {
    return toMatrixDD(dd, dd.makeDDNode(target, em));
  }

  auto it = controls.begin();
  const auto endIt = controls.end();
  wrapControlsUntil(dd, it, endIt, target, em);

  auto e = dd.makeDDNode(target, em);
  wrapControlsAbove(dd, it, endIt, e);
  return toMatrixDD(dd, e);
}

template <typename Matrix>
[[nodiscard]] mEdge
buildTwoQubitGateDD(Package& dd, const Matrix& mat, const Controls& controls,
                    const Qubit target0, const Qubit target1) {
  const std::array targets{target0, target1};
  ensureGateQubitsInRange(dd.qubits(), controls, targets);

  std::array<std::array<mCachedEdge, NEDGE>, NEDGE> em{};
  fillTerminalMatrix(em, mat);

  auto it = controls.begin();
  const auto endIt = controls.end();
  const auto smallerTarget = std::min(target0, target1);
  wrapControlsUntil(dd, it, endIt, smallerTarget, em);

  std::array<mCachedEdge, NEDGE> em0{};
  for (std::size_t row = 0; row < RADIX; ++row) {
    for (std::size_t col = 0; col < RADIX; ++col) {
      std::array<mCachedEdge, NEDGE> local{};
      if (target0 > target1) {
        for (std::size_t i = 0; i < RADIX; ++i) {
          for (std::size_t j = 0; j < RADIX; ++j) {
            local.at((i * RADIX) + j) =
                em.at((row * RADIX) + i).at((col * RADIX) + j);
          }
        }
      } else {
        for (std::size_t i = 0; i < RADIX; ++i) {
          for (std::size_t j = 0; j < RADIX; ++j) {
            local.at((i * RADIX) + j) =
                em.at((i * RADIX) + row).at((j * RADIX) + col);
          }
        }
      }
      em0.at((row * RADIX) + col) = dd.makeDDNode(smallerTarget, local);
    }
  }

  const auto largerTarget = std::max(target0, target1);
  wrapControlsUntil(dd, it, endIt, largerTarget, em0);

  auto e = dd.makeDDNode(largerTarget, em0);
  wrapControlsAbove(dd, it, endIt, e);
  return toMatrixDD(dd, e);
}

template <typename Matrix>
[[nodiscard]] mEdge
buildThreeQubitGateDD(Package& dd, const Matrix& mat, const Controls& controls,
                      const Qubit target0, const Qubit target1,
                      const Qubit target2) {
  /// Reduce 8x8 terminals to 4x4, then four edges, then the root, inserting
  /// controls between target levels. Matrix bits are MSB-first; DD levels
  /// follow ascending qubit indices.
  const std::array targets{target0, target1, target2};
  ensureGateQubitsInRange(dd.qubits(), controls, targets);

  std::array<std::array<mCachedEdge, THREE_QUBIT_GATE_DIM>,
             THREE_QUBIT_GATE_DIM>
      em{};
  fillTerminalMatrix(em, mat);

  std::array<std::pair<Qubit, std::uint8_t>, 3> ordered{
      {{target0, 2}, {target1, 1}, {target2, 0}},
  };
  std::ranges::sort(ordered, {}, &std::pair<Qubit, std::uint8_t>::first);
  const auto qLow = ordered[0].first;
  const auto qMid = ordered[1].first;
  const auto qHigh = ordered[2].first;
  const auto bLow = ordered[0].second;
  const auto bMid = ordered[1].second;
  const auto bHigh = ordered[2].second;

  auto it = controls.begin();
  const auto endIt = controls.end();
  wrapControlsUntil(dd, it, endIt, qLow, em);

  /// Remaining-bit index: bit(mid) + 2 * bit(high).
  std::array<std::array<mCachedEdge, NEDGE>, NEDGE> emMid{};
  for (std::size_t rMH = 0; rMH < NEDGE; ++rMH) {
    for (std::size_t cMH = 0; cMH < NEDGE; ++cMH) {
      const auto rMidBit = rMH & 1U;
      const auto rHighBit = rMH >> 1U;
      const auto cMidBit = cMH & 1U;
      const auto cHighBit = cMH >> 1U;
      std::array<mCachedEdge, NEDGE> local{};
      for (std::size_t i = 0; i < RADIX; ++i) {
        for (std::size_t j = 0; j < RADIX; ++j) {
          const auto rowIdx =
              (i << bLow) | (rMidBit << bMid) | (rHighBit << bHigh);
          const auto colIdx =
              (j << bLow) | (cMidBit << bMid) | (cHighBit << bHigh);
          local.at((i * RADIX) + j) = em.at(rowIdx).at(colIdx);
        }
      }
      emMid.at(rMH).at(cMH) = dd.makeDDNode(qLow, local);
    }
  }

  wrapControlsUntil(dd, it, endIt, qMid, emMid);

  std::array<mCachedEdge, NEDGE> emHigh{};
  for (std::size_t row = 0; row < RADIX; ++row) {
    for (std::size_t col = 0; col < RADIX; ++col) {
      std::array<mCachedEdge, NEDGE> local{};
      for (std::size_t i = 0; i < RADIX; ++i) {
        for (std::size_t j = 0; j < RADIX; ++j) {
          local.at((i * RADIX) + j) =
              emMid.at(i + (row * RADIX)).at(j + (col * RADIX));
        }
      }
      emHigh.at((row * RADIX) + col) = dd.makeDDNode(qMid, local);
    }
  }

  wrapControlsUntil(dd, it, endIt, qHigh, emHigh);

  auto e = dd.makeDDNode(qHigh, emHigh);
  wrapControlsAbove(dd, it, endIt, e);
  return toMatrixDD(dd, e);
}

} // namespace

mEdge Package::makeGateDD(const GateMatrix& mat, const Qubit target) {
  return makeGateDD(mat, Controls{}, target);
}
mEdge Package::makeGateDD(const GateMatrix& mat, const Control& control,
                          const Qubit target) {
  return makeGateDD(mat, Controls{control}, target);
}
mEdge Package::makeGateDD(const GateMatrix& mat, const Controls& controls,
                          const Qubit target) {
  return buildSingleQubitGateDD(*this, mat, controls, target);
}
mEdge Package::makeGateDD(const std::span<const std::complex<fp>, NEDGE> mat,
                          const Controls& controls, const Qubit target) {
  return buildSingleQubitGateDD(*this, mat, controls, target);
}
mEdge Package::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                  const Qubit target0, const Qubit target1) {
  return makeTwoQubitGateDD(mat, Controls{}, target0, target1);
}
mEdge Package::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                  const Control& control, const Qubit target0,
                                  const Qubit target1) {
  return makeTwoQubitGateDD(mat, Controls{control}, target0, target1);
}
mEdge Package::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                  const Controls& controls, const Qubit target0,
                                  const Qubit target1) {
  return buildTwoQubitGateDD(*this, mat, controls, target0, target1);
}
mEdge Package::makeTwoQubitGateDD(
    const std::span<const std::complex<fp>,
                    static_cast<std::size_t>(NEDGE) * NEDGE>
        mat,
    const Controls& controls, const Qubit target0, const Qubit target1) {
  return buildTwoQubitGateDD(*this, mat, controls, target0, target1);
}
mEdge Package::makeThreeQubitGateDD(const ThreeQubitGateMatrix& mat,
                                    const Qubit target0, const Qubit target1,
                                    const Qubit target2) {
  return makeThreeQubitGateDD(mat, Controls{}, target0, target1, target2);
}
mEdge Package::makeThreeQubitGateDD(const ThreeQubitGateMatrix& mat,
                                    const Control& control, const Qubit target0,
                                    const Qubit target1, const Qubit target2) {
  return makeThreeQubitGateDD(mat, Controls{control}, target0, target1,
                              target2);
}
mEdge Package::makeThreeQubitGateDD(const ThreeQubitGateMatrix& mat,
                                    const Controls& controls,
                                    const Qubit target0, const Qubit target1,
                                    const Qubit target2) {
  return buildThreeQubitGateDD(*this, mat, controls, target0, target1, target2);
}
mEdge Package::makeThreeQubitGateDD(
    const std::span<const std::complex<fp>,
                    static_cast<std::size_t>(THREE_QUBIT_GATE_DIM) *
                        THREE_QUBIT_GATE_DIM>
        mat,
    const Controls& controls, const Qubit target0, const Qubit target1,
    const Qubit target2) {
  return buildThreeQubitGateDD(*this, mat, controls, target0, target1, target2);
}

mEdge Package::makeGateDD(const std::span<const std::complex<fp>> matrix,
                          const std::span<const Qubit> targets,
                          const Controls& controls) {
  if (targets.size() >= std::numeric_limits<size_t>::digits / 2 ||
      matrix.size() != (size_t{1} << (2 * targets.size()))) {
    throw std::invalid_argument("Matrix size does not match its target count.");
  }
  switch (targets.size()) {
  case 1:
    return makeGateDD(matrix.first<NEDGE>(), controls, targets[0]);
  case 2:
    return makeTwoQubitGateDD(matrix.first<NEDGE * NEDGE>(), controls,
                              targets[0], targets[1]);
  case 3:
    return makeThreeQubitGateDD(
        matrix.first<THREE_QUBIT_GATE_DIM * THREE_QUBIT_GATE_DIM>(), controls,
        targets[0], targets[1], targets[2]);
  default:
    break;
  }
  if (!controls.empty()) {
    throw std::invalid_argument(
        "Sparse controls require one to three target qubits.");
  }
  if (targets.empty()) {
    return mEdge::terminal(cn.lookup(matrix[0]));
  }
  /// The matrix-size check bounds the number of operands by the size_t width.
  std::array<std::pair<Qubit, size_t>, std::numeric_limits<size_t>::digits / 2>
      storage{};
  const auto operands = std::span{storage}.first(targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    if (targets[i] >= qubits()) {
      throwGateQubitOutOfRange(qubits());
    }
    operands[i] = {targets[i], size_t{1} << (targets.size() - 1 - i)};
  }
  std::ranges::sort(operands, {}, &std::pair<Qubit, size_t>::first);
  if (std::ranges::adjacent_find(
          operands, {}, &std::pair<Qubit, size_t>::first) != operands.end()) {
    throwGateQubitsNotDistinct();
  }
  const auto dimension = size_t{1} << targets.size();
  const auto root = buildMatrixDD(
      [matrix, dimension](const size_t row, const size_t col) {
        return matrix[(row * dimension) + col];
      },
      [&operands](const size_t level) { return operands[level]; },
      targets.size() - 1, 0, 0);
  return toMatrixDD(*this, root);
}

mEdge Package::makeDDFromMatrix(const CMat& matrix) {
  if (std::ranges::any_of(matrix, [&matrix](const auto& row) {
        return row.size() != matrix.size();
      })) {
    throw std::invalid_argument("Matrix must be square.");
  }
  return makeDDFromMatrix(matrix.size(),
                          [&matrix](const size_t row, const size_t col) {
                            return matrix[row][col];
                          });
}
void Package::clearComputeTables() {
  vectorAdd.clear();
  matrixAdd.clear();
  vectorAddMagnitudes.clear();
  matrixAddMagnitudes.clear();
  conjugateVector.clear();
  conjugateMatrixTranspose.clear();
  matrixMatrixMultiplication.clear();
  matrixVectorMultiplication.clear();
  vectorInnerProduct.clear();
  vectorKronecker.clear();
  matrixKronecker.clear();
  matrixTrace.clear();
}
std::string Package::measureAll(vEdge& rootEdge, const bool collapse,
                                std::mt19937_64& mt, const fp epsilon) {
  if (std::abs(ComplexNumbers::mag2(rootEdge.w) - 1.0) > epsilon) {
    if (rootEdge.w.approximatelyZero()) {
      throw std::runtime_error(
          "Numerical instabilities led to a 0-vector! Abort simulation!");
    }
    std::cerr << "WARNING in MAll: numerical instability occurred during "
                 "simulation: |alpha|^2 + |beta|^2 = "
              << ComplexNumbers::mag2(rootEdge.w) << ", but should be 1!\n";
  }

  if (rootEdge.isTerminal()) {
    return "";
  }

  vEdge cur = rootEdge;
  const auto numberOfQubits = static_cast<std::size_t>(rootEdge.p->v) + 1U;

  std::string result(numberOfQubits, '0');

  std::uniform_real_distribution<fp> dist(0.0, 1.0);

  for (auto i = numberOfQubits; i > 0; --i) {
    fp p0 = ComplexNumbers::mag2(cur.p->e.at(0).w);
    const fp p1 = ComplexNumbers::mag2(cur.p->e.at(1).w);
    const fp tmp = p0 + p1;

    if (std::abs(tmp - 1.0) > epsilon) {
      throw std::runtime_error("Added probabilities differ from 1 by " +
                               std::to_string(std::abs(tmp - 1.0)));
    }
    p0 /= tmp;

    const fp threshold = dist(mt);
    if (threshold < p0) {
      cur = cur.p->e.at(0);
    } else {
      result[cur.p->v] = '1';
      cur = cur.p->e.at(1);
    }
  }

  if (collapse) {
    vEdge e = vEdge::one();
    std::array<vEdge, 2> edges{};
    for (std::size_t p = 0U; p < numberOfQubits; ++p) {
      if (result[p] == '0') {
        edges[0] = e;
        edges[1] = vEdge::zero();
      } else {
        edges[0] = vEdge::zero();
        edges[1] = e;
      }
      e = makeDDNode(static_cast<Qubit>(p), edges);
    }
    incRef(e);
    decRef(rootEdge);
    rootEdge = e;
  }

  std::ranges::reverse(result);
  return result;
}
fp Package::assignProbabilities(const vEdge& edge,
                                std::unordered_map<const vNode*, fp>& probs) {
  auto const it = probs.find(edge.p);
  if (it != probs.end()) {
    return ComplexNumbers::mag2(edge.w) * it->second;
  }
  double sum{1};
  if (!edge.isTerminal()) {
    sum = assignProbabilities(edge.p->e[0], probs) +
          assignProbabilities(edge.p->e[1], probs);
  }

  probs.insert({edge.p, sum});

  return ComplexNumbers::mag2(edge.w) * sum;
}
std::pair<fp, fp>
Package::determineMeasurementProbabilities(const vEdge& rootEdge,
                                           const Qubit index) {
  checkMeasurementQubit(rootEdge, index);
  if (rootEdge.p->v == index) {
    const auto probability = ComplexNumbers::mag2(rootEdge.w);
    const auto zero = static_cast<ComplexValue>(rootEdge.p->e[0].w);
    const auto one = static_cast<ComplexValue>(rootEdge.p->e[1].w);
    return {zero.approximatelyZero() ? 0. : probability * zero.mag2(),
            one.approximatelyZero() ? 0. : probability * one.mag2()};
  }

  std::unordered_map<const vNode*, fp> measurementProbabilities;
  std::queue<const vNode*> q;

  measurementProbabilities.emplace(rootEdge.p,
                                   ComplexNumbers::mag2(rootEdge.w));
  q.push(rootEdge.p);

  while (q.front()->v != index) {
    const auto* ptr = q.front();
    q.pop();
    const fp prob = measurementProbabilities.at(ptr);

    for (const auto& edge : ptr->e) {
      const auto weight = static_cast<ComplexValue>(edge.w);
      if (weight.approximatelyZero()) {
        continue;
      }
      if (edge.isTerminal()) {
        throw std::invalid_argument("Measurement qubit is outside the state.");
      }
      const fp contribution = prob * weight.mag2();
      auto [it, inserted] =
          measurementProbabilities.try_emplace(edge.p, contribution);
      if (inserted) {
        q.push(edge.p);
      } else {
        it->second += contribution;
      }
    }
  }

  fp pzero{0};
  fp pone{0};
  while (!q.empty()) {
    const auto* ptr = q.front();
    q.pop();
    const fp prob = measurementProbabilities.at(ptr);
    const auto& s0 = ptr->e[0];
    if (const auto s0w = static_cast<ComplexValue>(s0.w);
        !s0w.approximatelyZero()) {
      pzero += prob * s0w.mag2();
    }
    const auto& s1 = ptr->e[1];
    if (const auto s1w = static_cast<ComplexValue>(s1.w);
        !s1w.approximatelyZero()) {
      pone += prob * s1w.mag2();
    }
  }

  return {pzero, pone};
}
char Package::measureOneCollapsing(vEdge& rootEdge, const Qubit index,
                                   std::mt19937_64& mt, const fp epsilon) {
  const auto& [pzero, pone] =
      determineMeasurementProbabilities(rootEdge, index);
  const fp sum = pzero + pone;
  if (std::abs(sum - 1) > epsilon) {
    throw std::runtime_error(
        "Numerical instability occurred during measurement: |alpha|^2 + "
        "|beta|^2 = " +
        std::to_string(pzero) + " + " + std::to_string(pone) + " = " +
        std::to_string(pzero + pone) + ", but should be 1!");
  }
  std::uniform_real_distribution<fp> dist(0., 1.);
  if (const auto threshold = dist(mt); threshold < pzero / sum) {
    performCollapsingMeasurement(rootEdge, index, pzero, true);
    return '0';
  }
  performCollapsingMeasurement(rootEdge, index, pone, false);
  return '1';
}
void Package::performCollapsingMeasurement(vEdge& rootEdge, const Qubit index,
                                           const fp probability,
                                           const bool measureZero) {
  checkMeasurementQubit(rootEdge, index);
  vCachedEdge projected{};
  if (rootEdge.p->v == index) {
    std::array<vCachedEdge, RADIX> edges{};
    const auto& successor = rootEdge.p->e[measureZero ? 0 : 1];
    edges[measureZero ? 0 : 1] = {successor.p, successor.w};
    projected = makeDDNode(index, edges);
    projected.w = projected.w * static_cast<ComplexValue>(rootEdge.w);
  } else {
    const auto measurementGate =
        makeGateDD(measureZero ? MEAS_ZERO_MAT : MEAS_ONE_MAT, index);
    projected = project(rootEdge, measurementGate.p, measureZero);
  }
  auto e = cn.lookup(projected);

  assert(probability > 0.);
  e.w = cn.lookup(e.w / std::sqrt(probability));
  incRef(e);
  decRef(rootEdge);
  rootEdge = e;
}
vCachedEdge Package::project(const vEdge& state, mNode* projector,
                             const bool measureZero) {
  if (state.w.exactlyZero()) {
    return vCachedEdge::zero();
  }
  if (state.isTerminal()) {
    throw std::invalid_argument("Measurement qubit is outside the state.");
  }
  if (const auto* cached =
          matrixVectorMultiplication.lookup(projector, state.p);
      cached != nullptr) {
    return {cached->p, cached->w * static_cast<ComplexValue>(state.w)};
  }

  std::array<vCachedEdge, RADIX> edges{};
  if (state.p->v == projector->v) {
    const auto& successor = state.p->e[measureZero ? 0 : 1];
    edges[measureZero ? 0 : 1] = {successor.p, successor.w};
  } else {
    edges = {
        project(state.p->e[0], projector, measureZero),
        project(state.p->e[1], projector, measureZero),
    };
  }
  auto result = makeDDNode(state.p->v, edges);
  matrixVectorMultiplication.insert(projector, state.p, result);
  result.w = result.w * static_cast<ComplexValue>(state.w);
  return result;
}

vEdge Package::conjugate(const vEdge& a) {
  const auto r = conjugateRec(a);
  return {.p = r.p, .w = cn.lookup(r.w)};
}
vCachedEdge Package::conjugateRec(const vEdge& a) {
  if (a.isZeroTerminal()) {
    return vCachedEdge::zero();
  }

  if (a.isTerminal()) {
    return {a.p, ComplexNumbers::conj(a.w)};
  }

  if (const auto* r = conjugateVector.lookup(a.p); r != nullptr) {
    return {r->p, r->w * ComplexNumbers::conj(a.w)};
  }

  std::array<vCachedEdge, 2> e{};
  e[0] = conjugateRec(a.p->e[0]);
  e[1] = conjugateRec(a.p->e[1]);
  auto res = makeDDNode(a.p->v, e);
  conjugateVector.insert(a.p, res);
  res.w = res.w * ComplexNumbers::conj(a.w);
  return res;
}
mEdge Package::conjugateTranspose(const mEdge& a) {
  const auto r = conjugateTransposeRec(a);
  return {.p = r.p, .w = cn.lookup(r.w)};
}
mCachedEdge Package::conjugateTransposeRec(const mEdge& a) {
  if (a.isTerminal()) { // terminal case
    return {a.p, ComplexNumbers::conj(a.w)};
  }

  if (const auto* r = conjugateMatrixTranspose.lookup(a.p); r != nullptr) {
    return {r->p, r->w * ComplexNumbers::conj(a.w)};
  }

  std::array<mCachedEdge, NEDGE> e{};
  // conjugate transpose submatrices and rearrange as required
  for (auto i = 0U; i < RADIX; ++i) {
    for (auto j = 0U; j < RADIX; ++j) {
      e[(RADIX * i) + j] = conjugateTransposeRec(a.p->e[(RADIX * j) + i]);
    }
  }
  auto const res = makeDDNode(a.p->v, e);

  conjugateMatrixTranspose.insert(a.p, res);

  // adjust top weight including conjugate
  return {res.p, res.w * ComplexNumbers::conj(a.w)};
}
VectorDD Package::applyOperation(const MatrixDD& operation, const VectorDD& e) {
  const auto tmp = multiply(operation, e);
  incRef(tmp);
  decRef(e);
  garbageCollect();
  return tmp;
}
MatrixDD Package::applyOperation(const MatrixDD& operation, const MatrixDD& e,
                                 const bool applyFromLeft) {
  const MatrixDD tmp =
      applyFromLeft ? multiply(operation, e) : multiply(e, operation);
  incRef(tmp);
  decRef(e);
  garbageCollect();
  return tmp;
}
ComplexValue Package::innerProduct(const vEdge& x, const vEdge& y) {
  if (x.isTerminal() || y.isTerminal() || x.w.approximatelyZero() ||
      y.w.approximatelyZero()) { // the 0 case
    return 0;
  }

  const auto w = std::max(x.p->v, y.p->v);
  // Overall normalization factor needs to be conjugated
  // before input into recursive private function
  auto const xCopy = vEdge{.p = x.p, .w = ComplexNumbers::conj(x.w)};
  return innerProduct(xCopy, y, w + 1U);
}
fp Package::fidelity(const vEdge& x, const vEdge& y) {
  return innerProduct(x, y).mag2();
}
fp Package::fidelityOfMeasurementOutcomes(const vEdge& e,
                                          const SparsePVec& probs,
                                          const Permutation& permutation) {
  if (e.w.approximatelyZero()) {
    return 0.;
  }
  return fidelityOfMeasurementOutcomesRecursive(e, probs, 0, permutation,
                                                e.p->v + 1U);
}
ComplexValue Package::innerProduct(const vEdge& x, const vEdge& y,
                                   const Qubit var) {
  const auto xWeight = static_cast<ComplexValue>(x.w);
  if (xWeight.approximatelyZero()) {
    return 0;
  }
  const auto yWeight = static_cast<ComplexValue>(y.w);
  if (yWeight.approximatelyZero()) {
    return 0;
  }

  const auto rWeight = xWeight * yWeight;
  if (var == 0) { // Multiplies terminal weights
    return rWeight;
  }

  if (const auto* r = vectorInnerProduct.lookup(x.p, y.p); r != nullptr) {
    return r->w * rWeight;
  }

  const auto w = static_cast<Qubit>(var - 1U);
  ComplexValue sum = 0;
  // Iterates through edge weights recursively until terminal
  for (auto i = 0U; i < RADIX; i++) {
    vEdge e1{};
    if (!x.isTerminal() && x.p->v == w) {
      e1 = x.p->e[i];
      e1.w = ComplexNumbers::conj(e1.w);
    } else {
      e1 = {.p = x.p, .w = Complex::one()};
    }
    vEdge e2{};
    if (!y.isTerminal() && y.p->v == w) {
      e2 = y.p->e[i];
    } else {
      e2 = {.p = y.p, .w = Complex::one()};
    }
    sum += innerProduct(e1, e2, w);
  }
  vectorInnerProduct.insert(x.p, y.p, vCachedEdge::terminal(sum));
  return sum * rWeight;
}
fp Package::fidelityOfMeasurementOutcomesRecursive(
    const vEdge& e, const SparsePVec& probs, const std::size_t i,
    const Permutation& permutation, const std::size_t nQubits) {
  const auto top = ComplexNumbers::mag(e.w);
  if (e.isTerminal()) {
    auto idx = i;
    if (!permutation.empty()) {
      const auto binaryString = intToBinaryString(i, nQubits);
      std::string filteredString(permutation.size(), '0');
      for (const auto& [physical, logical] : permutation) {
        filteredString[logical] = binaryString[physical];
      }
      idx = std::stoull(filteredString, nullptr, 2);
    }
    if (auto const it = probs.find(idx); it != probs.end()) {
      return top * std::sqrt(it->second);
    }
    return 0.;
  }

  const std::size_t leftIdx = i;
  fp leftContribution = 0.;
  if (!e.p->e[0].w.approximatelyZero()) {
    leftContribution = fidelityOfMeasurementOutcomesRecursive(
        e.p->e[0], probs, leftIdx, permutation, nQubits);
  }

  const std::size_t rightIdx = i | (1ULL << e.p->v);
  auto rightContribution = 0.;
  if (!e.p->e[1].w.approximatelyZero()) {
    rightContribution = fidelityOfMeasurementOutcomesRecursive(
        e.p->e[1], probs, rightIdx, permutation, nQubits);
  }

  return top * (leftContribution + rightContribution);
}
fp Package::expectationValue(const mEdge& x, const vEdge& y) {
  assert(!x.isZeroTerminal() && !y.isTerminal());
  if (!x.isTerminal() && x.p->v > y.p->v) {
    throw std::invalid_argument(
        "Observable must not act on more qubits than the state to compute the"
        "expectation value.");
  }

  const auto yPrime = multiply(x, y);
  const ComplexValue expValue = innerProduct(y, yPrime);

  assert(RealNumber::approximatelyZero(expValue.i));
  return expValue.r;
}
mEdge Package::partialTrace(const mEdge& a,
                            const std::vector<bool>& eliminate) {
  std::vector<size_t> eliminatedBelow(eliminate.size() + 1);
  for (size_t q = 0; q < eliminate.size(); ++q) {
    eliminatedBelow[q + 1] =
        eliminatedBelow[q] + static_cast<size_t>(eliminate[q]);
  }
  auto const r = trace(a, eliminatedBelow);
  return {.p = r.p, .w = cn.lookup(r.w)};
}
ComplexValue Package::trace(const mEdge& a, const std::size_t numQubits) {
  if (a.isIdentity()) {
    return static_cast<ComplexValue>(a.w);
  }
  std::vector<size_t> eliminatedBelow(numQubits + 1);
  std::iota(eliminatedBelow.begin(), eliminatedBelow.end(), size_t{0});
  return trace(a, eliminatedBelow).w;
}
bool Package::isCloseToIdentity(const mEdge& m, const fp tol,
                                const std::vector<bool>& garbage,
                                const bool checkCloseToOne) const {
  std::unordered_set<decltype(m.p)> visited{};
  visited.reserve(mUniqueTable.getNumEntries());
  return isCloseToIdentityRecursive(m, visited, tol, garbage, checkCloseToOne);
}
mCachedEdge Package::trace(const mEdge& a,
                           const std::span<const size_t> eliminatedBelow) {
  const auto aWeight = static_cast<ComplexValue>(a.w);
  if (aWeight.approximatelyZero()) {
    return mCachedEdge::zero();
  }
  if (a.isIdentity()) {
    return {a.p, aWeight};
  }

  const auto v = a.p->v;
  const auto below = eliminatedBelow[v];
  const auto through = eliminatedBelow[static_cast<size_t>(v) + 1];
  if (through == 0) {
    return {a.p, aWeight};
  }

  if (through != below) {
    /// Only complete traces are independent of the elimination mask.
    const bool eliminateAll = through == static_cast<size_t>(v) + 1;
    if (eliminateAll) {
      if (const auto* r = getTraceComputeTable().lookup(a.p); r != nullptr) {
        return {r->p, r->w * aWeight};
      }
    }

    const auto nextLevel = static_cast<Qubit>(v == below ? 0 : v - below - 1);
    auto r = add2(trace(a.p->e[0], eliminatedBelow),
                  trace(a.p->e[3], eliminatedBelow), nextLevel);
    r.w = r.w / 2.0;
    if (eliminateAll) {
      getTraceComputeTable().insert(a.p, r);
    }
    r.w = r.w * aWeight;
    return r;
  }

  std::array<mCachedEdge, NEDGE> edge{};
  std::ranges::transform(a.p->e, edge.begin(),
                         [this, eliminatedBelow](const mEdge& e) {
                           return trace(e, eliminatedBelow);
                         });
  auto r = makeDDNode(static_cast<Qubit>(v - below), edge);
  r.w = r.w * aWeight;
  return r;
}
bool Package::isCloseToIdentityRecursive(
    const mEdge& m, std::unordered_set<decltype(m.p)>& visited, const fp tol,
    const std::vector<bool>& garbage, const bool checkCloseToOne) {
  // immediately return if this node is identical to the identity or zero
  if (m.isTerminal()) {
    return true;
  }

  if (visited.contains(m.p)) {
    return true;
  }

  const auto n = m.p->v;

  if (garbage.size() > n && garbage[n]) {
    return isCloseToIdentityRecursive(m.p->e[0U], visited, tol, garbage,
                                      checkCloseToOne) &&
           isCloseToIdentityRecursive(m.p->e[1U], visited, tol, garbage,
                                      checkCloseToOne) &&
           isCloseToIdentityRecursive(m.p->e[2U], visited, tol, garbage,
                                      checkCloseToOne) &&
           isCloseToIdentityRecursive(m.p->e[3U], visited, tol, garbage,
                                      checkCloseToOne);
  }

  // check whether any of the middle successors is non-zero, i.e., m = [ x 0 0
  // y ]
  const auto mag1 = dd::ComplexNumbers::mag2(m.p->e[1U].w);
  const auto mag2 = dd::ComplexNumbers::mag2(m.p->e[2U].w);
  if (mag1 > tol || mag2 > tol) {
    return false;
  }

  if (checkCloseToOne) {
    // check whether  m = [ ~1 0 0 y ]
    const auto mag0 = dd::ComplexNumbers::mag2(m.p->e[0U].w);
    if (std::abs(mag0 - 1.0) > tol) {
      return false;
    }
    const auto arg0 = dd::ComplexNumbers::arg(m.p->e[0U].w);
    if (std::abs(arg0) > tol) {
      return false;
    }

    // check whether m = [ x 0 0 ~1 ] or m = [ x 0 0 ~0 ] (the last case is
    // true for an ancillary qubit)
    const auto mag3 = dd::ComplexNumbers::mag2(m.p->e[3U].w);
    if (mag3 > tol) {
      if (std::abs(mag3 - 1.0) > tol) {
        return false;
      }
      const auto arg3 = dd::ComplexNumbers::arg(m.p->e[3U].w);
      if (std::abs(arg3) > tol) {
        return false;
      }
    }
  }
  // m either has the form [ ~1 0 0 ~1 ] or [ ~1 0 0 ~0 ]
  const auto ident0 = isCloseToIdentityRecursive(m.p->e[0U], visited, tol,
                                                 garbage, checkCloseToOne);

  if (!ident0) {
    return false;
  }
  // m either has the form [ I 0 0 ~1 ] or [ I 0 0 ~0 ]
  const auto ident3 = isCloseToIdentityRecursive(m.p->e[3U], visited, tol,
                                                 garbage, checkCloseToOne);

  visited.insert(m.p);
  return ident3;
}
mEdge Package::makeIdent() { return mEdge::one(); }
mEdge Package::createInitialMatrix(const std::vector<bool>& ancillary) {
  return reduceAncillae(makeIdent(), ancillary);
}
mEdge Package::reduceAncillae(mEdge e, const std::vector<bool>& ancillary,
                              const bool regular) {
  if (std::ranges::none_of(ancillary, [](const bool v) { return v; }) ||
      e.isZeroTerminal()) {
    return e;
  }

  if (e.isIdentity()) {
    auto g = e;
    for (auto i = 0U; i < ancillary.size(); ++i) {
      if (ancillary[i]) {
        g = makeDDNode(
            static_cast<Qubit>(i),
            std::array{g, mEdge::zero(), mEdge::zero(), mEdge::zero()});
      }
    }
    incRef(g);
    return g;
  }

  Qubit lowerbound = 0;
  for (auto i = 0U; i < ancillary.size(); ++i) {
    if (ancillary[i]) {
      lowerbound = static_cast<Qubit>(i);
      break;
    }
  }

  auto g = CachedEdge<mNode>{e.p, 1.};
  if (e.p->v >= lowerbound) {
    g = reduceAncillaeRecursion(e.p, ancillary, lowerbound, regular);
  }

  for (std::size_t i = e.p->v + 1; i < ancillary.size(); ++i) {
    if (ancillary[i]) {
      g = makeDDNode(static_cast<Qubit>(i), std::array{
                                                g,
                                                mCachedEdge::zero(),
                                                mCachedEdge::zero(),
                                                mCachedEdge::zero(),
                                            });
    }
  }
  const auto res = mEdge{.p = g.p, .w = cn.lookup(g.w * e.w)};
  incRef(res);
  decRef(e);
  return res;
}
vEdge Package::reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                             const bool normalizeWeights) {
  if (!normalizeWeights &&
      (std::ranges::none_of(garbage, [](bool v) { return v; }) ||
       e.isTerminal())) {
    return e;
  }
  Qubit lowerbound = 0;
  for (std::size_t i = 0U; i < garbage.size(); ++i) {
    if (garbage[i]) {
      lowerbound = static_cast<Qubit>(i);
      break;
    }
  }
  if (!normalizeWeights && e.p->v < lowerbound) {
    return e;
  }
  const auto f =
      reduceGarbageRecursion(e.p, garbage, lowerbound, normalizeWeights);
  auto weight = e.w * f.w;
  if (normalizeWeights) {
    weight = weight.mag();
  }
  const auto res = vEdge{.p = f.p, .w = cn.lookup(weight)};
  incRef(res);
  decRef(e);
  return res;
}
mEdge Package::reduceGarbage(const mEdge& e, const std::vector<bool>& garbage,
                             const bool regular, const bool normalizeWeights) {
  if (!normalizeWeights &&
      (std::ranges::none_of(garbage, [](bool v) { return v; }) ||
       e.isZeroTerminal())) {
    return e;
  }

  if (e.isIdentity()) {
    auto g = e;
    for (auto i = 0U; i < garbage.size(); ++i) {
      if (garbage[i]) {
        if (regular) {
          g = makeDDNode(static_cast<Qubit>(i),
                         std::array{g, g, mEdge::zero(), mEdge::zero()});
        } else {
          g = makeDDNode(static_cast<Qubit>(i),
                         std::array{g, mEdge::zero(), g, mEdge::zero()});
        }
      }
    }
    incRef(g);
    return g;
  }

  Qubit lowerbound = 0;
  for (auto i = 0U; i < garbage.size(); ++i) {
    if (garbage[i]) {
      lowerbound = static_cast<Qubit>(i);
      break;
    }
  }

  auto g = CachedEdge<mNode>{e.p, 1.};
  if (e.p->v >= lowerbound || normalizeWeights) {
    g = reduceGarbageRecursion(e.p, garbage, lowerbound, regular,
                               normalizeWeights);
  }

  for (std::size_t i = e.p->v + 1; i < garbage.size(); ++i) {
    if (garbage[i]) {
      if (regular) {
        g = makeDDNode(
            static_cast<Qubit>(i),
            std::array{g, g, mCachedEdge::zero(), mCachedEdge::zero()});
      } else {
        g = makeDDNode(
            static_cast<Qubit>(i),
            std::array{g, mCachedEdge::zero(), g, mCachedEdge::zero()});
      }
    }
  }

  auto weight = g.w * e.w;
  if (normalizeWeights) {
    weight = weight.mag();
  }
  const auto res = mEdge{.p = g.p, .w = cn.lookup(weight)};

  incRef(res);
  decRef(e);
  return res;
}
mCachedEdge Package::reduceAncillaeRecursion(mNode* p,
                                             const std::vector<bool>& ancillary,
                                             const Qubit lowerbound,
                                             const bool regular) {
  if (p->v < lowerbound) {
    return {p, 1.};
  }

  std::array<mCachedEdge, NEDGE> edges{};
  std::bitset<NEDGE> handled{};
  for (auto i = 0U; i < NEDGE; ++i) {
    // no need to reduce ancillaries for entries that will be zeroed anyway
    if (ancillary[p->v] &&
        ((i == 3) || (i == 1 && regular) || (i == 2 && !regular))) {
      continue;
    }
    if (handled.test(i)) {
      continue;
    }

    if (p->e[i].isZeroTerminal()) {
      edges[i] = {p->e[i].p, p->e[i].w};
      handled.set(i);
      continue;
    }

    if (p->e[i].isIdentity()) {
      auto g = mCachedEdge::one();
      for (auto j = lowerbound; j < p->v; ++j) {
        if (ancillary[j]) {
          g = makeDDNode(j, std::array{
                                g,
                                mCachedEdge::zero(),
                                mCachedEdge::zero(),
                                mCachedEdge::zero(),
                            });
        }
      }
      edges[i] = {g.p, p->e[i].w};
      handled.set(i);
      continue;
    }

    edges[i] =
        reduceAncillaeRecursion(p->e[i].p, ancillary, lowerbound, regular);
    for (Qubit j = p->e[i].p->v + 1U; j < p->v; ++j) {
      if (ancillary[j]) {
        edges[i] = makeDDNode(j, std::array{
                                     edges[i],
                                     mCachedEdge::zero(),
                                     mCachedEdge::zero(),
                                     mCachedEdge::zero(),
                                 });
      }
    }

    for (auto j = i + 1U; j < NEDGE; ++j) {
      if (p->e[i].p == p->e[j].p) {
        edges[j] = edges[i];
        edges[j].w = edges[j].w * p->e[j].w;
        handled.set(j);
      }
    }
    edges[i].w = edges[i].w * p->e[i].w;
    handled.set(i);
  }
  if (!ancillary[p->v]) {
    return makeDDNode(p->v, edges);
  }

  // something to reduce for this qubit
  if (regular) {
    return makeDDNode(p->v, std::array{
                                edges[0],
                                mCachedEdge::zero(),
                                edges[2],
                                mCachedEdge::zero(),
                            });
  }
  return makeDDNode(p->v, std::array{
                              edges[0],
                              edges[1],
                              mCachedEdge::zero(),
                              mCachedEdge::zero(),
                          });
}
vCachedEdge Package::reduceGarbageRecursion(vNode* p,
                                            const std::vector<bool>& garbage,
                                            const Qubit lowerbound,
                                            const bool normalizeWeights) {
  if (!normalizeWeights && p->v < lowerbound) {
    return {p, 1.};
  }

  std::array<vCachedEdge, RADIX> edges{};
  std::bitset<RADIX> handled{};
  for (auto i = 0U; i < RADIX; ++i) {
    if (!handled.test(i)) {
      if (p->e[i].isTerminal()) {
        const auto weight = normalizeWeights
                                ? ComplexNumbers::mag(p->e[i].w)
                                : static_cast<ComplexValue>(p->e[i].w);
        edges[i] = {p->e[i].p, weight};
      } else {
        edges[i] = reduceGarbageRecursion(p->e[i].p, garbage, lowerbound,
                                          normalizeWeights);
        for (auto j = i + 1; j < RADIX; ++j) {
          if (p->e[i].p == p->e[j].p) {
            edges[j] = edges[i];
            edges[j].w = edges[j].w * p->e[j].w;
            if (normalizeWeights) {
              edges[j].w = edges[j].w.mag();
            }
            handled.set(j);
          }
        }
        edges[i].w = edges[i].w * p->e[i].w;
        if (normalizeWeights) {
          edges[i].w = edges[i].w.mag();
        }
      }
      handled.set(i);
    }
  }
  if (!garbage[p->v]) {
    return makeDDNode(p->v, edges);
  }
  // something to reduce for this qubit
  return makeDDNode(p->v, std::array{
                              addMagnitudes(edges[0], edges[1], p->v - 1),
                              vCachedEdge::zero(),
                          });
}
mCachedEdge Package::reduceGarbageRecursion(mNode* p,
                                            const std::vector<bool>& garbage,
                                            const Qubit lowerbound,
                                            const bool regular,
                                            const bool normalizeWeights) {
  if (!normalizeWeights && p->v < lowerbound) {
    return {p, 1.};
  }

  std::array<mCachedEdge, NEDGE> edges{};
  std::bitset<NEDGE> handled{};
  for (auto i = 0U; i < NEDGE; ++i) {
    if (handled.test(i)) {
      continue;
    }

    if (p->e[i].isZeroTerminal()) {
      edges[i] = mCachedEdge::zero();
      handled.set(i);
      continue;
    }

    if (p->e[i].isIdentity()) {
      edges[i] = mCachedEdge::one();
      for (auto j = lowerbound; j < p->v; ++j) {
        if (garbage[j]) {
          if (regular) {
            edges[i] = makeDDNode(j, std::array{
                                         edges[i],
                                         edges[i],
                                         mCachedEdge::zero(),
                                         mCachedEdge::zero(),
                                     });
          } else {
            edges[i] = makeDDNode(j, std::array{
                                         edges[i],
                                         mCachedEdge::zero(),
                                         edges[i],
                                         mCachedEdge::zero(),
                                     });
          }
        }
      }
      if (normalizeWeights) {
        edges[i].w = edges[i].w * ComplexNumbers::mag(p->e[i].w);
      } else {
        edges[i].w = edges[i].w * p->e[i].w;
      }
      handled.set(i);
      continue;
    }

    edges[i] = reduceGarbageRecursion(p->e[i].p, garbage, lowerbound, regular,
                                      normalizeWeights);
    for (Qubit j = p->e[i].p->v + 1U; j < p->v; ++j) {
      if (garbage[j]) {
        if (regular) {
          edges[i] = makeDDNode(j, std::array{
                                       edges[i],
                                       edges[i],
                                       mCachedEdge::zero(),
                                       mCachedEdge::zero(),
                                   });
        } else {
          edges[i] = makeDDNode(j, std::array{
                                       edges[i],
                                       mCachedEdge::zero(),
                                       edges[i],
                                       mCachedEdge::zero(),
                                   });
        }
      }
    }

    for (auto j = i + 1; j < NEDGE; ++j) {
      if (p->e[i].p == p->e[j].p) {
        edges[j] = edges[i];
        edges[j].w = edges[j].w * p->e[j].w;
        if (normalizeWeights) {
          edges[j].w = edges[j].w.mag();
        }
        handled.set(j);
      }
    }
    edges[i].w = edges[i].w * p->e[i].w;
    if (normalizeWeights) {
      edges[i].w = edges[i].w.mag();
    }
    handled.set(i);
  }
  if (!garbage[p->v]) {
    return makeDDNode(p->v, edges);
  }

  if (regular) {
    return makeDDNode(p->v, std::array{
                                addMagnitudes(edges[0], edges[2], p->v - 1),
                                addMagnitudes(edges[1], edges[3], p->v - 1),
                                mCachedEdge::zero(),
                                mCachedEdge::zero(),
                            });
  }
  return makeDDNode(p->v, std::array{
                              addMagnitudes(edges[0], edges[1], p->v - 1),
                              mCachedEdge::zero(),
                              addMagnitudes(edges[2], edges[3], p->v - 1),
                              mCachedEdge::zero(),
                          });
}
} // namespace dd
