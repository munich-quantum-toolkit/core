/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include "dd/DensityNoiseTable.hpp"
#include "dd/Edge.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/StochasticNoiseOperationTable.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "dd/UniqueTable.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/Control.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dd {
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
  dUniqueTable.resize(nqubits);
  stochasticNoiseOperationCache.resize(nqubits);
}

void Package::reset() {
  clearUniqueTables();
  resetMemoryManagers();
  clearComputeTables();
  mark.reset();
}

void Package::resetMemoryManagers(const bool resizeToTotal) {
  vMemoryManager.reset(resizeToTotal);
  mMemoryManager.reset(resizeToTotal);
  dMemoryManager.reset(resizeToTotal);
  cMemoryManager.reset(resizeToTotal);
}

void Package::clearUniqueTables() {
  vUniqueTable.clear();
  mUniqueTable.clear();
  dUniqueTable.clear();
  cUniqueTable.clear();
}

bool Package::garbageCollect(bool force) {
  using flags = std::tuple<bool, bool, bool, bool>;

  // return immediately if no table needs collection
  if (!force && !vUniqueTable.possiblyNeedsCollection() &&
      !mUniqueTable.possiblyNeedsCollection() &&
      !dUniqueTable.possiblyNeedsCollection() &&
      !cUniqueTable.possiblyNeedsCollection()) {
    return false;
  }

  const auto sweep = [this, &force]() -> flags {
    const bool invC = cUniqueTable.garbageCollect(force) > 0;
    force |= invC;
    const bool invV = vUniqueTable.garbageCollect(force) > 0;
    const bool invM = mUniqueTable.garbageCollect(force) > 0;
    const bool invD = dUniqueTable.garbageCollect(force) > 0;
    return {invC, invV, invM, invD};
  };

  const auto [invC, invV, invM, invD] = mark.execute<flags>(sweep);

  // invalidate all compute tables involving vectors if any vector node has
  // been collected
  if (invV) {
    vectorAdd.clear();
    vectorInnerProduct.clear();
    vectorKronecker.clear();
    matrixVectorMultiplication.clear();
  }
  // invalidate all compute tables involving matrices if any matrix node has
  // been collected
  if (invM) {
    matrixAdd.clear();
    conjugateMatrixTranspose.clear();
    matrixKronecker.clear();
    matrixTrace.clear();
    matrixVectorMultiplication.clear();
    matrixMatrixMultiplication.clear();
    stochasticNoiseOperationCache.clear();
  }
  // invalidate all compute tables involving density matrices if any density
  // matrix node has been collected
  if (invD) {
    densityAdd.clear();
    densityDensityMultiplication.clear();
    densityNoise.clear();
    densityTrace.clear();
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
    stochasticNoiseOperationCache.clear();
    densityAdd.clear();
    densityDensityMultiplication.clear();
    densityNoise.clear();
    densityTrace.clear();
  }
  return invC || invV || invM || invD;
}

Package::ActiveCounts Package::computeActiveCounts() {
  const auto count = [this]() -> ActiveCounts {
    return {
        vUniqueTable.countMarkedEntries(), mUniqueTable.countMarkedEntries(),
        dUniqueTable.countMarkedEntries(), cUniqueTable.countMarkedEntries()};
  };
  return mark.execute<ActiveCounts>(count);
}

dEdge Package::makeZeroDensityOperator(const std::size_t n) {
  auto f = dEdge::one();
  for (std::size_t p = 0; p < n; p++) {
    f = makeDDNode(static_cast<Qubit>(p),
                   std::array{f, dEdge::zero(), dEdge::zero(), dEdge::zero()});
  }
  incRef(f);
  return f;
}

mEdge Package::makeGateDD(const GateMatrix& mat, const qc::Qubit target) {
  return makeGateDD(mat, qc::Controls{}, target);
}
mEdge Package::makeGateDD(const GateMatrix& mat, const qc::Control& control,
                          const qc::Qubit target) {
  return makeGateDD(mat, qc::Controls{control}, target);
}
mEdge Package::makeGateDD(const GateMatrix& mat, const qc::Controls& controls,
                          const qc::Qubit target) {
  if (std::any_of(controls.begin(), controls.end(),
                  [this](const auto& c) {
                    return c.qubit > static_cast<Qubit>(nqubits - 1U);
                  }) ||
      target > static_cast<Qubit>(nqubits - 1U)) {
    throw std::runtime_error{
        "Requested gate acting on qubit(s) with index larger than " +
        std::to_string(nqubits - 1U) +
        " while the package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }
  std::array<mCachedEdge, NEDGE> em{};
  for (auto i = 0U; i < NEDGE; ++i) {
    em[i] = mCachedEdge::terminal(mat[i]);
  }

  if (controls.empty()) {
    // Single qubit operation
    const auto e = makeDDNode(static_cast<Qubit>(target), em);
    return {e.p, cn.lookup(e.w)};
  }

  auto it = controls.begin();
  auto edges = std::array{mCachedEdge::zero(), mCachedEdge::zero(),
                          mCachedEdge::zero(), mCachedEdge::zero()};

  // process lines below target
  for (; it != controls.end() && it->qubit < target; ++it) {
    for (auto i1 = 0U; i1 < RADIX; ++i1) {
      for (auto i2 = 0U; i2 < RADIX; ++i2) {
        const auto i = (i1 * RADIX) + i2;
        if (it->type == qc::Control::Type::Neg) { // neg. control
          edges[0] = em[i];
          edges[3] = (i1 == i2) ? mCachedEdge::one() : mCachedEdge::zero();
        } else { // pos. control
          edges[0] = (i1 == i2) ? mCachedEdge::one() : mCachedEdge::zero();
          edges[3] = em[i];
        }
        em[i] = makeDDNode(static_cast<Qubit>(it->qubit), edges);
      }
    }
  }

  // target line
  auto e = makeDDNode(static_cast<Qubit>(target), em);

  // process lines above target
  for (; it != controls.end(); ++it) {
    if (it->type == qc::Control::Type::Neg) { // neg. control
      edges[0] = e;
      edges[3] = mCachedEdge::one();
      e = makeDDNode(static_cast<Qubit>(it->qubit), edges);
    } else { // pos. control
      edges[0] = mCachedEdge::one();
      edges[3] = e;
      e = makeDDNode(static_cast<Qubit>(it->qubit), edges);
    }
  }
  return {e.p, cn.lookup(e.w)};
}
mEdge Package::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                  const qc::Qubit target0,
                                  const qc::Qubit target1) {
  return makeTwoQubitGateDD(mat, qc::Controls{}, target0, target1);
}
mEdge Package::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                  const qc::Control& control,
                                  const qc::Qubit target0,
                                  const qc::Qubit target1) {
  return makeTwoQubitGateDD(mat, qc::Controls{control}, target0, target1);
}
mEdge Package::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                  const qc::Controls& controls,
                                  const qc::Qubit target0,
                                  const qc::Qubit target1) {
  // sanity check
  if (std::any_of(controls.begin(), controls.end(),
                  [this](const auto& c) {
                    return c.qubit > static_cast<Qubit>(nqubits - 1U);
                  }) ||
      target0 > static_cast<Qubit>(nqubits - 1U) ||
      target1 > static_cast<Qubit>(nqubits - 1U)) {
    throw std::runtime_error{
        "Requested gate acting on qubit(s) with index larger than " +
        std::to_string(nqubits - 1U) +
        " while the package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }

  // create terminal edge matrix
  std::array<std::array<mCachedEdge, NEDGE>, NEDGE> em{};
  for (auto i1 = 0U; i1 < NEDGE; i1++) {
    const auto& matRow = mat.at(i1);
    auto& emRow = em.at(i1);
    for (auto i2 = 0U; i2 < NEDGE; i2++) {
      emRow.at(i2) = mCachedEdge::terminal(matRow.at(i2));
    }
  }

  // process lines below smaller target
  auto it = controls.begin();
  const auto smallerTarget = std::min(target0, target1);

  auto edges = std::array{mCachedEdge::zero(), mCachedEdge::zero(),
                          mCachedEdge::zero(), mCachedEdge::zero()};

  for (; it != controls.end() && it->qubit < smallerTarget; ++it) {
    for (auto row = 0U; row < NEDGE; ++row) {
      for (auto col = 0U; col < NEDGE; ++col) {
        if (it->type == qc::Control::Type::Neg) { // negative control
          edges[0] = em[row][col];
          edges[3] = (row == col) ? mCachedEdge::one() : mCachedEdge::zero();
        } else { // positive control
          edges[0] = (row == col) ? mCachedEdge::one() : mCachedEdge::zero();
          edges[3] = em[row][col];
        }
        em[row][col] = makeDDNode(static_cast<Qubit>(it->qubit), edges);
      }
    }
  }

  // process the smaller target by taking the 16 submatrices and appropriately
  // combining them into four DDs.
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
      em0.at((row * RADIX) + col) =
          makeDDNode(static_cast<Qubit>(smallerTarget), local);
    }
  }

  // process lines between the two targets
  const auto largerTarget = std::max(target0, target1);
  for (; it != controls.end() && it->qubit < largerTarget; ++it) {
    for (auto i = 0U; i < NEDGE; ++i) {
      if (it->type == qc::Control::Type::Neg) { // negative control
        edges[0] = em0[i];
        edges[3] =
            (i == 0 || i == 3) ? mCachedEdge::one() : mCachedEdge::zero();
      } else { // positive control
        edges[0] =
            (i == 0 || i == 3) ? mCachedEdge::one() : mCachedEdge::zero();
        edges[3] = em0[i];
      }
      em0[i] = makeDDNode(static_cast<Qubit>(it->qubit), edges);
    }
  }

  // process the larger target by combining the four DDs from the smaller
  // target
  auto e = makeDDNode(static_cast<Qubit>(largerTarget), em0);

  // process lines above the larger target
  for (; it != controls.end(); ++it) {
    if (it->type == qc::Control::Type::Neg) { // negative control
      edges[0] = e;
      edges[3] = mCachedEdge::one();
    } else { // positive control
      edges[0] = mCachedEdge::one();
      edges[3] = e;
    }
    e = makeDDNode(static_cast<Qubit>(it->qubit), edges);
  }

  return {e.p, cn.lookup(e.w)};
}
mEdge Package::makeDDFromMatrix(const CMat& matrix) {
  if (matrix.empty()) {
    return mEdge::one();
  }

  const auto& length = matrix.size();
  if ((length & (length - 1)) != 0) {
    throw std::invalid_argument("Matrix must have a length of a power of two.");
  }

  const auto& width = matrix[0].size();
  if (length != width) {
    throw std::invalid_argument("Matrix must be square.");
  }

  if (length == 1) {
    return mEdge::terminal(cn.lookup(matrix[0][0]));
  }

  const auto level = static_cast<Qubit>(std::log2(length) - 1);
  const auto matrixDD = makeDDFromMatrix(matrix, level, 0, length, 0, width);
  return {matrixDD.p, cn.lookup(matrixDD.w)};
}
mCachedEdge Package::makeDDFromMatrix(const CMat& matrix, const Qubit level,
                                      const std::size_t rowStart,
                                      const std::size_t rowEnd,
                                      const std::size_t colStart,
                                      const std::size_t colEnd) {
  // base case
  if (level == 0U) {
    assert(rowEnd - rowStart == 2);
    assert(colEnd - colStart == 2);
    return makeDDNode<mNode, CachedEdge>(
        0U, {mCachedEdge::terminal(matrix[rowStart][colStart]),
             mCachedEdge::terminal(matrix[rowStart][colStart + 1]),
             mCachedEdge::terminal(matrix[rowStart + 1][colStart]),
             mCachedEdge::terminal(matrix[rowStart + 1][colStart + 1])});
  }

  // recursively call the function on all quadrants
  const auto rowMid = (rowStart + rowEnd) / 2;
  const auto colMid = (colStart + colEnd) / 2;
  const auto l = static_cast<Qubit>(level - 1U);

  return makeDDNode<mNode, CachedEdge>(
      level, {makeDDFromMatrix(matrix, l, rowStart, rowMid, colStart, colMid),
              makeDDFromMatrix(matrix, l, rowStart, rowMid, colMid, colEnd),
              makeDDFromMatrix(matrix, l, rowMid, rowEnd, colStart, colMid),
              makeDDFromMatrix(matrix, l, rowMid, rowEnd, colMid, colEnd)});
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

  stochasticNoiseOperationCache.clear();
  densityAdd.clear();
  densityDensityMultiplication.clear();
  densityNoise.clear();
  densityTrace.clear();
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

  std::uniform_real_distribution<fp> dist(0.0, 1.0L);

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

  return std::string{result.rbegin(), result.rend()};
}
fp Package::assignProbabilities(const vEdge& edge,
                                std::unordered_map<const vNode*, fp>& probs) {
  auto it = probs.find(edge.p);
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
  std::map<const vNode*, fp> measurementProbabilities;
  std::set<const vNode*> visited;
  std::queue<const vNode*> q;

  measurementProbabilities[rootEdge.p] = ComplexNumbers::mag2(rootEdge.w);
  visited.insert(rootEdge.p);
  q.push(rootEdge.p);

  while (q.front()->v != index) {
    const auto* ptr = q.front();
    q.pop();
    const fp prob = measurementProbabilities[ptr];

    const auto& s0 = ptr->e[0];
    if (const auto s0w = static_cast<ComplexValue>(s0.w);
        !s0w.approximatelyZero()) {
      const fp tmp1 = prob * s0w.mag2();
      if (visited.find(s0.p) != visited.end()) {
        measurementProbabilities[s0.p] = measurementProbabilities[s0.p] + tmp1;
      } else {
        measurementProbabilities[s0.p] = tmp1;
        visited.insert(s0.p);
        q.push(s0.p);
      }
    }

    const auto& s1 = ptr->e[1];
    if (const auto s1w = static_cast<ComplexValue>(s1.w);
        !s1w.approximatelyZero()) {
      const fp tmp1 = prob * s1w.mag2();
      if (visited.find(s1.p) != visited.end()) {
        measurementProbabilities[s1.p] = measurementProbabilities[s1.p] + tmp1;
      } else {
        measurementProbabilities[s1.p] = tmp1;
        visited.insert(s1.p);
        q.push(s1.p);
      }
    }
  }

  fp pzero{0};
  fp pone{0};
  while (!q.empty()) {
    const auto* ptr = q.front();
    q.pop();
    const auto& s0 = ptr->e[0];
    if (const auto s0w = static_cast<ComplexValue>(s0.w);
        !s0w.approximatelyZero()) {
      pzero += measurementProbabilities[ptr] * s0w.mag2();
    }
    const auto& s1 = ptr->e[1];
    if (const auto s1w = static_cast<ComplexValue>(s1.w);
        !s1w.approximatelyZero()) {
      pone += measurementProbabilities[ptr] * s1w.mag2();
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
char Package::measureOneCollapsing(dEdge& e, const Qubit index,
                                   std::mt19937_64& mt) {
  char measuredResult = '0';
  dEdge::alignDensityEdge(e);
  const auto nrQubits = e.p->v + 1U;
  dEdge::setDensityMatrixTrue(e);

  auto const measZeroDd = makeGateDD(MEAS_ZERO_MAT, index);

  auto tmp0 = conjugateTranspose(measZeroDd);
  auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
  auto tmp2 = multiply(densityFromMatrixEdge(measZeroDd), tmp1, true);
  auto densityMatrixTrace = trace(tmp2, nrQubits);

  std::uniform_real_distribution<fp> dist(0., 1.);
  if (const auto threshold = dist(mt); threshold > densityMatrixTrace.r) {
    auto const measOneDd = makeGateDD(MEAS_ONE_MAT, index);
    tmp0 = conjugateTranspose(measOneDd);
    tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
    tmp2 = multiply(densityFromMatrixEdge(measOneDd), tmp1, true);
    measuredResult = '1';
    densityMatrixTrace = trace(tmp2, nrQubits);
  }

  incRef(tmp2);
  dEdge::alignDensityEdge(e);
  decRef(e);
  e = tmp2;
  dEdge::setDensityMatrixTrue(e);

  // Normalize density matrix
  e.w = cn.lookup(e.w / densityMatrixTrace);
  return measuredResult;
}
void Package::performCollapsingMeasurement(vEdge& rootEdge, const Qubit index,
                                           const fp probability,
                                           const bool measureZero) {
  const GateMatrix measurementMatrix =
      measureZero ? MEAS_ZERO_MAT : MEAS_ONE_MAT;

  const auto measurementGate = makeGateDD(measurementMatrix, index);

  vEdge e = multiply(measurementGate, rootEdge);

  assert(probability > 0.);
  e.w = cn.lookup(e.w / std::sqrt(probability));
  incRef(e);
  decRef(rootEdge);
  rootEdge = e;
}
vEdge Package::conjugate(const vEdge& a) {
  const auto r = conjugateRec(a);
  return {r.p, cn.lookup(r.w)};
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
  return {r.p, cn.lookup(r.w)};
}
mCachedEdge Package::conjugateTransposeRec(const mEdge& a) {
  if (a.isTerminal()) { // terminal case
    return {a.p, ComplexNumbers::conj(a.w)};
  }

  // check if in compute table
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
  // create new top node
  auto res = makeDDNode(a.p->v, e);

  // put it in the compute table
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
dEdge Package::applyOperationToDensity(dEdge& e, const mEdge& operation) {
  const auto tmp0 = conjugateTranspose(operation);
  const auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
  const auto tmp2 = multiply(densityFromMatrixEdge(operation), tmp1, true);
  incRef(tmp2);
  dEdge::alignDensityEdge(e);
  decRef(e);
  e = tmp2;
  dEdge::setDensityMatrixTrue(e);
  return e;
}
ComplexValue Package::innerProduct(const vEdge& x, const vEdge& y) {
  if (x.isTerminal() || y.isTerminal() || x.w.approximatelyZero() ||
      y.w.approximatelyZero()) { // the 0 case
    return 0;
  }

  const auto w = std::max(x.p->v, y.p->v);
  // Overall normalization factor needs to be conjugated
  // before input into recursive private function
  auto xCopy = vEdge{x.p, ComplexNumbers::conj(x.w)};
  return innerProduct(xCopy, y, w + 1U);
}
fp Package::fidelity(const vEdge& x, const vEdge& y) {
  return innerProduct(x, y).mag2();
}
fp Package::fidelityOfMeasurementOutcomes(const vEdge& e,
                                          const SparsePVec& probs,
                                          const qc::Permutation& permutation) {
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
      e1 = {x.p, Complex::one()};
    }
    vEdge e2{};
    if (!y.isTerminal() && y.p->v == w) {
      e2 = y.p->e[i];
    } else {
      e2 = {y.p, Complex::one()};
    }
    sum += innerProduct(e1, e2, w);
  }
  vectorInnerProduct.insert(x.p, y.p, vCachedEdge::terminal(sum));
  return sum * rWeight;
}
fp Package::fidelityOfMeasurementOutcomesRecursive(
    const vEdge& e, const SparsePVec& probs, const std::size_t i,
    const qc::Permutation& permutation, const std::size_t nQubits) {
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
    if (auto it = probs.find(idx); it != probs.end()) {
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
  auto r = trace(a, eliminate, eliminate.size());
  return {r.p, cn.lookup(r.w)};
}
bool Package::isCloseToIdentity(const mEdge& m, const fp tol,
                                const std::vector<bool>& garbage,
                                const bool checkCloseToOne) const {
  std::unordered_set<decltype(m.p)> visited{};
  visited.reserve(mUniqueTable.getNumEntries());
  return isCloseToIdentityRecursive(m, visited, tol, garbage, checkCloseToOne);
}
bool Package::isCloseToIdentityRecursive(
    const mEdge& m, std::unordered_set<decltype(m.p)>& visited, const fp tol,
    const std::vector<bool>& garbage, const bool checkCloseToOne) {
  // immediately return if this node is identical to the identity or zero
  if (m.isTerminal()) {
    return true;
  }

  // immediately return if this node has already been visited
  if (visited.find(m.p) != visited.end()) {
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
  // return if no more ancillaries left
  if (std::none_of(ancillary.begin(), ancillary.end(),
                   [](const bool v) { return v; }) ||
      e.isZeroTerminal()) {
    return e;
  }

  // if we have only identities and no other nodes
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
      g = makeDDNode(static_cast<Qubit>(i),
                     std::array{g, mCachedEdge::zero(), mCachedEdge::zero(),
                                mCachedEdge::zero()});
    }
  }
  const auto res = mEdge{g.p, cn.lookup(g.w * e.w)};
  incRef(res);
  decRef(e);
  return res;
}
vEdge Package::reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                             const bool normalizeWeights) {
  // return if no more garbage left
  if (!normalizeWeights &&
      (std::none_of(garbage.begin(), garbage.end(), [](bool v) { return v; }) ||
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
  const auto res = vEdge{f.p, cn.lookup(weight)};
  incRef(res);
  decRef(e);
  return res;
}
mEdge Package::reduceGarbage(const mEdge& e, const std::vector<bool>& garbage,
                             const bool regular, const bool normalizeWeights) {
  // return if no more garbage left
  if (!normalizeWeights &&
      (std::none_of(garbage.begin(), garbage.end(), [](bool v) { return v; }) ||
       e.isZeroTerminal())) {
    return e;
  }

  // if we have only identities and no other nodes
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
  const auto res = mEdge{g.p, cn.lookup(weight)};

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
    if (ancillary[p->v]) {
      // no need to reduce ancillaries for entries that will be zeroed anyway
      if ((i == 3) || (i == 1 && regular) || (i == 2 && !regular)) {
        continue;
      }
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
          g = makeDDNode(j,
                         std::array{g, mCachedEdge::zero(), mCachedEdge::zero(),
                                    mCachedEdge::zero()});
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
        edges[i] =
            makeDDNode(j, std::array{edges[i], mCachedEdge::zero(),
                                     mCachedEdge::zero(), mCachedEdge::zero()});
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
    return makeDDNode(p->v, std::array{edges[0], mCachedEdge::zero(), edges[2],
                                       mCachedEdge::zero()});
  }
  return makeDDNode(p->v, std::array{edges[0], edges[1], mCachedEdge::zero(),
                                     mCachedEdge::zero()});
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
  return makeDDNode(p->v,
                    std::array{addMagnitudes(edges[0], edges[1], p->v - 1),
                               vCachedEdge ::zero()});
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
            edges[i] = makeDDNode(j, std::array{edges[i], edges[i],
                                                mCachedEdge::zero(),
                                                mCachedEdge::zero()});
          } else {
            edges[i] = makeDDNode(j, std::array{edges[i], mCachedEdge::zero(),
                                                edges[i], mCachedEdge::zero()});
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
          edges[i] =
              makeDDNode(j, std::array{edges[i], edges[i], mCachedEdge::zero(),
                                       mCachedEdge::zero()});
        } else {
          edges[i] = makeDDNode(j, std::array{edges[i], mCachedEdge::zero(),
                                              edges[i], mCachedEdge::zero()});
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
    return makeDDNode(p->v,
                      std::array{addMagnitudes(edges[0], edges[2], p->v - 1),
                                 addMagnitudes(edges[1], edges[3], p->v - 1),
                                 mCachedEdge::zero(), mCachedEdge::zero()});
  }
  return makeDDNode(p->v,
                    std::array{addMagnitudes(edges[0], edges[1], p->v - 1),
                               mCachedEdge::zero(),
                               addMagnitudes(edges[2], edges[3], p->v - 1),
                               mCachedEdge::zero()});
}
} // namespace dd
