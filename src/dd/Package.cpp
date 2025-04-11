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
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
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
  vectors().resize(nqubits);
  matrices().resize(nqubits);
  densities().resize(nqubits);
  stochasticNoiseOperationCache.resize(nqubits);
}

void Package::reset() {
  cUt.clear();

  vectors().reset();
  matrices().reset();
  densities().reset();
}

bool Package::garbageCollect(bool force) {
  // return immediately if no table needs collection
  if (!force && !cUt.possiblyNeedsCollection() &&
      !matrices().possiblyNeedsCollection() &&
      !densities().possiblyNeedsCollection() &&
      !vectors().possiblyNeedsCollection()) {
    return false;
  }

  const auto cCollect = cUt.garbageCollect(force);
  if (cCollect > 0) {
    // Collecting garbage in the complex numbers table requires collecting the
    // node tables as well
    force = true;
  }
  const auto vCollect = vectors().garbageCollect(force);
  const auto mCollect = matrices().garbageCollect(force);
  const auto dCollect = densities().garbageCollect(force);

  // invalidate all compute tables involving vectors if any vector node has
  // been collected
  if (vCollect > 0) {
    matrixVectorMultiplication.clear();
  }
  // invalidate all compute tables involving matrices if any matrix node has
  // been collected
  if (mCollect > 0) {
    matrixTrace.clear();
    matrixVectorMultiplication.clear();
    matrixMatrixMultiplication.clear();
    stochasticNoiseOperationCache.clear();
  }
  // invalidate all compute tables involving density matrices if any density
  // matrix node has been collected
  if (dCollect > 0) {
    densityDensityMultiplication.clear();
    densityNoise.clear();
    densityTrace.clear();
  }
  // invalidate all compute tables where any component of the entry contains
  // numbers from the complex table if any complex numbers were collected
  if (cCollect > 0) {
    matrixVectorMultiplication.clear();
    matrixMatrixMultiplication.clear();
    matrixTrace.clear();
    stochasticNoiseOperationCache.clear();
    densityDensityMultiplication.clear();
    densityNoise.clear();
    densityTrace.clear();
  }
  return vCollect > 0 || mCollect > 0 || cCollect > 0;
}

void Package::performCollapsingMeasurement(vEdge& rootEdge, const Qubit index,
                                           const fp probability,
                                           const bool measureZero) {
  const GateMatrix measurementMatrix =
      measureZero ? MEAS_ZERO_MAT : MEAS_ONE_MAT;

  const auto measurementGate = matrices().makeGateDD(measurementMatrix, index);

  vEdge e = multiply(measurementGate, rootEdge);

  assert(probability > 0.);
  e.w = cn.lookup(e.w / std::sqrt(probability));
  incRef(e);
  decRef(rootEdge);
  rootEdge = e;
}

char Package::measureOneCollapsing(vEdge& rootEdge, const Qubit index,
                                   std::mt19937_64& mt, const fp epsilon) {
  const auto& [pzero, pone] =
      vectors().determineMeasurementProbabilities(rootEdge, index);
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

  auto const measZeroDd = matrices().makeGateDD(MEAS_ZERO_MAT, index);

  auto tmp0 = matrices().conjugateTranspose(measZeroDd);
  auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
  auto tmp2 = multiply(densityFromMatrixEdge(measZeroDd), tmp1, true);
  auto densityMatrixTrace = trace(tmp2, nrQubits);

  std::uniform_real_distribution<fp> dist(0., 1.);
  if (const auto threshold = dist(mt); threshold > densityMatrixTrace.r) {
    auto const measOneDd = matrices().makeGateDD(MEAS_ONE_MAT, index);
    tmp0 = matrices().conjugateTranspose(measOneDd);
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
  auto result = e.w / densityMatrixTrace;
  cn.decRef(e.w);
  e.w = cn.lookup(result);
  cn.incRef(e.w);
  return measuredResult;
}

mEdge Package::applyOperation(const mEdge& operation, const mEdge& e,
                              const bool applyFromLeft) {
  const mEdge tmp =
      applyFromLeft ? multiply(operation, e) : multiply(e, operation);
  incRef(tmp);
  decRef(e);
  garbageCollect();
  return tmp;
}

vEdge Package::applyOperation(const mEdge& operation, const vEdge& e) {
  const auto tmp = multiply(operation, e);
  incRef(tmp);
  decRef(e);
  garbageCollect();
  return tmp;
}

dEdge Package::applyOperationToDensity(dEdge& e, const mEdge& operation) {
  const auto tmp0 = matrices().conjugateTranspose(operation);
  const auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
  const auto tmp2 = multiply(densityFromMatrixEdge(operation), tmp1, true);
  incRef(tmp2);
  dEdge::alignDensityEdge(e);
  decRef(e);
  e = tmp2;
  dEdge::setDensityMatrixTrue(e);
  return e;
}

fp Package::expectationValue(const mEdge& x, const vEdge& y) {
  assert(!x.isZeroTerminal() && !y.isTerminal());
  if (!x.isTerminal() && x.p->v > y.p->v) {
    throw std::invalid_argument(
        "Observable must not act on more qubits than the state to compute the"
        "expectation value.");
  }

  const auto yPrime = multiply(x, y);
  const ComplexValue expValue = vectors().innerProduct(y, yPrime);

  assert(RealNumber::approximatelyZero(expValue.i));
  return expValue.r;
}
mEdge Package::partialTrace(const mEdge& a,
                            const std::vector<bool>& eliminate) {
  auto r = trace(a, eliminate, eliminate.size());
  return {r.p, cn.lookup(r.w)};
}

} // namespace dd
