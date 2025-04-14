/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/VectorDDContainer.hpp"

#include "dd/GateMatrixDefinitions.hpp"

#include <queue>

namespace dd {

vEdge VectorDDContainer::makeZeroState(const std::size_t n,
                                       const std::size_t start) {
  if (n + start > nqubits) {
    throw std::runtime_error{
        "Requested state with " + std::to_string(n + start) +
        " qubits, but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }
  auto f = vEdge::one();
  for (std::size_t p = start; p < n + start; p++) {
    f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero()});
  }
  incRef(f);
  return f;
}

vEdge VectorDDContainer::makeBasisState(const std::size_t n,
                                        const std::vector<bool>& state,
                                        const std::size_t start) {
  if (n + start > nqubits) {
    throw std::runtime_error{
        "Requested state with " + std::to_string(n + start) +
        " qubits, but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }
  auto f = vEdge::one();
  for (std::size_t p = start; p < n + start; ++p) {
    if (!state[p]) {
      f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero()});
    } else {
      f = makeDDNode(static_cast<Qubit>(p), std::array{vEdge::zero(), f});
    }
  }
  incRef(f);
  return f;
}

vEdge VectorDDContainer::makeBasisState(const std::size_t n,
                                        const std::vector<BasisStates>& state,
                                        const std::size_t start) {
  if (n + start > nqubits) {
    throw std::runtime_error{
        "Requested state with " + std::to_string(n + start) +
        " qubits, but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }
  if (state.size() < n) {
    throw std::runtime_error("Insufficient qubit states provided. Requested " +
                             std::to_string(n) + ", but received " +
                             std::to_string(state.size()));
  }

  auto f = vCachedEdge::one();
  for (std::size_t p = start; p < n + start; ++p) {
    switch (state[p]) {
    case BasisStates::zero:
      f = makeDDNode(static_cast<Qubit>(p), std::array{f, vCachedEdge::zero()});
      break;
    case BasisStates::one:
      f = makeDDNode(static_cast<Qubit>(p), std::array{vCachedEdge::zero(), f});
      break;
    case BasisStates::plus:
      f = makeDDNode(static_cast<Qubit>(p),
                     std::array<vCachedEdge, RADIX>{
                         {{f.p, dd::SQRT2_2}, {f.p, dd::SQRT2_2}}});
      break;
    case BasisStates::minus:
      f = makeDDNode(static_cast<Qubit>(p),
                     std::array<vCachedEdge, RADIX>{
                         {{f.p, dd::SQRT2_2}, {f.p, -dd::SQRT2_2}}});
      break;
    case BasisStates::right:
      f = makeDDNode(static_cast<Qubit>(p),
                     std::array<vCachedEdge, RADIX>{
                         {{f.p, dd::SQRT2_2}, {f.p, {0, dd::SQRT2_2}}}});
      break;
    case BasisStates::left:
      f = makeDDNode(static_cast<Qubit>(p),
                     std::array<vCachedEdge, RADIX>{
                         {{f.p, dd::SQRT2_2}, {f.p, {0, -dd::SQRT2_2}}}});
      break;
    }
  }
  const vEdge e{f.p, getCn().lookup(f.w)};
  incRef(e);
  return e;
}

vEdge VectorDDContainer::makeGHZState(const std::size_t n) {
  if (n > nqubits) {
    throw std::runtime_error{
        "Requested state with " + std::to_string(n) +
        " qubits, but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }

  if (n == 0U) {
    return vEdge::one();
  }

  auto leftSubtree = vEdge::one();
  auto rightSubtree = vEdge::one();

  for (std::size_t p = 0; p < n - 1; ++p) {
    leftSubtree = makeDDNode(static_cast<Qubit>(p),
                             std::array{leftSubtree, vEdge::zero()});
    rightSubtree = makeDDNode(static_cast<Qubit>(p),
                              std::array{vEdge::zero(), rightSubtree});
  }

  const vEdge e = makeDDNode(
      static_cast<Qubit>(n - 1),
      std::array<vEdge, RADIX>{
          {{leftSubtree.p, {&constants::sqrt2over2, &constants::zero}},
           {rightSubtree.p, {&constants::sqrt2over2, &constants::zero}}}});
  incRef(e);
  return e;
}

vEdge VectorDDContainer::makeWState(const std::size_t n) {
  if (n > nqubits) {
    throw std::runtime_error{
        "Requested state with " + std::to_string(n) +
        " qubits, but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }

  if (n == 0U) {
    return vEdge::one();
  }

  auto leftSubtree = vEdge::zero();
  if ((1. / sqrt(static_cast<double>(n))) < RealNumber::eps) {
    throw std::runtime_error(
        "Requested qubit size for generating W-state would lead to an "
        "underflow due to 1 / sqrt(n) being smaller than the currently set "
        "tolerance " +
        std::to_string(RealNumber::eps) +
        ". If you still wanna run the computation, please lower "
        "the tolerance accordingly.");
  }

  auto rightSubtree = vEdge::terminal(getCn().lookup(1. / std::sqrt(n)));
  for (size_t p = 0; p < n; ++p) {
    leftSubtree = makeDDNode(static_cast<Qubit>(p),
                             std::array{leftSubtree, rightSubtree});
    if (p != n - 1U) {
      rightSubtree = makeDDNode(static_cast<Qubit>(p),
                                std::array{rightSubtree, vEdge::zero()});
    }
  }
  incRef(leftSubtree);
  return leftSubtree;
}

vEdge VectorDDContainer::makeStateFromVector(const CVec& stateVector) {
  if (stateVector.empty()) {
    return vEdge::one();
  }
  const auto& length = stateVector.size();
  if ((length & (length - 1)) != 0) {
    throw std::invalid_argument(
        "State vector must have a length of a power of two.");
  }

  if (length == 1) {
    return vEdge::terminal(getCn().lookup(stateVector[0]));
  }

  const auto level = static_cast<Qubit>(std::log2(length) - 1);
  const auto state =
      makeStateFromVector(stateVector.begin(), stateVector.end(), level);
  const vEdge e{state.p, getCn().lookup(state.w)};
  incRef(e);
  return e;
}

vCachedEdge
VectorDDContainer::makeStateFromVector(const CVec::const_iterator& begin,
                                       const CVec::const_iterator& end,
                                       const Qubit level) {
  if (level == 0U) {
    assert(std::distance(begin, end) == 2);
    const auto zeroSuccessor = vCachedEdge::terminal(*begin);
    const auto oneSuccessor = vCachedEdge::terminal(*(begin + 1));
    return makeDDNode<CachedEdge>(0, {zeroSuccessor, oneSuccessor});
  }

  const auto half = std::distance(begin, end) / 2;
  const auto zeroSuccessor =
      makeStateFromVector(begin, begin + half, level - 1);
  const auto oneSuccessor = makeStateFromVector(begin + half, end, level - 1);
  return makeDDNode<CachedEdge>(level, {zeroSuccessor, oneSuccessor});
}

std::string VectorDDContainer::measureAll(vEdge& rootEdge, const bool collapse,
                                          std::mt19937_64& mt,
                                          const fp epsilon) {
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

fp VectorDDContainer::assignProbabilities(
    const vEdge& edge, std::unordered_map<const vNode*, fp>& probs) {
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
VectorDDContainer::determineMeasurementProbabilities(const vEdge& rootEdge,
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

vEdge VectorDDContainer::conjugate(const vEdge& a) {
  const auto r = conjugateRec(a);
  return {r.p, getCn().lookup(r.w)};
}

vCachedEdge VectorDDContainer::conjugateRec(const vEdge& a) {
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

ComplexValue VectorDDContainer::innerProduct(const vEdge& x, const vEdge& y) {
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

fp VectorDDContainer::fidelity(const vEdge& x, const vEdge& y) {
  return innerProduct(x, y).mag2();
}

fp VectorDDContainer::fidelityOfMeasurementOutcomes(
    const vEdge& e, const SparsePVec& probs,
    const qc::Permutation& permutation) {
  if (e.w.approximatelyZero()) {
    return 0.;
  }
  return fidelityOfMeasurementOutcomesRecursive(e, probs, 0, permutation,
                                                e.p->v + 1U);
}

ComplexValue VectorDDContainer::innerProduct(const vEdge& x, const vEdge& y,
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

fp VectorDDContainer::fidelityOfMeasurementOutcomesRecursive(
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

vEdge VectorDDContainer::reduceGarbage(vEdge& e,
                                       const std::vector<bool>& garbage,
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
  const auto res = vEdge{f.p, getCn().lookup(weight)};
  incRef(res);
  decRef(e);
  return res;
}

vCachedEdge VectorDDContainer::reduceGarbageRecursion(
    vNode* p, const std::vector<bool>& garbage, const Qubit lowerbound,
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
} // namespace dd
