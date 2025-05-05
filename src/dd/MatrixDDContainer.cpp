/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/MatrixDDContainer.hpp"

namespace dd {

mEdge MatrixDDContainer::makeGateDD(const GateMatrix& mat,
                                    const qc::Qubit target) {
  return makeGateDD(mat, qc::Controls{}, target);
}

mEdge MatrixDDContainer::makeGateDD(const GateMatrix& mat,
                                    const qc::Control& control,
                                    const qc::Qubit target) {
  return makeGateDD(mat, qc::Controls{control}, target);
}

mEdge MatrixDDContainer::makeGateDD(const GateMatrix& mat,
                                    const qc::Controls& controls,
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
    return {e.p, getCn().lookup(e.w)};
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
  return {e.p, getCn().lookup(e.w)};
}

mEdge MatrixDDContainer::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                            const qc::Qubit target0,
                                            const qc::Qubit target1) {
  return makeTwoQubitGateDD(mat, qc::Controls{}, target0, target1);
}

mEdge MatrixDDContainer::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                            const qc::Control& control,
                                            const qc::Qubit target0,
                                            const qc::Qubit target1) {
  return makeTwoQubitGateDD(mat, qc::Controls{control}, target0, target1);
}

mEdge MatrixDDContainer::makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
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

  return {e.p, getCn().lookup(e.w)};
}

mEdge MatrixDDContainer::makeDDFromMatrix(const CMat& matrix) {
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
    return mEdge::terminal(getCn().lookup(matrix[0][0]));
  }

  const auto level = static_cast<Qubit>(std::log2(length) - 1);
  const auto MatrixDDContainer =
      makeDDFromMatrix(matrix, level, 0, length, 0, width);
  return {MatrixDDContainer.p, getCn().lookup(MatrixDDContainer.w)};
}

mCachedEdge MatrixDDContainer::makeDDFromMatrix(const CMat& matrix,
                                                const Qubit level,
                                                const std::size_t rowStart,
                                                const std::size_t rowEnd,
                                                const std::size_t colStart,
                                                const std::size_t colEnd) {
  // base case
  if (level == 0U) {
    assert(rowEnd - rowStart == 2);
    assert(colEnd - colStart == 2);
    return makeDDNode<CachedEdge>(
        0U, {mCachedEdge::terminal(matrix[rowStart][colStart]),
             mCachedEdge::terminal(matrix[rowStart][colStart + 1]),
             mCachedEdge::terminal(matrix[rowStart + 1][colStart]),
             mCachedEdge::terminal(matrix[rowStart + 1][colStart + 1])});
  }

  // recursively call the function on all quadrants
  const auto rowMid = (rowStart + rowEnd) / 2;
  const auto colMid = (colStart + colEnd) / 2;
  const auto l = static_cast<Qubit>(level - 1U);

  return makeDDNode<CachedEdge>(
      level, {makeDDFromMatrix(matrix, l, rowStart, rowMid, colStart, colMid),
              makeDDFromMatrix(matrix, l, rowStart, rowMid, colMid, colEnd),
              makeDDFromMatrix(matrix, l, rowMid, rowEnd, colStart, colMid),
              makeDDFromMatrix(matrix, l, rowMid, rowEnd, colMid, colEnd)});
}

mEdge MatrixDDContainer::conjugateTranspose(const mEdge& a) {
  const auto r = conjugateTransposeRec(a);
  return {r.p, getCn().lookup(r.w)};
}

mCachedEdge MatrixDDContainer::conjugateTransposeRec(const mEdge& a) {
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

bool MatrixDDContainer::isCloseToIdentity(const mEdge& m, const fp tol,
                                          const std::vector<bool>& garbage,
                                          const bool checkCloseToOne) const {
  std::unordered_set<decltype(m.p)> visited{};
  visited.reserve(ut.getNumActiveEntries());
  return isCloseToIdentityRecursive(m, visited, tol, garbage, checkCloseToOne);
}
bool MatrixDDContainer::isCloseToIdentityRecursive(
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
mEdge MatrixDDContainer::makeIdent() { return mEdge::one(); }
mEdge MatrixDDContainer::createInitialMatrix(
    const std::vector<bool>& ancillary) {
  return reduceAncillae(makeIdent(), ancillary);
}
mEdge MatrixDDContainer::reduceAncillae(mEdge e,
                                        const std::vector<bool>& ancillary,
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
  const auto res = mEdge{g.p, getCn().lookup(g.w * e.w)};
  incRef(res);
  decRef(e);
  return res;
}

mEdge MatrixDDContainer::reduceGarbage(const mEdge& e,
                                       const std::vector<bool>& garbage,
                                       const bool regular,
                                       const bool normalizeWeights) {
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
  const auto res = mEdge{g.p, getCn().lookup(weight)};

  incRef(res);
  decRef(e);
  return res;
}
mCachedEdge MatrixDDContainer::reduceAncillaeRecursion(
    mNode* p, const std::vector<bool>& ancillary, const Qubit lowerbound,
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

mCachedEdge MatrixDDContainer::reduceGarbageRecursion(
    mNode* p, const std::vector<bool>& garbage, const Qubit lowerbound,
    const bool regular, const bool normalizeWeights) {
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
