/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Complex.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <functional>

namespace dd {

class ComplexNumbers;
class MemoryManager;

/**
 * @brief A DD node with a cached edge weight
 * @details Some DD operations create intermediate results that are not part of
 * the final result. To avoid storing these intermediate results in the unique
 * table, they are represented via cached numbers.
 * @tparam Node Type of the DD node
 */
template <typename Node> struct CachedEdge {
  Node* p{};
  ComplexValue w;

  CachedEdge() = default;
  CachedEdge(Node* n, const ComplexValue& v) : p(n), w(v) {}
  CachedEdge(Node* n, const Complex& c)
      : p(n), w(static_cast<ComplexValue>(c)) {}

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close enough"
  /// according to a given tolerance this notion of equivalence is chosen to
  /// counter floating point inaccuracies
  bool operator==(const CachedEdge& other) const {
    return p == other.p && w.approximatelyEquals(other.w);
  }
  bool operator!=(const CachedEdge& other) const { return !operator==(other); }

  /**
   * @brief Create a terminal edge with the given weight.
   * @param w The weight of the terminal edge.
   * @return A terminal edge with the given weight.
   */
  [[nodiscard]] static constexpr CachedEdge terminal(const ComplexValue& w) {
    return CachedEdge{Node::getTerminal(), w};
  }

  /**
   * @brief Create a terminal edge with the given weight.
   * @param w The weight of the terminal edge.
   * @return A terminal edge with the given weight.
   */
  [[nodiscard]] static constexpr CachedEdge
  terminal(const std::complex<fp>& w) {
    return CachedEdge{Node::getTerminal(), static_cast<ComplexValue>(w)};
  }

  /**
   * @brief Create a terminal edge with the given weight.
   * @param w The weight of the terminal edge.
   * @return A terminal edge with the given weight.
   */
  [[nodiscard]] static constexpr CachedEdge terminal(const Complex& w) {
    return terminal(static_cast<ComplexValue>(w));
  }

  /**
   * @brief Create a zero terminal edge.
   * @return A zero terminal edge.
   */
  [[nodiscard]] static constexpr CachedEdge zero() {
    return terminal(ComplexValue(0.));
  }

  /**
   * @brief Create a one terminal edge.
   * @return A one terminal edge.
   */
  [[nodiscard]] static constexpr CachedEdge one() {
    return terminal(ComplexValue(1.));
  }

  /**
   * @brief Check whether this is a terminal.
   * @return whether this is a terminal
   */
  [[nodiscard]] constexpr bool isTerminal() const {
    return Node::isTerminal(p);
  }

  /**
   * @brief Get a normalized vector DD from a fresh node and a list of edges.
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized vector DD
   */
  static auto normalize(Node* p, const std::array<CachedEdge, RADIX>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> CachedEdge
    requires IsVector<Node>;

  /**
   * @brief Get a normalized matrix DD from a fresh node and a list
   * of edges.
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized matrix DD
   */
  static auto normalize(Node* p, const std::array<CachedEdge, NEDGE>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> CachedEdge
    requires IsMatrix<Node>;

  /**
   * @brief Check whether the matrix represented by the DD is the identity.
   * @return whether the matrix is the identity
   */
  [[nodiscard]] bool isIdentity(const bool upToGlobalPhase = true) const
    requires IsMatrix<Node>
  {
    if (!isTerminal()) {
      return false;
    }
    if (upToGlobalPhase) {
      return !w.exactlyZero();
    }
    return w.exactlyOne();
  }
};

// Deduction guide for constructor: CachedEdge(Node*, const ComplexValue&)
template <class Node>
CachedEdge(Node*, const ComplexValue&) -> CachedEdge<Node>;

// Deduction guide for constructor: CachedEdge(Node*, const Complex&)
template <class Node> CachedEdge(Node*, const Complex&) -> CachedEdge<Node>;

} // namespace dd

template <class Node> struct std::hash<dd::CachedEdge<Node>> {
  auto operator()(dd::CachedEdge<Node> const& e) const noexcept -> std::size_t;
};
