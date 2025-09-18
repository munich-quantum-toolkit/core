/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include <type_traits>

namespace dd {

struct vNode; // NOLINT(readability-identifier-naming)
struct mNode; // NOLINT(readability-identifier-naming)
struct dNode; // NOLINT(readability-identifier-naming)
class ComplexNumbers;
class MemoryManager;

template <typename T>
using isVector = std::enable_if_t<std::is_same_v<T, vNode>, bool>;
template <typename T>
using isMatrix = std::enable_if_t<std::is_same_v<T, mNode>, bool>;
template <typename T>
using isMatrixVariant =
    std::enable_if_t<std::is_same_v<T, mNode> || std::is_same_v<T, dNode>,
                     bool>;

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
   * @tparam T template parameter to enable this method only for vNode
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized vector DD
   */
  template <typename T = Node, isVector<T> = true>
  static auto normalize(Node* p, const std::array<CachedEdge, RADIX>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> CachedEdge;

  /**
   * @brief Get a normalized (density) matrix) DD from a fresh node and a list
   * of edges.
   * @tparam T template parameter to enable this method only for matrix nodes
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized (density) matrix DD
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  static auto normalize(Node* p, const std::array<CachedEdge, NEDGE>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> CachedEdge;

  /**
   * @brief Check whether the matrix represented by the DD is the identity.
   * @tparam T template parameter to enable this function only for matrix nodes
   * @return whether the matrix is the identity
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  [[nodiscard]] bool isIdentity(const bool upToGlobalPhase = true) const {
    if (!isTerminal()) {
      return false;
    }
    if (upToGlobalPhase) {
      return !w.exactlyZero();
    }
    return w.exactlyOne();
  }
};

} // namespace dd

template <class Node> struct std::hash<dd::CachedEdge<Node>> {
  auto operator()(dd::CachedEdge<Node> const& e) const noexcept -> std::size_t;
};
