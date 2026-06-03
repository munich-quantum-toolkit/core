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
#include "dd/DDDefinitions.hpp"
#include "dd/RealNumber.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_set>

namespace dd {

class ComplexNumbers;
class MemoryManager;

using AmplitudeFunc = std::function<void(std::size_t, const std::complex<fp>&)>;
using ProbabilityFunc = std::function<void(std::size_t, const fp&)>;
using MatrixEntryFunc =
    std::function<void(std::size_t, std::size_t, const std::complex<fp>&)>;

/**
 * @brief A weighted edge pointing to a DD node
 * @details This struct is used to represent the core data structure of the DD
 * package. It is a wrapper around a pointer to a DD node and a complex edge
 * weight.
 * @tparam Node Type of the DD node
 */
template <class Node> struct Edge {
  Node* p;
  Complex w;

  /// Comparing two DD edges with another involves comparing the respective
  /// pointers and checking whether the corresponding weights are "close enough"
  /// according to a given tolerance this notion of equivalence is chosen to
  /// counter floating point inaccuracies
  constexpr bool operator==(const Edge& other) const {
    return p == other.p && w.approximatelyEquals(other.w);
  }
  constexpr bool operator!=(const Edge& other) const {
    return !operator==(other);
  }

  /**
   * @brief Get the static zero terminal
   * @return the zero terminal
   */
  static constexpr Edge zero() { return terminal(Complex::zero()); }

  /**
   * @brief Get the static one terminal
   * @return the one terminal
   */
  static constexpr Edge one() { return terminal(Complex::one()); }

  /**
   * @brief Get a terminal DD with a given edge weight
   * @param w the edge weight
   * @return the terminal DD representing (w)
   */
  [[nodiscard]] static constexpr Edge terminal(const Complex& w) {
    return Edge{Node::getTerminal(), w};
  }

  /**
   * @brief Check whether an edge requires tracking.
   * @param e The edge to check.
   * @return Whether the edge requires tracking.
   */
  [[nodiscard]] static constexpr bool trackingRequired(const Edge& e) {
    return !e.isTerminal() || !constants::isStaticNumber(e.w.r) ||
           !constants::isStaticNumber(e.w.i);
  }

  /**
   * @brief Check whether this is a terminal
   * @return whether this is a terminal
   */
  [[nodiscard]] constexpr bool isTerminal() const {
    return Node::isTerminal(p);
  }

  /**
   * @brief Check whether this is a zero terminal
   * @return whether this is a zero terminal
   */
  [[nodiscard]] constexpr bool isZeroTerminal() const {
    return isTerminal() && w.exactlyZero();
  }

  /**
   * @brief Check whether this is a one terminal
   * @return whether this is a one terminal
   */
  [[nodiscard]] constexpr bool isOneTerminal() const {
    return isTerminal() && w.exactlyOne();
  }

  /**
   * @brief Get a single element of the vector or matrix represented by the DD
   * @param numQubits number of qubits in the considered DD
   * @param decisions string {0, 1, 2, 3}^n describing which outgoing edge
   * should be followed (for vectors entries are limited to 0 and 1) If string
   * is longer than required, the additional characters are ignored.
   * @return the complex amplitude of the specified element
   */
  [[nodiscard]] std::complex<fp>
  getValueByPath(std::size_t numQubits, const std::string& decisions) const;

  /**
   * @brief Get the size of the DD
   * @details The size of a DD is defined as the number of nodes (including the
   * terminal node) in the DD.
   * @return the size of the DD
   */
  [[nodiscard]] std::size_t size() const;

  /// @brief Mark the edge as used.
  void mark() const noexcept;

  /// @brief Unmark the edge.
  void unmark() const noexcept;

private:
  /**
   * @brief Recursively traverse the DD and count the number of nodes
   * @param visited set of visited nodes
   * @return the size of the DD
   */
  [[nodiscard]] std::size_t
  size(std::unordered_set<const Node*>& visited) const;

public:
  /**
   * @brief Get a normalized vector DD from a fresh node and a list of edges
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized vector DD
   */
  static auto normalize(Node* p, const std::array<Edge, RADIX>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> Edge
    requires IsVector<Node>;

  /**
   * @brief Get a single element of the vector represented by the DD
   * @param i index of the element
   * @return the complex value of the amplitude
   */
  [[nodiscard]] std::complex<fp> getValueByIndex(std::size_t i) const
    requires IsVector<Node>;

  /**
   * @brief Get the vector represented by the DD
   * @param threshold amplitudes with a magnitude below this threshold will be
   * ignored
   * @return the vector
   */
  [[nodiscard]] CVec getVector(fp threshold = 0.) const
    requires IsVector<Node>;

  /**
   * @brief Get the sparse vector represented by the DD
   * @param threshold amplitudes with a magnitude below this threshold will be
   * ignored
   * @return the sparse vector
   */
  [[nodiscard]] SparseCVec getSparseVector(fp threshold = 0.) const
    requires IsVector<Node>;

  /**
   * @brief Print the vector represented by the DD
   * @note This function scales exponentially with the number of qubits.
   */
  void printVector() const
    requires IsVector<Node>;

  /**
   * @brief Add the amplitudes of a vector DD to a vector
   * @param amplitudes the vector to add to
   */
  void addToVector(CVec& amplitudes) const
    requires IsVector<Node>;

private:
  /**
   * @brief Recursively traverse the DD and call a function for each non-zero
   * amplitude.
   * @details Scales with the number of non-zero amplitudes.
   * @param amp the accumulated amplitude from previous traversals
   * @param i the current index in the vector
   * @param f This function is called for each non-zero amplitude with the
   * index and the amplitude as arguments.
   * @param threshold amplitude with a magnitude below this threshold will be
   * ignored
   */
  void traverseVector(const std::complex<fp>& amp, std::size_t i,
                      AmplitudeFunc f, fp threshold = 0.) const
    requires IsVector<Node>;

public:
  /**
   * @brief Get a normalized matrix DD from a fresh node and a list
   * of edges
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized matrix DD
   */
  static auto normalize(Node* p, const std::array<Edge, NEDGE>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> Edge
    requires IsMatrix<Node>;

  /**
   * @brief Check whether the matrix represented by the DD is the identity
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

  /**
   * @brief Get a single element of the matrix represented by the DD
   * @param numQubits number of qubits in the considered DD
   * @param i row index of the element
   * @param j column index of the element
   * @return the complex value of the entry
   */
  [[nodiscard]] std::complex<fp>
  getValueByIndex(std::size_t numQubits, std::size_t i, std::size_t j) const
    requires IsMatrix<Node>;

  /**
   * @brief Get the matrix represented by the DD
   * @param numQubits number of qubits in the considered DD
   * @param threshold entries with a magnitude below this threshold will be
   * ignored
   * @return the matrix
   */
  [[nodiscard]] CMat getMatrix(std::size_t numQubits, fp threshold = 0.) const
    requires IsMatrix<Node>;

  /**
   * @brief Get the sparse matrix represented by the DD
   * @param numQubits number of qubits in the considered DD
   * @param threshold entries with a magnitude below this threshold will be
   * ignored
   * @return the sparse matrix
   */
  [[nodiscard]] SparseCMat getSparseMatrix(std::size_t numQubits,
                                           fp threshold = 0.) const
    requires IsMatrix<Node>;

  /**
   * @brief Print the matrix represented by the DD
   * @param numQubits number of qubits in the considered DD
   * @note This function scales exponentially with the number of qubits.
   */
  void printMatrix(std::size_t numQubits) const
    requires IsMatrix<Node>;

  /**
   * @brief Recursively traverse the DD and call a function for each non-zero
   * matrix entry.
   * @param amp the accumulated amplitude from previous traversals
   * @param i the current row index in the matrix
   * @param j the current column index in the matrix
   * @param f This function is called for each non-zero matrix entry with the
   * row index, the column index and the amplitude as arguments.
   * @param level the current level in the DD (ranges from 1 to n for regular
   * nodes and is 0 for the terminal node)
   * @param threshold entries with a magnitude below this threshold will be
   * ignored
   */
  void traverseMatrix(const std::complex<fp>& amp, std::size_t i, std::size_t j,
                      MatrixEntryFunc f, std::size_t level,
                      fp threshold = 0.) const
    requires IsMatrix<Node>;
};
} // namespace dd

template <class Node> struct std::hash<dd::Edge<Node>> {
  std::size_t operator()(dd::Edge<Node> const& e) const noexcept;
};
