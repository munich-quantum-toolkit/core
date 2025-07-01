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
#include "dd/DDDefinitions.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace dd {

struct vNode;
struct mNode;
struct dNode;
class ComplexNumbers;
class MemoryManager;

template <typename T>
using isVector = std::enable_if_t<std::is_same_v<T, vNode>, bool>;
template <typename T>
using isMatrix = std::enable_if_t<std::is_same_v<T, mNode>, bool>;
template <typename T>
using isDensityMatrix = std::enable_if_t<std::is_same_v<T, dNode>, bool>;
template <typename T>
using isMatrixVariant =
    std::enable_if_t<std::is_same_v<T, mNode> || std::is_same_v<T, dNode>,
                     bool>;

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
    return !(e.isTerminal() && constants::isStaticNumber(e.w.r) &&
             constants::isStaticNumber(e.w.i));
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
   * @tparam T template parameter to enable this function only for vNode
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized vector DD
   */
  template <typename T = Node, isVector<T> = true>
  static auto normalize(Node* p, const std::array<Edge, RADIX>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> Edge;

  /**
   * @brief Get a single element of the vector represented by the DD
   * @tparam T template parameter to enable this function only for vNode
   * @param i index of the element
   * @return the complex value of the amplitude
   */
  template <typename T = Node, isVector<T> = true>
  [[nodiscard]] std::complex<fp> getValueByIndex(std::size_t i) const;

  /**
   * @brief Get the vector represented by the DD
   * @tparam T template parameter to enable this function only for vNode
   * @param threshold amplitudes with a magnitude below this threshold will be
   * ignored
   * @return the vector
   */
  template <typename T = Node, isVector<T> = true>
  [[nodiscard]] CVec getVector(fp threshold = 0.) const;

  /**
   * @brief Get the sparse vector represented by the DD
   * @tparam T template parameter to enable this function only for vNode
   * @param threshold amplitudes with a magnitude below this threshold will be
   * ignored
   * @return the sparse vector
   */
  template <typename T = Node, isVector<T> = true>
  [[nodiscard]] SparseCVec getSparseVector(fp threshold = 0.) const;

  /**
   * @brief Print the vector represented by the DD
   * @tparam T template parameter to enable this function only for vNode
   * @note This function scales exponentially with the number of qubits.
   */
  template <typename T = Node, isVector<T> = true> void printVector() const;

  /**
   * @brief Add the amplitudes of a vector DD to a vector
   * @tparam T template parameter to enable this function only for vNode
   * @param amplitudes the vector to add to
   */
  template <typename T = Node, isVector<T> = true>
  void addToVector(CVec& amplitudes) const;

private:
  /**
   * @brief Recursively traverse the DD and call a function for each non-zero
   * amplitude.
   * @details Scales with the number of non-zero amplitudes.
   * @tparam T template parameter to enable this function only for vNode
   * @param amp the accumulated amplitude from previous traversals
   * @param i the current index in the vector
   * @param f This function is called for each non-zero amplitude with the
   * index and the amplitude as arguments.
   * @param threshold amplitude with a magnitude below this threshold will be
   * ignored
   */
  template <typename T = Node, isVector<T> = true>
  void traverseVector(const std::complex<fp>& amp, std::size_t i,
                      AmplitudeFunc f, fp threshold = 0.) const;

public:
  /**
   * @brief Get a normalized (density) matrix DD from a fresh node and a list
   * of edges
   * @tparam T template parameter to enable this function only for matrix nodes
   * @param p the fresh node
   * @param e the list of edges that form the successor nodes
   * @param mm a reference to the memory manager (for returning unused nodes)
   * @param cn a reference to the complex number manager (for adding new
   * complex numbers)
   * @return the normalized (density) matrix DD
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  static auto normalize(Node* p, const std::array<Edge, NEDGE>& e,
                        MemoryManager& mm, ComplexNumbers& cn) -> Edge;

  /**
   * @brief Check whether the matrix represented by the DD is the identity
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

  /**
   * @brief Get a single element of the matrix represented by the DD
   * @tparam T template parameter to enable this function only for matrix nodes
   * @param numQubits number of qubits in the considered DD
   * @param i row index of the element
   * @param j column index of the element
   * @return the complex value of the entry
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  [[nodiscard]] std::complex<fp>
  getValueByIndex(std::size_t numQubits, std::size_t i, std::size_t j) const;

  /**
   * @brief Get the matrix represented by the DD
   * @tparam T template parameter to enable this function only for matrix nodes
   * @param numQubits number of qubits in the considered DD
   * @param threshold entries with a magnitude below this threshold will be
   * ignored
   * @return the matrix
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  [[nodiscard]] CMat getMatrix(std::size_t numQubits, fp threshold = 0.) const;

  /**
   * @brief Get the sparse matrix represented by the DD
   * @tparam T template parameter to enable this function only for matrix nodes
   * @param numQubits number of qubits in the considered DD
   * @param threshold entries with a magnitude below this threshold will be
   * ignored
   * @return the sparse matrix
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  [[nodiscard]] SparseCMat getSparseMatrix(std::size_t numQubits,
                                           fp threshold = 0.) const;

  /**
   * @brief Print the matrix represented by the DD
   * @tparam T template parameter to enable this function only for matrix nodes
   * @param numQubits number of qubits in the considered DD
   * @note This function scales exponentially with the number of qubits.
   */
  template <typename T = Node, isMatrixVariant<T> = true>
  void printMatrix(std::size_t numQubits) const;

  /**
   * @brief Recursively traverse the DD and call a function for each non-zero
   * matrix entry.
   * @tparam T template parameter to enable this function only for matrix nodes
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
  template <typename T = Node, isMatrixVariant<T> = true>
  void traverseMatrix(const std::complex<fp>& amp, std::size_t i, std::size_t j,
                      MatrixEntryFunc f, std::size_t level,
                      fp threshold = 0.) const;

  template <typename T = Node, isDensityMatrix<T> = true>
  [[maybe_unused]] static void setDensityConjugateTrue(Edge& e) {
    Node::setConjugateTempFlagTrue(e.p);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  [[maybe_unused]] static void setFirstEdgeDensityPathTrue(Edge& e) {
    Node::setNonReduceTempFlagTrue(e.p);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  static void setDensityMatrixTrue(Edge& e) {
    Node::setDensityMatTempFlagTrue(e.p);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  static void alignDensityEdge(Edge& e) {
    Node::alignDensityNode(e.p);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  static void revertDmChangesToEdges(Edge& x, Edge& y) {
    revertDmChangesToEdge(x);
    revertDmChangesToEdge(y);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  static void revertDmChangesToEdge(Edge& x) {
    Node::revertDmChangesToNode(x.p);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  static void applyDmChangesToEdges(Edge& x, Edge& y) {
    applyDmChangesToEdge(x);
    applyDmChangesToEdge(y);
  }

  template <typename T = Node, isDensityMatrix<T> = true>
  static void applyDmChangesToEdge(Edge& x) {
    Node::applyDmChangesToNode(x.p);
  }

  /**
   * @brief Get the sparse probability vector for the underlying density matrix
   * @tparam T template parameter to enable this function only for dNode
   * @param numQubits number of qubits in the considered DD
   * @param threshold probabilities below this threshold will be ignored
   * @return the sparse probability vector
   */
  template <typename T = Node, isDensityMatrix<T> = true>
  [[nodiscard]] SparsePVec getSparseProbabilityVector(std::size_t numQubits,
                                                      fp threshold = 0.) const;

  /**
   * @brief Get the sparse probability vector for the underlying density matrix
   * @tparam T template parameter to enable this function only for dNode
   * @param numQubits number of qubits in the considered DD
   * @param threshold probabilities below this threshold will be ignored
   * @return the sparse probability vector (using strings as keys)
   */
  template <typename T = Node, isDensityMatrix<T> = true>
  [[nodiscard]] SparsePVecStrKeys
  getSparseProbabilityVectorStrKeys(std::size_t numQubits,
                                    fp threshold = 0.) const;

private:
  /**
   * @brief Recursively traverse diagonal of the DD and call a function for each
   * non-zero entry.
   * @tparam T template parameter to enable this function only for dNode
   * @param prob the accumulated probability from previous traversals
   * @param i the current diagonal index in the matrix
   * @param f This function is called for each non-zero entry with the
   * diagonal index and the probability as arguments.
   * @param level the current level in the DD (ranges from 1 to n for regular
   * nodes and is 0 for the terminal node)
   * @param threshold probabilities below this threshold will be ignored
   */
  template <typename T = Node, isDensityMatrix<T> = true>
  void traverseDiagonal(const fp& prob, std::size_t i, ProbabilityFunc f,
                        std::size_t level, fp threshold = 0.) const;
};
} // namespace dd

template <class Node> struct std::hash<dd::Edge<Node>> {
  std::size_t operator()(dd::Edge<Node> const& e) const noexcept;
};
