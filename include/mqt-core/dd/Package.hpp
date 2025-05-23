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

#include "dd/CachedEdge.hpp"
#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/DensityDDContainer.hpp"
#include "dd/DensityNoiseTable.hpp"
#include "dd/Edge.hpp"
#include "dd/MatrixDDContainer.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package_fwd.hpp" // IWYU pragma: export
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/StochasticNoiseOperationTable.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "dd/UniqueTable.hpp"
#include "dd/VectorDDContainer.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/Control.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <stack>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dd {

/**
 * @brief The DD package class
 *
 * @details This is the main class of the decision diagram module in MQT Core.
 * It contains the core functionality for working with quantum decision
 * diagrams. Specifically, it provides the means to
 * - represent quantum states as decision diagrams,
 * - represent quantum operations as decision diagrams,
 * - multiply decision diagrams (MxV, MxM, etc.),
 * - perform collapsing measurements on decision diagrams,
 * - sample from decision diagrams.
 *
 * To this end, it maintains several internal data structures, such as unique
 * tables, compute tables, and memory managers, which are used to manage the
 * nodes of the decision diagrams.
 */
class Package {

  ///
  /// Construction, destruction, information, and reset
  ///
public:
  static constexpr std::size_t MAX_POSSIBLE_QUBITS =
      static_cast<std::size_t>(std::numeric_limits<Qubit>::max()) + 1U;
  static constexpr std::size_t DEFAULT_QUBITS = 32U;
  /**
   * @brief Construct a new DD Package instance
   *
   * @param nq The maximum number of qubits to allocate memory for. This can
   * always be extended later using @ref resize.
   * @param config The configuration of the package
   */
  explicit Package(std::size_t nq = DEFAULT_QUBITS,
                   const DDPackageConfig& config = DDPackageConfig{});
  ~Package() = default;
  Package(const Package& package) = delete;

  Package& operator=(const Package& package) = delete;

  /**
   * @brief Resize the package to a new number of qubits
   *
   * @details This method will resize all the unique tables appropriately so
   * that they can handle the new number of qubits.
   *
   * @param nq The new number of qubits
   */
  void resize(std::size_t nq);

  /// Reset package state
  void reset();

  /// Get the number of qubits
  [[nodiscard]] auto qubits() const { return nqubits; }

public:
  std::size_t nqubits;
  DDPackageConfig config_;

  MemoryManager cMm{MemoryManager::create<RealNumber>()};
  RealNumberUniqueTable cUt{cMm};
  ComplexNumbers cn{cUt};

private:
  VectorDDContainer vectors_{nqubits, cUt, cn, {/*todo: add config here */}};
  MatrixDDContainer matrices_{nqubits, cUt, cn, {/*todo: add config here */}};
  DensityDDContainer densities_{nqubits, cUt, cn, {/*todo: add config here */}};

public:
  VectorDDContainer& vectors() { return vectors_; }
  const VectorDDContainer& vectors() const { return vectors_; }

  MatrixDDContainer& matrices() { return matrices_; }
  const MatrixDDContainer& matrices() const { return matrices_; }

  DensityDDContainer& densities() { return densities_; }
  const DensityDDContainer& densities() const { return densities_; }

  void incRef(const vEdge& e) { vectors().incRef(e); }
  void incRef(const mEdge& e) { matrices().incRef(e); }
  void incRef(const dEdge& e) { densities().incRef(e); }

  void decRef(const vEdge& e) { vectors().decRef(e); }
  void decRef(const mEdge& e) { matrices().decRef(e); }
  void decRef(const dEdge& e) { densities().decRef(e); }

  [[nodiscard]] vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                                    bool normalizeWeights = false) {
    return vectors().reduceGarbage(e, garbage, normalizeWeights);
  }

  [[nodiscard]] mEdge reduceGarbage(const mEdge& e,
                                    const std::vector<bool>& garbage,
                                    bool regular = true,
                                    bool normalizeWeights = false) {
    return matrices().reduceGarbage(e, garbage, regular, normalizeWeights);
  }

  vEdge makeDDNode(
      const Qubit var,
      const std::array<vEdge, std::tuple_size_v<decltype(vNode::e)>>& edges,
      [[maybe_unused]] const bool generateDensityMatrix = false) {
    return vectors().makeDDNode(var, edges, generateDensityMatrix);
  }

  vCachedEdge
  makeDDNode(const Qubit var,
             const std::array<vCachedEdge,
                              std::tuple_size_v<decltype(vNode::e)>>& edges,
             [[maybe_unused]] const bool generateDensityMatrix = false) {
    return vectors().makeDDNode(var, edges, generateDensityMatrix);
  }

  mEdge makeDDNode(
      const Qubit var,
      const std::array<mEdge, std::tuple_size_v<decltype(mNode::e)>>& edges,
      [[maybe_unused]] const bool generateDensityMatrix = false) {
    return matrices().makeDDNode(var, edges, generateDensityMatrix);
  }
  mCachedEdge
  makeDDNode(const Qubit var,
             const std::array<mCachedEdge,
                              std::tuple_size_v<decltype(mNode::e)>>& edges,
             [[maybe_unused]] const bool generateDensityMatrix = false) {
    return matrices().makeDDNode(var, edges, generateDensityMatrix);
  }

  dEdge makeDDNode(
      const Qubit var,
      const std::array<dEdge, std::tuple_size_v<decltype(dNode::e)>>& edges,
      [[maybe_unused]] const bool generateDensityMatrix = false) {
    return densities().makeDDNode(var, edges, generateDensityMatrix);
  }
  dCachedEdge
  makeDDNode(const Qubit var,
             const std::array<dCachedEdge,
                              std::tuple_size_v<decltype(dNode::e)>>& edges,
             [[maybe_unused]] const bool generateDensityMatrix = false) {
    return densities().makeDDNode(var, edges, generateDensityMatrix);
  }

  vCachedEdge add(const vCachedEdge& x, const vCachedEdge& y, Qubit var) {
    return vectors().add(x, y, var);
  }
  mCachedEdge add(const mCachedEdge& x, const mCachedEdge& y, Qubit var) {
    return matrices().add(x, y, var);
  }

  dCachedEdge add(const dCachedEdge& x, const dCachedEdge& y, Qubit var) {
    return densities().add(x, y, var);
  }

public:
  /**
   * @brief Trigger garbage collection in all unique tables
   *
   * @details Garbage collection is the process of removing all nodes from the
   * unique tables that have a reference count of zero.
   * Such nodes are considered "dead" and they can be safely removed from the
   * unique tables. This process is necessary to free up memory that is no
   * longer needed. By default, garbage collection is only triggered if the
   * unique table indicates that it possibly needs collection. Whenever some
   * nodes are recollected, some compute tables need to be invalidated as well.
   *
   * @param force
   * @return
   */
  bool garbageCollect(bool force = false);

  /**
   * @brief Performs a specific measurement on the given state vector decision
   * diagram. Collapses the state according to the measurement result.
   * @param rootEdge the root edge of the state vector decision diagram
   * @param index the index of the qubit to be measured
   * @param probability the probability of the measurement result (required for
   * normalization)
   * @param measureZero whether or not to measure '0' (otherwise '1' is
   * measured)
   */
  void performCollapsingMeasurement(vEdge& rootEdge, Qubit index,
                                    fp probability, bool measureZero);

  /**
   * @brief Measures the qubit with the given index in the given state vector
   * decision diagram. Collapses the state according to the measurement result.
   * @param rootEdge the root edge of the state vector decision diagram
   * @param index the index of the qubit to be measured
   * @param mt the random number generator
   * @param epsilon the numerical precision used for checking the normalization
   * of the state vector decision diagram
   * @return the measurement result ('0' or '1')
   * @throws std::runtime_error if a numerical instability is detected during
   * the measurement.
   */
  char measureOneCollapsing(vEdge& rootEdge, Qubit index, std::mt19937_64& mt,
                            fp epsilon = 0.001);

  char measureOneCollapsing(dEdge& e, Qubit index, std::mt19937_64& mt);

public:
  ///
  /// Multiplication
  ///
  ComputeTable<mNode*, vNode*, vCachedEdge> matrixVectorMultiplication{
      config_.ctMatVecMultNumBucket};
  ComputeTable<mNode*, mNode*, mCachedEdge> matrixMatrixMultiplication{
      config_.ctMatMatMultNumBucket};
  ComputeTable<dNode*, dNode*, dCachedEdge> densityDensityMultiplication{
      config_.ctDmDmMultNumBucket};

  /**
   * @brief Get the compute table for multiplication operations.
   *
   * @tparam RightOperandNode The type of the right operand node.
   * @return A reference to the appropriate compute table for the given node
   * type.
   */
  template <class RightOperandNode>
  [[nodiscard]] auto& getMultiplicationComputeTable() {
    if constexpr (std::is_same_v<RightOperandNode, vNode>) {
      return matrixVectorMultiplication;
    } else if constexpr (std::is_same_v<RightOperandNode, mNode>) {
      return matrixMatrixMultiplication;
    } else if constexpr (std::is_same_v<RightOperandNode, dNode>) {
      return densityDensityMultiplication;
    }
  }

  /**
   * @brief Applies a matrix operation to a vector.
   *
   * @details The reference count of the input vector is decreased,
   * while the reference count of the result is increased. After the operation,
   * garbage collection is triggered.
   *
   * @param operation Matrix operation to apply
   * @param e Vector to apply the operation to
   * @return The appropriately reference-counted result.
   */
  VectorDD applyOperation(const MatrixDD& operation, const VectorDD& e);

  /**
   * @brief Applies a matrix operation to a matrix.
   *
   * @details The reference count of the input matrix is decreased,
   * while the reference count of the result is increased. After the operation,
   * garbage collection is triggered.
   *
   * @param operation Matrix operation to apply
   * @param e Matrix to apply the operation to
   * @param applyFromLeft Flag to indicate if the operation should be applied
   * from the left (default) or right.
   * @return The appropriately reference-counted result.
   */
  MatrixDD applyOperation(const MatrixDD& operation, const MatrixDD& e,
                          bool applyFromLeft = true);

  dEdge applyOperationToDensity(dEdge& e, const mEdge& operation);

  /**
   * @brief Multiplies two decision diagrams.
   *
   * @tparam LeftOperandNode The type of the left operand node.
   * @tparam RightOperandNode The type of the right operand node.
   * @param x The left operand decision diagram.
   * @param y The right operand decision diagram.
   * @param generateDensityMatrix Flag to indicate if a density matrix node
   * should be generated.
   * @return The resulting decision diagram after multiplication.
   *
   * @details This function performs the multiplication of two decision diagrams
   * (DDs). It uses a compute table to cache intermediate results and avoid
   * redundant computations. The multiplication is conducted recursively, where
   * the function traverses the nodes of the DDs, multiplies corresponding
   * edges, and normalizes the resulting edges. If the nodes are terminal, their
   * weights are directly multiplied. The function ensures that the resulting DD
   * is properly normalized and stored in the unique table to maintain the
   * canonical form.
   */
  template <class LeftOperandNode, class RightOperandNode>
  Edge<RightOperandNode>
  multiply(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
           [[maybe_unused]] const bool generateDensityMatrix = false) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    static_assert(std::disjunction_v<std::is_same<LEdge, mEdge>,
                                     std::is_same<LEdge, dEdge>>,
                  "Left operand must be a matrix or density matrix");
    static_assert(std::disjunction_v<std::is_same<REdge, vEdge>,
                                     std::is_same<REdge, mEdge>,
                                     std::is_same<REdge, dEdge>>,
                  "Right operand must be a vector, matrix or density matrix");
    Qubit var{};
    if constexpr (std::is_same_v<LEdge, dEdge>) {
      auto xCopy = x;
      auto yCopy = y;
      dEdge::applyDmChangesToEdges(xCopy, yCopy);

      if (!xCopy.isTerminal()) {
        var = xCopy.p->v;
      }
      if (!y.isTerminal() && yCopy.p->v > var) {
        var = yCopy.p->v;
      }

      const auto e = multiply2(xCopy, yCopy, var, generateDensityMatrix);
      dEdge::revertDmChangesToEdges(xCopy, yCopy);
      return cn.lookup(e);
    } else {
      if (!x.isTerminal()) {
        var = x.p->v;
      }
      if (!y.isTerminal() && y.p->v > var) {
        var = y.p->v;
      }
      const auto e = multiply2(x, y, var);
      return cn.lookup(e);
    }
  }

private:
  /**
   * @brief Internal function to multiply two decision diagrams.
   *
   * This function is used internally to multiply two decision diagrams (DDs) of
   * type Node. It is not intended to be called directly.
   *
   * @tparam LeftOperandNode The type of the left operand node.
   * @tparam RightOperandNode The type of the right operand node.
   * @param x The left operand decision diagram.
   * @param y The right operand decision diagram.
   * @param var The variable associated with the current level of recursion.
   * @param generateDensityMatrix Flag to indicate if a density matrix node
   * should be generated.
   * @return The resulting DD after multiplication.
   */
  template <class LeftOperandNode, class RightOperandNode>
  CachedEdge<RightOperandNode>
  multiply2(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
            const Qubit var,
            [[maybe_unused]] const bool generateDensityMatrix = false) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    using ResultEdge = CachedEdge<RightOperandNode>;

    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return ResultEdge::zero();
    }

    const auto xWeight = static_cast<ComplexValue>(x.w);
    const auto yWeight = static_cast<ComplexValue>(y.w);
    const auto rWeight = xWeight * yWeight;
    if (x.isIdentity()) {
      if constexpr (!std::is_same_v<RightOperandNode, dNode>) {
        return {y.p, rWeight};
      } else {
        if (y.isIdentity() ||
            (dNode::isDensityMatrixTempFlagSet(y.p->flags) &&
             generateDensityMatrix) ||
            (!dNode::isDensityMatrixTempFlagSet(y.p->flags) &&
             !generateDensityMatrix)) {
          return {y.p, rWeight};
        }
      }
    }

    if constexpr (std::is_same_v<RightOperandNode, mNode> ||
                  std::is_same_v<RightOperandNode, dNode>) {
      if (y.isIdentity()) {
        if constexpr (!std::is_same_v<LeftOperandNode, dNode>) {
          return {x.p, rWeight};
        } else {
          if (x.isIdentity() ||
              (dNode::isDensityMatrixTempFlagSet(x.p->flags) &&
               generateDensityMatrix) ||
              (!dNode::isDensityMatrixTempFlagSet(x.p->flags) &&
               !generateDensityMatrix)) {
            return {x.p, rWeight};
          }
        }
      }
    }

    auto& computeTable = getMultiplicationComputeTable<RightOperandNode>();
    if (const auto* r = computeTable.lookup(x.p, y.p, generateDensityMatrix);
        r != nullptr) {
      return {r->p, r->w * rWeight};
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(y.p->e)>;

    constexpr std::size_t rows = RADIX;
    constexpr std::size_t cols = n == NEDGE ? RADIX : 1U;

    std::array<ResultEdge, n> edge{};
    for (auto i = 0U; i < rows; i++) {
      for (auto j = 0U; j < cols; j++) {
        auto idx = (cols * i) + j;
        edge[idx] = ResultEdge::zero();
        for (auto k = 0U; k < rows; k++) {
          const auto xIdx = (rows * i) + k;
          LEdge e1{};
          if (x.p != nullptr && x.p->v == var) {
            e1 = x.p->e[xIdx];
          } else {
            if (xIdx == 0 || xIdx == 3) {
              e1 = LEdge{x.p, Complex::one()};
            } else {
              e1 = LEdge::zero();
            }
          }

          const auto yIdx = j + (cols * k);
          REdge e2{};
          if (y.p != nullptr && y.p->v == var) {
            e2 = y.p->e[yIdx];
          } else {
            if (yIdx == 0 || yIdx == 3) {
              e2 = REdge{y.p, Complex::one()};
            } else {
              e2 = REdge::zero();
            }
          }

          const auto v = static_cast<Qubit>(var - 1);
          if constexpr (std::is_same_v<LeftOperandNode, dNode>) {
            dCachedEdge m;
            dEdge::applyDmChangesToEdges(e1, e2);
            if (!generateDensityMatrix || idx == 1) {
              // When generateDensityMatrix is false or I have the first edge I
              // don't optimize anything and set generateDensityMatrix to false
              // for all child edges
              m = multiply2(e1, e2, v, false);
            } else if (idx == 2) {
              // When I have the second edge and generateDensityMatrix == false,
              // then edge[2] == edge[1]
              if (k == 0) {
                if (edge[1].w.approximatelyZero()) {
                  edge[2] = ResultEdge::zero();
                } else {
                  edge[2] = edge[1];
                }
              }
              continue;
            } else {
              m = multiply2(e1, e2, v, generateDensityMatrix);
            }

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              dNode::applyDmChangesToNode(edge[idx].p);
              dNode::applyDmChangesToNode(m.p);
              edge[idx] = add(edge[idx], m, v);
              dNode::revertDmChangesToNode(m.p);
              dNode::revertDmChangesToNode(edge[idx].p);
            }
            // Undo modifications on density matrices
            dEdge::revertDmChangesToEdges(e1, e2);
          } else {
            auto m = multiply2(e1, e2, v);

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              edge[idx] = add(edge[idx], m, v);
            }
          }
        }
      }
    }

    auto e = makeDDNode(var, edge, generateDensityMatrix);
    computeTable.insert(x.p, y.p, e);

    e.w = e.w * rWeight;
    return e;
  }

public:
  /**
   * @brief Calculates the expectation value of an operator with respect to a
   * quantum state.
   *
   * @param x A matrix decision diagram (DD) representing the operator.
   * @param y A vector decision diagram (DD) representing the quantum state.
   * @return A floating-point value representing the expectation value of the
   * operator with respect to the quantum state.
   * @throws std::runtime_error if the edges are not on the same level or if the
   * expectation value is non-real.
   *
   * @details This function calls the multiply() function to apply the operator
   * to the quantum state, then calls innerProduct() to calculate the overlap
   * between the original state and the applied state (i.e., <Psi| Psi'> = <Psi|
   * (Op|Psi>)). It also calls the garbageCollect() function to free up any
   * unused memory.
   */
  fp expectationValue(const mEdge& x, const vEdge& y);

  ///
  /// (Partial) trace
  ///
public:
  UnaryComputeTable<dNode*, dCachedEdge> densityTrace{
      config_.ctDmTraceNumBucket};
  UnaryComputeTable<mNode*, mCachedEdge> matrixTrace{
      config_.ctMatTraceNumBucket};

  /**
   * @brief Get the compute table for trace operations.
   *
   * @tparam Node The type of the node.
   * @return A reference to the appropriate compute table for the given node
   * type.
   */
  template <class Node> [[nodiscard]] auto& getTraceComputeTable() {
    if constexpr (std::is_same_v<Node, mNode>) {
      return matrixTrace;
    } else {
      return densityTrace;
    }
  }

  /**
   * @brief Computes the partial trace of a matrix decision diagram.
   *
   * @param a The matrix decision diagram.
   * @param eliminate A vector of booleans indicating which qubits to trace out.
   * @return The resulting matrix decision diagram after the partial trace.
   */
  mEdge partialTrace(const mEdge& a, const std::vector<bool>& eliminate);

  /**
   * @brief Computes the trace of a decision diagram.
   *
   * @tparam Node The type of the node.
   * @param a The decision diagram.
   * @param numQubits The number of qubits in the decision diagram.
   * @return The trace of the decision diagram as a complex value.
   */
  template <class Node>
  ComplexValue trace(const Edge<Node>& a, const std::size_t numQubits) {
    if (a.isIdentity()) {
      return static_cast<ComplexValue>(a.w);
    }
    const auto eliminate = std::vector<bool>(numQubits, true);
    return trace(a, eliminate, numQubits).w;
  }

private:
  /**
   * @brief Computes the normalized (partial) trace using a compute table to
   * store results for eliminated nodes.
   * @details At each level, perform a lookup and store results in the compute
   * table only if all lower-level qubits are eliminated as well.
   *
   * This optimization allows the full trace
   * computation to scale linearly with respect to the number of nodes.
   * However, the partial trace computation still scales with the number of
   * paths to the lowest level in the DD that should be traced out.
   *
   * For matrices, normalization is continuously applied, dividing by two at
   * each level marked for elimination, thereby ensuring that the result is
   * mapped to the interval [0,1] (as opposed to the interval [0,2^N]).
   *
   * For density matrices, such normalization is not applied as the trace of
   * density matrices is always 1 by definition.
   */
  template <class Node>
  CachedEdge<Node> trace(const Edge<Node>& a,
                         const std::vector<bool>& eliminate, std::size_t level,
                         std::size_t alreadyEliminated = 0) {
    const auto aWeight = static_cast<ComplexValue>(a.w);
    if (aWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }

    // If `a` is the identity matrix or there is nothing left to eliminate,
    // then simply return `a`
    if (a.isIdentity() ||
        std::none_of(eliminate.begin(),
                     eliminate.begin() +
                         static_cast<std::vector<bool>::difference_type>(level),
                     [](bool v) { return v; })) {
      return CachedEdge<Node>{a.p, aWeight};
    }

    const auto v = a.p->v;
    if (eliminate[v]) {
      // Lookup nodes marked for elimination in the compute table if all
      // lower-level qubits are eliminated as well: if the trace has already
      // been computed, return the result
      const auto eliminateAll = std::all_of(
          eliminate.begin(),
          eliminate.begin() +
              static_cast<std::vector<bool>::difference_type>(level),
          [](bool e) { return e; });
      if (eliminateAll) {
        if (const auto* r = getTraceComputeTable<Node>().lookup(a.p);
            r != nullptr) {
          return {r->p, r->w * aWeight};
        }
      }

      const auto elims = alreadyEliminated + 1;
      auto r = add(trace(a.p->e[0], eliminate, level - 1, elims),
                   trace(a.p->e[3], eliminate, level - 1, elims), v - 1);

      // The resulting weight is continuously normalized to the range [0,1] for
      // matrix nodes
      if constexpr (std::is_same_v<Node, mNode>) {
        r.w = r.w / 2.0;
      }

      // Insert result into compute table if all lower-level qubits are
      // eliminated as well
      if (eliminateAll) {
        getTraceComputeTable<Node>().insert(a.p, r);
      }
      r.w = r.w * aWeight;
      return r;
    }

    std::array<CachedEdge<Node>, NEDGE> edge{};
    std::transform(a.p->e.cbegin(), a.p->e.cend(), edge.begin(),
                   [this, &eliminate, &alreadyEliminated,
                    &level](const Edge<Node>& e) -> CachedEdge<Node> {
                     return trace(e, eliminate, level - 1, alreadyEliminated);
                   });
    const auto adjustedV =
        static_cast<Qubit>(static_cast<std::size_t>(a.p->v) -
                           (static_cast<std::size_t>(std::count(
                                eliminate.begin(), eliminate.end(), true)) -
                            alreadyEliminated));
    auto r = makeDDNode(adjustedV, edge);
    r.w = r.w * aWeight;
    return r;
  }

public:
  ///
  /// Noise Operations
  ///
  StochasticNoiseOperationTable<mEdge> stochasticNoiseOperationCache{
      nqubits, config_.stochasticCacheOps};
  DensityNoiseTable<dEdge, dEdge> densityNoise{config_.ctDmNoiseNumBucket};
};

} // namespace dd
