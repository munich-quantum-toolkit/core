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
#include "dd/DensityNoiseTable.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package_fwd.hpp" // IWYU pragma: export
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

private:
  std::size_t nqubits;
  DDPackageConfig config_;

public:
  /// The memory manager for vector nodes
  MemoryManager vMemoryManager{
      MemoryManager::create<vNode>(config_.utVecInitialAllocationSize)};
  /// The memory manager for matrix nodes
  MemoryManager mMemoryManager{
      MemoryManager::create<mNode>(config_.utMatInitialAllocationSize)};
  /// The memory manager for density matrix nodes
  MemoryManager dMemoryManager{
      MemoryManager::create<dNode>(config_.utDmInitialAllocationSize)};
  /**
   * @brief The memory manager for complex numbers
   * @note The real and imaginary part of complex numbers are treated
   * separately. Hence, it suffices for the manager to only manage real numbers.
   */
  MemoryManager cMemoryManager{MemoryManager::create<RealNumber>()};

  /**
   * @brief Get the memory manager for a given type
   * @tparam T The type to get the manager for
   * @return A reference to the manager
   */
  template <class T> [[nodiscard]] auto& getMemoryManager() {
    if constexpr (std::is_same_v<T, vNode>) {
      return vMemoryManager;
    } else if constexpr (std::is_same_v<T, mNode>) {
      return mMemoryManager;
    } else if constexpr (std::is_same_v<T, dNode>) {
      return dMemoryManager;
    } else if constexpr (std::is_same_v<T, RealNumber>) {
      return cMemoryManager;
    }
  }

  /**
   * @brief Reset all memory managers
   * @arg resizeToTotal If set to true, each manager allocates one chunk of
   * memory as large as all chunks combined before the reset.
   * @see MemoryManager::reset
   */
  void resetMemoryManagers(bool resizeToTotal = false);

  /// The unique table used for vector nodes
  UniqueTable vUniqueTable{vMemoryManager, {0U, config_.utVecNumBucket}};
  /// The unique table used for matrix nodes
  UniqueTable mUniqueTable{mMemoryManager, {0U, config_.utMatNumBucket}};
  /// The unique table used for density matrix nodes
  UniqueTable dUniqueTable{dMemoryManager, {0U, config_.utDmNumBucket}};
  /**
   * @brief The unique table used for complex numbers
   * @note The table actually only stores real numbers in the interval [0, 1],
   * but is used to manages all complex numbers throughout the package.
   * @see RealNumberUniqueTable
   */
  RealNumberUniqueTable cUniqueTable{cMemoryManager};
  ComplexNumbers cn{cUniqueTable};

  /**
   * @brief Get the unique table for a given type
   * @tparam T The type to get the unique table for
   * @return A reference to the unique table
   */
  template <class T> [[nodiscard]] auto& getUniqueTable() {
    if constexpr (std::is_same_v<T, vNode>) {
      return vUniqueTable;
    } else if constexpr (std::is_same_v<T, mNode>) {
      return mUniqueTable;
    } else if constexpr (std::is_same_v<T, dNode>) {
      return dUniqueTable;
    } else if constexpr (std::is_same_v<T, RealNumber>) {
      return cUniqueTable;
    }
  }

  /**
   * @brief Clear all unique tables
   * @see UniqueTable::clear
   * @see RealNumberUniqueTable::clear
   */
  void clearUniqueTables();

  /**
   * @brief Add the DD to a tracking hashset and update its reference count.
   * @tparam Node The node type of the edge.
   * @param e The edge to increase the reference count of.
   */
  template <class Node> void incRef(const Edge<Node>& e) noexcept {
    if (Edge<Node>::trackingRequired(e)) {
      roots.addToRoots(e);
    }
  }

  /**
   * @brief Decrease the DD's reference count and remove it from the tracking
   * hashset if the count hits zero.
   * @tparam Node The node type of the edge.
   * @param e The edge to decrease the reference count of.
   * @throws std::invalid_argument If the edge is not part of the tracking
   * hashset.
   */
  template <class Node> void decRef(const Edge<Node>& e) {
    if (Edge<Node>::trackingRequired(e)) {
      roots.removeFromRoots(e);
    }
  }

  template <class Node> [[nodiscard]] auto& getRootSet() noexcept {
    return roots.getRoots<Node>();
  }

private:
  struct RootSetManager {
  public:
    template <class Node>
    using RootSet = std::unordered_map<Edge<Node>, std::size_t>;

    /// @brief Add to respective root set.
    template <class Node> void addToRoots(const Edge<Node>& e) noexcept {
      getRoots<Node>()[e]++;
    }

    /// @brief Remove from respective root set.
    template <class Node> void removeFromRoots(const Edge<Node>& e) {
      auto& set = getRoots<Node>();
      auto it = set.find(e);
      if (it == set.end()) {
        throw std::invalid_argument("Edge is not part of the root set.");
      }
      if (--it->second == 0U) {
        set.erase(it);
      }
    }

    /// @brief Execute mark() -> op() -> unmark().
    template <class Result, typename Fn> Result execute(Fn& op) noexcept {
      mark();
      Result res = op();
      unmark();
      return res;
    }

    /// @brief Clear all root sets.
    void reset() {
      vRoots.clear();
      mRoots.clear();
      dRoots.clear();
    }

  private:
    /// @brief Mark edges contained in @p roots.
    template <class Node> static void mark(const RootSet<Node>& roots) {
      for (auto& [edge, _] : roots) {
        edge.mark();
      }
    }

    /// @brief Unmark edges contained in @p roots.
    template <class Node> static void unmark(const RootSet<Node>& roots) {
      for (auto& [edge, _] : roots) {
        edge.unmark();
      }
    }

    /// @brief Mark edges contained in all root sets.
    void mark() noexcept {
      RootSetManager::mark(vRoots);
      RootSetManager::mark(mRoots);
      RootSetManager::mark(dRoots);
    }

    /// @brief Unmark edges contained in all root sets.
    void unmark() noexcept {
      RootSetManager::unmark(vRoots);
      RootSetManager::unmark(mRoots);
      RootSetManager::unmark(dRoots);
    }

    /// @brief Return vector roots.
    template <class Node,
              std::enable_if_t<std::is_same_v<Node, vNode>, bool> = true>
    auto& getRoots() noexcept {
      return vRoots;
    }

    /// @brief Return matrix roots.
    template <class Node,
              std::enable_if_t<std::is_same_v<Node, mNode>, bool> = true>
    auto& getRoots() noexcept {
      return mRoots;
    }

    /// @brief Return density roots.
    template <class Node,
              std::enable_if_t<std::is_same_v<Node, dNode>, bool> = true>
    auto& getRoots() noexcept {
      return dRoots;
    }

    RootSet<vNode> vRoots;
    RootSet<mNode> mRoots;
    RootSet<dNode> dRoots;

    template <class Node> friend auto& Package::getRootSet() noexcept;
  };

  RootSetManager roots;

public:
  /**
   * @brief Trigger garbage collection on all unique tables.
   * @details Mark-and-sweep algorithm: First, mark all nodes and complex
   * numbers tracked in @p roots. Second, remove any unmarked nodes and numbers
   * from the respective unique tables. Lastly, unmark all nodes and complex
   * numbers again.
   * @note By default, garbage collection is only triggered if the unique tables
   * report that a collection might be necessary.
   *
   * @param force Force garbage collect, regardless of whether any
   * table reports that it may need collecting.
   * @returns Whether at least one vector, matrix, density-matrix node, or
   *          any complex number was reclaimed.
   */
  bool garbageCollect(bool force = false);

  struct ActiveCounts {
    std::size_t vector = 0U;
    std::size_t matrix = 0U;
    std::size_t density = 0U;
    std::size_t reals = 0U;
  };
  /**
   * @brief Compute the active number of nodes and numbers
   * @note This traverses every currently tracked DD twice.
   */
  [[nodiscard]] ActiveCounts computeActiveCounts();

  ///
  /// Vector nodes, edges and quantum states
  ///

  /**
   * @brief Construct the all-zero density operator
            \f$|0...0\rangle\langle0...0|\f$
   * @param n The number of qubits
   * @return A decision diagram for the all-zero density operator
   */
  dEdge makeZeroDensityOperator(std::size_t n);

  ///
  /// Matrix nodes, edges and quantum gates
  ///

  /**
   * @brief Construct the DD for a single-qubit gate
   * @param mat The matrix representation of the gate
   * @param target The target qubit
   * @return A decision diagram for the gate
   */
  mEdge makeGateDD(const GateMatrix& mat, qc::Qubit target);

  /**
   * @brief Construct the DD for a single-qubit controlled gate
   * @param mat The matrix representation of the gate
   * @param control The control qubit
   * @param target The target qubit
   * @return A decision diagram for the gate
   */
  mEdge makeGateDD(const GateMatrix& mat, const qc::Control& control,
                   qc::Qubit target);

  /**
   * @brief Construct the DD for a multi-controlled single-qubit gate
   * @param mat The matrix representation of the gate
   * @param controls The control qubits
   * @param target The target qubit
   * @return A decision diagram for the gate
   */
  mEdge makeGateDD(const GateMatrix& mat, const qc::Controls& controls,
                   qc::Qubit target);

  /**
   * @brief Creates the DD for a two-qubit gate
   * @param mat Matrix representation of the gate
   * @param target0 First target qubit
   * @param target1 Second target qubit
   * @return DD representing the gate
   * @throws std::runtime_error if the number of qubits is larger than the
   * package configuration
   */
  mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat, qc::Qubit target0,
                           qc::Qubit target1);

  /**
   * @brief Creates the DD for a two-qubit gate
   * @param mat Matrix representation of the gate
   * @param control Control qubit of the two-qubit gate
   * @param target0 First target qubit
   * @param target1 Second target qubit
   * @return DD representing the gate
   * @throws std::runtime_error if the number of qubits is larger than the
   * package configuration
   */
  mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                           const qc::Control& control, qc::Qubit target0,
                           qc::Qubit target1);

  /**
   * @brief Creates the DD for a two-qubit gate
   * @param mat Matrix representation of the gate
   * @param controls Control qubits of the two-qubit gate
   * @param target0 First target qubit
   * @param target1 Second target qubit
   * @return DD representing the gate
   * @throws std::runtime_error if the number of qubits is larger than the
   * package configuration
   */
  mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                           const qc::Controls& controls, qc::Qubit target0,
                           qc::Qubit target1);

  /**
   * @brief Converts a given matrix to a decision diagram
   * @param matrix A complex matrix to convert to a DD.
   * @return A decision diagram representing the matrix.
   * @throws std::invalid_argument If the given matrix is not square or its
   * length is not a power of two.
   */
  mEdge makeDDFromMatrix(const CMat& matrix);

private:
  /**
   * @brief Constructs a decision diagram (DD) from a complex matrix using a
   * recursive algorithm.
   *
   * @param matrix The complex matrix from which to create the DD.
   * @param level The current level of recursion. Starts at the highest level of
   * the matrix (log base 2 of the matrix size - 1).
   * @param rowStart The starting row of the quadrant being processed.
   * @param rowEnd The ending row of the quadrant being processed.
   * @param colStart The starting column of the quadrant being processed.
   * @param colEnd The ending column of the quadrant being processed.
   * @return An mCachedEdge representing the root node of the created DD.
   *
   * @details This function recursively breaks down the matrix into quadrants
   * until each quadrant has only one element. At each level of recursion, four
   * new edges are created, one for each quadrant of the matrix. The four
   * resulting decision diagram edges are used to create a new decision diagram
   * node at the current level, and this node is returned as the result of the
   * current recursive call. At the base case of recursion, the matrix has only
   * one element, which is converted into a terminal node of the decision
   * diagram.
   *
   * @note This function assumes that the matrix size is a power of two.
   */
  mCachedEdge makeDDFromMatrix(const CMat& matrix, Qubit level,
                               std::size_t rowStart, std::size_t rowEnd,
                               std::size_t colStart, std::size_t colEnd);

public:
  /**
   * @brief Create a normalized DD node and return an edge pointing to it.
   *
   * @details The node is not recreated if it already exists. This function
   * retrieves a node from the memory manager, sets its variable, and normalizes
   * the edges. If the node resembles the identity, it is skipped. The function
   * then looks up the node in the unique table and returns an edge pointing to
   * it.
   *
   * @tparam Node The type of the node.
   * @tparam EdgeType The type of the edge.
   * @param var The variable associated with the node.
   * @param edges The edges of the node.
   * @param generateDensityMatrix Flag to indicate if a density matrix node
   * should be generated.
   * @return An edge pointing to the normalized DD node.
   */
  template <class Node, template <class> class EdgeType>
  EdgeType<Node>
  makeDDNode(const Qubit var,
             const std::array<EdgeType<Node>,
                              std::tuple_size_v<decltype(Node::e)>>& edges,
             [[maybe_unused]] const bool generateDensityMatrix = false) {
    auto& memoryManager = getMemoryManager<Node>();
    auto p = memoryManager.template get<Node>();

    p->v = var;
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      p->flags = 0;
      if constexpr (std::is_same_v<Node, dNode>) {
        p->setDensityMatrixNodeFlag(generateDensityMatrix);
      }
    }

    auto e = EdgeType<Node>::normalize(p, edges, memoryManager, cn);
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      if (!e.isTerminal()) {
        const auto& es = e.p->e;
        // Check if node resembles the identity. If so, skip it.
        if ((es[0].p == es[3].p) &&
            (es[0].w.exactlyOne() && es[1].w.exactlyZero() &&
             es[2].w.exactlyZero() && es[3].w.exactlyOne())) {
          auto* ptr = es[0].p;
          memoryManager.returnEntry(*e.p);
          return EdgeType<Node>{ptr, e.w};
        }
      }
    }

    // look it up in the unique tables
    auto& uniqueTable = getUniqueTable<Node>();
    auto* l = uniqueTable.lookup(e.p);

    return EdgeType<Node>{l, e.w};
  }

  /**
   * @brief Delete an edge from the decision diagram.
   *
   * @tparam Node The type of the node.
   * @param e The edge to delete.
   * @param v The variable associated with the edge.
   * @param edgeIdx The index of the edge to delete.
   * @return The modified edge after deletion.
   */
  template <class Node>
  Edge<Node> deleteEdge(const Edge<Node>& e, const Qubit v,
                        const std::size_t edgeIdx) {
    std::unordered_map<Node*, Edge<Node>> nodes{};
    return deleteEdge(e, v, edgeIdx, nodes);
  }

  /**
   * @brief Helper function to delete an edge from the decision diagram.
   *
   * @tparam Node The type of the node.
   * @param e The edge to delete.
   * @param v The variable associated with the edge.
   * @param edgeIdx The index of the edge to delete.
   * @param nodes A map to keep track of processed nodes.
   * @return The modified edge after deletion.
   */
  template <class Node>
  Edge<Node> deleteEdge(const Edge<Node>& e, const Qubit v,
                        const std::size_t edgeIdx,
                        std::unordered_map<Node*, Edge<Node>>& nodes) {
    if (e.isTerminal()) {
      return e;
    }

    const auto& nodeIt = nodes.find(e.p);
    Edge<Node> r{};
    if (nodeIt != nodes.end()) {
      r = nodeIt->second;
    } else {
      constexpr std::size_t n = std::tuple_size_v<decltype(e.p->e)>;
      std::array<Edge<Node>, n> edges{};
      if (e.p->v == v) {
        for (std::size_t i = 0; i < n; i++) {
          edges[i] = i == edgeIdx
                         ? Edge<Node>::zero()
                         : e.p->e[i]; // optimization -> node cannot occur below
                                      // again, since dd is assumed to be free
        }
      } else {
        for (std::size_t i = 0; i < n; i++) {
          edges[i] = deleteEdge(e.p->e[i], v, edgeIdx, nodes);
        }
      }

      r = makeDDNode(e.p->v, edges);
      nodes[e.p] = r;
    }
    r.w = cn.lookup(r.w * e.w);
    return r;
  }

  ///
  /// Compute table definitions
  ///

  /**
   * @brief Clear all compute tables.
   *
   * @details This method clears all entries in the compute tables used for
   * various operations. It resets the state of the compute tables, making them
   * ready for new computations.
   */
  void clearComputeTables();

  ///
  /// Measurements from state decision diagrams
  ///

  /**
   * @brief Measure all qubits in the given decision diagram.
   *
   * @details This function measures all qubits in the decision diagram
   * represented by `rootEdge`. It checks for numerical instabilities and
   * collapses the state if requested.
   *
   * @param rootEdge The decision diagram to measure.
   * @param collapse If true, the state is collapsed after measurement.
   * @param mt A random number generator.
   * @param epsilon The tolerance for numerical instabilities.
   * @return A string representing the measurement result.
   * @throws std::runtime_error If numerical instabilities are detected or if
   * probabilities do not sum to 1.
   */
  std::string measureAll(vEdge& rootEdge, bool collapse, std::mt19937_64& mt,
                         fp epsilon = 0.001);

private:
  /**
   * @brief Assigns probabilities to nodes in a decision diagram.
   *
   * @details This function recursively assigns probabilities to nodes in a
   * decision diagram. It calculates the probability of reaching each node and
   * stores the result in a map.
   *
   * @param edge The edge to start the probability assignment from.
   * @param probs A map to store the probabilities of each node.
   * @return The probability of the given edge.
   */
  static fp assignProbabilities(const vEdge& edge,
                                std::unordered_map<const vNode*, fp>& probs);

public:
  /**
   * @brief Determine the measurement probabilities for a given qubit index.
   *
   * @param rootEdge The root edge of the decision diagram.
   * @param index The qubit index to determine the measurement probabilities
   * for.
   * @return A pair of floating-point values representing the probabilities of
   * measuring 0 and 1, respectively.
   *
   * @details This function calculates the probabilities of measuring 0 and 1
   * for a given qubit index in the decision diagram. It uses a breadth-first
   * search to traverse the decision diagram and accumulate the measurement
   * probabilities. The function maintains a map of measurement probabilities
   * for each node and a set of visited nodes to avoid redundant calculations.
   * It also uses a queue to process nodes level by level.
   */
  static std::pair<fp, fp>
  determineMeasurementProbabilities(const vEdge& rootEdge, Qubit index);

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

  ///
  /// Addition
  ///
  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge> vectorAdd{
      config_.ctVecAddNumBucket};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge> matrixAdd{
      config_.ctMatAddNumBucket};
  ComputeTable<dCachedEdge, dCachedEdge, dCachedEdge> densityAdd{
      config_.ctDmAddNumBucket};

  /**
   * @brief Get the compute table for addition operations.
   *
   * @tparam Node The type of the node.
   * @return A reference to the appropriate compute table for the given node
   * type.
   */
  template <class Node> [[nodiscard]] auto& getAddComputeTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vectorAdd;
    } else if constexpr (std::is_same_v<Node, mNode>) {
      return matrixAdd;
    } else if constexpr (std::is_same_v<Node, dNode>) {
      return densityAdd;
    }
  }

  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge> vectorAddMagnitudes{
      config_.ctVecAddMagNumBucket};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge> matrixAddMagnitudes{
      config_.ctMatAddMagNumBucket};

  /**
   * @brief Get the compute table for addition operations with magnitudes.
   *
   * @tparam Node The type of the node.
   * @return A reference to the appropriate compute table for the given node
   * type.
   */
  template <class Node> [[nodiscard]] auto& getAddMagnitudesComputeTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vectorAddMagnitudes;
    } else if constexpr (std::is_same_v<Node, mNode>) {
      return matrixAddMagnitudes;
    }
  }

  /**
   * @brief Add two decision diagrams.
   *
   * @tparam Node The type of the node.
   * @param x The first DD.
   * @param y The second DD.
   * @return The resulting DD after addition.
   *
   * @details This function performs the addition of two decision diagrams
   * (DDs). It uses a compute table to cache intermediate results and avoid
   * redundant computations. The addition is conducted recursively, where the
   * function traverses the nodes of the DDs, adds corresponding edges, and
   * normalizes the resulting edges. If the nodes are terminal, their weights
   * are directly added. The function ensures that the resulting DD is properly
   * normalized and stored in the unique table to maintain the canonical form.
   */
  template <class Node>
  Edge<Node> add(const Edge<Node>& x, const Edge<Node>& y) {
    Qubit var{};
    if (!x.isTerminal()) {
      var = x.p->v;
    }
    if (!y.isTerminal() && (y.p->v) > var) {
      var = y.p->v;
    }

    const auto result = add2(CachedEdge{x.p, x.w}, {y.p, y.w}, var);
    return cn.lookup(result);
  }

  /**
   * @brief Internal function to add two decision diagrams.
   *
   * This function is used internally to add two decision diagrams (DDs) of type
   * Node. It is not intended to be called directly.
   *
   * @tparam Node The type of the node.
   * @param x The first DD.
   * @param y The second DD.
   * @param var The variable associated with the current level of recursion.
   * @return The resulting DD after addition.
   */
  template <class Node>
  CachedEdge<Node> add2(const CachedEdge<Node>& x, const CachedEdge<Node>& y,
                        const Qubit var) {
    if (x.w.exactlyZero()) {
      if (y.w.exactlyZero()) {
        return CachedEdge<Node>::zero();
      }
      return y;
    }
    if (y.w.exactlyZero()) {
      return x;
    }
    if (x.p == y.p) {
      const auto rWeight = x.w + y.w;
      return {x.p, rWeight};
    }

    auto& computeTable = getAddComputeTable<Node>();
    if (const auto* r = computeTable.lookup(x, y); r != nullptr) {
      return *r;
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<CachedEdge<Node>, n> edge{};
    for (std::size_t i = 0U; i < n; i++) {
      CachedEdge<Node> e1{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (x.isIdentity() || x.p->v < var) {
          // [ 0 | 1 ]   [ x | 0 ]
          // --------- = ---------
          // [ 2 | 3 ]   [ 0 | x ]
          if (i == 0 || i == 3) {
            e1 = x;
          }
        } else {
          auto& xSuccessor = x.p->e[i];
          e1 = {xSuccessor.p, 0};
          if (!xSuccessor.w.exactlyZero()) {
            e1.w = x.w * xSuccessor.w;
          }
        }
      } else {
        auto& xSuccessor = x.p->e[i];
        e1 = {xSuccessor.p, 0};
        if (!xSuccessor.w.exactlyZero()) {
          e1.w = x.w * xSuccessor.w;
        }
      }
      CachedEdge<Node> e2{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (y.isIdentity() || y.p->v < var) {
          // [ 0 | 1 ]   [ y | 0 ]
          // --------- = ---------
          // [ 2 | 3 ]   [ 0 | y ]
          if (i == 0 || i == 3) {
            e2 = y;
          }
        } else {
          auto& ySuccessor = y.p->e[i];
          e2 = {ySuccessor.p, 0};
          if (!ySuccessor.w.exactlyZero()) {
            e2.w = y.w * ySuccessor.w;
          }
        }
      } else {
        auto& ySuccessor = y.p->e[i];
        e2 = {ySuccessor.p, 0};
        if (!ySuccessor.w.exactlyZero()) {
          e2.w = y.w * ySuccessor.w;
        }
      }

      if constexpr (std::is_same_v<Node, dNode>) {
        dNode::applyDmChangesToNode(e1.p);
        dNode::applyDmChangesToNode(e2.p);
        edge[i] = add2(e1, e2, var - 1);
        dNode::revertDmChangesToNode(e2.p);
        dNode::revertDmChangesToNode(e1.p);
      } else {
        edge[i] = add2(e1, e2, var - 1);
      }
    }
    auto r = makeDDNode(var, edge);
    computeTable.insert(x, y, r);
    return r;
  }

  /**
   * @brief Compute the element-wise magnitude sum of two vectors or matrices.
   *
   * For two vectors (or matrices) \p x and \p y, this function returns a result
   * \p r such that for each index \p i:
   * \f$ r[i] = \sqrt{|x[i]|^2 + |y[i]|^2} \f$
   *
   * @param x DD representation of the first operand.
   * @param y DD representation of the second operand.
   * @param var Number of qubits in the DD.
   * @return DD representing the result.
   */
  template <class Node>
  CachedEdge<Node> addMagnitudes(const CachedEdge<Node>& x,
                                 const CachedEdge<Node>& y, const Qubit var) {
    if (x.w.exactlyZero()) {
      if (y.w.exactlyZero()) {
        return CachedEdge<Node>::zero();
      }
      const auto rWeight = y.w.mag();
      return {y.p, rWeight};
    }
    if (y.w.exactlyZero()) {
      const auto rWeight = x.w.mag();
      return {x.p, rWeight};
    }
    if (x.p == y.p) {
      const auto rWeight = std::sqrt(x.w.mag2() + y.w.mag2());
      return {x.p, rWeight};
    }

    auto& computeTable = getAddMagnitudesComputeTable<Node>();
    if (const auto* r = computeTable.lookup(x, y); r != nullptr) {
      return *r;
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<CachedEdge<Node>, n> edge{};
    for (std::size_t i = 0U; i < n; i++) {
      CachedEdge<Node> e1{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (x.isIdentity() || x.p->v < var) {
          if (i == 0 || i == 3) {
            e1 = x;
          }
        } else {
          auto& xSuccessor = x.p->e[i];
          e1 = {xSuccessor.p, 0};
          if (!xSuccessor.w.exactlyZero()) {
            e1.w = x.w * xSuccessor.w;
          }
        }
      } else {
        auto& xSuccessor = x.p->e[i];
        e1 = {xSuccessor.p, 0};
        if (!xSuccessor.w.exactlyZero()) {
          e1.w = x.w * xSuccessor.w;
        }
      }
      CachedEdge<Node> e2{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (y.isIdentity() || y.p->v < var) {
          if (i == 0 || i == 3) {
            e2 = y;
          }
        } else {
          auto& ySuccessor = y.p->e[i];
          e2 = {ySuccessor.p, 0};
          if (!ySuccessor.w.exactlyZero()) {
            e2.w = y.w * ySuccessor.w;
          }
        }
      } else {
        auto& ySuccessor = y.p->e[i];
        e2 = {ySuccessor.p, 0};
        if (!ySuccessor.w.exactlyZero()) {
          e2.w = y.w * ySuccessor.w;
        }
      }
      edge[i] = addMagnitudes(e1, e2, var - 1);
    }
    auto r = makeDDNode(var, edge);
    computeTable.insert(x, y, r);
    return r;
  }

  ///
  /// Vector conjugation
  ///
  UnaryComputeTable<vNode*, vCachedEdge> conjugateVector{
      config_.ctVecConjNumBucket};

  /**
   * @brief Conjugates a given decision diagram edge.
   *
   * @param a The decision diagram edge to conjugate.
   * @return The conjugated decision diagram edge.
   */
  vEdge conjugate(const vEdge& a);
  /**
   * @brief Recursively conjugates a given decision diagram edge.
   *
   * @param a The decision diagram edge to conjugate.
   * @return The conjugated decision diagram edge.
   */
  vCachedEdge conjugateRec(const vEdge& a);

  ///
  /// Matrix (conjugate) transpose
  ///
  UnaryComputeTable<mNode*, mCachedEdge> conjugateMatrixTranspose{
      config_.ctMatConjTransNumBucket};

  /**
   * @brief Computes the conjugate transpose of a given matrix edge.
   *
   * @param a The matrix edge to conjugate transpose.
   * @return The conjugated transposed matrix edge.
   */
  mEdge conjugateTranspose(const mEdge& a);
  /**
   * @brief Recursively computes the conjugate transpose of a given matrix edge.
   *
   * @param a The matrix edge to conjugate transpose.
   * @return The conjugated transposed matrix edge.
   */
  mCachedEdge conjugateTransposeRec(const mEdge& a);

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
              edge[idx] = add2(edge[idx], m, v);
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
              edge[idx] = add2(edge[idx], m, v);
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

  ///
  /// Inner product, fidelity, expectation value
  ///
public:
  ComputeTable<vNode*, vNode*, vCachedEdge> vectorInnerProduct{
      config_.ctVecInnerProdNumBucket};

  /**
   * @brief Calculates the inner product of two vector decision diagrams.
   *
   * @param x A vector DD representing a quantum state.
   * @param y A vector DD representing a quantum state.
   * @return A complex number representing the scalar product of the DDs.
   */
  ComplexValue innerProduct(const vEdge& x, const vEdge& y);

  /**
   * @brief Calculates the fidelity between two vector decision diagrams.
   *
   * @param x A vector DD representing a quantum state.
   * @param y A vector DD representing a quantum state.
   * @return The fidelity between the two quantum states.
   */
  fp fidelity(const vEdge& x, const vEdge& y);

  /**
   * @brief Calculates the fidelity between a vector decision diagram and a
   * sparse probability vector.
   *
   * @details This function computes the fidelity between a quantum state
   * represented by a vector decision diagram and a sparse probability vector.
   * The optional permutation of qubits can be provided to match the qubit
   * ordering.
   *
   * @param e The root edge of the decision diagram.
   * @param probs A map of probabilities for each measurement outcome.
   * @param permutation An optional permutation of qubits.
   * @return The fidelity of the measurement outcomes.
   */
  static fp
  fidelityOfMeasurementOutcomes(const vEdge& e, const SparsePVec& probs,
                                const qc::Permutation& permutation = {});

private:
  /**
   * @brief Recursively calculates the inner product of two vector decision
   * diagrams.
   *
   * @param x A vector DD representing a quantum state.
   * @param y A vector DD representing a quantum state.
   * @param var The number of levels contained in each vector DD.
   * @return A complex number representing the scalar product of the DDs.
   * @note This function is called recursively such that the number of levels
   *       decreases each time to traverse the DDs.
   */
  ComplexValue innerProduct(const vEdge& x, const vEdge& y, Qubit var);

  /**
   * @brief Recursively calculates the fidelity of measurement outcomes.
   *
   * @details This function computes the fidelity between a quantum state
   * represented by a vector decision diagram and a sparse probability vector.
   * It traverses the decision diagram recursively, calculating the contribution
   * of each path to the overall fidelity. An optional permutation of qubits can
   * be provided to match the qubit ordering.
   *
   * @param e The root edge of the decision diagram.
   * @param probs A map of probabilities for each measurement outcome.
   * @param i The current index in the decision diagram traversal.
   * @param permutation An optional permutation of qubits.
   * @param nQubits The number of qubits in the decision diagram.
   * @return The fidelity of the measurement outcomes.
   */
  static fp fidelityOfMeasurementOutcomesRecursive(
      const vEdge& e, const SparsePVec& probs, std::size_t i,
      const qc::Permutation& permutation, std::size_t nQubits);

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
  /// Kronecker/tensor product
  ///

  ComputeTable<vNode*, vNode*, vCachedEdge> vectorKronecker{
      config_.ctVecKronNumBucket};
  ComputeTable<mNode*, mNode*, mCachedEdge> matrixKronecker{
      config_.ctMatKronNumBucket};

  /**
   * @brief Get the compute table for Kronecker product operations.
   *
   * @tparam Node The type of the node.
   * @return A reference to the appropriate compute table for the given node
   * type.
   */
  template <class Node> [[nodiscard]] auto& getKroneckerComputeTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vectorKronecker;
    } else {
      return matrixKronecker;
    }
  }

  /**
   * @brief Computes the Kronecker product of two decision diagrams.
   *
   * @tparam Node The type of the node.
   * @param x The first decision diagram.
   * @param y The second decision diagram.
   * @param yNumQubits The number of qubits in the second decision diagram.
   * @param incIdx Whether to increment the index of the nodes in the second
   * decision diagram.
   * @return The resulting decision diagram after computing the Kronecker
   * product.
   * @throws std::invalid_argument if the node type is `dNode` (density
   * matrices).
   */
  template <class Node>
  Edge<Node> kronecker(const Edge<Node>& x, const Edge<Node>& y,
                       const std::size_t yNumQubits, const bool incIdx = true) {
    if constexpr (std::is_same_v<Node, dNode>) {
      throw std::invalid_argument(
          "Kronecker is currently not supported for density matrices");
    }

    const auto e = kronecker2(x, y, yNumQubits, incIdx);
    return cn.lookup(e);
  }

private:
  /**
   * @brief Internal function to compute the Kronecker product of two decision
   * diagrams.
   *
   * This function is used internally to compute the Kronecker product of two
   * decision diagrams (DDs) of type Node. It is not intended to be called
   * directly.
   *
   * @tparam Node The type of the node.
   * @param x The first decision diagram.
   * @param y The second decision diagram.
   * @param yNumQubits The number of qubits in the second decision diagram.
   * @param incIdx Whether to increment the qubit index.
   * @return The resulting decision diagram after the Kronecker product.
   */
  template <class Node>
  CachedEdge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y,
                              const std::size_t yNumQubits,
                              const bool incIdx = true) {
    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return CachedEdge<Node>::zero();
    }
    const auto xWeight = static_cast<ComplexValue>(x.w);
    if (xWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }
    const auto yWeight = static_cast<ComplexValue>(y.w);
    if (yWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }
    const auto rWeight = xWeight * yWeight;
    if (rWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }

    if (x.isTerminal() && y.isTerminal()) {
      return {x.p, rWeight};
    }

    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      if (x.isIdentity()) {
        return {y.p, rWeight};
      }
    } else {
      if (x.isTerminal()) {
        return {y.p, rWeight};
      }
      if (y.isTerminal()) {
        return {x.p, rWeight};
      }
    }

    // check if we already computed the product before and return the result
    auto& computeTable = getKroneckerComputeTable<Node>();
    if (const auto* r = computeTable.lookup(x.p, y.p); r != nullptr) {
      return {r->p, rWeight};
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<CachedEdge<Node>, n> edge{};
    for (auto i = 0U; i < n; ++i) {
      edge[i] = kronecker2(x.p->e[i], y, yNumQubits, incIdx);
    }

    // Increase the qubit index
    Qubit idx = x.p->v;
    if (incIdx) {
      // use the given number of qubits if y is an identity
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (y.isIdentity()) {
          idx += static_cast<Qubit>(yNumQubits);
        } else {
          idx += static_cast<Qubit>(y.p->v + 1U);
        }
      } else {
        idx += static_cast<Qubit>(y.p->v + 1U);
      }
    }
    auto e = makeDDNode(idx, edge, true);
    computeTable.insert(x.p, y.p, {e.p, e.w});
    return {e.p, rWeight};
  }

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

  /**
   * @brief Checks if a given matrix is close to the identity matrix.
   * @details This function checks if a given matrix is close to the identity
   * matrix, while ignoring any potential garbage qubits and ignoring the
   * diagonal weights if `checkCloseToOne` is set to false.
   * @param m An mEdge that represents the DD of the matrix.
   * @param tol The accepted tolerance for the edge weights of the DD.
   * @param garbage A vector of boolean values that defines which qubits are
   * considered garbage qubits. If it's empty, then no qubit is considered to be
   * a garbage qubit.
   * @param checkCloseToOne If false, the function only checks if the matrix is
   * close to a diagonal matrix.
   */
  [[nodiscard]] bool isCloseToIdentity(const mEdge& m, fp tol = 1e-10,
                                       const std::vector<bool>& garbage = {},
                                       bool checkCloseToOne = true) const;

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
      auto r = add2(trace(a.p->e[0], eliminate, level - 1, elims),
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

  /**
   * @brief Recursively checks if a given matrix is close to the identity
   * matrix.
   *
   * @param m The matrix edge to check.
   * @param visited A set of visited nodes to avoid redundant checks.
   * @param tol The tolerance for comparing edge weights.
   * @param garbage A vector of boolean values indicating which qubits are
   * considered garbage.
   * @param checkCloseToOne A flag to indicate whether to check if diagonal
   * elements are close to one.
   * @return True if the matrix is close to the identity matrix, false
   * otherwise.
   */
  static bool isCloseToIdentityRecursive(
      const mEdge& m, std::unordered_set<decltype(m.p)>& visited, fp tol,
      const std::vector<bool>& garbage, bool checkCloseToOne);

public:
  ///
  /// Identity matrices
  ///

  /// Create identity DD represented by the one-terminal.
  static mEdge makeIdent();

  mEdge createInitialMatrix(const std::vector<bool>& ancillary);

  ///
  /// Noise Operations
  ///
  StochasticNoiseOperationTable<mEdge> stochasticNoiseOperationCache{
      nqubits, config_.stochasticCacheOps};
  DensityNoiseTable<dEdge, dEdge> densityNoise{config_.ctDmNoiseNumBucket};

  ///
  /// Ancillary and garbage reduction
  ///

  /**
   * @brief Reduces the decision diagram by handling ancillary qubits.
   *
   * @param e The matrix decision diagram edge to be reduced.
   * @param ancillary A boolean vector indicating which qubits are ancillary
   * (true) or not (false).
   * @param regular Flag indicating whether to perform regular (true) or inverse
   * (false) reduction.
   * @return The reduced matrix decision diagram edge.
   *
   * @details This function modifies the decision diagram to account for
   * ancillary qubits by:
   * 1. Early returning if there are no ancillary qubits or if the edge is zero
   * 2. Special handling for identity matrices by creating appropriate zero
   * nodes
   * 3. Finding the lowest ancillary qubit as a starting point
   * 4. Recursively reducing nodes starting from the lowest ancillary qubit
   * 5. Adding zero nodes for any remaining higher ancillary qubits
   *
   * The function maintains proper reference counting by incrementing the
   * reference count of the result and decrementing the reference count of the
   * input edge.
   */
  mEdge reduceAncillae(mEdge e, const std::vector<bool>& ancillary,
                       bool regular = true);

  /**
   * @brief Reduces the given decision diagram by summing entries for garbage
   * qubits.
   *
   * For each garbage qubit q, this function sums all the entries for q = 0 and
   * q = 1, setting the entry for q = 0 to the sum and the entry for q = 1 to
   * zero. To ensure that the probabilities of the resulting state are the sum
   * of the probabilities of the initial state, the function computes
   * `sqrt(|a|^2 + |b|^2)` for two entries `a` and `b`.
   *
   * @param e DD representation of the matrix/vector.
   * @param garbage Vector that describes which qubits are garbage and which
   * ones are not. If garbage[i] = true, then qubit q_i is considered garbage.
   * @param normalizeWeights By default set to `false`. If set to `true`, the
   * function changes all weights in the DD to their magnitude, also for
   *                         non-garbage qubits. This is used for checking
   * partial equivalence of circuits. For partial equivalence, only the
   *                         measurement probabilities are considered, so we
   * need to consider only the magnitudes of each entry.
   * @return DD representing the reduced matrix/vector.
   */
  vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                      bool normalizeWeights = false);

  /**
   * @brief Reduces garbage qubits in a matrix decision diagram.
   *
   * @param e The matrix decision diagram edge to be reduced.
   * @param garbage A boolean vector indicating which qubits are garbage (true)
   * or not (false).
   * @param regular Flag indicating whether to apply regular (true) or inverse
   * (false) reduction. In regular mode, garbage entries are summed in the first
   * two components, in inverse mode, they are summed in the first and third
   * components.
   * @param normalizeWeights Flag indicating whether to normalize weights to
   * their magnitudes. When true, all weights in the DD are changed to their
   * magnitude, also for non-garbage qubits. This is used for checking partial
   * equivalence where only measurement probabilities matter.
   * @return The reduced matrix decision diagram edge.
   *
   * @details For each garbage qubit q, this function sums all the entries for
   * q=0 and q=1, setting the entry for q=0 to the sum and the entry for q=1 to
   * zero. To maintain proper probabilities, the function computes sqrt(|a|^2 +
   * |b|^2) for two entries a and b. The function handles special cases like
   * zero terminals and identity matrices separately and maintains proper
   * reference counting throughout the reduction process.
   */
  mEdge reduceGarbage(const mEdge& e, const std::vector<bool>& garbage,
                      bool regular = true, bool normalizeWeights = false);

private:
  mCachedEdge reduceAncillaeRecursion(mNode* p,
                                      const std::vector<bool>& ancillary,
                                      Qubit lowerbound, bool regular = true);

  vCachedEdge reduceGarbageRecursion(vNode* p, const std::vector<bool>& garbage,
                                     Qubit lowerbound,
                                     bool normalizeWeights = false);
  mCachedEdge reduceGarbageRecursion(mNode* p, const std::vector<bool>& garbage,
                                     Qubit lowerbound, bool regular = true,
                                     bool normalizeWeights = false);

  ///
  /// Vector and matrix extraction from DDs
  ///
public:
  /// transfers a decision diagram from another package to this package
  template <class Node> Edge<Node> transfer(Edge<Node>& original) {
    if (original.isTerminal()) {
      return {original.p, cn.lookup(original.w)};
    }

    // POST ORDER TRAVERSAL USING ONE STACK
    // https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
    Edge<Node> root{};
    std::stack<Edge<Node>*> stack;

    std::unordered_map<decltype(original.p), decltype(original.p)> mappedNode{};

    Edge<Node>* currentEdge = &original;
    constexpr std::size_t n = std::tuple_size_v<decltype(original.p->e)>;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      while (currentEdge != nullptr && !currentEdge->isTerminal()) {
        for (std::size_t i = n - 1; i > 0; --i) {
          auto& edge = currentEdge->p->e[i];
          if (edge.isTerminal()) {
            continue;
          }
          if (edge.w.approximatelyZero()) {
            continue;
          }
          if (mappedNode.contains(edge.p)) {
            continue;
          }

          // non-zero edge to be included
          stack.push(&edge);
        }
        stack.push(currentEdge);
        currentEdge = &currentEdge->p->e[0];
      }
      currentEdge = stack.top();
      stack.pop();

      bool hasChild = false;
      for (std::size_t i = 1; i < n && !hasChild; ++i) {
        auto& edge = currentEdge->p->e[i];
        if (edge.w.approximatelyZero()) {
          continue;
        }
        if (mappedNode.contains(edge.p)) {
          continue;
        }
        hasChild = edge.p == stack.top()->p;
      }

      if (hasChild) {
        Edge<Node>* temp = stack.top();
        stack.pop();
        stack.push(currentEdge);
        currentEdge = temp;
      } else {
        if (!mappedNode.contains(currentEdge->p)) {
          currentEdge = nullptr;
          continue;
        }
        std::array<Edge<Node>, n> edges{};
        for (std::size_t i = 0; i < n; i++) {
          if (currentEdge->p->e[i].isTerminal()) {
            edges[i].p = currentEdge->p->e[i].p;
          } else {
            edges[i].p = mappedNode[currentEdge->p->e[i].p];
          }
          edges[i].w = cn.lookup(currentEdge->p->e[i].w);
        }
        root = makeDDNode(currentEdge->p->v, edges);
        mappedNode[currentEdge->p] = root.p;
        currentEdge = nullptr;
      }
    } while (!stack.empty());
    root.w = cn.lookup(original.w * root.w);
    return root;
  }

  ///
  /// Deserialization
  /// Note: do not rely on the binary format being portable across different
  /// architectures/platforms
  ///

  template <class Node, class Edge = Edge<Node>,
            std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  Edge deserialize(std::istream& is, const bool readBinary = false) {
    auto result = CachedEdge<Node>{};
    ComplexValue rootweight{};

    std::unordered_map<std::int64_t, Node*> nodes{};
    std::int64_t nodeIndex{};
    Qubit v{};
    std::array<ComplexValue, N> edgeWeights{};
    std::array<std::int64_t, N> edgeIndices{};
    edgeIndices.fill(-2);

    if (readBinary) {
      std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
      is.read(reinterpret_cast<char*>(&version),
              sizeof(decltype(SERIALIZATION_VERSION)));
      if (version != SERIALIZATION_VERSION) {
        throw std::runtime_error(
            "Wrong Version of serialization file version. version of file: " +
            std::to_string(version) +
            "; current version: " + std::to_string(SERIALIZATION_VERSION));
      }

      if (!is.eof()) {
        rootweight.readBinary(is);
      }

      while (is.read(reinterpret_cast<char*>(&nodeIndex),
                     sizeof(decltype(nodeIndex)))) {
        is.read(reinterpret_cast<char*>(&v), sizeof(decltype(v)));
        for (std::size_t i = 0U; i < N; i++) {
          is.read(reinterpret_cast<char*>(&edgeIndices[i]),
                  sizeof(decltype(edgeIndices[i])));
          edgeWeights[i].readBinary(is);
        }
        result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
      }
    } else {
      std::string version;
      std::getline(is, version);
      if (std::stoi(version) != SERIALIZATION_VERSION) {
        throw std::runtime_error(
            "Wrong Version of serialization file version. version of file: " +
            version +
            "; current version: " + std::to_string(SERIALIZATION_VERSION));
      }

      const std::string complexRealRegex =
          R"(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![ \d\.]*(?:[eE][+-])?\d*[iI]))?)";
      const std::string complexImagRegex =
          R"(( ?[+-]? ?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?[iI])?)";
      const std::string edgeRegex =
          " \\(((-?\\d+) (" + complexRealRegex + complexImagRegex + "))?\\)";
      const std::regex complexWeightRegex(complexRealRegex + complexImagRegex);

      std::string lineConstruct = "(\\d+) (\\d+)";
      for (std::size_t i = 0U; i < N; ++i) {
        lineConstruct += "(?:" + edgeRegex + ")";
      }
      lineConstruct += " *(?:#.*)?";
      const std::regex lineRegex(lineConstruct);
      std::smatch m;

      std::string line;
      if (std::getline(is, line)) {
        if (!std::regex_match(line, m, complexWeightRegex)) {
          throw std::runtime_error("Regex did not match second line: " + line);
        }
        rootweight.fromString(m.str(1), m.str(2));
      }

      while (std::getline(is, line)) {
        if (line.empty() || line.size() == 1) {
          continue;
        }

        if (!std::regex_match(line, m, lineRegex)) {
          throw std::runtime_error("Regex did not match line: " + line);
        }

        // match 1: node_idx
        // match 2: qubit_idx

        // repeats for every edge
        // match 3: edge content
        // match 4: edge_target_idx
        // match 5: real + imag (without i)
        // match 6: real
        // match 7: imag (without i)
        nodeIndex = std::stoi(m.str(1));
        v = static_cast<Qubit>(std::stoi(m.str(2)));

        for (auto edgeIdx = 3U, i = 0U; i < N; i++, edgeIdx += 5) {
          if (m.str(edgeIdx).empty()) {
            continue;
          }

          edgeIndices[i] = std::stoi(m.str(edgeIdx + 1));
          edgeWeights[i].fromString(m.str(edgeIdx + 3), m.str(edgeIdx + 4));
        }

        result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
      }
    }
    return {result.p, cn.lookup(result.w * rootweight)};
  }

  template <class Node, class Edge = Edge<Node>>
  Edge deserialize(const std::string& inputFilename, const bool readBinary) {
    auto ifs = std::ifstream(inputFilename, std::ios::binary);

    if (!ifs.good()) {
      throw std::invalid_argument("Cannot open serialized file: " +
                                  inputFilename);
    }

    return deserialize<Node>(ifs, readBinary);
  }

private:
  template <class Node, std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  CachedEdge<Node>
  deserializeNode(const std::int64_t index, const Qubit v,
                  std::array<std::int64_t, N>& edgeIdx,
                  const std::array<ComplexValue, N>& edgeWeight,
                  std::unordered_map<std::int64_t, Node*>& nodes) {
    if (index == -1) {
      return CachedEdge<Node>::zero();
    }

    std::array<CachedEdge<Node>, N> edges{};
    for (auto i = 0U; i < N; ++i) {
      if (edgeIdx[i] == -2) {
        edges[i] = CachedEdge<Node>::zero();
      } else {
        if (edgeIdx[i] == -1) {
          edges[i] = CachedEdge<Node>::one();
        } else {
          edges[i].p = nodes[edgeIdx[i]];
        }
        edges[i].w = edgeWeight[i];
      }
    }
    // reset
    edgeIdx.fill(-2);

    auto r = makeDDNode(v, edges);
    nodes[index] = r.p;
    return r;
  }
};

} // namespace dd
