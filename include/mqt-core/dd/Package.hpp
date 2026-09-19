/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Package.hpp
/// Main decision-diagram package and its construction operations.

#pragma once

#include "dd/CachedEdge.hpp"
#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/Edge.hpp"
#include "dd/Error.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package_fwd.hpp" // IWYU pragma: export
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "dd/UniqueTable.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <stack>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace dd {

/// The DD package class
///
/// This is the main class of the decision diagram module in MQT Core.
/// It contains the core functionality for working with quantum decision
/// diagrams. Specifically, it provides the means to
/// - represent quantum states as decision diagrams,
/// - represent quantum operations as decision diagrams,
/// - multiply decision diagrams (MxV, MxM, etc.),
/// - perform collapsing measurements on decision diagrams,
/// - sample from decision diagrams.
///
/// To this end, it maintains several internal data structures, such as unique
/// tables, compute tables, and memory managers, which are used to manage the
/// nodes of the decision diagrams.
class Package {

  ///
  /// Construction, destruction, information, and reset
  ///
public:
  static constexpr std::size_t MAX_POSSIBLE_QUBITS =
      static_cast<std::size_t>(std::numeric_limits<Qubit>::max()) + 1U;
  static constexpr std::size_t DEFAULT_QUBITS = 32U;
  /// Construct a new DD Package instance
  ///
  /// @param nq The maximum number of qubits to allocate memory for. This can
  /// always be extended later using @ref resize.
  /// @param config The configuration of the package
  [[nodiscard]] static Result<std::unique_ptr<Package>>
  create(size_t nq = DEFAULT_QUBITS, const DDPackageConfig& config = {});
  ~Package() = default;
  Package(const Package& package) = delete;

  Package& operator=(const Package& package) = delete;

  /// Resize the package to a new number of qubits
  ///
  /// This method will resize all the unique tables appropriately so
  /// that they can handle the new number of qubits.
  ///
  /// @param nq The new number of qubits
  [[nodiscard]] std::optional<Error> resize(size_t nq);

  /// Reset package state
  void reset();

  /// Get the number of qubits
  [[nodiscard]] auto qubits() const { return nqubits; }

private:
  Package(size_t nq, const DDPackageConfig& config);
  std::size_t nqubits;
  DDPackageConfig config_;

public:
  /// The memory manager for vector nodes
  MemoryManager vMemoryManager{
      MemoryManager::create<vNode>(config_.utVecInitialAllocationSize),
  };
  /// The memory manager for matrix nodes
  MemoryManager mMemoryManager{
      MemoryManager::create<mNode>(config_.utMatInitialAllocationSize),
  };
  /// The memory manager for complex numbers
  /// @note The real and imaginary part of complex numbers are treated
  /// separately. Hence, it suffices for the manager to only manage real
  /// numbers.
  MemoryManager cMemoryManager{MemoryManager::create<RealNumber>()};

  /// Get the memory manager for a given type
  /// @tparam T The type to get the manager for
  /// @return A reference to the manager
  template <class T> [[nodiscard]] auto& getMemoryManager() {
    if constexpr (std::is_same_v<T, vNode>) {
      return vMemoryManager;
    } else if constexpr (std::is_same_v<T, mNode>) {
      return mMemoryManager;
    } else if constexpr (std::is_same_v<T, RealNumber>) {
      return cMemoryManager;
    }
  }

  /// Reset all memory managers
  /// @arg resizeToTotal If set to true, each manager allocates one chunk of
  /// memory as large as all chunks combined before the reset.
  /// @see MemoryManager::reset
  void resetMemoryManagers(bool resizeToTotal = false);

  /// The unique table used for vector nodes
  UniqueTable vUniqueTable{vMemoryManager,
                           {.nVars = 0U, .nBuckets = config_.utVecNumBucket}};
  /// The unique table used for matrix nodes
  UniqueTable mUniqueTable{mMemoryManager,
                           {.nVars = 0U, .nBuckets = config_.utMatNumBucket}};
  /// The unique table used for complex numbers
  /// @note The table actually only stores real numbers in the interval [0, 1],
  /// but is used to manages all complex numbers throughout the package.
  /// @see RealNumberUniqueTable
  RealNumberUniqueTable cUniqueTable{cMemoryManager};
  ComplexNumbers cn{cUniqueTable};

  /// Get the unique table for a given type
  /// @tparam T The type to get the unique table for
  /// @return A reference to the unique table
  template <class T> [[nodiscard]] auto& getUniqueTable() {
    if constexpr (std::is_same_v<T, vNode>) {
      return vUniqueTable;
    } else if constexpr (std::is_same_v<T, mNode>) {
      return mUniqueTable;
    } else if constexpr (std::is_same_v<T, RealNumber>) {
      return cUniqueTable;
    }
  }

  /// Clear all unique tables
  /// @see UniqueTable::clear
  /// @see RealNumberUniqueTable::clear
  void clearUniqueTables();

  /// Add the DD to a tracking hashset and update its reference count.
  /// @tparam Node The node type of the edge.
  /// @param e The edge to increase the reference count of.
  template <class Node> void incRef(const Edge<Node>& e) noexcept {
    if (Edge<Node>::trackingRequired(e)) {
      roots.addToRoots(e);
    }
  }

  /// Decrease the DD's reference count and remove it from the tracking
  /// hashset if the count hits zero.
  /// @tparam Node The node type of the edge.
  /// @param e The edge to decrease the reference count of.
  /// Returns an error if the edge is not part of the tracking
  /// hashset.
  template <class Node>
  [[nodiscard]] std::optional<Error> decRef(const Edge<Node>& e) {
    if (Edge<Node>::trackingRequired(e)) {
      return roots.removeFromRoots(e);
    }
    return std::nullopt;
  }

  template <class Node> [[nodiscard]] auto& getRootSet() noexcept {
    return roots.getRoots<Node>();
  }

private:
  struct RootSetManager {
    template <class Node>
    using RootSet = std::unordered_map<Edge<Node>, std::size_t>;

    /// Add to respective root set.
    template <class Node> void addToRoots(const Edge<Node>& e) noexcept {
      ++getRoots<Node>()[e];
    }

    /// Remove from respective root set.
    template <class Node>
    std::optional<Error> removeFromRoots(const Edge<Node>& e) {
      auto& set = getRoots<Node>();
      auto it = set.find(e);
      if (it == set.end()) {
        return Error{"Edge is not part of the root set."};
      }
      if (--it->second == 0U) {
        set.erase(it);
      }
      return std::nullopt;
    }

    /// Execute mark() → op() → unmark().
    template <class Result, typename Fn> Result execute(Fn& op) noexcept {
      mark();
      Result res = op();
      unmark();
      return res;
    }

    /// Clear all root sets.
    void reset() {
      vRoots.clear();
      mRoots.clear();
    }

  private:
    /// Mark edges contained in @p roots.
    template <class Node> static void mark(const RootSet<Node>& roots) {
      for (auto& edge : roots | std::views::keys) {
        edge.mark();
      }
    }

    /// Unmark edges contained in @p roots.
    template <class Node> static void unmark(const RootSet<Node>& roots) {
      for (auto& edge : roots | std::views::keys) {
        edge.unmark();
      }
    }

    /// Mark edges contained in all root sets.
    void mark() noexcept {
      RootSetManager::mark(vRoots);
      RootSetManager::mark(mRoots);
    }

    /// Unmark edges contained in all root sets.
    void unmark() noexcept {
      RootSetManager::unmark(vRoots);
      RootSetManager::unmark(mRoots);
    }

    /// Return vector roots.
    template <class Node>
    auto& getRoots() noexcept
      requires(IsVector<Node>)
    {
      return vRoots;
    }

    /// Return matrix roots.
    template <class Node>
    auto& getRoots() noexcept
      requires(IsMatrix<Node>)
    {
      return mRoots;
    }

    RootSet<vNode> vRoots;
    RootSet<mNode> mRoots;

    template <class Node> friend auto& Package::getRootSet() noexcept;
  };

  RootSetManager roots;

public:
  /// Trigger garbage collection on all unique tables.
  ///
  /// Mark-and-sweep algorithm: First, mark all nodes and complex
  /// numbers tracked in @p roots. Second, remove any unmarked nodes and numbers
  /// from the respective unique tables. Lastly, unmark all nodes and complex
  /// numbers again.
  /// @note By default, garbage collection is only triggered if the unique
  /// tables report that a collection might be necessary.
  ///
  /// @param force Force garbage collect, regardless of whether any
  /// table reports that it may need collecting.
  /// @returns Whether at least one vector, matrix, or any complex number was
  /// reclaimed.
  bool garbageCollect(bool force = false);

  struct ActiveCounts {
    std::size_t vector = 0U;
    std::size_t matrix = 0U;
    std::size_t reals = 0U;
  };
  /// Compute the active number of nodes and numbers
  /// @note This traverses every currently tracked DD twice.
  [[nodiscard]] ActiveCounts computeActiveCounts();

  //
  // Matrix nodes, edges and quantum gates
  //

  /// Construct the DD for a single-qubit gate
  /// @param mat The matrix representation of the gate
  /// @param target The target qubit
  /// @return A decision diagram for the gate
  [[nodiscard]] Result<mEdge> makeGateDD(const GateMatrix& mat, Qubit target);

  /// Construct the DD for a single-qubit controlled gate
  /// @param mat The matrix representation of the gate
  /// @param control The control qubit
  /// @param target The target qubit
  /// @return A decision diagram for the gate
  [[nodiscard]] Result<mEdge> makeGateDD(const GateMatrix& mat,
                                         const Control& control, Qubit target);

  /// Construct the DD for a multi-controlled single-qubit gate
  /// @param mat The matrix representation of the gate
  /// @param controls The control qubits
  /// @param target The target qubit
  /// @return A decision diagram for the gate
  [[nodiscard]] Result<mEdge>
  makeGateDD(const GateMatrix& mat, const Controls& controls, Qubit target);

  /// Construct a single-qubit gate DD from a row-major matrix view.
  [[nodiscard]] Result<mEdge>
  makeGateDD(std::span<const std::complex<fp>, NEDGE> mat,
             const Controls& controls, Qubit target);

  /// Creates the DD for a two-qubit gate
  /// @param mat Matrix representation of the gate
  /// @param target0 First target qubit
  /// @param target1 Second target qubit
  /// @return DD representing the gate
  /// Returns an error if the number of qubits is larger than the
  /// package configuration
  [[nodiscard]] Result<mEdge> makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                                 Qubit target0, Qubit target1);

  /// Creates the DD for a two-qubit gate
  /// @param mat Matrix representation of the gate
  /// @param control Control qubit of the two-qubit gate
  /// @param target0 First target qubit
  /// @param target1 Second target qubit
  /// @return DD representing the gate
  /// Returns an error if the number of qubits is larger than the
  /// package configuration
  [[nodiscard]] Result<mEdge> makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                                 const Control& control,
                                                 Qubit target0, Qubit target1);

  /// Creates the DD for a two-qubit gate
  /// @param mat Matrix representation of the gate
  /// @param controls Control qubits of the two-qubit gate
  /// @param target0 First target qubit
  /// @param target1 Second target qubit
  /// @return DD representing the gate
  /// Returns an error if the number of qubits is larger than the
  /// package configuration
  [[nodiscard]] Result<mEdge> makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                                 const Controls& controls,
                                                 Qubit target0, Qubit target1);

  /// Construct a two-qubit gate DD from a row-major matrix view.
  [[nodiscard]] Result<mEdge>
  makeTwoQubitGateDD(std::span<const std::complex<fp>,
                               static_cast<std::size_t>(NEDGE) * NEDGE> mat,
                     const Controls& controls, Qubit target0, Qubit target1);

  /// Creates the DD for a three-qubit gate
  /// @param mat Matrix representation of the gate
  /// @param target0 First target qubit
  /// @param target1 Second target qubit
  /// @param target2 Third target qubit
  /// @return DD representing the gate
  /// Returns an error if the number of qubits is larger than the
  /// package configuration
  [[nodiscard]] Result<mEdge>
  makeThreeQubitGateDD(const ThreeQubitGateMatrix& mat, Qubit target0,
                       Qubit target1, Qubit target2);

  /// Creates the DD for a three-qubit gate
  /// @param mat Matrix representation of the gate
  /// @param control Control qubit of the three-qubit gate
  /// @param target0 First target qubit
  /// @param target1 Second target qubit
  /// @param target2 Third target qubit
  /// @return DD representing the gate
  /// Returns an error if the number of qubits is larger than the
  /// package configuration
  [[nodiscard]] Result<mEdge>
  makeThreeQubitGateDD(const ThreeQubitGateMatrix& mat, const Control& control,
                       Qubit target0, Qubit target1, Qubit target2);

  /// Creates the DD for a three-qubit gate
  /// @param mat Matrix representation of the gate
  /// @param controls Control qubits of the three-qubit gate
  /// @param target0 First target qubit
  /// @param target1 Second target qubit
  /// @param target2 Third target qubit
  /// @return DD representing the gate
  /// Returns an error if the number of qubits is larger than the
  /// package configuration
  [[nodiscard]] Result<mEdge>
  makeThreeQubitGateDD(const ThreeQubitGateMatrix& mat,
                       const Controls& controls, Qubit target0, Qubit target1,
                       Qubit target2);

  /// Construct a three-qubit gate DD from a row-major matrix view.
  [[nodiscard]] Result<mEdge> makeThreeQubitGateDD(
      std::span<const std::complex<fp>,
                static_cast<std::size_t>(THREE_QUBIT_GATE_DIM) *
                    THREE_QUBIT_GATE_DIM> mat,
      const Controls& controls, Qubit target0, Qubit target1, Qubit target2);

  /// Converts a given matrix to a decision diagram
  /// @param matrix A complex matrix to convert to a DD.
  /// @return A decision diagram representing the matrix.
  /// Returns an error if the given matrix is not square or its
  /// length is not a power of two.
  /// Returns an error if the matrix exceeds the package capacity.
  [[nodiscard]] Result<mEdge> makeDDFromMatrix(const CMat& matrix);

  /// Construct a matrix DD without copying its storage.
  /// @param dimension Number of rows and columns; zero yields the identity.
  /// @param entry Callable returning the complex entry at (row, column).
  /// @pre entry is valid for all indices smaller than dimension.
  /// Returns an error if dimension is not a power of two.
  /// Returns an error if the matrix exceeds the package capacity.
  template <class MatrixEntry>
  [[nodiscard]] Result<mEdge> makeDDFromMatrix(const size_t dimension,
                                               const MatrixEntry& entry) {
    if (dimension == 0) {
      return mEdge::one();
    }
    if (!std::has_single_bit(dimension)) {
      return Error{"Matrix must have a length of a power of two."};
    }
    const auto levels = std::bit_width(dimension) - 1;
    if (levels > qubits()) {
      return Error{"Matrix exceeds the package qubit capacity."};
    }
    if (levels == 0) {
      return mEdge::terminal(cn.lookup(entry(0, 0)));
    }
    const auto operand = [](const size_t level) {
      return std::pair{static_cast<Qubit>(level), size_t{1} << level};
    };
    const auto root = buildMatrixDD(entry, operand, levels - 1, 0, 0);
    return mEdge{.p = root.p, .w = cn.lookup(root.w)};
  }

  /// Embed a row-major local matrix on targets in most-significant-bit order.
  /// Missing DD levels represent identity wires. An empty target list takes a
  /// single scalar entry. Controls are supported for one to three targets.
  /// Returns an error if the matrix size does not match the target
  /// count or controls accompany zero or more than three targets.
  /// Returns an error if qubits exceed package capacity, targets are
  /// duplicated, controls have conflicting polarities, or controls overlap
  /// targets.
  [[nodiscard]] Result<mEdge>
  makeGateDD(std::span<const std::complex<fp>> matrix,
             std::span<const Qubit> targets, const Controls& controls = {});

private:
  /// Read matrix bits in DD level order, which may differ from operand order.
  template <class MatrixEntry, class Operand>
  mCachedEdge buildMatrixDD(const MatrixEntry& entry, const Operand& operand,
                            const size_t level, const size_t row,
                            const size_t col) {
    const auto [wire, mask] = operand(level);
    if (level == 0) {
      return makeDDNode<mNode, CachedEdge>(
          wire, {mCachedEdge::terminal(entry(row, col)),
                 mCachedEdge::terminal(entry(row, col | mask)),
                 mCachedEdge::terminal(entry(row | mask, col)),
                 mCachedEdge::terminal(entry(row | mask, col | mask))});
    }
    return makeDDNode<mNode, CachedEdge>(
        wire,
        {buildMatrixDD(entry, operand, level - 1, row, col),
         buildMatrixDD(entry, operand, level - 1, row, col | mask),
         buildMatrixDD(entry, operand, level - 1, row | mask, col),
         buildMatrixDD(entry, operand, level - 1, row | mask, col | mask)});
  }

public:
  /// Create a normalized DD node and return an edge pointing to it.
  ///
  /// Reuses the unique-table entry for an existing normalized node and omits
  /// matrix nodes that represent an identity level.
  ///
  /// @tparam Node The type of the node.
  /// @tparam EdgeType The type of the edge.
  /// @param var The variable associated with the node.
  /// @param edges The edges of the node.
  /// @return An edge pointing to the normalized DD node.
  template <class Node, template <class> class EdgeType>
  EdgeType<Node>
  makeDDNode(const Qubit var,
             const std::array<EdgeType<Node>,
                              std::tuple_size_v<decltype(Node::e)>>& edges) {
    auto& memoryManager = getMemoryManager<Node>();
    auto* p = memoryManager.template get<Node>();

    p->v = var;
    if constexpr (IsMatrix<Node>) {
      p->flags = 0;
    }

    auto e = EdgeType<Node>::normalize(p, edges, memoryManager, cn);
    if constexpr (IsMatrix<Node>) {
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

  /// Delete an edge from the decision diagram.
  ///
  /// @tparam Node The type of the node.
  /// @param e The edge to delete.
  /// @param v The variable associated with the edge.
  /// @param edgeIdx The index of the edge to delete.
  /// @return The modified edge after deletion.
  template <class Node>
  Edge<Node> deleteEdge(const Edge<Node>& e, const Qubit v,
                        const std::size_t edgeIdx) {
    std::unordered_map<Node*, Edge<Node>> nodes{};
    return deleteEdge(e, v, edgeIdx, nodes);
  }

  /// Helper function to delete an edge from the decision diagram.
  ///
  /// @tparam Node The type of the node.
  /// @param e The edge to delete.
  /// @param v The variable associated with the edge.
  /// @param edgeIdx The index of the edge to delete.
  /// @param nodes A map to keep track of processed nodes.
  /// @return The modified edge after deletion.
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
                         : e.p->e[i]; // optimization → node cannot occur below
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

  //
  // Compute table definitions
  //

  /// Clear all compute tables.
  ///
  /// This method clears all entries in the compute tables used for
  /// various operations. It resets the state of the compute tables, making them
  /// ready for new computations.
  void clearComputeTables();

  //
  // Measurements from state decision diagrams
  //

  /// Measure all qubits in the given decision diagram.
  ///
  /// @param rootEdge The decision diagram to measure.
  /// @param collapse If true, the state is collapsed after measurement.
  /// @param mt A random number generator.
  /// @param epsilon The tolerance for numerical instabilities.
  /// @return A string representing the measurement result.
  /// Returns an error if numerical instabilities are detected or if
  /// probabilities do not sum to 1.
  [[nodiscard]] Result<std::string> measureAll(vEdge& rootEdge, bool collapse,
                                               std::mt19937_64& mt,
                                               fp epsilon = 0.001);

private:
  /// Assigns probabilities to nodes in a decision diagram.
  ///
  /// @param edge The edge to start the probability assignment from.
  /// @param probs A map to store the probabilities of each node.
  /// @return The probability of the given edge.
  static fp assignProbabilities(const vEdge& edge,
                                std::unordered_map<const vNode*, fp>& probs);

  /// Collapse after determineMeasurementProbabilities checked the state shape.
  std::optional<Error> collapse(vEdge& rootEdge, Qubit index, fp probability,
                                bool measureZero);

  /// Project a state, caching the result without its incoming weight.
  vCachedEdge project(const vEdge& state, mNode* projector, bool measureZero);

public:
  /// Determine the measurement probabilities for a given qubit index.
  ///
  /// @param rootEdge The root edge of the decision diagram.
  /// @param index The qubit index to determine the measurement probabilities
  /// for.
  /// @return A pair of floating-point values representing the probabilities of
  /// measuring 0 and 1, respectively.
  ///
  /// Returns an error if the qubit is outside the state.
  [[nodiscard]] static Result<std::pair<fp, fp>>
  determineMeasurementProbabilities(const vEdge& rootEdge, Qubit index);

  /// Measures the qubit with the given index in the given state vector
  /// decision diagram. Collapses the state according to the measurement result.
  /// @param rootEdge the root edge of the state vector decision diagram
  /// @param index the index of the qubit to be measured
  /// @param mt the random number generator
  /// @param epsilon the numerical precision used for checking the normalization
  /// of the state vector decision diagram
  /// @return the measurement result ('0' or '1')
  /// Returns an error if a numerical instability is detected during
  /// the measurement.
  /// Returns an error if the qubit is outside the state.
  [[nodiscard]] Result<char> measureOneCollapsing(vEdge& rootEdge, Qubit index,
                                                  std::mt19937_64& mt,
                                                  fp epsilon = 0.001);

  /// Performs a specific measurement on the given state vector decision
  /// diagram. Collapses the state according to the measurement result.
  /// @param rootEdge the root edge of the state vector decision diagram
  /// @param index the index of the qubit to be measured
  /// @param probability the probability of the measurement result (required for
  /// normalization)
  /// @param measureZero whether or not to measure '0' (otherwise '1' is
  /// measured)
  /// Returns an error if the qubit is outside the state.
  [[nodiscard]] std::optional<Error>
  performCollapsingMeasurement(vEdge& rootEdge, Qubit index, fp probability,
                               bool measureZero);

  ///
  /// Addition
  ///
  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge> vectorAdd{
      config_.ctVecAddNumBucket};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge> matrixAdd{
      config_.ctMatAddNumBucket};

  /// Get the compute table for addition operations.
  ///
  /// @tparam Node The type of the node.
  /// @return A reference to the appropriate compute table for the given node
  /// type.
  template <class Node> [[nodiscard]] auto& getAddComputeTable() {
    if constexpr (IsVector<Node>) {
      return vectorAdd;
    } else if constexpr (IsMatrix<Node>) {
      return matrixAdd;
    }
  }

  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge> vectorAddMagnitudes{
      config_.ctVecAddMagNumBucket};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge> matrixAddMagnitudes{
      config_.ctMatAddMagNumBucket};

  /// Get the compute table for addition operations with magnitudes.
  ///
  /// @tparam Node The type of the node.
  /// @return A reference to the appropriate compute table for the given node
  /// type.
  template <class Node> [[nodiscard]] auto& getAddMagnitudesComputeTable() {
    if constexpr (IsVector<Node>) {
      return vectorAddMagnitudes;
    } else if constexpr (IsMatrix<Node>) {
      return matrixAddMagnitudes;
    }
  }

  /// Add two decision diagrams.
  ///
  /// @tparam Node The type of the node.
  /// @param x The first DD.
  /// @param y The second DD.
  /// @return The resulting DD after addition.
  ///
  template <class Node>
  Edge<Node> add(const Edge<Node>& x, const Edge<Node>& y) {
    Qubit var{};
    if (!x.isTerminal()) {
      var = x.p->v;
    }
    if (!y.isTerminal() && y.p->v > var) {
      var = y.p->v;
    }

    const auto result = add2(CachedEdge{x.p, x.w}, {y.p, y.w}, var);
    return cn.lookup(result);
  }

private:
  /// Select a successor at this level and apply the incoming edge weight.
  template <class Node>
  static CachedEdge<Node> weightedSuccessor(const CachedEdge<Node>& edge,
                                            const Qubit var,
                                            const std::size_t index) {
    if constexpr (IsMatrix<Node>) {
      if (edge.isIdentity() || edge.p->v < var) {
        return index == 0 || index == 3 ? edge : CachedEdge<Node>{};
      }
    }
    const auto& successor = edge.p->e[index];
    auto result = CachedEdge<Node>{successor.p, 0};
    if (!successor.w.exactlyZero()) {
      result.w = edge.w * successor.w;
    }
    return result;
  }

public:
  /// Internal function to add two decision diagrams.
  ///
  /// @tparam Node The type of the node.
  /// @param x The first DD.
  /// @param y The second DD.
  /// @param var The variable associated with the current level of recursion.
  /// @return The resulting DD after addition.
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
      edge[i] = add2(weightedSuccessor(x, var, i), weightedSuccessor(y, var, i),
                     var - 1);
    }
    auto r = makeDDNode(var, edge);
    computeTable.insert(x, y, r);
    return r;
  }

  /// Compute the element-wise magnitude sum of two vectors or matrices.
  ///
  /// For two vectors (or matrices) \p x and \p y, this function returns a
  /// result
  /// \p r such that for each index \p i:
  /// \f$ r[i] = \sqrt{|x[i]|^2 + |y[i]|^2} \f$
  ///
  /// @param x DD representation of the first operand.
  /// @param y DD representation of the second operand.
  /// @param var Number of qubits in the DD.
  /// @return DD representing the result.
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
      edge[i] = addMagnitudes(weightedSuccessor(x, var, i),
                              weightedSuccessor(y, var, i), var - 1);
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

  /// Conjugates a given decision diagram edge.
  ///
  /// @param a The decision diagram edge to conjugate.
  /// @return The conjugated decision diagram edge.
  vEdge conjugate(const vEdge& a);
  /// Recursively conjugates a given decision diagram edge.
  ///
  /// @param a The decision diagram edge to conjugate.
  /// @return The conjugated decision diagram edge.
  vCachedEdge conjugateRec(const vEdge& a);

  ///
  /// Matrix (conjugate) transpose
  ///
  UnaryComputeTable<mNode*, mCachedEdge> conjugateMatrixTranspose{
      config_.ctMatConjTransNumBucket};

  /// Computes the conjugate transpose of a given matrix edge.
  ///
  /// @param a The matrix edge to conjugate transpose.
  /// @return The conjugated transposed matrix edge.
  mEdge conjugateTranspose(const mEdge& a);
  /// Recursively computes the conjugate transpose of a given matrix edge.
  ///
  /// @param a The matrix edge to conjugate transpose.
  /// @return The conjugated transposed matrix edge.
  mCachedEdge conjugateTransposeRec(const mEdge& a);

  ///
  /// Multiplication
  ///
  ComputeTable<mNode*, vNode*, vCachedEdge> matrixVectorMultiplication{
      config_.ctMatVecMultNumBucket};
  ComputeTable<mNode*, mNode*, mCachedEdge> matrixMatrixMultiplication{
      config_.ctMatMatMultNumBucket};

  /// Get the compute table for multiplication operations.
  ///
  /// @tparam RightOperandNode The type of the right operand node.
  /// @return A reference to the appropriate compute table for the given node
  /// type.
  template <class RightOperandNode>
  [[nodiscard]] auto& getMultiplicationComputeTable() {
    if constexpr (std::is_same_v<RightOperandNode, vNode>) {
      return matrixVectorMultiplication;
    } else if constexpr (std::is_same_v<RightOperandNode, mNode>) {
      return matrixMatrixMultiplication;
    }
  }

  /// Applies a matrix operation to a vector.
  ///
  /// The reference count of the input vector is decreased,
  /// while the reference count of the result is increased. After the operation,
  /// garbage collection is triggered.
  ///
  /// @param operation Matrix operation to apply
  /// @param e Vector to apply the operation to
  /// @return The appropriately reference-counted result.
  Result<VectorDD> applyOperation(const MatrixDD& operation, const VectorDD& e);

  /// Applies a matrix operation to a matrix.
  ///
  /// The reference count of the input matrix is decreased,
  /// while the reference count of the result is increased. After the operation,
  /// garbage collection is triggered.
  ///
  /// @param operation Matrix operation to apply
  /// @param e Matrix to apply the operation to
  /// @param applyFromLeft Flag to indicate if the operation should be applied
  /// from the left (default) or right.
  /// @return The appropriately reference-counted result.
  Result<MatrixDD> applyOperation(const MatrixDD& operation, const MatrixDD& e,
                                  bool applyFromLeft = true);

  /// Multiplies two decision diagrams.
  ///
  /// @tparam LeftOperandNode The type of the left operand node.
  /// @tparam RightOperandNode The type of the right operand node.
  /// @param x The left operand decision diagram.
  /// @param y The right operand decision diagram.
  /// @return The resulting decision diagram after multiplication.
  ///
  template <class LeftOperandNode, class RightOperandNode>
    requires IsMatrix<LeftOperandNode> &&
             (IsVector<RightOperandNode> || IsMatrix<RightOperandNode>)
  Edge<RightOperandNode> multiply(const Edge<LeftOperandNode>& x,
                                  const Edge<RightOperandNode>& y) {
    Qubit var{};

    if (!x.isTerminal()) {
      var = x.p->v;
    }
    if (!y.isTerminal() && y.p->v > var) {
      var = y.p->v;
    }
    const auto e = multiply2(x, y, var);
    return cn.lookup(e);
  }

private:
  /// Internal function to multiply two decision diagrams.
  ///
  /// @tparam LeftOperandNode The type of the left operand node.
  /// @tparam RightOperandNode The type of the right operand node.
  /// @param x The left operand decision diagram.
  /// @param y The right operand decision diagram.
  /// @param var The variable associated with the current level of recursion.
  /// @return The resulting DD after multiplication.
  template <class LeftOperandNode, class RightOperandNode>
  CachedEdge<RightOperandNode> multiply2(const Edge<LeftOperandNode>& x,
                                         const Edge<RightOperandNode>& y,
                                         Qubit var) {
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
      return {y.p, rWeight};
    }

    if constexpr (std::is_same_v<RightOperandNode, mNode>) {
      if (y.isIdentity()) {
        return {x.p, rWeight};
      }
    }

    auto& computeTable = getMultiplicationComputeTable<RightOperandNode>();
    if (const auto* r = computeTable.lookup(x.p, y.p); r != nullptr) {
      return {r->p, r->w * rWeight};
    }

    if constexpr (IsMatrix<RightOperandNode>) {
      var = std::max(x.p->v, y.p->v);
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(y.p->e)>;

    constexpr std::size_t rows = RADIX;
    constexpr std::size_t cols = n == NEDGE ? RADIX : 1U;

    std::array<ResultEdge, n> edge{};
    if (x.p->v < var && !y.isTerminal() && y.p->v == var) {
      /// The left operand acts as identity at this level.
      for (std::size_t i = 0; i < n; ++i) {
        edge[i] = multiply2(LEdge{x.p, Complex::one()}, y.p->e[i], var - 1);
      }
    } else {
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

    auto e = makeDDNode(var, edge);
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

  /// Calculates the inner product of two vector decision diagrams.
  ///
  /// @param x A vector DD representing a quantum state.
  /// @param y A vector DD representing a quantum state.
  /// @return A complex number representing the scalar product of the DDs.
  ComplexValue innerProduct(const vEdge& x, const vEdge& y);

  /// Calculates the fidelity between two vector decision diagrams.
  ///
  /// @param x A vector DD representing a quantum state.
  /// @param y A vector DD representing a quantum state.
  /// @return The fidelity between the two quantum states.
  fp fidelity(const vEdge& x, const vEdge& y);

  /// Calculates the fidelity between a vector decision diagram and a
  /// sparse probability vector.
  ///
  /// @param e The root edge of the decision diagram.
  /// @param probs A map of probabilities for each measurement outcome.
  /// @param permutation Optional permutation matching the measurement order.
  /// @return The fidelity of the measurement outcomes.
  [[nodiscard]] static Result<fp>
  fidelityOfMeasurementOutcomes(const vEdge& e, const SparsePVec& probs,
                                const Permutation& permutation = {});

private:
  /// Recursively calculates the inner product of two vector decision
  /// diagrams.
  ///
  /// @param x A vector DD representing a quantum state.
  /// @param y A vector DD representing a quantum state.
  /// @param var The number of levels contained in each vector DD.
  /// @return A complex number representing the scalar product of the DDs.
  ComplexValue innerProduct(const vEdge& x, const vEdge& y, Qubit var);

  /// Recursively calculates the fidelity of measurement outcomes.
  ///
  /// @param e The root edge of the decision diagram.
  /// @param probs A map of probabilities for each measurement outcome.
  /// @param i The current index in the decision diagram traversal.
  /// @param permutation Optional permutation matching the measurement order.
  /// @param nQubits The number of qubits in the decision diagram.
  /// @return The fidelity of the measurement outcomes.
  static fp fidelityOfMeasurementOutcomesRecursive(
      const vEdge& e, const SparsePVec& probs, std::size_t i,
      const Permutation& permutation, std::size_t nQubits);

public:
  /// Compute the real expectation value <y|x|y>.
  ///
  /// @param x An observable whose expectation value is real.
  /// @param y A non-terminal state vector DD.
  /// @return The real part of the expectation value.
  /// Returns an error if the observable acts on a qubit outside
  /// the state.
  /// @pre The observable is not the zero terminal. Debug assertions also
  /// require a non-terminal state and an approximately zero imaginary part.
  [[nodiscard]] Result<fp> expectationValue(const mEdge& x, const vEdge& y);

  ///
  /// Kronecker/tensor product
  ///

  ComputeTable<vNode*, vNode*, vCachedEdge> vectorKronecker{
      config_.ctVecKronNumBucket};
  ComputeTable<mNode*, mNode*, mCachedEdge> matrixKronecker{
      config_.ctMatKronNumBucket};

  /// Get the compute table for Kronecker product operations.
  ///
  /// @tparam Node The type of the node.
  /// @return A reference to the appropriate compute table for the given node
  /// type.
  template <class Node> [[nodiscard]] auto& getKroneckerComputeTable() {
    if constexpr (IsVector<Node>) {
      return vectorKronecker;
    } else {
      return matrixKronecker;
    }
  }

  /// Computes the Kronecker product of two decision diagrams.
  ///
  /// @tparam Node The type of the node.
  /// @param x The first decision diagram.
  /// @param y The second decision diagram.
  /// @param yNumQubits The number of qubits in the second decision diagram.
  /// @param incIdx Whether to shift the first DD above the second DD.
  ///
  /// Matrix widths include leading identity levels omitted from the DD.
  /// The compute table is reused while the index shift remains unchanged.
  /// @return The resulting decision diagram after computing the Kronecker
  /// product.
  template <class Node>
  Edge<Node> kronecker(const Edge<Node>& x, const Edge<Node>& y,
                       const std::size_t yNumQubits, const bool incIdx = true) {
    size_t shift = 0;
    if (incIdx) {
      if constexpr (IsMatrix<Node>) {
        shift = yNumQubits;
      } else if (!y.isTerminal()) {
        shift = static_cast<size_t>(y.p->v) + 1;
      }
    }
    auto& cachedShift =
        IsVector<Node> ? vectorKroneckerShift_ : matrixKroneckerShift_;
    if (cachedShift != shift) {
      getKroneckerComputeTable<Node>().clear();
      cachedShift = shift;
    }
    const auto e = kronecker2(x, y, shift);
    return cn.lookup(e);
  }

private:
  size_t vectorKroneckerShift_ = 0;
  size_t matrixKroneckerShift_ = 0;

  /// Internal function to compute the Kronecker product of two decision
  /// diagrams.
  ///
  /// @tparam Node The type of the node.
  /// @param x The first decision diagram.
  /// @param y The second decision diagram.
  /// @param shift The qubit index offset for nodes from the first DD.
  /// @return The resulting decision diagram after the Kronecker product.
  template <class Node>
  CachedEdge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y,
                              const size_t shift) {
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

    if constexpr (IsMatrix<Node>) {
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
      edge[i] = kronecker2(x.p->e[i], y, shift);
    }

    auto e = makeDDNode(static_cast<Qubit>(x.p->v + shift), edge);
    computeTable.insert(x.p, y.p, {e.p, e.w});
    return {e.p, rWeight};
  }

  ///
  /// (Partial) trace
  ///
public:
  UnaryComputeTable<mNode*, mCachedEdge> matrixTrace{
      config_.ctMatTraceNumBucket};

  /// Get the compute table for trace operations.
  ///
  /// @tparam Node The type of the node.
  /// @return A reference to the appropriate compute table for the given node
  /// type.
  [[nodiscard]] auto& getTraceComputeTable() { return matrixTrace; }

  /// Computes the partial trace of a matrix decision diagram.
  ///
  /// @param a The matrix decision diagram.
  /// @param eliminate A vector of booleans indicating which qubits to trace
  /// out.
  /// @return The normalized partial trace, divided by two per eliminated qubit.
  mEdge partialTrace(const mEdge& a, const std::vector<bool>& eliminate);

  /// Computes the trace of a matrix decision diagram.
  ///
  /// @param a The decision diagram.
  /// @param numQubits The number of qubits in the decision diagram.
  /// @return The normalized trace, divided by the matrix dimension.
  ComplexValue trace(const mEdge& a, std::size_t numQubits);

  /// Checks if a given matrix is close to the identity matrix.
  ///
  /// This function checks if a given matrix is close to the identity
  /// matrix, while ignoring any potential garbage qubits and ignoring the
  /// diagonal weights if `checkCloseToOne` is set to false.
  /// @param m An mEdge that represents the DD of the matrix.
  /// @param tol The accepted tolerance for the edge weights of the DD.
  /// @param garbage A vector of boolean values that defines which qubits are
  /// considered garbage qubits. If it's empty, then no qubit is considered to
  /// be a garbage qubit.
  /// @param checkCloseToOne If false, the function only checks if the matrix is
  /// close to a diagonal matrix.
  [[nodiscard]] bool isCloseToIdentity(const mEdge& m, fp tol = 1e-10,
                                       const std::vector<bool>& garbage = {},
                                       bool checkCloseToOne = true) const;

private:
  /// Computes the normalized (partial) trace using a compute table to
  /// store results for eliminated nodes.
  ///
  /// At each level, perform a lookup and store results in the compute
  /// table only if all lower-level qubits are eliminated as well.
  ///
  /// This optimization allows the full trace
  /// computation to scale linearly with respect to the number of nodes.
  /// However, the partial trace computation still scales with the number of
  /// paths to the lowest level in the DD that should be traced out.
  ///
  /// For matrices, normalization is continuously applied, dividing by two at
  /// each level marked for elimination, thereby ensuring that the result is
  /// mapped to the interval [0,1] (as opposed to the interval [0,2^N]).
  mCachedEdge trace(const mEdge& a, std::span<const size_t> eliminatedBelow);

  /// Recursively checks if a given matrix is close to the identity
  /// matrix.
  ///
  /// @param m The matrix edge to check.
  /// @param visited A set of visited nodes to avoid redundant checks.
  /// @param tol The tolerance for comparing edge weights.
  /// @param garbage A vector of boolean values indicating which qubits are
  /// considered garbage.
  /// @param checkCloseToOne A flag to indicate whether to check if diagonal
  /// elements are close to one.
  /// @return True if the matrix is close to the identity matrix, false
  /// otherwise.
  static bool isCloseToIdentityRecursive(
      const mEdge& m, std::unordered_set<decltype(m.p)>& visited, fp tol,
      const std::vector<bool>& garbage, bool checkCloseToOne);

public:
  //
  // Identity matrices
  //

  /// Create identity DD represented by the one-terminal.
  static mEdge makeIdent();

  Result<mEdge> createInitialMatrix(const std::vector<bool>& ancillary);

  //
  // Ancillary and garbage reduction
  //

  /// Reduces the decision diagram by handling ancillary qubits.
  ///
  /// @param e The matrix decision diagram edge to be reduced.
  /// @param ancillary A boolean vector indicating which qubits are ancillary
  /// (true) or not (false).
  /// @param regular Flag indicating whether to perform regular (true) or
  /// inverse (false) reduction.
  /// @return The reduced matrix decision diagram edge.
  ///
  /// Transfers the input edge's reference to the result when reduction changes
  /// a non-terminal DD. The input reference is unchanged for a no-op.
  /// Identity inputs retain their reference and yield an incremented result.
  Result<mEdge> reduceAncillae(mEdge e, const std::vector<bool>& ancillary,
                               bool regular = true);

  /// Reduces the given decision diagram by summing entries for garbage
  /// qubits.
  ///
  /// For each garbage qubit q, this function sums all the entries for q = 0 and
  /// q = 1, setting the entry for q = 0 to the sum and the entry for q = 1 to
  /// zero. To ensure that the probabilities of the resulting state are the sum
  /// of the probabilities of the initial state, the function computes
  /// `sqrt(|a|^2 + |b|^2)` for two entries `a` and `b`.
  ///
  /// @param e DD representation of the matrix/vector.
  /// @param garbage Vector that describes which qubits are garbage and which
  /// ones are not. If garbage[i] = true, then qubit q_i is considered garbage.
  /// @param normalizeWeights By default set to `false`. If set to `true`, the
  /// function changes all weights in the DD to their magnitude, also for
  /// non-garbage qubits. This is used for checking partial equivalence of
  /// circuits. For partial equivalence, only the measurement probabilities are
  /// considered, so we need to consider only the magnitudes of each entry.
  /// @return DD representing the reduced matrix/vector.
  Result<vEdge> reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                              bool normalizeWeights = false);

  /// Reduces garbage qubits in a matrix decision diagram.
  ///
  /// @param e The matrix decision diagram edge to be reduced.
  /// @param garbage A boolean vector indicating which qubits are garbage (true)
  /// or not (false).
  /// @param regular Flag indicating whether to apply regular (true) or inverse
  /// (false) reduction. In regular mode, garbage entries are summed in the
  /// first two components, in inverse mode, they are summed in the first and
  /// third components.
  /// @param normalizeWeights Flag indicating whether to normalize weights to
  /// their magnitudes. When true, all weights in the DD are changed to their
  /// magnitude, also for non-garbage qubits. This is used for checking partial
  /// equivalence where only measurement probabilities matter.
  /// @return The reduced matrix decision diagram edge.
  ///
  /// For each garbage qubit q, this function sums all the entries for
  /// q=0 and q=1, setting the entry for q=0 to the sum and the entry for q=1 to
  /// zero. To maintain proper probabilities, the function computes sqrt(|a|^2 +
  /// |b|^2) for two entries a and b.
  Result<mEdge> reduceGarbage(const mEdge& e, const std::vector<bool>& garbage,
                              bool regular = true,
                              bool normalizeWeights = false);

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

  //
  // Vector and matrix extraction from DDs
  //
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
        if (mappedNode.contains(currentEdge->p)) {
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

  /// Streams must report I/O failures through their state flags.
  template <class Node, class Edge = Edge<Node>,
            size_t N = std::tuple_size_v<decltype(Node::e)>>
  [[nodiscard]] Result<Edge> deserialize(std::istream& is,
                                         const bool readBinary = false) {
    if (is.exceptions() != std::ios::goodbit) {
      return Error{
          "DD deserialization requires a stream without exception flags.",
          Error::Kind::IO};
    }
    auto result = CachedEdge<Node>::one();
    ComplexValue rootweight{};
    std::unordered_map<int64_t, Node*> nodes;
    const auto invalid = [] {
      return Error{"Invalid or truncated serialized DD."};
    };
    const auto readInteger = []<typename T>(std::string_view& input, T& value) {
      if (input.empty()) {
        return false;
      }
      const auto parsed =
          std::from_chars(input.data(), input.data() + input.size(), value);
      if (parsed.ec != std::errc{}) {
        return false;
      }
      input.remove_prefix(static_cast<size_t>(parsed.ptr - input.data()));
      return true;
    };
    if (readBinary) {
      std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
      is.read(reinterpret_cast<char*>(&version), sizeof(version));
      if (!is || version != SERIALIZATION_VERSION) {
        return invalid();
      }
      rootweight.readBinary(is);
      if (!is || !std::isfinite(rootweight.r) || !std::isfinite(rootweight.i)) {
        return invalid();
      }
      while (true) {
        int64_t index{};
        is.read(reinterpret_cast<char*>(&index), sizeof(index));
        if (!is) {
          if (is.eof() && is.gcount() == 0) {
            break;
          }
          return invalid();
        }
        Qubit wire{};
        is.read(reinterpret_cast<char*>(&wire), sizeof(wire));
        std::array<int64_t, N> indices{};
        std::array<ComplexValue, N> weights{};
        for (size_t i = 0; i < N; ++i) {
          is.read(reinterpret_cast<char*>(&indices[i]), sizeof(indices[i]));
          weights[i].readBinary(is);
        }
        if (!is) {
          return invalid();
        }
        auto node = deserializeNode(index, wire, indices, weights, nodes);
        if (auto* error = std::get_if<Error>(&node)) {
          return std::move(*error);
        }
        result = std::get<0>(node);
      }
    } else {
      std::string line;
      if (!std::getline(is, line)) {
        return invalid();
      }
      std::string_view text = line;
      uint64_t version{};
      if (!readInteger(text, version) || !text.empty() ||
          version != SERIALIZATION_VERSION) {
        return invalid();
      }
      if (!std::getline(is, line)) {
        return invalid();
      }
      auto weight = ComplexValue::parse(line);
      if (auto* error = std::get_if<Error>(&weight)) {
        return std::move(*error);
      }
      rootweight = std::get<0>(weight);
      while (std::getline(is, line)) {
        if (line.empty() || line.size() == 1) {
          continue;
        }
        text = line;
        int64_t index{};
        size_t wire{};
        if (!readInteger(text, index) || index < 0 || !text.starts_with(' ')) {
          return invalid();
        }
        text.remove_prefix(1);
        if (!readInteger(text, wire) || wire >= qubits()) {
          return invalid();
        }
        std::array<int64_t, N> indices{};
        indices.fill(-2);
        std::array<ComplexValue, N> weights{};
        for (size_t i = 0; i < N; ++i) {
          if (!text.starts_with(" (")) {
            return invalid();
          }
          text.remove_prefix(2);
          const auto end = text.find(')');
          if (end == std::string_view::npos) {
            return invalid();
          }
          auto edge = text.substr(0, end);
          text.remove_prefix(end + 1);
          if (edge.empty()) {
            continue;
          }
          if (!readInteger(edge, indices[i]) || !edge.starts_with(' ')) {
            return invalid();
          }
          edge.remove_prefix(1);
          auto edgeWeight = ComplexValue::parse(edge);
          if (auto* error = std::get_if<Error>(&edgeWeight)) {
            return std::move(*error);
          }
          weights[i] = std::get<0>(edgeWeight);
        }
        while (text.starts_with(' ')) {
          text.remove_prefix(1);
        }
        if (!text.empty() && !text.starts_with('#')) {
          return invalid();
        }
        auto node = deserializeNode(index, static_cast<Qubit>(wire), indices,
                                    weights, nodes);
        if (auto* error = std::get_if<Error>(&node)) {
          return std::move(*error);
        }
        result = std::get<0>(node);
      }
    }
    if (is.bad()) {
      return Error{"Cannot read serialized DD.", Error::Kind::IO};
    }
    return Edge{result.p, cn.lookup(result.w * rootweight)};
  }

  template <class Node, class Edge = Edge<Node>>
  [[nodiscard]] Result<Edge> deserialize(const std::string& inputFilename,
                                         const bool readBinary) {
    auto input = std::ifstream(inputFilename, std::ios::binary);
    if (!input) {
      return Error{"Cannot open serialized file: " + inputFilename,
                   Error::Kind::IO};
    }
    return deserialize<Node>(input, readBinary);
  }

private:
  template <class Node, std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  Result<CachedEdge<Node>>
  deserializeNode(const std::int64_t index, const Qubit v,
                  const std::array<std::int64_t, N>& edgeIdx,
                  const std::array<ComplexValue, N>& edgeWeight,
                  std::unordered_map<std::int64_t, Node*>& nodes) {
    if (index == -1) {
      return CachedEdge<Node>::zero();
    }

    if (index < 0 || v >= qubits() || nodes.contains(index)) {
      return Error{"Invalid serialized DD node index or qubit."};
    }

    std::array<CachedEdge<Node>, N> edges{};
    for (auto i = 0U; i < N; ++i) {
      if (edgeIdx[i] == -2) {
        edges[i] = CachedEdge<Node>::zero();
      } else {
        if (edgeIdx[i] == -1) {
          edges[i] = CachedEdge<Node>::one();
        } else {
          const auto child = nodes.find(edgeIdx[i]);
          if (child == nodes.end() ||
              (!Node::isTerminal(child->second) && child->second->v >= v)) {
            return Error{"Serialized DD edges must refer to preceding nodes on "
                         "lower qubits."};
          }
          edges[i].p = child->second;
        }
        if (!std::isfinite(edgeWeight[i].r) ||
            !std::isfinite(edgeWeight[i].i)) {
          return Error{"Serialized DD weights must be finite."};
        }
        edges[i].w = edgeWeight[i];
      }
    }
    auto r = makeDDNode(v, edges);
    nodes[index] = r.p;
    return r;
  }
};

} // namespace dd
