/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DesicionDiagramContainer.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "ir/Permutation.hpp"

namespace dd {

class VectorDDContainer : public DDContainer<vNode> {
public:
  struct Config : public DDContainer<vNode>::Config {
    std::size_t ctVecConjNumBucket =
        UnaryComputeTable<vNode*, vCachedEdge>::DEFAULT_NUM_BUCKETS;
    std::size_t vectorInnerProduct =
        ComputeTable<vNode*, vNode*, vCachedEdge>::DEFAULT_NUM_BUCKETS;
  };

  VectorDDContainer(std::size_t nqubits, RealNumberUniqueTable& cUt,
                    ComplexNumbers& cn, const Config& config)
      : DDContainer<vNode>(nqubits, cUt, cn, config),
        conjugateVector(config.ctVecConjNumBucket),
        vectorInnerProduct(config.vectorInnerProduct) {}

  void reset();

  bool garbageCollect(bool force);

  /**
   * @brief Construct the all-zero state \f$|0...0\rangle\f$
   * @param n The number of qubits
   * @param start The starting qubit index. Default is 0.
   * @return A decision diagram for the all-zero state
   */
  vEdge makeZeroState(std::size_t n, std::size_t start = 0);

  /**
   * @brief Construct a computational basis state \f$|b_{n-1}...b_0\rangle\f$
   * @param n The number of qubits
   * @param state The state to construct
   * @param start The starting qubit index. Default is 0.
   * @return A decision diagram for the computational basis state
   */
  vEdge makeBasisState(std::size_t n, const std::vector<bool>& state,
                       std::size_t start = 0);

  /**
   * @brief Construct a product state out of
   *        \f$\{0, 1, +, -, R, L\}^{\otimes n}\f$.
   * @param n The number of qubits
   * @param state The state to construct
   * @param start The starting qubit index. Default is 0.
   * @return A decision diagram for the product state
   */
  vEdge makeBasisState(std::size_t n, const std::vector<BasisStates>& state,
                       std::size_t start = 0);

  /**
   * @brief Construct a GHZ state \f$|0...0\rangle + |1...1\rangle\f$
   * @param n The number of qubits
   * @return A decision diagram for the GHZ state
   */
  vEdge makeGHZState(std::size_t n);

  /**
   * @brief Construct a W state
   * @details The W state is defined as
   * \f[
   * |0...01\rangle + |0...10\rangle + |10...0\rangle
   * \f]
   * @param n The number of qubits
   * @return A decision diagram for the W state
   */
  vEdge makeWState(std::size_t n);

  /**
   * @brief Construct a decision diagram from an arbitrary state vector
   * @param stateVector The state vector to convert to a DD
   * @return A decision diagram for the state
   */
  vEdge makeStateFromVector(const CVec& stateVector);

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

  /**
   * @brief Constructs a decision diagram (DD) from a state vector using a
   * recursive algorithm.
   *
   * @param begin Iterator pointing to the beginning of the state vector.
   * @param end Iterator pointing to the end of the state vector.
   * @param level The current level of recursion. Starts at the highest level of
   * the state vector (log base 2 of the vector size - 1).
   * @return A vCachedEdge representing the root node of the created DD.
   *
   * @details This function recursively breaks down the state vector into halves
   * until each half has only one element. At each level of recursion, two new
   * edges are created, one for each half of the state vector. The two resulting
   * decision diagram edges are used to create a new decision diagram node at
   * the current level, and this node is returned as the result of the current
   * recursive call. At the base case of recursion, the state vector has only
   * two elements, which are converted into terminal nodes of the decision
   * diagram.
   *
   * @note This function assumes that the state vector size is a power of two.
   */
  vCachedEdge makeStateFromVector(const CVec::const_iterator& begin,
                                  const CVec::const_iterator& end, Qubit level);

public:
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
  [[nodiscard]] vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                                    bool normalizeWeights = false);

private:
  vCachedEdge reduceGarbageRecursion(vNode* p, const std::vector<bool>& garbage,
                                     Qubit lowerbound,
                                     bool normalizeWeights = false);

  UnaryComputeTable<vNode*, vCachedEdge> conjugateVector;

  ComputeTable<vNode*, vNode*, vCachedEdge> vectorInnerProduct;
};

} // namespace dd
