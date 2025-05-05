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

namespace dd {

class MatrixDDContainer : public DDContainer<mNode> {
public:
  struct Config : public DDContainer<mNode>::Config {
    std::size_t ctMatConjTransposeNumBucket =
        UnaryComputeTable<mNode*, mCachedEdge>::DEFAULT_NUM_BUCKETS;
  };

  MatrixDDContainer(std::size_t nqubits, RealNumberUniqueTable& cUt,
                    ComplexNumbers& cn, const Config& config)
      : DDContainer<mNode>(nqubits, cUt, cn, config),
        conjugateMatrixTranspose(config.ctMatConjTransposeNumBucket) {}

  inline void reset() {
    DDContainer<mNode>::reset();
    conjugateMatrixTranspose.clear();
  }

  inline bool garbageCollect(bool force) {
    const bool collect = DDContainer<mNode>::garbageCollect(force);
    if (collect) {
      conjugateMatrixTranspose.clear();
    }
    return collect;
  }

  /**
   * @brief Construct the DD for a single-qubit gate
   * @param mat The matrix representation of the gate
   * @param target The target qubit
   * @return A decision diagram for the gate
   */
  [[nodiscard]] mEdge makeGateDD(const GateMatrix& mat, qc::Qubit target);

  /**
   * @brief Construct the DD for a single-qubit controlled gate
   * @param mat The matrix representation of the gate
   * @param control The control qubit
   * @param target The target qubit
   * @return A decision diagram for the gate
   */
  [[nodiscard]] mEdge makeGateDD(const GateMatrix& mat,
                                 const qc::Control& control, qc::Qubit target);

  /**
   * @brief Construct the DD for a multi-controlled single-qubit gate
   * @param mat The matrix representation of the gate
   * @param controls The control qubits
   * @param target The target qubit
   * @return A decision diagram for the gate
   */
  [[nodiscard]] mEdge makeGateDD(const GateMatrix& mat,
                                 const qc::Controls& controls,
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
  [[nodiscard]] mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                         qc::Qubit target0, qc::Qubit target1);

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
  [[nodiscard]] mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                         const qc::Control& control,
                                         qc::Qubit target0, qc::Qubit target1);

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
  [[nodiscard]] mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                                         const qc::Controls& controls,
                                         qc::Qubit target0, qc::Qubit target1);

  /**
   * @brief Converts a given matrix to a decision diagram
   * @param matrix A complex matrix to convert to a DD.
   * @return A decision diagram representing the matrix.
   * @throws std::invalid_argument If the given matrix is not square or its
   * length is not a power of two.
   */
  [[nodiscard]] mEdge makeDDFromMatrix(const CMat& matrix);

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
  [[nodiscard]] mCachedEdge makeDDFromMatrix(const CMat& matrix, Qubit level,
                                             std::size_t rowStart,
                                             std::size_t rowEnd,
                                             std::size_t colStart,
                                             std::size_t colEnd);

public:
  /**
   * @brief Computes the conjugate transpose of a given matrix edge.
   *
   * @param a The matrix edge to conjugate transpose.
   * @return The conjugated transposed matrix edge.
   */
  [[nodiscard]] mEdge conjugateTranspose(const mEdge& a);

  /**
   * @brief Recursively computes the conjugate transpose of a given matrix edge.
   *
   * @param a The matrix edge to conjugate transpose.
   * @return The conjugated transposed matrix edge.
   */
  [[nodiscard]] mCachedEdge conjugateTransposeRec(const mEdge& a);

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

  /// Create identity DD represented by the one-terminal.
  [[nodiscard]] static mEdge makeIdent();

  [[nodiscard]] mEdge createInitialMatrix(const std::vector<bool>& ancillary);

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
  [[nodiscard]] mEdge reduceGarbage(const mEdge& e,
                                    const std::vector<bool>& garbage,
                                    bool regular = true,
                                    bool normalizeWeights = false);

private:
  [[nodiscard]] mCachedEdge
  reduceAncillaeRecursion(mNode* p, const std::vector<bool>& ancillary,
                          Qubit lowerbound, bool regular = true);

  [[nodiscard]] mCachedEdge
  reduceGarbageRecursion(mNode* p, const std::vector<bool>& garbage,
                         Qubit lowerbound, bool regular = true,
                         bool normalizeWeights = false);

  UnaryComputeTable<mNode*, mCachedEdge> conjugateMatrixTranspose;
};

} // namespace dd
