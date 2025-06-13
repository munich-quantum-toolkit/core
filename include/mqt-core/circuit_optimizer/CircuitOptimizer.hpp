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

#include "ir/QuantumComputation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>

namespace qc {

class CircuitOptimizer {
public:
  CircuitOptimizer() = default;

  using DAG = std::vector<std::deque<std::unique_ptr<Operation>*>>;
  using DAGIterator = std::deque<std::unique_ptr<Operation>*>::iterator;
  using DAGReverseIterator =
      std::deque<std::unique_ptr<Operation>*>::reverse_iterator;
  using DAGIterators = std::vector<DAGIterator>;
  using DAGReverseIterators = std::vector<DAGReverseIterator>;

  static DAG constructDAG(QuantumComputation& qc);

  static void swapReconstruction(QuantumComputation& qc);

  static void singleQubitGateFusion(QuantumComputation& qc);

  static void removeIdentities(QuantumComputation& qc);

  static void removeOperation(QuantumComputation& qc,
                              const std::unordered_set<OpType>& opTypes,
                              size_t opSize);

  static void removeDiagonalGatesBeforeMeasure(QuantumComputation& qc);

  static void removeFinalMeasurements(QuantumComputation& qc);

  static void decomposeSWAP(QuantumComputation& qc,
                            bool isDirectedArchitecture);

  static void eliminateResets(QuantumComputation& qc);

  static void deferMeasurements(QuantumComputation& qc);

  static void flattenOperations(QuantumComputation& qc,
                                bool customGatesOnly = false);

  static void cancelCNOTs(QuantumComputation& qc);

  /**
   * @brief Replaces all MCX gates with MCZ gates (and H gates surrounding the
   * target qubit) in the given circuit.
   * @param qc the quantum circuit
   */
  static void replaceMCXWithMCZ(QuantumComputation& qc);

  /**
   * @brief Backpropagates the output permutation through the circuit.
   * @details Starts at the end of the circuit with a potentially incomplete
   * output permutation and backpropagates it through the circuit until the
   * beginning of the circuit is reached. The tracked permutation is updated
   * with every SWAP gate and, eventually, the initial layout of the circuit is
   * set to the tracked permutation. Any unassigned qubit in the initial layout
   * is assigned to the first available position (favoring an identity mapping).
   * @param qc the quantum circuit
   */
  static void backpropagateOutputPermutation(QuantumComputation& qc);

  /**
   * @brief Collects all operations in the circuit into blocks of a maximum
   * size.
   * @details The circuit is traversed and operations are collected into blocks
   * of a maximum size. The blocks are then appended to the circuit in the order
   * they were collected. Each block is realized as a compound operation.
   * Light optimizations are applied to the blocks, such as removing identity
   * gates and fusing single-qubit gates.
   * @param qc the quantum circuit
   * @param maxBlockSize the maximum size of a block
   * @param onlyCollectCliffords whether to only group Clifford operations
   */
  static void collectBlocks(QuantumComputation& qc, std::size_t maxBlockSize,
                            bool onlyCollectCliffords = false);

  /**
   * @brief Elide permutations by propagating them through the circuit.
   * @details The circuit is traversed and any SWAP gate is eliminated by
   * propagating the permutation through the circuit. The final layout of the
   * circuit is updated accordingly. This pass works well together with the
   * `swapReconstruction` pass.
   * @param qc the quantum circuit
   */
  static void elidePermutations(QuantumComputation& qc);
};
} // namespace qc
