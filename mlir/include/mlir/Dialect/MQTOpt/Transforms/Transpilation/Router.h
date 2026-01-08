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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <algorithm>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace mqt::ir::opt {

class NaiveRouter {
public:
  [[nodiscard]] static mlir::SmallVector<QubitIndexPair>
  route(QubitIndexPair gate, const ThinLayout& layout,
        const Architecture& arch) {
    mlir::SmallVector<QubitIndexPair, 64> swaps;
    const auto hw0 = layout.getHardwareIndex(gate.first);
    const auto hw1 = layout.getHardwareIndex(gate.second);
    return arch.shortestSWAPsBetween(hw0, hw1);
  }
};

/// @brief Specifies the weights for different terms in the cost function f.
struct HeuristicWeights {
  float alpha;
  mlir::SmallVector<float> lambdas;

  HeuristicWeights(const float alpha, const float lambda,
                   const std::size_t nlookahead)
      : alpha(alpha), lambdas(1 + nlookahead) {
    lambdas[0] = 1.;
    for (std::size_t i = 1; i < lambdas.size(); ++i) {
      lambdas[i] = lambdas[i - 1] * lambda;
    }
  }
};

class AStarHeuristicRouter {
public:
  explicit AStarHeuristicRouter(HeuristicWeights weights)
      : weights_(std::move(weights)) {}

private:
  struct Node {
    mlir::SmallVector<QubitIndexPair, 64> sequence;
    ThinLayout layout;
    float f;

    /**
     * @brief Construct a root node with the given layout. Initialize the
     * sequence with an empty vector and set the cost to zero.
     */
    explicit Node(ThinLayout layout) : layout(std::move(layout)), f(0) {}

    /**
     * @brief Construct a non-root node from its parent node. Apply the given
     * swap to the layout of the parent node and evaluate the cost.
     */
    Node(const Node& parent, QubitIndexPair swap,
         mlir::SmallVector<mlir::ArrayRef<QubitIndexPair>> window,
         const Architecture& arch, const HeuristicWeights& weights)
        : sequence(parent.sequence), layout(parent.layout), f(0) {
      /// Apply node-specific swap to given layout.
      layout.swap(layout.getProgramIndex(swap.first),
                  layout.getProgramIndex(swap.second));

      /// Add swap to sequence.
      sequence.push_back(swap);

      /// Evaluate cost function.
      f = g(weights) + h(window, arch, weights); // NOLINT
    }

    /**
     * @brief Return true if the current sequence of SWAPs makes all gates
     * executable.
     */
    [[nodiscard]] bool isGoal(mlir::ArrayRef<QubitIndexPair> layer,
                              const Architecture& arch) const {
      return std::ranges::all_of(layer, [&](const QubitIndexPair gate) {
        return arch.areAdjacent(layout.getHardwareIndex(gate.first),
                                layout.getHardwareIndex(gate.second));
      });
    }

    /**
     * @returns The depth in the search tree.
     */
    [[nodiscard]] std::size_t depth() const { return sequence.size(); }

    [[nodiscard]] bool operator>(const Node& rhs) const { return f > rhs.f; }

  private:
    /**
     * @brief Calculate the path cost for the A* search algorithm.
     *
     * The path cost function is the weighted sum of the currently required
     * SWAPs.
     */
    [[nodiscard]] float g(const HeuristicWeights& weights) const {
      return (weights.alpha * static_cast<float>(depth()));
    }

    /**
     * @brief Calculate the heuristic cost for the A* search algorithm.
     *
     * Computes the minimal number of SWAPs required to route each gate in each
     * layer. For each gate, this is determined by the shortest distance between
     * its hardware qubits. Intuitively, this is the number of SWAPs that a
     * naive router would insert to route the layers.
     */
    [[nodiscard]] float
    h(mlir::SmallVector<mlir::ArrayRef<QubitIndexPair>> window,
      const Architecture& arch, const HeuristicWeights& weights) const {
      float nn{0};
      for (const auto [i, layer] : llvm::enumerate(window)) {
        for (const auto [prog0, prog1] : layer) {
          const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
          const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
          nn += weights.lambdas[i] * static_cast<float>(nswaps);
        }
      }
      return nn;
    }
  };

  using MinQueue = std::priority_queue<Node, std::vector<Node>, std::greater<>>;

public:
  [[nodiscard]] std::optional<mlir::SmallVector<QubitIndexPair, 64>>
  route(mlir::SmallVector<mlir::ArrayRef<QubitIndexPair>> window,
        const ThinLayout& layout, const Architecture& arch) const {
    Node root(layout);

    /// Early exit. No SWAPs required:
    if (root.isGoal(window.front(), arch)) {
      return mlir::SmallVector<QubitIndexPair, 64>{};
    }

    /// Initialize queue.
    MinQueue frontier{};
    frontier.emplace(root);

    /// Iterative searching and expanding.
    while (!frontier.empty()) {
      Node curr = frontier.top();
      frontier.pop();

      if (curr.isGoal(window.front(), arch)) {
        return curr.sequence;
      }

      /// Expand frontier with all neighbouring SWAPs in the current front.
      expand(frontier, curr, window, arch);
    }

    return std::nullopt;
  }

private:
  /// @brief Expand frontier with all neighbouring SWAPs in the current front.
  void expand(MinQueue& frontier, const Node& parent,
              mlir::SmallVector<mlir::ArrayRef<QubitIndexPair>> window,
              const Architecture& arch) const {
    llvm::SmallDenseSet<QubitIndexPair, 64> expansionSet{};

    /// Currently: Don't revert last SWAP.
    /// TODO: Idea? Don't revert "front" (independent) SWAPs?
    if (!parent.sequence.empty()) {
      expansionSet.insert(parent.sequence.back());
    }

    for (const QubitIndexPair gate : window.front()) {
      for (const auto prog : {gate.first, gate.second}) {
        const auto hw0 = parent.layout.getHardwareIndex(prog);
        for (const auto hw1 : arch.neighboursOf(hw0)) {
          /// Ensure consistent hashing/comparison.
          const QubitIndexPair swap = std::minmax(hw0, hw1);
          if (!expansionSet.insert(swap).second) {
            continue;
          }

          frontier.emplace(parent, swap, window, arch, weights_);
        }
      }
    }
  }

  HeuristicWeights weights_;
};

} // namespace mqt::ir::opt
