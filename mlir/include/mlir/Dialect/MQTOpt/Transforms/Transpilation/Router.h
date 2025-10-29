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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Scheduler.h"

#include <algorithm>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt::ir::opt {

/**
 * @brief A vector of SWAPs.
 */
using RouterResult = mlir::SmallVector<QubitIndexPair>;

/**
 * @brief A planner determines the sequence of swaps required to route an array
of gates.
*/
struct RouterBase {
  virtual ~RouterBase() = default;
  [[nodiscard]] virtual RouterResult route(const Layers&, const ThinLayout&,
                                           const Architecture&) const = 0;
};

/**
 * @brief Use shortest path swapping to make one gate executable.
 */
struct NaiveRouter final : RouterBase {
  [[nodiscard]] RouterResult route(const Layers& layers,
                                   const ThinLayout& layout,
                                   const Architecture& arch) const override {
    if (layers.size() != 1 || layers.front().size() != 1) {
      throw std::invalid_argument(
          "NaiveRouter expects exactly one layer with one gate");
    }

    /// This assumes an avg. of 16 SWAPs per gate.
    mlir::SmallVector<QubitIndexPair, 16> swaps;
    for (const auto [prog0, prog1] : layers.front()) {
      const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
      const auto path = arch.shortestPathBetween(hw0, hw1);
      for (std::size_t i = 0; i < path.size() - 2; ++i) {
        swaps.emplace_back(path[i], path[i + 1]);
      }
    }
    return swaps;
  }
};

/**
 * @brief Specifies the weights for different terms in the cost function f.
 */
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

/**
 * @brief Use A*-search to make all gates executable.
 */
struct AStarHeuristicRouter final : RouterBase {
  explicit AStarHeuristicRouter(HeuristicWeights weights)
      : weights_(std::move(weights)) {}

private:
  using ClosedSet = mlir::DenseSet<ThinLayout>;

  struct Node {
    mlir::SmallVector<QubitIndexPair> sequence;
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
    Node(const Node& parent, QubitIndexPair swap, const Layers& layers,
         const Architecture& arch, const HeuristicWeights& weights)
        : sequence(parent.sequence), layout(parent.layout), f(0) {
      /// Apply node-specific swap to given layout.
      layout.swap(layout.getProgramIndex(swap.first),
                  layout.getProgramIndex(swap.second));

      /// Add swap to sequence.
      sequence.push_back(swap);

      /// Evaluate cost function.
      f = g(weights) + h(layers, arch, weights); // NOLINT
    }

    /**
     * @brief Return true if the current sequence of SWAPs makes all gates
     * executable.
     */
    [[nodiscard]] bool isGoal(const mlir::ArrayRef<QubitIndexPair>& gates,
                              const Architecture& arch) const {
      return std::ranges::all_of(gates, [&](const QubitIndexPair gate) {
        return arch.areAdjacent(layout.getHardwareIndex(gate.first),
                                layout.getHardwareIndex(gate.second));
      });
    }

    [[nodiscard]] bool operator>(const Node& rhs) const { return f > rhs.f; }

  private:
    /**
     * @brief Calculate the path cost for the A* search algorithm.
     *
     * The path cost function is the weighted sum of the currently required
     * SWAPs.
     */
    [[nodiscard]] float g(const HeuristicWeights& weights) {
      return (weights.alpha * static_cast<float>(sequence.size()));
    }

    /**
     * @brief Calculate the heuristic cost for the A* search algorithm.
     *
     * Computes the minimal number of SWAPs required to route each gate in each
     * layer. For each gate, this is determined by the shortest distance between
     * its hardware qubits. Intuitively, this is the number of SWAPs that a
     * naive router would insert to route the layers.
     */
    [[nodiscard]] float h(const Layers& layers, const Architecture& arch,
                          const HeuristicWeights& weights) const {
      float nn{0};
      for (const auto [i, layer] : llvm::enumerate(layers)) {
        for (const auto [prog0, prog1] : layer) {
          const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
          const std::size_t dist = arch.distanceBetween(hw0, hw1);
          const std::size_t nswaps = dist < 2 ? 0 : dist - 2;
          nn += weights.lambdas[i] * static_cast<float>(nswaps);
        }
      }
      return nn;
    }
  };

  using MinQueue = std::priority_queue<Node, std::vector<Node>, std::greater<>>;

public:
  [[nodiscard]] RouterResult route(const Layers& layers,
                                   const ThinLayout& layout,
                                   const Architecture& arch) const override {
    Node root(layout);

    /// Early exit. No SWAPs required:
    if (root.isGoal(layers.front(), arch)) {
      return {};
    }

    /// Initialize queue.
    MinQueue frontier{};
    frontier.emplace(root);

    /// Iterative searching and expanding.
    while (!frontier.empty()) {
      Node curr = frontier.top();
      frontier.pop();

      if (curr.isGoal(layers.front(), arch)) {
        return curr.sequence;
      }

      /// Expand frontier with all neighbouring SWAPs in the current front.
      expand(frontier, curr, layers, arch);
    }

    return {};
  }

private:
  /**
   * @brief Expand frontier with all neighbouring SWAPs in the current front.
   * @returns SWAP sequence if a goal node is expanded. Otherwise: std::nullopt.
   */
  void expand(MinQueue& frontier, const Node& parent, const Layers& layers,
              const Architecture& arch) const {
    llvm::SmallDenseSet<QubitIndexPair, 64> swaps{};
    for (const QubitIndexPair gate : layers.front()) {
      for (const auto prog : {gate.first, gate.second}) {
        const auto hw0 = parent.layout.getHardwareIndex(prog);
        for (const auto hw1 : arch.neighboursOf(hw0)) {
          /// Ensure consistent hashing/comparison.
          const QubitIndexPair swap = std::minmax(hw0, hw1);
          if (!swaps.insert(swap).second) {
            continue;
          }

          frontier.emplace(parent, swap, layers, arch, weights_);
        }
      }
    }
  }

  HeuristicWeights weights_;
};
} // namespace mqt::ir::opt
