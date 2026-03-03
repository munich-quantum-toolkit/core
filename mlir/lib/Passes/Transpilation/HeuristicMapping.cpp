/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Passes/Passes.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "heuristic-mapping-pass"

namespace mlir::qco {

#define GEN_PASS_DEF_HEURISTICMAPPINGPASS
#include "mlir/Passes/Passes.h.inc"

struct HeuristicMappingPass
    : impl::HeuristicMappingPassBase<HeuristicMappingPass> {
private:
  using QubitValue = TypedValue<QubitType>;
  using IndexGate = std::pair<std::size_t, std::size_t>;
  using IndexGateSet = DenseSet<IndexGate>;
  using Layer = SetVector<IndexGate>;

  enum class Direction : std::uint8_t { Forward, Backward };

  /**
   * @brief A quantum accelerator's architecture.
   */
  class [[nodiscard]] Architecture {
  public:
    using CouplingSet = DenseSet<std::pair<uint32_t, uint32_t>>;
    using NeighbourVector = SmallVector<SmallVector<uint32_t, 4>>;

    explicit Architecture(std::string name, std::size_t nqubits,
                          CouplingSet couplingSet)
        : name_(std::move(name)), nqubits_(nqubits),
          couplingSet_(std::move(couplingSet)), neighbours_(nqubits),
          dist_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)),
          prev_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)) {
      floydWarshallWithPathReconstruction();
      collectNeighbours();
    }

    /**
     * @brief Return the architecture's name.
     */
    [[nodiscard]] constexpr std::string_view name() const { return name_; }

    /**
     * @brief Return the architecture's number of qubits.
     */
    [[nodiscard]] constexpr std::size_t nqubits() const { return nqubits_; }

    /**
     * @brief Return true if @p u and @p v are adjacent.
     */
    [[nodiscard]] bool areAdjacent(uint32_t u, uint32_t v) const {
      return couplingSet_.contains({u, v});
    }

    /**
     * @brief Return the length of the shortest path between @p u and @p v.
     */
    [[nodiscard]] std::size_t distanceBetween(uint32_t u, uint32_t v) const {
      if (dist_[u][v] == UINT64_MAX) {
        throw std::domain_error("No path between qubits " + std::to_string(u) +
                                " and " + std::to_string(v));
      }
      return dist_[u][v];
    }

    /**
     * @brief Collect all neighbours of @p u.
     */
    [[nodiscard]] SmallVector<uint32_t, 4> neighboursOf(uint32_t u) const {
      return neighbours_[u];
    }

  private:
    using Matrix = SmallVector<SmallVector<std::size_t>>;

    /**
     * @brief Find all shortest paths in the coupling map between two qubits.
     * @details Vertices are the qubits. Edges connected two qubits. Has a time
     * and memory complexity of O(nqubits^3) and O(nqubits^2), respectively.
     * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
     */
    void floydWarshallWithPathReconstruction() {
      for (const auto& [u, v] : couplingSet_) {
        dist_[u][v] = 1;
        prev_[u][v] = u;
      }
      for (std::size_t v = 0; v < nqubits(); ++v) {
        dist_[v][v] = 0;
        prev_[v][v] = v;
      }

      for (std::size_t k = 0; k < nqubits(); ++k) {
        for (std::size_t i = 0; i < nqubits(); ++i) {
          for (std::size_t j = 0; j < nqubits(); ++j) {
            if (dist_[i][k] == UINT64_MAX || dist_[k][j] == UINT64_MAX) {
              continue; // avoid overflow with "infinite" distances
            }
            const std::size_t sum = dist_[i][k] + dist_[k][j];
            if (dist_[i][j] > sum) {
              dist_[i][j] = sum;
              prev_[i][j] = prev_[k][j];
            }
          }
        }
      }
    }

    /**
     * @brief Collect the neighbours of all qubits.
     * @details Has a time complexity of O(nqubits)
     */
    void collectNeighbours() {
      for (const auto& [u, v] : couplingSet_) {
        neighbours_[u].push_back(v);
      }
    }

    std::string name_;
    std::size_t nqubits_;
    CouplingSet couplingSet_;
    NeighbourVector neighbours_;

    Matrix dist_;
    Matrix prev_;
  };

  /**
   * @brief A qubit layout that maps program and hardware indices without
   * storing Values. Used for efficient memory usage when Value tracking isn't
   * needed.
   *
   * Note that we use the terminology "hardware" and "program" qubits here,
   * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
   * are C++ keywords.
   */
  class [[nodiscard]] Layout {
  public:
    /**
     * @brief Constructs the identity (i->i) layout.
     * @param nqubits The number of qubits.
     * @return The identity layout.
     */
    static Layout identity(const std::size_t nqubits) {
      Layout layout(nqubits);
      for (std::size_t i = 0; i < nqubits; ++i) {
        layout.add(i, i);
      }
      return layout;
    }

    explicit Layout(const std::size_t nqubits)
        : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}

    /**
     * @brief Insert program:hardware index mapping.
     * @param prog The program index.
     * @param hw The hardware index.
     */
    void add(uint32_t prog, uint32_t hw) {
      assert(prog < programToHardware_.size() &&
             "add: program index out of bounds");
      assert(hw < hardwareToProgram_.size() &&
             "add: hardware index out of bounds");
      programToHardware_[prog] = hw;
      hardwareToProgram_[hw] = prog;
    }

    /**
     * @brief Look up program index for a hardware index.
     * @param hw The hardware index.
     * @return The program index of the respective hardware index.
     */
    [[nodiscard]] uint32_t getProgramIndex(const uint32_t hw) const {
      assert(hw < hardwareToProgram_.size() &&
             "getProgramIndex: hardware index out of bounds");
      return hardwareToProgram_[hw];
    }

    /**
     * @brief Look up hardware index for a program index.
     * @param prog The program index.
     * @return The hardware index of the respective program index.
     */
    [[nodiscard]] uint32_t getHardwareIndex(const uint32_t prog) const {
      assert(prog < programToHardware_.size() &&
             "getHardwareIndex: program index out of bounds");
      return programToHardware_[prog];
    }

    /**
     * @brief Convenience function to lookup multiple hardware indices at once.
     * @param progs The program indices.
     * @return A tuple of hardware indices.
     */
    template <typename... ProgIndices>
      requires(sizeof...(ProgIndices) > 0) &&
              ((std::is_convertible_v<ProgIndices, uint32_t>) && ...)
    [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
      return std::tuple{getHardwareIndex(static_cast<uint32_t>(progs))...};
    }

    /**
     * @brief Convenience function to lookup multiple program indices at once.
     * @param hws The hardware indices.
     * @return A tuple of program indices.
     */
    template <typename... HwIndices>
      requires(sizeof...(HwIndices) > 0) &&
              ((std::is_convertible_v<HwIndices, uint32_t>) && ...)
    [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
      return std::tuple{getProgramIndex(static_cast<uint32_t>(hws))...};
    }

    /**
     * @brief Swap the mapping to program indices of two hardware indices.
     */
    void swap(const uint32_t hw0, const uint32_t hw1) {
      const uint32_t prog0 = hardwareToProgram_[hw0];
      const uint32_t prog1 = hardwareToProgram_[hw1];

      std::swap(hardwareToProgram_[hw0], hardwareToProgram_[hw1]);
      std::swap(programToHardware_[prog0], programToHardware_[prog1]);
    }

    /**
     * @returns the number of qubits handled by the layout.
     */
    [[nodiscard]] std::size_t getNumQubits() const {
      return programToHardware_.size();
    }

    void dump() {
      llvm::dbgs() << "prog= ";
      for (std::size_t i = 0; i < getNumQubits(); ++i) {
        llvm::dbgs() << i << " ";
      }
      llvm::dbgs() << "\nhw=   ";
      for (std::size_t i = 0; i < getNumQubits(); ++i) {
        llvm::dbgs() << programToHardware_[i] << ' ';
      }
      llvm::dbgs() << '\n';
    }

  protected:
    /**
     * @brief Maps a program qubit index to its hardware index.
     */
    SmallVector<uint32_t> programToHardware_;

    /**
     * @brief Maps a hardware qubit index to its program index.
     */
    SmallVector<uint32_t> hardwareToProgram_;
  };

  class [[nodiscard]] AStarSearchEngine {
  public:
    explicit AStarSearchEngine(const float alpha, const float lambda,
                               const std::size_t nlookahead, Architecture& arch)
        : w_(alpha, lambda, nlookahead), arch_(&arch) {}

  private:
    struct Weights {
      Weights(const float alpha, const float lambda,
              const std::size_t nlookahead)
          : alpha(alpha), decay(1 + nlookahead) {
        decay[0] = 1.;
        for (std::size_t i = 1; i < decay.size(); ++i) {
          decay[i] = decay[i - 1] * lambda;
        }
      }

      float alpha;
      SmallVector<float> decay;
    };

    struct Node {
      SmallVector<IndexGate> sequence;
      Layout layout;
      float f;

      /**
       * @brief Construct a root node with the given layout. Initialize the
       * sequence with an empty vector and set the cost to zero.
       */
      explicit Node(Layout layout) : layout(std::move(layout)), f(0) {}

      /**
       * @brief Construct a non-root node from its parent node. Apply the given
       * swap to the layout of the parent node and evaluate the cost.
       */
      Node(const Node& parent, IndexGate swap, ArrayRef<Layer> layers,
           const Architecture& arch, const Weights& w)
          : sequence(parent.sequence), layout(parent.layout), f(0) {
        /// Apply node-specific swap to given layout.
        layout.swap(swap.first, swap.second);

        // Add swap to sequence.
        sequence.push_back(swap);

        // Evaluate cost function.
        f = g(w.alpha) + h(layers, arch, w); // NOLINT
      }

      /**
       * @returns true if the current sequence of SWAPs makes all gates
       * executable.
       */
      [[nodiscard]] bool isGoal(const Layer& layer,
                                const Architecture& arch) const {
        return llvm::all_of(layer, [&](const IndexGate gate) {
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
      [[nodiscard]] float g(float alpha) const {
        return alpha * static_cast<float>(sequence.size());
      }

      /**
       * @brief Calculate the heuristic cost for the A* search algorithm.
       *
       * Computes the minimal number of SWAPs required to route each gate in
       * each layer. For each gate, this is determined by the shortest distance
       * between its hardware qubits. Intuitively, this is the number of SWAPs
       * that a naive router would insert to route the layers.
       */
      [[nodiscard]] float h(ArrayRef<Layer> layers, const Architecture& arch,
                            const Weights& w) const {
        float costs{0};
        for (const auto [i, layer] : llvm::enumerate(layers)) {
          for (const auto [prog0, prog1] : layer) {
            const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
            const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
            costs += w.decay[i] * static_cast<float>(nswaps);
          }
        }
        return costs;
      }
    };

    using MinQueue =
        std::priority_queue<Node, std::vector<Node>, std::greater<>>;

  public:
    [[nodiscard]] llvm::FailureOr<SmallVector<IndexGate>>
    search(ArrayRef<Layer> layers, const Layout& layout) {
      Node root(layout);

      // Early exit. No SWAPs required:
      if (root.isGoal(layers.front(), *arch_)) {
        return SmallVector<IndexGate>{};
      }

      // Initialize queue.
      MinQueue frontier{};
      frontier.emplace(root);

      // Iterative searching and expanding.
      while (!frontier.empty()) {
        Node curr = frontier.top();
        frontier.pop();

        if (curr.isGoal(layers.front(), *arch_)) {
          return curr.sequence;
        }

        expand(frontier, curr, layers);
      }

      return llvm::failure();
    }

  private:
    void expand(MinQueue& frontier, const Node& node, ArrayRef<Layer> layers) {
      expansionSet_.clear();

      if (!node.sequence.empty()) {
        expansionSet_.insert(node.sequence.back());
      }

      for (const IndexGate& gate : layers.front()) {
        for (const auto prog : {gate.first, gate.second}) {
          const auto hw0 = node.layout.getHardwareIndex(prog);
          for (const auto hw1 : arch_->neighboursOf(hw0)) {
            /// Ensure consistent hashing/comparison.
            const IndexGate swap = std::minmax(hw0, hw1);
            if (!expansionSet_.insert(swap).second) {
              continue;
            }

            frontier.emplace(node, swap, layers, *arch_, w_);
          }
        }
      }
    }
    Weights w_;
    Architecture* arch_;
    DenseSet<IndexGate> expansionSet_;
  };

public:
  using HeuristicMappingPassBase::HeuristicMappingPassBase;

  void runOnOperation() override {
    constexpr std::size_t repeats = 10;
    constexpr std::size_t nlookahead = 3;

    // TODO: Hardcoded architecture.
    Architecture arch("RigettiNovera", 9,
                      {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                       {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                       {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                       {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});

    IRRewriter rewriter(&getContext());

    AStarSearchEngine engine(0.5, 0.8, nlookahead, arch);
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      const auto dyn = collectDynamicQubits(func.getFunctionBody());
      const auto [ltr, rtl] = computeBidirectionalLayers(dyn);

      // Use the SABRE Approach to improve the initial layout choice (here:
      // identity): Traverse the layers from left-to-right-to-left and
      // cold-route along the way. Repeat this procedure "repeats" times.

      Layout layout = Layout::identity(arch.nqubits());
      for (std::size_t r = 0; r < repeats; ++r) {
        if (failed(routeCold(ltr, layout, nlookahead, engine))) {
          signalPassFailure();
          return;
        }
        if (failed(routeCold(ltr, layout, nlookahead, engine))) {
          signalPassFailure();
          return;
        }
      }
      layout.dump();

      // Once the initial layout is found, replace the dynamic with static
      // qubits ("placement") and hot-route the circuit layer-by-layer.

      const auto stat = place(dyn, layout, func.getFunctionBody(), rewriter);
      if (failed(routeHot(ltr, layout, nlookahead, stat, engine, rewriter))) {
        signalPassFailure();
        return;
      };
      sortTopologically(&func.getFunctionBody().front());
    }
  }

private:
  /**
   * @brief Collect dynamic qubits contained in the given function body.
   * @returns a vector of SSA values produced by qco.alloc operations.
   */
  [[nodiscard]] static SmallVector<QubitValue>
  collectDynamicQubits(Region& funcBody) {
    return SmallVector<QubitValue>(map_range(
        funcBody.getOps<AllocOp>(), [](AllocOp op) { return op.getResult(); }));
  }

  /**
   * @brief Computes forwards and backwards layers.
   * @returns a pair of vectors of layers, where [0]=forward and [1]=backward.
   */
  [[nodiscard]] static std::pair<SmallVector<Layer>, SmallVector<Layer>>
  computeBidirectionalLayers(ArrayRef<QubitValue> dyn) {
    auto wires = toWires(dyn);
    const auto ltr = collectLayers<Direction::Forward>(wires);
    const auto rtl = collectLayers<Direction::Backward>(wires);
    return std::make_pair(ltr, rtl);
  }

  [[nodiscard]] static SmallVector<QubitValue>
  place(const ArrayRef<QubitValue> dynQubits, const Layout& layout,
        Region& funcBody, IRRewriter& rewriter) {
    SmallVector<QubitValue> statics(layout.getNumQubits());

    // 1. Replace existing dynamic allocations with mapped static ones.
    for (const auto [p, q] : llvm::enumerate(dynQubits)) {
      const auto hw = layout.getHardwareIndex(p);
      rewriter.setInsertionPoint(q.getDefiningOp());
      auto op = rewriter.replaceOpWithNewOp<StaticOp>(q.getDefiningOp(), hw);
      statics[hw] = op.getQubit();
    }

    // 2. Create static qubits for the remaining (unused) hardware indices.
    for (std::size_t p = dynQubits.size(); p < layout.getNumQubits(); ++p) {
      rewriter.setInsertionPointToStart(&funcBody.front());
      const auto hw = layout.getHardwareIndex(p);
      auto op = rewriter.create<StaticOp>(rewriter.getUnknownLoc(), hw);
      rewriter.setInsertionPoint(funcBody.back().getTerminator());
      rewriter.create<DeallocOp>(rewriter.getUnknownLoc(), op.getQubit());
      statics[hw] = op.getQubit();
    }

    return statics;
  }

  /**
   * @brief "Cold" routing of the given layers.
   * @details Iterates over a sliding window of layers and uses the A* search
   * engine to find a sequence of SWAPs that makes that layer executable.
   * Instead of inserted these SWAPs into the IR, this function only updates the
   * layout.
   * @returns llvm::failure() if A* search isn't able to find a solution.
   */
  static llvm::LogicalResult routeCold(ArrayRef<Layer> layers, Layout& layout,
                                       const std::size_t nlookahead,
                                       AStarSearchEngine& engine) {
    for (std::size_t i = 0; i < layers.size(); ++i) {
      const std::size_t len = std::min(1 + nlookahead, layers.size() - i);
      const auto window = layers.slice(i, len);
      const auto swaps = engine.search(window, layout);
      if (failed(swaps)) {
        return llvm::failure();
      }

      for (const auto [hw0, hw1] : *swaps) {
        layout.swap(hw0, hw1);
      }
    }

    return llvm::success();
  }

  /**
   * @brief "Hot" routing of the given layers.
   * @details Iterates over a sliding window of layers and uses the A* search
   * engine to find a sequence of SWAPs that makes that layer executable.
   * This function inserts SWAP ops.
   * @returns llvm::failure() if A* search isn't able to find a solution.
   */
  static LogicalResult routeHot(ArrayRef<Layer> ltr, Layout& layout,
                                const std::size_t nlookahead,
                                ArrayRef<QubitValue> statics,
                                AStarSearchEngine& engine,
                                IRRewriter& rewriter) {
    constexpr auto advanceFront = [](WireIterator& it) {
      while (true) {
        const auto next = std::next(it);
        if (isa<DeallocOp>(next.operation())) {
          break;
        }

        auto op = dyn_cast<UnitaryOpInterface>(next.operation());
        if (op && op.getNumQubits() > 1) {
          break;
        }

        std::ranges::advance(it, 1);
      }
    };

    auto wires = toWires(statics);
    for (const auto [i, layer] : llvm::enumerate(ltr)) {
      llvm::for_each(wires, advanceFront);

      const auto len = std::min(1 + nlookahead, ltr.size() - i);
      const auto window = ltr.slice(i, len);
      const auto swaps = engine.search(window, layout);
      if (failed(swaps)) {
        return llvm::failure();
      }

      const auto unknown = rewriter.getUnknownLoc();
      for (const auto [hw0, hw1] : *swaps) {
        const auto in0 = wires[hw0].qubit();
        const auto in1 = wires[hw1].qubit();

        auto op = rewriter.create<SWAPOp>(unknown, in0, in1);
        const auto out0 = op.getQubit0Out();
        const auto out1 = op.getQubit1Out();

        rewriter.replaceAllUsesExcept(in0, out1, op);
        rewriter.replaceAllUsesExcept(in1, out0, op);

        // Jump over the SWAPOp.
        std::ranges::advance(wires[hw0], 1);
        std::ranges::advance(wires[hw1], 1);

        layout.swap(hw0, hw1);
      }

      // Jump over two-qubit gates contained in the layer.
      for (const auto [prog0, prog1] : layer) {
        std::ranges::advance(wires[layout.getHardwareIndex(prog0)], 1);
        std::ranges::advance(wires[layout.getHardwareIndex(prog1)], 1);
      }
    }

    return llvm::success();
  }

  /**
   * @brief Transform a range of qubit values to a vector of wire iterators.
   * @returns a vector of wire iterators.
   */
  template <typename QubitRange>
  static SmallVector<WireIterator> toWires(QubitRange qubits) {
    return SmallVector<WireIterator>(
        map_range(qubits, [](auto q) { return WireIterator(q); }));
  }

  template <Direction d>
  static SmallVector<Layer> collectLayers(MutableArrayRef<WireIterator> wires) {
    constexpr std::size_t step = d == Direction::Forward ? 1 : -1;
    const auto stop = [](const WireIterator& it) {
      if constexpr (d == Direction::Forward) {
        return it != std::default_sentinel;
      } else {
        return !isa<AllocOp>(it.operation());
      }
    };

    SmallVector<Layer> layers;
    DenseMap<UnitaryOpInterface, std::size_t> hits;

    while (true) {
      Layer layer{};
      for (const auto [index, it] : llvm::enumerate(wires)) {
        while (stop(it)) {
          const auto res =
              TypeSwitch<Operation*, WalkResult>(it.operation())
                  .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                    assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                    if (op.getNumQubits() == 1) {
                      std::ranges::advance(it, step);
                      return WalkResult::advance();
                    }

                    if (hits.contains(op)) {
                      const auto otherIndex = hits[op];
                      layer.insert(std::make_pair(index, otherIndex));

                      std::ranges::advance(wires[index], step);
                      std::ranges::advance(wires[otherIndex], step);

                      hits.erase(op);
                    } else {
                      hits.try_emplace(op, index);
                    }

                    return WalkResult::interrupt();
                  })
                  .template Case<AllocOp, StaticOp, ResetOp, MeasureOp,
                                 DeallocOp>([&](auto) {
                    std::ranges::advance(it, step);
                    return WalkResult::advance();
                  })
                  .Default([&](Operation* op) {
                    report_fatal_error("unknown op in wireiterator use: " +
                                       op->getName().getStringRef());
                    return WalkResult::interrupt();
                  });

          if (res.wasInterrupted()) {
            break;
          }
        }
      }

      if (layer.empty()) {
        break;
      }

      layers.emplace_back(layer);
      hits.clear();
    }

    return layers;
  }
};
} // namespace mlir::qco
