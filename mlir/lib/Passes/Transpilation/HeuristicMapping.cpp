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

#include <cstdint>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
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

  enum Direction : std::uint8_t { Forward, Backward };

  /// @details Using SetVector ensure a deterministic order.
  struct Layer {
    SetVector<IndexGate> indices;
    SetVector<Operation*> ops;

    /// @brief Add a two-qubit gate to the layer.
    void add(const IndexGate& gate, Operation* op) {
      indices.insert(gate);
      ops.insert(op);
    }

    /// @returns true if there are no two-qubit gates in this layer.
    [[nodiscard]] bool empty() const { return ops.empty(); }
  };

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
     * @brief Collect the shortest SWAP sequence to make @p u and @p v adjacent.
     * @returns The SWAP sequence from the destination (v) to source (u) qubit.
     */
    [[nodiscard]] llvm::SmallVector<std::pair<uint32_t, uint32_t>>
    shortestSWAPsBetween(uint32_t u, uint32_t v) const {
      if (u == v) {
        return {};
      }

      if (prev_[u][v] == UINT64_MAX) {
        throw std::domain_error("No path between qubits " + std::to_string(u) +
                                " and " + std::to_string(v));
      }

      llvm::SmallVector<std::pair<uint32_t, uint32_t>> swaps;
      uint32_t last = v;
      uint32_t curr = prev_[u][v];

      while (curr != u) {
        swaps.emplace_back(last, curr); // Insert SWAP(last, curr).
        last = curr;
        curr = prev_[u][curr];
      }

      return swaps;
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
     * @brief Swap the mapping to hardware indices of two program indices.
     */
    void swap(const uint32_t prog0, const uint32_t prog1) {
      const uint32_t hw0 = programToHardware_[prog0];
      const uint32_t hw1 = programToHardware_[prog1];

      std::swap(programToHardware_[prog0], programToHardware_[prog1]);
      std::swap(hardwareToProgram_[hw0], hardwareToProgram_[hw1]);
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
    explicit AStarSearchEngine(float alpha) { params_.alpha = alpha; }

  private:
    struct Parameters {
      float alpha{};
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
      Node(const Node& parent, IndexGate swap, ArrayRef<IndexGate> layer,
           const Architecture& arch, const Parameters& params)
          : sequence(parent.sequence), layout(parent.layout), f(0) {
        /// Apply node-specific swap to given layout.
        layout.swap(layout.getProgramIndex(swap.first),
                    layout.getProgramIndex(swap.second));

        // Add swap to sequence.
        sequence.push_back(swap);

        // Evaluate cost function.
        f = g(params.alpha) + h(layer, arch); // NOLINT
      }

      /**
       * @brief Return true if the current sequence of SWAPs makes all gates
       * executable.
       */
      [[nodiscard]] bool isGoal(ArrayRef<IndexGate> layer,
                                const Architecture& arch) const {
        return llvm::all_of(layer, [&](const IndexGate gate) {
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
      [[nodiscard]] float g(float alpha) const {
        return (alpha * static_cast<float>(depth()));
      }

      /**
       * @brief Calculate the heuristic cost for the A* search algorithm.
       *
       * Computes the minimal number of SWAPs required to route each gate in
       * each layer. For each gate, this is determined by the shortest distance
       * between its hardware qubits. Intuitively, this is the number of SWAPs
       * that a naive router would insert to route the layers.
       */
      [[nodiscard]] float h(ArrayRef<IndexGate> layer,
                            const Architecture& arch) const {
        float costs{0};
        for (const auto [prog0, prog1] : layer) {
          const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
          const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
          costs += static_cast<float>(nswaps);
        }
        return costs;
      }
    };

    using MinQueue =
        std::priority_queue<Node, std::vector<Node>, std::greater<>>;

  public:
    [[nodiscard]] std::optional<SmallVector<IndexGate>>
    route(ArrayRef<IndexGate> layer, const Layout& layout,
          const Architecture& arch) const {
      Node root(layout);

      /// Early exit. No SWAPs required:
      if (root.isGoal(layer, arch)) {
        return SmallVector<IndexGate>{};
      }

      /// Initialize queue.
      MinQueue frontier{};
      frontier.emplace(root);

      /// Iterative searching and expanding.
      while (!frontier.empty()) {
        Node curr = frontier.top();
        frontier.pop();

        if (curr.isGoal(layer, arch)) {
          return curr.sequence;
        }

        /// Expand frontier with all neighbouring SWAPs in the current front.
        expand(frontier, curr, layer, arch);
      }

      return std::nullopt;
    }

  private:
    /// @brief Expand frontier with all neighbouring SWAPs in the current front.
    void expand(MinQueue& frontier, const Node& parent,
                ArrayRef<IndexGate> layer, const Architecture& arch) const {
      DenseSet<IndexGate> expansionSet{};

      if (!parent.sequence.empty()) {
        expansionSet.insert(parent.sequence.back());
      }

      for (const IndexGate& gate : layer) {
        for (const auto prog : {gate.first, gate.second}) {
          const auto hw0 = parent.layout.getHardwareIndex(prog);
          for (const auto hw1 : arch.neighboursOf(hw0)) {
            /// Ensure consistent hashing/comparison.
            const IndexGate swap = std::minmax(hw0, hw1);
            if (!expansionSet.insert(swap).second) {
              continue;
            }

            frontier.emplace(parent, swap, layer, arch, params_);
          }
        }
      }
    }

    Parameters params_;
  };

public:
  using HeuristicMappingPassBase::HeuristicMappingPassBase;

  void runOnOperation() override {
    constexpr std::size_t repeats = 10;

    // TODO: Hardcoded architecture.
    Architecture arch("RigettiNovera", 9,
                      {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                       {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                       {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                       {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});
    AStarSearchEngine engine(0.5);

    IRRewriter rewriter(&getContext());
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      rewriter.setInsertionPointToStart(&func.getFunctionBody().front());

      SmallVector<QubitValue> qubits(
          map_range(func.getFunctionBody().getOps<AllocOp>(),
                    [](AllocOp op) { return op.getResult(); }));

      //// ---- TODO: Add a nice description what happens here.

      auto wires = toWires(qubits);
      const auto fLayers = collectLayers<Forward>(wires);
      const auto bLayers = collectLayers<Backward>(wires);

      Layout layout = Layout::identity(arch.nqubits());
      for (std::size_t r = 0; r < repeats; ++r) {
        route(fLayers, layout, engine, arch);
        route(bLayers, layout, engine, arch);
      }

      layout.dump();

      qubits = place(qubits, layout, func.getFunctionBody(), rewriter);
      wires = toWires(qubits);

      //// ----

      //// ---- TODO: Add a nice description what happens here.

      route(fLayers, layout, func.getFunctionBody(), rewriter, engine, arch);

      //// ----

      sortTopologically(&func.getFunctionBody().front());
    }
  }

private:
  [[nodiscard]] static SmallVector<QubitValue>
  place(const ArrayRef<QubitValue> dynQubits, const Layout& layout,
        Region& functionBody, IRRewriter& rewriter) {
    SmallVector<QubitValue> statQubits(layout.getNumQubits());

    // 1. Replace existing dynamic allocations with mapped static ones.
    for (const auto [prog, q] : llvm::enumerate(dynQubits)) {
      rewriter.setInsertionPoint(q.getDefiningOp());

      const auto hw = layout.getHardwareIndex(prog);
      statQubits[hw] =
          rewriter.replaceOpWithNewOp<StaticOp>(q.getDefiningOp(), hw)
              .getQubit();
    }

    // 2. Create static qubits for the remaining (unused) hardware indices.
    for (std::size_t prog = dynQubits.size(); prog < layout.getNumQubits();
         ++prog) {
      rewriter.setInsertionPointToStart(&functionBody.front());

      const auto hw = layout.getHardwareIndex(prog);
      const auto q =
          rewriter.create<StaticOp>(rewriter.getUnknownLoc(), hw).getQubit();
      statQubits[hw] = q;

      rewriter.setInsertionPoint(functionBody.back().getTerminator());
      rewriter.create<DeallocOp>(rewriter.getUnknownLoc(), q);
    }

    return statQubits;
  }

  static void route(ArrayRef<Layer> layers, Layout& layout,
                    const AStarSearchEngine& engine, const Architecture& arch) {

    for (const auto& layer : layers) {
      const auto ref = layer.indices.getArrayRef();
      const auto swaps = engine.route(ref, layout, arch);
      for (const auto [hw0, hw1] : *swaps) {
        const auto [prog0, prog1] = layout.getProgramIndices(hw0, hw1);

        layout.swap(prog0, prog1);
      }
    }
  }

  static void route(ArrayRef<Layer> layers, Layout& layout,
                    Region& functionBody, IRRewriter& rewriter,
                    const AStarSearchEngine& engine, const Architecture& arch) {
    // Collect all static qubits in array s.t. qubits[i] is the i-th static
    // qubit.
    SmallVector<QubitValue> qubits(layout.getNumQubits());
    for (auto staticOp : functionBody.getOps<StaticOp>()) {
      qubits[staticOp.getIndex()] = staticOp.getQubit();
    }

    // Transform qubits to wires.
    auto wires = toWires(qubits);
    for (const auto& layer : layers) {
      llvm::dbgs() << "--- layer start ---\n";
      for (auto& it : wires) {

        while (true) {
          auto next = std::next(it);
          if (isa<DeallocOp>(next.operation())) {
            break;
          }

          if (auto op = dyn_cast<UnitaryOpInterface>(next.operation())) {
            if (op.getNumQubits() > 1) {
              break;
            }
          }

          ++it;
        }
      }
      llvm::dbgs() << "wires:\n";

      for (auto& wire : wires) {
        wire.qubit().dump();
      }

      const auto swaps =
          engine.route(layer.indices.getArrayRef(), layout, arch);
      for (const auto [hw0, hw1] : *swaps) {
        const auto [prog0, prog1] = layout.getProgramIndices(hw0, hw1);

        const auto in0 = wires[hw0].qubit();
        const auto in1 = wires[hw1].qubit();

        auto swapOp =
            rewriter.create<SWAPOp>(rewriter.getUnknownLoc(), in0, in1);
        const auto out0 = swapOp.getQubit0Out();
        const auto out1 = swapOp.getQubit1Out();

        rewriter.replaceAllUsesExcept(in0, out1, swapOp);
        rewriter.replaceAllUsesExcept(in1, out0, swapOp);

        ++wires[hw0];
        ++wires[hw1];

        layout.swap(prog0, prog1);
      }

      for (const auto [prog0, prog1] : layer.indices) {
        ++wires[layout.getHardwareIndex(prog0)];
        ++wires[layout.getHardwareIndex(prog1)];
      }

      llvm::dbgs() << "--- layer end ---\n";
    }
  }

  template <typename QubitRange>
  static SmallVector<WireIterator> toWires(QubitRange qubits) {
    return SmallVector<WireIterator>(
        map_range(qubits, [](auto q) { return WireIterator(q); }));
  }

  template <Direction d> Layer advance(MutableArrayRef<WireIterator> wires) {
    constexpr std::size_t step = d == Forward ? 1 : -1;
    const auto stop = [](const WireIterator& it) {
      if constexpr (d == Forward) {
        return it != std::default_sentinel;
      } else {
        return !isa<AllocOp>(it.operation());
      }
    };

    Layer layer{};
    DenseMap<UnitaryOpInterface, std::size_t> hits;
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
                    layer.add(std::make_pair(index, otherIndex), op);
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

    return layer;
  }

  template <Direction d>
  SmallVector<Layer> collectLayers(MutableArrayRef<WireIterator> wires) {
    constexpr std::size_t step = d == Forward ? 1 : -1;

    SmallVector<Layer> layers;
    while (true) {
      const auto layer = advance<d>(wires);
      if (layer.empty()) {
        break;
      }

      layers.emplace_back(layer);

      for (const auto [i1, i2] : layer.indices) {
        std::ranges::advance(wires[i1], step);
        std::ranges::advance(wires[i2], step);
      }
    }

    return layers;
  }
};
} // namespace mlir::qco
