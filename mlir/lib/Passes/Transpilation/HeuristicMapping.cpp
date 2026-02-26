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
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
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

  struct Layer {
    /// @brief Set of two-qubit gates.
    DenseSet<std::pair<std::size_t, std::size_t>> gates;

#ifndef NDEBUG
    LLVM_DUMP_METHOD void dump(llvm::raw_ostream& os = llvm::dbgs()) const {
      os << "gates= ";
      for (const auto [i1, i2] : gates) {
        os << "(" << i1 << ", " << i2 << ") ";
      }
      os << "\n";
    }
#endif
  };

  enum Direction : std::uint8_t { Forward, Backward };

  /**
   * @brief A quantum accelerator's architecture.
   * @details Computes all-shortest paths at construction.
   */
  class Architecture {
  public:
    using CouplingSet = mlir::DenseSet<std::pair<uint32_t, uint32_t>>;
    using NeighbourVector = mlir::SmallVector<mlir::SmallVector<uint32_t, 4>>;

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
    [[nodiscard]] llvm::SmallVector<uint32_t, 4>
    neighboursOf(uint32_t u) const {
      return neighbours_[u];
    }

  private:
    using Matrix = llvm::SmallVector<llvm::SmallVector<std::size_t>>;

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

  protected:
    /**
     * @brief Maps a program qubit index to its hardware index.
     */
    mlir::SmallVector<uint32_t> programToHardware_;

    /**
     * @brief Maps a hardware qubit index to its program index.
     */
    mlir::SmallVector<uint32_t> hardwareToProgram_;
  };

  class AStarSearchEngine {
  public:
    explicit AStarSearchEngine(float alpha) { params_.alpha = alpha; }

  private:
    struct Parameters {
      float alpha{};
    };

    struct Node {
      SmallVector<std::pair<std::size_t, std::size_t>> sequence;
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
      Node(const Node& parent, std::pair<std::size_t, std::size_t> swap,
           const Layer& layer, const Architecture& arch,
           const Parameters& params)
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
      [[nodiscard]] bool isGoal(Layer layer, const Architecture& arch) const {
        return llvm::all_of(
            layer.gates, [&](const std::pair<std::size_t, std::size_t> gate) {
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
      [[nodiscard]] float h(const Layer& layer,
                            const Architecture& arch) const {
        float costs{0};
        for (const auto [prog0, prog1] : layer.gates) {
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
    [[nodiscard]] std::optional<
        SmallVector<std::pair<std::size_t, std::size_t>>>
    route(const Layer& layer, const Layout& layout,
          const Architecture& arch) const {
      Node root(layout);

      /// Early exit. No SWAPs required:
      if (root.isGoal(layer, arch)) {
        return SmallVector<std::pair<std::size_t, std::size_t>>{};
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
    void expand(MinQueue& frontier, const Node& parent, const Layer& layer,
                const Architecture& arch) const {
      DenseSet<std::pair<std::size_t, std::size_t>> expansionSet{};

      if (!parent.sequence.empty()) {
        expansionSet.insert(parent.sequence.back());
      }

      for (const std::pair<std::size_t, std::size_t> gate : layer.gates) {
        for (const auto prog : {gate.first, gate.second}) {
          const auto hw0 = parent.layout.getHardwareIndex(prog);
          for (const auto hw1 : arch.neighboursOf(hw0)) {
            /// Ensure consistent hashing/comparison.
            const std::pair<std::size_t, std::size_t> swap =
                std::minmax(hw0, hw1);
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
    // TODO: Hardcoded architecture.
    Architecture arch("RigettiNovera", 9,
                      {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                       {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                       {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                       {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});
    AStarSearchEngine engine(0.5);

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      Region& region = func.getFunctionBody();

      SmallVector<QubitValue> dynQubits(map_range(
          region.getOps<AllocOp>(), [](AllocOp op) { return op.getResult(); }));

      //// ---- TODO: Add a nice description what happens here.

      SmallVector<WireIterator> wires(
          llvm::map_range(dynQubits, [](auto q) { return WireIterator(q); }));
      SmallVector<WireIterator> wireStarts(wires);

      const auto forwardLayers =
          getLayers<Forward>(wires, [&](std::size_t, const WireIterator& it) {
            return it != std::default_sentinel;
          });
      const auto backwardLayers = getLayers<Backward>(
          wires, [&](std::size_t idx, const WireIterator& it) {
            return it != wireStarts[idx];
          });

      //// ----

      llvm::dbgs() << "forward:\n";
      for (const auto& layer : forwardLayers) {
        layer.dump();
        llvm::dbgs() << " --- layer end ---\n";
      }

      llvm::dbgs() << "backward:\n";
      for (const auto& layer : backwardLayers) {
        layer.dump();
        llvm::dbgs() << " --- layer end ---\n";
      }

      //// ---- TODO: Add a nice description what happens here.

      auto layout = Layout::identity(arch.nqubits());
      const std::size_t repeats = 10;
      for (std::size_t i = 0; i < repeats; ++i) {
        process(forwardLayers, engine, arch, layout);
        process(backwardLayers, engine, arch, layout);
      }

      //// ----

      //// ---- TODO: Add a nice description what happens here.

      IRRewriter rewriter(&getContext());
      rewriter.setInsertionPointToStart(&region.front());
      const auto statQubits = place(dynQubits, layout, rewriter);

      //// ----
    }
  }

private:
  template <Direction d, typename EndCheckF>
  [[nodiscard]] static SmallVector<Layer>
  getLayers(MutableArrayRef<WireIterator> wires, EndCheckF endCheck) {
    constexpr std::size_t step = d == Forward ? 1 : -1;

    SmallVector<Layer> layers;
    DenseMap<UnitaryOpInterface, std::size_t> occ;

    while (true) {
      Layer l;
      // SetVector<QubitValue ssaFront;

      for (const auto [index, it] : llvm::enumerate(wires)) {
        while (endCheck(index, it)) {
          const auto res =
              TypeSwitch<Operation*, WalkResult>(it.operation())
                  .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                    assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                    if (op.getNumQubits() == 1) {
                      std::ranges::advance(it, step);
                      return WalkResult::advance();
                    }

                    // ssaFront.insert(
                    //     cast<QubitValue(std::prev(it).qubit()));

                    if (occ.contains(op)) {
                      const auto otherIndex = occ[op];
                      l.gates.insert(std::make_pair(index, otherIndex));
                      occ.erase(op);
                    } else {
                      occ.try_emplace(op, index);
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

      // If there are no more two-qubit gates, early exit.
      if (l.gates.empty()) {
        break;
      }

      for (auto [i1, i2] : l.gates) {
        std::ranges::advance(wires[i1], step);
        std::ranges::advance(wires[i2], step);
      }

      layers.emplace_back(l);
    }

    return layers;
  }

  static SmallVector<QubitValue>
  place(const ArrayRef<const QubitValue> dynQubits, const Layout& layout,
        IRRewriter& rewriter) {
    SmallVector<QubitValue> statQubits;
    statQubits.reserve(layout.getNumQubits());

    // 1. Replace existing dynamic allocations with mapped static ones.
    for (const auto [prog, q] : llvm::enumerate(dynQubits)) {
      const auto hw = layout.getHardwareIndex(prog);
      Operation* op = q.getDefiningOp();

      rewriter.setInsertionPoint(op);
      auto staticOp = rewriter.replaceOpWithNewOp<StaticOp>(op, hw);
      statQubits.push_back(staticOp.getQubit());
    }

    // Ensure subsequent insertions happen after the last replaced op.
    if (!statQubits.empty()) {
      rewriter.setInsertionPointAfter(statQubits.back().getDefiningOp());
    }

    // 2. Create static qubits for the remaining (unused) hardware indices.
    for (std::size_t prog = dynQubits.size(); prog < layout.getNumQubits();
         ++prog) {
      const auto hw = layout.getHardwareIndex(prog);
      auto staticOp = rewriter.create<StaticOp>(rewriter.getUnknownLoc(), hw);
      statQubits.push_back(staticOp.getQubit());
    }

    return statQubits;
  }

  /// TODO: Naming.
  static void process(ArrayRef<const Layer> layers,
                      const AStarSearchEngine& engine, const Architecture& arch,
                      Layout& layout) {
    for (const auto& layer : layers) {
      /// TODO: Check optional.
      const auto swaps = engine.route({layer}, layout, arch);
      for (const auto [hw0, hw1] : *swaps) {
        const auto [prog0, prog1] = layout.getProgramIndices(hw0, hw1);
        layout.swap(prog0, prog1);
      }
    }
  }
};
} // namespace mlir::qco
