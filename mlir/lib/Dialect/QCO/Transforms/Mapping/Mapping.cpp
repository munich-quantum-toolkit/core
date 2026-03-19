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
#include "mlir/Dialect/QCO/Transforms/Mapping/Architecture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Threading.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mapping-pass"

namespace mlir::qco {

#define GEN_PASS_DEF_MAPPINGPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using QubitValue = TypedValue<QubitType>;
  using IndexType = std::size_t;
  using IndexGate = std::pair<IndexType, IndexType>;
  using IndexGateSet = DenseSet<IndexGate>;
  using Layer = DenseSet<IndexGate>;

  /**
   * @brief Specifies the layering direction.
   */
  enum class Direction : std::uint8_t { Forward, Backward };

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

    /**
     * @brief Constructs a random layout.
     * @param nqubits The number of qubits.
     * @param seed A seed for randomization.
     * @return The random layout.
     */
    static Layout random(const std::size_t nqubits, const std::size_t seed) {
      SmallVector<IndexType> mapping(nqubits);
      std::iota(mapping.begin(), mapping.end(), IndexType{0});
      std::ranges::shuffle(mapping, std::mt19937_64{seed});

      Layout layout(nqubits);
      for (const auto [prog, hw] : enumerate(mapping)) {
        layout.add(prog, hw);
      }

      return layout;
    }

    /**
     * @brief Insert program:hardware index mapping.
     * @param prog The program index.
     * @param hw The hardware index.
     */
    void add(IndexType prog, IndexType hw) {
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
    [[nodiscard]] IndexType getProgramIndex(const IndexType hw) const {
      assert(hw < hardwareToProgram_.size() &&
             "getProgramIndex: hardware index out of bounds");
      return hardwareToProgram_[hw];
    }

    /**
     * @brief Look up hardware index for a program index.
     * @param prog The program index.
     * @return The hardware index of the respective program index.
     */
    [[nodiscard]] IndexType getHardwareIndex(const IndexType prog) const {
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
              ((std::is_convertible_v<ProgIndices, IndexType>) && ...)
    [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
      return std::tuple{getHardwareIndex(static_cast<IndexType>(progs))...};
    }

    /**
     * @brief Convenience function to lookup multiple program indices at once.
     * @param hws The hardware indices.
     * @return A tuple of program indices.
     */
    template <typename... HwIndices>
      requires(sizeof...(HwIndices) > 0) &&
              ((std::is_convertible_v<HwIndices, std::size_t>) && ...)
    [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
      return std::tuple{getProgramIndex(static_cast<IndexType>(hws))...};
    }

    /**
     * @brief Swap the mapping to program indices of two hardware indices.
     */
    void swap(const IndexType hw0, const IndexType hw1) {
      const auto prog0 = hardwareToProgram_[hw0];
      const auto prog1 = hardwareToProgram_[hw1];

      std::swap(hardwareToProgram_[hw0], hardwareToProgram_[hw1]);
      std::swap(programToHardware_[prog0], programToHardware_[prog1]);
    }

    /**
     * @returns the number of qubits managed by the layout.
     */
    [[nodiscard]] std::size_t nqubits() const {
      return programToHardware_.size();
    }

    void dump() {
      llvm::dbgs() << "prog= ";
      for (std::size_t i = 0; i < nqubits(); ++i) {
        llvm::dbgs() << i << " ";
      }
      llvm::dbgs() << "\nhw=   ";
      for (std::size_t i = 0; i < nqubits(); ++i) {
        llvm::dbgs() << programToHardware_[i] << ' ';
      }
      llvm::dbgs() << '\n';
    }

  protected:
    /**
     * @brief Maps a program qubit index to its hardware index.
     */
    SmallVector<IndexType> programToHardware_;

    /**
     * @brief Maps a hardware qubit index to its program index.
     */
    SmallVector<IndexType> hardwareToProgram_;

  private:
    explicit Layout(const std::size_t nqubits)
        : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}
  };

  /**
   * @brief Parameters influencing the behavior of the A* search algorithm.
   */
  struct Parameters {
    Parameters(const float alpha, const float lambda,
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

  /**
   * @brief Describes a node in the A* search graph.
   */
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
         const Architecture& arch, const Parameters& params)
        : sequence(parent.sequence), layout(parent.layout), f(0) {
      // Apply node-specific swap to given layout.
      layout.swap(swap.first, swap.second);

      // Add swap to sequence.
      sequence.emplace_back(swap);

      // Evaluate cost function.
      f = g(params.alpha) + h(layers, arch, params); // NOLINT
    }

    /**
     * @returns true if the current sequence of SWAPs makes all gates
     * executable.
     */
    [[nodiscard]] bool isGoal(const Layer& front,
                              const Architecture& arch) const {
      return all_of(front, [&](const IndexGate& gate) {
        return arch.areAdjacent(layout.getHardwareIndex(gate.first),
                                layout.getHardwareIndex(gate.second));
      });
    }

    /**
     * @returns true iff. the costs of this node are higher than the one of @p
     * rhs.
     */
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
     * that a naive router would insert to route the layers (with a constant
     * layout).
     */
    [[nodiscard]] float h(ArrayRef<Layer> layers, const Architecture& arch,
                          const Parameters& params) const {
      float costs{0};
      for (const auto& [decay, layer] : zip(params.decay, layers)) {
        for (const auto& [prog0, prog1] : layer) {
          const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
          const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
          costs += decay * static_cast<float>(nswaps);
        }
      }
      return costs;
    }
  };

  using MinQueue = std::priority_queue<Node, std::vector<Node>, std::greater<>>;

public:
  using MappingPassBase::MappingPassBase;

protected:
  void runOnOperation() override {
    std::mt19937_64 rng{this->seed};
    IRRewriter rewriter(&getContext());

    Parameters params(this->alpha, this->lambda, this->nlookahead);
    // TODO: Hardcoded architecture.
    Architecture arch("RigettiNovera", 9,
                      {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                       {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                       {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                       {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      const auto dyn = collectDynamicQubits(func.getFunctionBody());
      if (dyn.size() > arch.nqubits()) {
        func.emitError() << "the targeted architecture supports "
                         << arch.nqubits() << " qubits, got " << dyn.size();
        signalPassFailure();
        return;
      }

      const auto [ltr, rtl] = computeBidirectionalLayers(dyn);

      // Create trials. Currently this includes the identity layout and
      // `ntrials` many random layouts.

      SmallVector<Layout> trials;
      trials.reserve(1 + this->ntrials);
      trials.emplace_back(Layout::identity(arch.nqubits()));
      for (std::size_t i = 0; i < this->ntrials; ++i) {
        trials.emplace_back(Layout::random(arch.nqubits(), rng()));
      }

      // Execute each of the trials (possibly in parallel). Collect the results
      // and find the one with the fewest SWAPs.

      SmallVector<std::optional<TrialResult>> results(trials.size());
      parallelForEach(
          &getContext(), enumerate(trials), [&, this](auto indexedTrial) {
            auto [idx, layout] = indexedTrial;
            auto res = runMappingTrial(ltr, rtl, arch, params, layout);
            if (succeeded(res)) {
              results[idx] = std::move(*res);
            }
          });

      TrialResult* best = findBestTrial(results);
      if (best == nullptr) {
        signalPassFailure();
        return;
      }

      commitTrial(*best, dyn, func.getFunctionBody(), rewriter);
    }
  }

private:
  struct [[nodiscard]] TrialResult {
    explicit TrialResult(Layout layout) : layout(std::move(layout)) {}

    /// @brief The computed initial layout.
    Layout layout;
    /// @brief A vector of SWAPs for each layer.
    SmallVector<SmallVector<IndexGate>> swaps;
    /// @brief The number of inserted SWAPs.
    std::size_t nswaps{};
  };

  /**
   * @brief Find the best trial result in terms of the number of SWAPs.
   * @returns the best trial result or nullptr if no result is valid.
   */
  [[nodiscard]] static TrialResult*
  findBestTrial(MutableArrayRef<std::optional<TrialResult>> results) {
    TrialResult* best = nullptr;
    for (auto& opt : results) {
      if (opt.has_value()) {
        if (best == nullptr || best->nswaps > opt->nswaps) {
          best = &opt.value();
        }
      }
    }
    return best;
  }

  /**
   * @brief Run a mapping trial.
   * @details Use the SABRE Approach to improve the initial layout:
   * Traverse the layers from left-to-right-to-left and cold-route
   * along the way. Repeat this procedure "niterations" times.
   * @returns the trial result or failure() on failure.
   */
  FailureOr<TrialResult> runMappingTrial(ArrayRef<Layer> ltr,
                                         ArrayRef<Layer> rtl,
                                         const Architecture& arch,
                                         const Parameters& params,
                                         Layout& layout) {
    // Perform forwards and backwards traversals.
    for (std::size_t i = 0; i < this->niterations; ++i) {
      if (failed(route(ltr, arch, params, layout, [](const auto&) {}))) {
        return failure();
      }
      if (failed(route(rtl, arch, params, layout, [](const auto&) {}))) {
        return failure();
      }
    }

    TrialResult result(layout); // Copies the final initial layout.

    // Helper function that adds the SWAPs to the trial result.
    const auto collectSwaps = [&](ArrayRef<IndexGate> swaps) {
      result.nswaps += swaps.size();
      result.swaps.emplace_back(swaps);
    };

    // Perform final left-to-right traversal whilst collecting SWAPs.
    if (failed(route(ltr, arch, params, layout, collectSwaps))) {
      return failure();
    }

    return result;
  }

  /**
   * @brief Collect dynamic qubits contained in the given function body.
   * @returns a vector of SSA values produced by qco.alloc operations.
   */
  [[nodiscard]] static SmallVector<QubitValue>
  collectDynamicQubits(Region& funcBody) {
    return SmallVector<QubitValue>(
        map_range(funcBody.getOps<AllocOp>(), [](AllocOp op) {
          return llvm::cast<TypedValue<QubitType>>(op.getResult());
        }));
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

  /**
   * @brief Perform placement.
   * @details Replaces dynamic with static qubits. Extends the computation with
   * as many static qubits the architecture supports.
   * @returns vector of SSA values produced by qco.static operations, ordered by
   * the static index s.t. [i] = qco.static i.
   */
  [[nodiscard]] static SmallVector<QubitValue>
  place(ArrayRef<QubitValue> dynQubits, const Layout& layout, Region& funcBody,
        IRRewriter& rewriter) {
    SmallVector<QubitValue> statics(layout.nqubits());

    // 1. Replace existing dynamic allocations with mapped static ones.
    for (const auto [p, q] : enumerate(dynQubits)) {
      const auto hw = layout.getHardwareIndex(p);
      rewriter.setInsertionPoint(q.getDefiningOp());
      auto op = rewriter.replaceOpWithNewOp<StaticOp>(q.getDefiningOp(), hw);
      statics[hw] = llvm::cast<TypedValue<QubitType>>(op.getQubit());
    }

    // 2. Create static qubits for the remaining (unused) hardware indices.
    for (std::size_t p = dynQubits.size(); p < layout.nqubits(); ++p) {
      rewriter.setInsertionPointToStart(&funcBody.front());
      const auto hw = layout.getHardwareIndex(p);
      auto op = StaticOp::create(rewriter, rewriter.getUnknownLoc(), hw);
      rewriter.setInsertionPoint(funcBody.back().getTerminator());
      DeallocOp::create(rewriter, rewriter.getUnknownLoc(), op.getQubit());
      statics[hw] = llvm::cast<TypedValue<QubitType>>(op.getQubit());
    }

    return statics;
  }

  /**
   * @brief Perform A* search to find a sequence of SWAPs that makes the
   * two-qubit operations inside the first layer (the front) executable.
   * @details
   * The iteration budget is then b^{3}, which corresponds to
   * exhausting all paths of length up to b^{2} in a search tree with branching
   * factor b. A hard cap prevents impractical runtimes on larger architectures.
   *
   * The branching factor b of the A* search is the product of the
   * architecture's maximum qubit degree and the maximum number of two-qubit
   * gates in any layer:
   *
   * b = maxDegree × ⌈N/2⌉
   *
   * @returns a vector of hardware-index pairs (each denoting a SWAP) or
   * failure() if A* fails.
   */
  [[nodiscard]] static FailureOr<SmallVector<IndexGate>>
  search(ArrayRef<Layer> layers, const Layout& layout, const Architecture& arch,
         const Parameters& params) {
    constexpr std::size_t cap = 25'000'000UL;
    const std::size_t b = arch.maxDegree() * ((arch.nqubits() + 1) / 2);
    const std::size_t budget = std::min(b * b * b, cap);

    Node root(layout);
    if (root.isGoal(layers.front(), arch)) {
      return SmallVector<IndexGate>{};
    }

    MinQueue frontier{};
    frontier.emplace(root);
    DenseSet<IndexGate> expansionSet;

    std::size_t i = 0;
    while (!frontier.empty() && i < budget) {
      Node curr = frontier.top();
      frontier.pop();

      if (curr.isGoal(layers.front(), arch)) {
        return curr.sequence;
      }

      // Given a layout, create child-nodes for each possible SWAP between
      // two neighbouring hardware qubits.

      expansionSet.clear();
      if (!curr.sequence.empty()) {
        expansionSet.insert(curr.sequence.back());
      }

      for (const IndexGate& gate : layers.front()) {
        for (const auto prog : {gate.first, gate.second}) {
          const auto hw0 = curr.layout.getHardwareIndex(prog);
          for (const auto hw1 : arch.neighboursOf(hw0)) {
            /// Ensure consistent hashing/comparison.
            const IndexGate swap = std::minmax(hw0, hw1);
            if (!expansionSet.insert(swap).second) {
              continue;
            }

            frontier.emplace(curr, swap, layers, arch, params);
          }
        }
      }

      ++i;
    }

    return failure();
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

  /**
   * @brief Collect the layers of independently executable two-qubit gates of a
   * circuit.
   * @details Depending on the template parameter, the function collects the
   * layers in forward or backward direction, respectively. Towards that end,
   * the function traverses the def-use chain of each qubit until a two-qubit
   * gate is found. If a two-qubit gate is visited twice, it is considered ready
   * and inserted into the layer. This process is repeated until no more
   * two-qubit are found anymore.
   * @returns a vector of layers.
   */
  template <Direction d>
  static SmallVector<Layer> collectLayers(MutableArrayRef<WireIterator> wires) {
    constexpr auto step = d == Direction::Forward ? 1 : -1;
    const auto shouldContinue = [](const WireIterator& it) {
      if constexpr (d == Direction::Forward) {
        return it != std::default_sentinel;
      } else {
        return !isa<AllocOp>(it.operation());
      }
    };

    SmallVector<Layer> layers;
    DenseMap<UnitaryOpInterface, std::size_t> visited;

    while (true) {
      Layer layer{};
      for (const auto [index, it] : enumerate(wires)) {
        while (shouldContinue(it)) {
          const auto res =
              TypeSwitch<Operation*, WalkResult>(it.operation())
                  .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                    assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                    if (op.getNumQubits() == 1) {
                      std::ranges::advance(it, step);
                      return WalkResult::advance();
                    }

                    if (visited.contains(op)) {
                      const auto otherIndex = visited[op];
                      layer.insert(std::make_pair(index, otherIndex));

                      std::ranges::advance(wires[index], step);
                      std::ranges::advance(wires[otherIndex], step);

                      visited.erase(op);
                    } else {
                      visited.try_emplace(op, index);
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
      visited.clear();
    }

    return layers;
  }

  /**
   * @brief "Cold" routing of the given layers.
   * @details Iterates over a sliding window of layers and uses the A* search
   * engine to find a sequence of SWAPs that makes that layer executable.
   * Instead of inserting these SWAPs into the IR, this function only updates
   * (and hence modifies) the layout. The function calls the callback @p onSwaps
   * for each layer with the found sequence of SWAPs.
   * @returns failure() if A* search isn't able to find a solution.
   */
  template <typename OnSwaps>
  LogicalResult route(ArrayRef<Layer> layers, const Architecture& arch,
                      const Parameters& params, Layout& layout,
                      OnSwaps&& onSwaps) {
    auto&& callback = std::forward<OnSwaps>(onSwaps);

    for (std::size_t i = 0; i < layers.size(); ++i) {
      const std::size_t len = std::min(1 + nlookahead, layers.size() - i);
      const auto window = layers.slice(i, len);
      const auto swaps = search(window, layout, arch, params);
      if (failed(swaps)) {
        return failure();
      }

      for (const auto& [hw0, hw1] : *swaps) {
        layout.swap(hw0, hw1);
      }

      std::invoke(callback, *swaps);
    }

    return success();
  }

  /**
   * @brief Performs placement and inserts SWAPs into the IR.
   * @details Replace the dynamic with static qubits ("placement") and inserts
   * the SWAPs of the trial result into the IR.
   */
  static void commitTrial(const TrialResult& result,
                          ArrayRef<QubitValue> dynQubits, Region& funcBody,
                          IRRewriter& rewriter) {
    // Helper function that advances the iterator to the input qubit (the
    // operation producing it) of a deallocation or two-qubit op.
    const auto advFront = [](WireIterator& it) {
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

    auto wires = toWires(place(dynQubits, result.layout, funcBody, rewriter));

    DenseMap<Operation*, WireIterator*> seen;
    for (const auto [i, swaps] : enumerate(result.swaps)) {
      // Advance all wires to the next front of one-qubit outputs
      // (the SSA values).
      for_each(wires, advFront);

      // Apply the sequence of SWAPs and rewire the qubit SSA values.
      for (const auto& [hw0, hw1] : swaps) {
        Operation* op0 = wires[hw0].operation();
        Operation* op1 = wires[hw1].operation();
        const auto in0 = wires[hw0].qubit();
        const auto in1 = wires[hw1].qubit();

        // Reorder to avoid SSA dominance issues.
        assert(op0->getBlock()->isOpOrderValid() &&
               "An invalid op order leads to a significant runtime overhead.");
        if (op0->isBeforeInBlock(op1)) {
          rewriter.setInsertionPointAfterValue(in1);
        } else {
          rewriter.setInsertionPointAfterValue(in0);
        }

        auto op = SWAPOp::create(rewriter, rewriter.getUnknownLoc(), in0, in1);
        const auto out0 = op.getQubit0Out();
        const auto out1 = op.getQubit1Out();

        rewriter.replaceAllUsesExcept(in0, out1, op);
        rewriter.replaceAllUsesExcept(in1, out0, op);

        // Jump over the SWAPOp.
        std::ranges::advance(wires[hw0], 1);
        std::ranges::advance(wires[hw1], 1);
      }

      // Jump over "ready" two-qubit gates.
      for (auto& it : wires) {
        auto op = dyn_cast<UnitaryOpInterface>(std::next(it).operation());
        if (op && op.getNumQubits() > 1) {
          if (seen.contains(op)) {
            std::ranges::advance(it, 1);
            std::ranges::advance(*seen[op], 1);
            continue;
          }

          seen.try_emplace(op, &it);
        }
      }

      seen.clear(); // Prepare for next iteration.
    }
  }
};

} // namespace

} // namespace mlir::qco
