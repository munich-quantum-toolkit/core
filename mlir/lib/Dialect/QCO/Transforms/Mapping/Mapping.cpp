/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Architecture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
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
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mapping-pass"

namespace mlir::qco {

using namespace mlir::qtensor;

#define GEN_PASS_DEF_MAPPINGPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

LogicalResult isExecutable(Region& region, const Architecture& arch) {
  bool executable = true;
  walkUnit(region, [&](Operation* curr, const Qubits& qubits) {
    if (auto op = dyn_cast<UnitaryOpInterface>(curr)) {
      if (isa<BarrierOp>(op)) {
        return WalkResult::advance();
      }
      if (op.getNumQubits() > 1) {
        const auto q0 = cast<TypedValue<QubitType>>(op.getInputQubit(0));
        const auto q1 = cast<TypedValue<QubitType>>(op.getInputQubit(1));
        const auto i0 = qubits.getIndex(q0);
        const auto i1 = qubits.getIndex(q1);
        if (!arch.areAdjacent(i0, i1)) {
          executable = false;
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  if (executable) {
    return success();
  }

  return failure();
}

namespace {

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using QubitValue = TypedValue<QubitType>;
  using IndexType = std::size_t;
  using IndexPairType = std::pair<IndexType, IndexType>;

  struct LayerItem {
    Operation* op;
    IndexPairType progs;
  };

  using Layer = SmallVector<LayerItem>;
  using Window = SmallVector<Layer, 0>;

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

    /**
     * @returns the program to hardware mapping.
     */
    [[nodiscard]] ArrayRef<IndexType> getProgramToHardware() const {
      return programToHardware_;
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

  struct [[nodiscard]] Trial {
    explicit Trial(Layout layout) : layout(std::move(layout)) {}

    Layout layout;
    std::size_t nswaps{};
    bool success{false};
  };

  /**
   * @brief Parameters influencing the behavior of the A* search algorithm.
   */
  struct [[nodiscard]] Parameters {
    float alpha;
    float lambda;
  };

  /**
   * @brief Describes a node in the A* search graph.
   */
  struct [[nodiscard]] Node {
    struct ComparePointer {
      bool operator()(const Node* lhs, const Node* rhs) const {
        return lhs->f > rhs->f;
      }
    };

    Layout layout;
    IndexPairType swap;
    Node* parent;
    std::size_t depth;
    float f;

    /**
     * @brief Construct a root node with the given layout. Initialize the
     * sequence with an empty vector and set the cost to zero.
     */
    explicit Node(Layout layout)
        : layout(std::move(layout)), parent(nullptr), depth(0), f(0) {}

    /**
     * @brief Construct a non-root node from its parent node. Apply the given
     * swap to the layout of the parent node.
     */
    Node(Node* parent, const IndexPairType& swap, const Window& window,
         const Architecture& arch, const Parameters& params)
        : layout(parent->layout), swap(swap), parent(parent),
          depth(parent->depth + 1), f(0) {
      layout.swap(swap.first, swap.second);
      f = g(params.alpha) + h(window, arch, params); // NOLINT
    }

    /**
     * @returns true if the current sequence of SWAPs makes some gates
     * executable.
     */
    [[nodiscard]] bool isPartialGoal(ArrayRef<LayerItem> front,
                                     const Architecture& arch) const {
      return any_of(front, [&](const LayerItem& item) {
        const auto& [prog0, prog1] = item.progs;
        return arch.areAdjacent(layout.getHardwareIndex(prog0),
                                layout.getHardwareIndex(prog1));
      });
    }

    /**
     * @returns true if the current sequence of SWAPs makes all gates
     * executable.
     */
    [[nodiscard]] bool isGoal(ArrayRef<LayerItem> front,
                              const Architecture& arch) const {
      return all_of(front, [&](const LayerItem& item) {
        const auto& [prog0, prog1] = item.progs;
        return arch.areAdjacent(layout.getHardwareIndex(prog0),
                                layout.getHardwareIndex(prog1));
      });
    }

    /**
     * @returns a vector of "ready" two-qubit ops.
     */
    [[nodiscard]] DenseSet<Operation*>
    getReadyOps(ArrayRef<LayerItem> front, const Architecture& arch) const {
      DenseSet<Operation*> ops;
      for (const auto& item : front) {
        const auto& [prog0, prog1] = item.progs;
        if (arch.areAdjacent(layout.getHardwareIndex(prog0),
                             layout.getHardwareIndex(prog1))) {
          ops.insert(item.op);
        }
      }
      return ops;
    }

  private:
    /**
     * @brief Calculate the path cost for the A* search algorithm.
     *
     * The path cost function is the weighted sum of the currently required
     * SWAPs.
     */
    [[nodiscard]] float g(const float alpha) const {
      return alpha * static_cast<float>(depth);
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
    [[nodiscard]] float h(const Window& window, const Architecture& arch,
                          const Parameters& params) const {
      float costs{0};
      float decay{1.};

      for (const auto& [i, layer] : enumerate(window)) {
        for (const auto& [_, progs] : layer) {
          const auto [prog0, prog1] = progs;
          const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
          const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
          costs += decay * static_cast<float>(nswaps);
        }
        decay *= params.lambda;
      }
      return costs;
    }
  };

protected:
  using MappingPassBase::MappingPassBase;

  void runOnOperation() override {
    std::mt19937_64 rng{seed};
    IRRewriter rewriter(&getContext());

    // TODO: Hardcoded architecture.
    Architecture arch("RigettiNovera", 9,
                      {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                       {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                       {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                       {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      // TODO: 1) Not necessarily correct because deallocations could happen
      // in-between.
      // TODO: 2) Include QTensors.
      if (range_size(func.getOps<AllocOp>()) > arch.nqubits()) {
        func.emitError() << "the targeted architecture supports "
                         << arch.nqubits() << " qubits, got "
                         << range_size(func.getOps<AllocOp>());
        signalPassFailure();
        return;
      }

      // Create trials for initial layout refining. Currently this includes
      // `ntrials` many random layouts.
      SmallVector<Trial> trials;
      trials.reserve(ntrials);
      for (std::size_t i = 0; i < ntrials; ++i) {
        trials.emplace_back(Layout::random(arch.nqubits(), rng()));
      }

      // Execute each of the trials (possibly in parallel). Collect the results
      // and find the one with the fewest SWAPs on the final backwards pass.
      parallelForEach(&getContext(), trials, [&, this](Trial& trial) {
        refineLayout(func, arch, trial);
      });

      Trial* best = findBestTrial(trials);
      if (best == nullptr) {
        func.emitError() << "failed to find a best initial layout trial";
        signalPassFailure();
        return;
      }

      // Perform placement and hot routing.
      place(func, best->layout, rewriter);
      if (failed(route(func, arch, best->layout, rewriter))) {
        func.emitError() << "failed to map the function";
        signalPassFailure();
      }

      assert(isExecutable(func.getFunctionBody(), arch).succeeded());
    }
  }

private:
  /**
   * @brief Perform placement.
   * @details Replaces dynamic with static qubits. Extends the computation with
   * as many static qubits as the architecture supports.
   */
  static void place(func::FuncOp func, const Layout& layout,
                    IRRewriter& rewriter) {
    // 0. Materialize to avoid iterator invalidation after replacement.
    const auto qubitAllocs = to_vector(func.getOps<AllocOp>());

    const auto tensorAllocs = to_vector(func.getOps<qtensor::AllocOp>());
    const auto tensorDeallocs = to_vector(func.getOps<qtensor::DeallocOp>());
    auto tensorExtracts = to_vector(func.getOps<ExtractOp>());
    const auto tensorInserts = to_vector(func.getOps<InsertOp>());

    // 1. Replace existing dynamic allocations with mapped static ones.
    std::size_t p = 0;
    for (AllocOp op : qubitAllocs) {
      const auto hw = layout.getHardwareIndex(p);
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<StaticOp>(op, hw);
      ++p;
    }

    // 1.1 Handle tensors.
    for (qtensor::DeallocOp op : tensorDeallocs) {
      rewriter.eraseOp(op);
    }

    for (InsertOp op : reverse(tensorInserts)) {
      rewriter.setInsertionPoint(op);
      rewriter.create<SinkOp>(op.getLoc(), op.getScalar());
      rewriter.eraseOp(op);
    }

    for (auto [i, extractOp] : enumerate(reverse(tensorExtracts))) {
      const auto hw =
          layout.getHardwareIndex(p + tensorExtracts.size() - 1 - i);
      rewriter.setInsertionPoint(extractOp);
      auto op = StaticOp::create(rewriter, extractOp.getLoc(), hw);
      rewriter.replaceAllUsesWith(extractOp.getResult(), op.getQubit());
      rewriter.eraseOp(extractOp);
    }

    p += tensorExtracts.size();

    for (qtensor::AllocOp op : tensorAllocs) {
      rewriter.eraseOp(op);
    }

    // 2. Create static qubits for the remaining (unused) hardware indices.
    const auto location = rewriter.getInsertionPoint()->getLoc();
    for (; p < layout.nqubits(); ++p) {
      rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
      const auto hw = layout.getHardwareIndex(p);
      auto op = StaticOp::create(rewriter, location, hw);
      rewriter.setInsertionPoint(func.getFunctionBody().back().getTerminator());
      SinkOp::create(rewriter, rewriter.getUnknownLoc(), op.getQubit());
    }
  }

  /**
   * @brief Find the best trial result in terms of the number of SWAPs.
   * @returns the best trial result or nullptr if no result is valid.
   */
  [[nodiscard]] static Trial* findBestTrial(MutableArrayRef<Trial> trials) {
    Trial* best = nullptr;
    for (auto& trial : trials) {
      if (!trial.success) {
        continue;
      }

      if (best == nullptr || best->nswaps > trial.nswaps) {
        best = &trial;
      }
    }

    return best;
  }

  /**
   * @brief Refine the trial's layout and count #swaps for the final backwards
   * pass.
   * @details Use the SABRE Approach to improve the initial layout:
   * Traverse the layers from left-to-right-to-left and cold-route
   * along the way. Repeat this procedure "niterations" times.
   * @returns failure() if routing fails.
   */
  void refineLayout(func::FuncOp func, const Architecture& arch, Trial& trial) {
    if (niterations == 0) {
      trial.success = true;
      return;
    }

    SmallVector<WireIterator> wires;
    for (auto op : func.getOps<AllocOp>()) {
      wires.emplace_back(op.getResult());
    }

    for (auto op : func.getOps<ExtractOp>()) {
      wires.emplace_back(op.getResult());
    }

    SmallVector<std::size_t> enumeration;
    enumeration.reserve(wires.size());
    for (std::size_t i = 0; i < wires.size(); ++i) {
      enumeration.emplace_back(i);
    }

    for (std::size_t i = 0; i < niterations; ++i) {
      if (failed(layoutRoute(wires, enumeration, WalkDirection::Forward, arch,
                             trial))) {
        return;
      }
      if (failed(layoutRoute(wires, enumeration, WalkDirection::Backward, arch,
                             trial))) {
        return;
      }
    }

    trial.success = true;
  }

  /**
   * @brief Perform A* search to find a sequence of SWAPs that makes the
   * two-qubit operations inside the first layer (the front) executable.
   * @details
   * The iteration budget is b^{3} node expansions, i.e. roughly a depth-3
   * search in a tree with branching factor b. A hard cap prevents impractical
   * runtimes on larger architectures.
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
  [[nodiscard]] FailureOr<
      std::pair<DenseSet<Operation*>, SmallVector<IndexPairType>>>
  search(const Window& window, const Layout& layout, const Architecture& arch) {
    constexpr std::size_t cap = 25'000'000UL;
    const std::size_t b = arch.maxDegree() * ((arch.nqubits() + 1) / 2);
    const std::size_t budget = std::min(b * b * b, cap);

    const Parameters params{.alpha = alpha, .lambda = lambda};

    llvm::SpecificBumpPtrAllocator<Node> arena;
    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(window.front(), arch)) {
      return std::make_pair(root->getReadyOps(window.front(), arch),
                            SmallVector<IndexPairType>{});
    }
    frontier.emplace(root);

    DenseMap<ArrayRef<IndexType>, std::size_t> bestDepth;
    DenseSet<IndexPairType> expansionSet;

    std::size_t i = 0;
    while (!frontier.empty() && i < budget) {
      Node* curr = frontier.top();
      frontier.pop();

      // Multiple sequences of SWAPs can lead to the same layout and the same
      // layout creates the same child-nodes. Thus, if we've seen a layout
      // already at a lower depth don't reexpand the current node (and hence
      // recreate the same child nodes).

      const auto [it, inserted] = bestDepth.try_emplace(
          curr->layout.getProgramToHardware(), curr->depth);
      if (!inserted) {
        const auto otherDepth = it->getSecond();
        if (curr->depth >= otherDepth) {
          ++i;
          continue;
        }

        it->second = curr->depth;
      }

      // If the currently visited node is a goal node, reconstruct the sequence
      // of SWAPs from this node to the root.

      if (curr->isPartialGoal(window.front(), arch)) {
        const auto ready = curr->getReadyOps(window.front(), arch);

        SmallVector<IndexPairType> seq(curr->depth);
        std::size_t j = seq.size() - 1;
        for (Node* n = curr; n->parent != nullptr; n = n->parent) {
          seq[j] = n->swap;
          --j;
        }

        return std::make_pair(ready, seq);
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighbouring hardware qubits.

      expansionSet.clear();
      for (const auto& [_, progs] : window.front()) {
        const auto& [q0, q1] = progs;
        for (const auto prog : {q0, q1}) {
          for (const auto hw0 = curr->layout.getHardwareIndex(prog);
               const auto hw1 : arch.neighboursOf(hw0)) {
            // Ensure consistent hashing/comparison.
            const IndexPairType swap = std::minmax(hw0, hw1);
            if (!expansionSet.insert(swap).second) {
              continue;
            }

            frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                               window, arch, params));
          }
        }
      }

      ++i;
    }

    return failure();
  }

  /**
   * @brief Recompute window of layers into @p window.
   * @details
   * Traverses the circuit-layers until the desired window sizes is reached.
   * The callback function determines the index of the two-qubit gates inside
   * the layers:
   *    (std::size_t) -> (std::size_t)
   * For cold routing, we can simply use the enumeration (i->i)
   * because the enumeration is the program index assignment.
   * For hot routing, we lookup the program index of the hardware index
   * (provided by the enumeration) via the layout object.
   *
   * The size of the window is 1 + nlookahead.
   */
  template <class Fn>
  void rebuildWindow(ArrayRef<WireIterator> base,
                     ArrayRef<std::size_t> enumeration, WalkDirection direction,
                     Window& window, Fn&& fn) {
    assert(base.size() == enumeration.size());

    const auto lookup = std::forward<Fn>(fn);

    window.clear();                        // Clear window.
    SmallVector<WireIterator> wires(base); // Work on local copy.

    std::ignore = walkCircuitGraph(
        wires, direction,
        [&](FrontArrayRef front, ReleasedIterators& released) {
          if (front.empty()) {
            return WalkResult::advance();
          }

          // Construct layer from wire iterators.
          SmallVector<LayerItem> layer;
          for (ArrayRef<WireIterator*> its : front) {
            assert(its.size() == 2);
            assert(its[0] != nullptr && its[1] != nullptr);

            WireIterator& first = *its[0];
            WireIterator& second = *its[1];

            assert(first.operation() == second.operation());

            const auto idxFirst = std::distance(wires.data(), &first);
            const auto idxSecond = std::distance(wires.data(), &second);

            layer.emplace_back(first.operation(),
                               std::make_pair(lookup(enumeration[idxFirst]),
                                              lookup(enumeration[idxSecond])));

            SmallVector<WireIterator, 2> pair{first, second};
            walkQubitPairBlock(
                pair, direction,
                [&](const WireIterator& a, const WireIterator& b) {
                  first = a;
                  second = b;
                });

            released.append(its.begin(), its.end());
          }

          // Append layer.
          window.emplace_back(layer);

          // Stop, if the desired window size is reached.
          if (window.size() == 1 + nlookahead) {
            return WalkResult::interrupt();
          }

          return WalkResult::advance();
        });

    LLVM_DEBUG({
      llvm::dbgs() << "----- window start -----\n";
      for (const auto& layer : window) {
        for (const auto& [op, progs] : layer) {
          const auto& [i0, i1] = progs;
          llvm::dbgs() << op->getName() << "=(" << i0 << ", " << i1 << ") ";
        }
        llvm::dbgs() << '\n';
      }
      llvm::dbgs() << "----- window end -----\n";
    });
  }

  /**
   * @brief "Cold" routing.
   * @details Iterates over a dynamically computed window of layers and uses A*
   * search to find a sequence of SWAPs that makes that layer executable.
   * Instead of inserting these SWAPs into the IR, this function only updates
   * (and hence modifies) the layout.
   * @returns failure() if A* search isn't able to find a solution.
   */
  LogicalResult layoutRoute(MutableArrayRef<WireIterator> wires,
                            ArrayRef<std::size_t> enumeration,
                            const WalkDirection& direction,
                            const Architecture& arch, Trial& trial) {
    Window window;
    window.reserve(1 + nlookahead);

    trial.nswaps = 0; // Reset the SWAP count.

    return walkCircuitGraph(
        wires, direction,
        [&](FrontArrayRef front, ReleasedIterators& released) {
          if (front.empty()) {
            return WalkResult::advance();
          }

          // Recompute window of layers.
          rebuildWindow(wires, enumeration, direction, window,
                        [](std::size_t i) { return i; });

          // Perform A* search.
          const auto searchResult = search(window, trial.layout, arch);
          if (failed(searchResult)) {
            return WalkResult::interrupt();
          }

          const auto& [readyOps, swaps] = searchResult.value();
          // assert(readyOps.size() == front.size());

          // Collect trial statistics.
          trial.nswaps += swaps.size();

          // Apply SWAPS.
          for (const auto& [hw0, hw1] : swaps) {
            trial.layout.swap(hw0, hw1);
          }

          // Skip two-qubit blocks and release iterators.
          for (ArrayRef<WireIterator*> its : front) {
            WireIterator& first = *its[0];
            WireIterator& second = *its[1];

            if (readyOps.contains(first.operation())) {
              SmallVector<WireIterator, 2> pair{first, second};
              walkQubitPairBlock(
                  pair, direction,
                  [&](const WireIterator& a, const WireIterator& b) {
                    first = a;
                    second = b;
                  });

              released.append(its.begin(), its.end());
            }
          }

          return WalkResult::advance();
        });
  }

  /**
   * @brief "Hot" routing.
   * @details Iterates over a dynamically computed window of layers and uses A*
   * search to find a sequence of SWAPs that makes that layer executable.
   * @returns failure() if A* search isn't able to find a solution.
   */
  LogicalResult route(func::FuncOp func, const Architecture& arch,
                      Layout& layout, IRRewriter& rewriter) {
    constexpr auto direction = WalkDirection::Forward;

    Window window;
    window.reserve(1 + nlookahead);

    SmallVector<WireIterator> wires;
    wires.reserve(range_size(func.getOps<StaticOp>()));
    for (StaticOp op : func.getOps<StaticOp>()) {
      wires.emplace_back(op.getQubit());
    }

    SmallVector<std::size_t> enumeration;
    for (const auto& it : wires) {
      StaticOp op = cast<StaticOp>(it.operation());
      enumeration.emplace_back(op.getIndex());
    }

    SmallVector<std::size_t> revEnumeration(enumeration.size());
    for (std::size_t i = 0; i < enumeration.size(); ++i) {
      revEnumeration[enumeration[i]] = i;
    }

    DenseSet<Operation*> frontSet;
    frontSet.reserve((wires.size() + 1) / 2);

    DenseMap<Operation*, SmallVector<WireIterator*, 2>> others;
    others.reserve((wires.size() + 1) / 2);

    const auto fn = [&](FrontArrayRef front, ReleasedIterators& released) {
      if (front.empty()) {
        return WalkResult::advance();
      }

      // Recompute window of layers.
      rebuildWindow(wires, enumeration, direction, window,
                    [&](std::size_t i) { return layout.getProgramIndex(i); });

      // Perform A* search.
      const auto searchResult = search(window, layout, arch);
      if (failed(searchResult)) {
        return WalkResult::interrupt();
      }

      const auto& [readyOps, swaps] = searchResult.value();
      // assert(readyOps.size() == front.size());

      frontSet.clear();
      for (ArrayRef<WireIterator*> its : front) {
        assert(its.size() == 2);
        assert(its[0]->operation() == its[1]->operation());
        frontSet.insert(its[0]->operation());
      }

      // Prepare insertion points: Each wire iterator points at a two-qubit
      // operation of the current or next layers. Consequently, the wire
      // iterator also points to the qubit SSA value the respective two-qubit
      // operations produce. To point at the operation which produces the inputs
      // of the two-qubit operations, decrement each of the wire iterators.
      // Because sinks don't produce values, decrement a second time.

      for (auto& it : wires) {
        std::ranges::advance(it, -1);
        if (isa<SinkOp>(it.operation())) {
          std::ranges::advance(it, -1);
        }
      }

      // Collect pass statistics.
      numSwaps += swaps.size();

      // Apply and insert SWAPs.
      for (const auto& [hw0, hw1] : swaps) {
        layout.swap(hw0, hw1);

        WireIterator& first = wires[revEnumeration[hw0]];
        WireIterator& second = wires[revEnumeration[hw1]];

        assert(!isa<SinkOp>(first.operation()));
        assert(!isa<SinkOp>(second.operation()));

        const auto in0 = first.qubit();
        const auto in1 = second.qubit();

        rewriter.setInsertionPointAfter(in0.getDefiningOp());
        auto swapOp = SWAPOp::create(rewriter, in0.getLoc(), in0, in1);

        const auto out0 = swapOp.getQubit0Out();
        const auto out1 = swapOp.getQubit1Out();

        rewriter.replaceAllUsesExcept(in0, out1, swapOp);
        rewriter.replaceAllUsesExcept(in1, out0, swapOp);

        ++first;
        ++second;

        assert(isa<SWAPOp>(first.operation()));
        assert(isa<SWAPOp>(second.operation()));
      }

      // Find the iterators of the front gates after SWAP insertion. This is
      // required because the replaceAllUsesExcept swaps wire iterators values.

      others.clear();
      for (auto& it : wires) {
        std::ranges::advance(it, 1);

        if (frontSet.contains(it.operation())) {
          const auto [mapIt, inserted] =
              others.try_emplace(it.operation(), SmallVector{&it});
          if (!inserted) {
            mapIt->second.emplace_back(&it);
          }
        }
      }

      // Skip two-qubit blocks and release iterators.
      for (const auto& its : others.values()) {
        WireIterator& first = *its[0];
        WireIterator& second = *its[1];

        if (readyOps.contains(first.operation())) {
          SmallVector<WireIterator, 2> pair{first, second};
          walkQubitPairBlock(pair, direction,
                             [&](const WireIterator& a, const WireIterator& b) {
                               first = a;
                               second = b;
                             });

          released.append(its.begin(), its.end());
        }
      }

      return WalkResult::advance();
    };

    const auto res = walkCircuitGraph(wires, direction, fn);
    if (failed(res)) {
      return failure();
    }

    for_each(func.getFunctionBody().getBlocks(),
             [](Block& b) { sortTopologically(&b); });

    return success();
  }
};

} // namespace

} // namespace mlir::qco
