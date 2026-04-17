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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Allocator.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Threading.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
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

  enum class RoutingMode : std::uint8_t { Cold, Hot };

  struct LayerItem {
    Operation* op;
    IndexPairType progs;
  };

  using Window = SmallVector<LayerItem>;

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
     * @returns true if the current SWAP sequence makes all gates in the front
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

      for (const auto& [i, item] : enumerate(window)) {
        const auto [prog0, prog1] = item.progs;
        const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
        const std::size_t nswaps = arch.distanceBetween(hw0, hw1) - 1;
        costs += decay * static_cast<float>(nswaps);
        decay *= params.lambda;
      }
      return costs;
    }
  };

public:
  explicit MappingPass()
      : arch(std::make_shared<Architecture>(getEmptyArchitecture())) {}

  explicit MappingPass(MappingPassOptions options)
      : arch(std::make_shared<Architecture>(getEmptyArchitecture())),
        MappingPassBase<MappingPass>(options) {}

  explicit MappingPass(std::shared_ptr<Architecture> arch,
                       MappingPassOptions options)
      : arch(std::move(arch)), MappingPassBase<MappingPass>(options) {}

protected:
  void runOnOperation() override {
    std::mt19937_64 rng{seed};
    IRRewriter rewriter(&getContext());

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      // TODO: 1) Not necessarily correct because deallocations could happen
      // in-between.
      // TODO: 2) Include QTensors.
      if (range_size(func.getOps<AllocOp>()) > arch->nqubits()) {
        func.emitError() << "the targeted architecture supports "
                         << arch->nqubits() << " qubits, got "
                         << range_size(func.getOps<AllocOp>());
        signalPassFailure();
        return;
      }

      // Create trials for initial layout refining. Currently this includes
      // `ntrials` many random layouts.
      SmallVector<Trial> trials;
      trials.reserve(ntrials);
      for (std::size_t i = 0; i < ntrials; ++i) {
        trials.emplace_back(Layout::random(arch->nqubits(), rng()));
      }

      // Execute each of the trials (possibly in parallel). Collect the results
      // and find the one with the fewest SWAPs on the final backwards pass.
      parallelForEach(&getContext(), trials, [&, this](Trial& trial) {
        const auto res = refineLayout(func, trial.layout);
        if (succeeded(res)) {
          trial.success = true;
          trial.nswaps = *res;
        }
      });

      Trial* best = findBestTrial(trials);
      if (best == nullptr) {
        func.emitError() << "failed to find a best initial layout trial";
        signalPassFailure();
        return;
      }

      // Perform placement and hot routing.
      place(func, best->layout, rewriter);

      // Collect wire iterators for static qubits.
      // The i-th wire iterator belongs to the i-th program qubit.
      SmallVector<WireIterator> wires(range_size(func.getOps<StaticOp>()));
      for (StaticOp op : func.getOps<StaticOp>()) {
        const auto hw = op.getIndex();
        const auto prog = best->layout.getProgramIndex(hw);
        wires[prog] = WireIterator(op.getQubit());
      }

      // Perform hot routing by inserting SWAPs into the IR.
      const auto res = route<RoutingMode::Hot>(wires, WalkDirection::Forward,
                                               best->layout, &rewriter);
      if (failed(res)) {
        func.emitError() << "failed to map the function";
        signalPassFailure();
      }

      // Collect statistics.
      numSwaps += *res;

      // Fix SSA Dominance issues.
      for_each(func.getFunctionBody().getBlocks(),
               [](Block& b) { sortTopologically(&b); });

      assert(isExecutable(func.getFunctionBody(), *arch).succeeded());
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
  FailureOr<std::size_t> refineLayout(func::FuncOp func, Layout& layout) {
    if (niterations == 0) {
      return 0;
    }

    SmallVector<WireIterator> wires;
    for (auto op : func.getOps<AllocOp>()) {
      wires.emplace_back(op.getResult());
    }

    for (auto op : func.getOps<ExtractOp>()) {
      wires.emplace_back(op.getResult());
    }

    std::size_t nswaps{0};
    for (std::size_t i = 0; i < niterations; ++i) {
      const auto resF = route(wires, WalkDirection::Forward, layout);
      if (failed(resF)) {
        return failure();
      }

      const auto resB = route(wires, WalkDirection::Backward, layout);
      if (failed(resB)) {
        return failure();
      }
      nswaps = *resB;
    }

    return nswaps;
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
  [[nodiscard]] FailureOr<SmallVector<IndexPairType>>
  search(const Window& window, const Layout& layout) {
    constexpr std::size_t cap = 25'000'000UL;

    const std::size_t b = arch->maxDegree() * ((arch->nqubits() + 1) / 2);
    const std::size_t budget = std::min(b * b * b, cap);

    const Parameters params{.alpha = alpha, .lambda = lambda};

    llvm::SpecificBumpPtrAllocator<Node> arena;
    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    // Early exit, if the root node is a goal node already.
    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(window.front(), *arch)) {
      return SmallVector<IndexPairType>{};
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

      if (curr->isGoal(window.front(), *arch)) {
        SmallVector<IndexPairType> seq(curr->depth);
        std::size_t j = seq.size() - 1;
        for (Node* n = curr; n->parent != nullptr; n = n->parent) {
          seq[j] = n->swap;
          --j;
        }

        return seq;
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighbouring hardware qubits.

      expansionSet.clear();
      const auto& [q0, q1] = window.front().progs;
      for (const auto prog : {q0, q1}) {
        for (const auto hw0 = curr->layout.getHardwareIndex(prog);
             const auto hw1 : arch->neighboursOf(hw0)) {
          // Ensure consistent hashing/comparison.
          const IndexPairType swap = std::minmax(hw0, hw1);
          if (!expansionSet.insert(swap).second) {
            continue;
          }

          frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                             window, *arch, params));
        }
      }

      ++i;
    }

    return failure();
  }

  /**
   * @brief Skip a qubit-pair block.
   * @details Traverses the pair of wire iterators in tandem until a two-qubit
   * operation is found. If the two-qubit operation is equivalent, continue.
   * Otherwise stop.
   */
  static void skipQubitPairBlock(WireIterator& w0, WireIterator& w1,
                                 WalkDirection direction) {
    const auto step = direction == WalkDirection::Forward ? 1 : -1;
    const auto proceed = [&](const WireIterator& it) {
      if (direction == WalkDirection::Forward) {
        return it != std::default_sentinel;
      }

      if (it.operation() == nullptr) {
        return false;
      }

      return !isa<qco::AllocOp, StaticOp, qtensor::ExtractOp>(it.operation());
    };

    WireIterator curr0(w0);
    WireIterator curr1(w1);
    while (true) {
      while (proceed(curr0)) {
        std::ranges::advance(curr0, step);
      }

      if (curr0 == std::default_sentinel) {
        return;
      }

      while (proceed(curr1)) {
        std::ranges::advance(curr1, step);
      }

      if (curr1 == std::default_sentinel) {
        return;
      }

      if (curr0.operation() != curr1.operation()) {
        return;
      }

      // Handle two-qubit barrier edge case explicitly.
      if (auto barrier = dyn_cast<BarrierOp>(curr0.operation())) {
        if (barrier.getNumQubits() != 2) {
          return;
        }
      }

      w0 = curr0;
      w1 = curr1;
    }
  }

  /**
   * @brief Build and return window of layers.
   * @details Traverses the circuit-layers until the desired window sizes is
   * reached. Assumes that wires[i] = i-th program qubit. The size of the window
   * is 1 + nlookahead.
   * @returns window of layers.
   */
  Window getWindow(ArrayRef<WireIterator> baseWires, WalkDirection direction) {
    Window window;
    window.reserve(1 + nlookahead);

    SmallVector<WireIterator> wires(baseWires);
    std::ignore = walkCircuitGraph(
        wires, direction, [&](const ReadyRange& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          // Construct layer from wire iterators.
          for (const auto& [op, progs] : ready) {
            if (isa<BarrierOp>(op)) {
              released.emplace_back(op);
              continue;
            }

            window.emplace_back(op, std::make_pair(progs[0], progs[1]));
            if (window.size() == 1 + nlookahead) {
              return WalkResult::interrupt();
            }

            skipQubitPairBlock(wires[progs[0]], wires[progs[1]], direction);
            released.emplace_back(wires[progs[0]].operation());
          }

          return WalkResult::advance();
        });

    return window;
  }

  /**
   * @brief Advance past all executable gates.
   * @details Traverses the multi-qubit gates of the circuit until no more
   * executable gates are found.
   */
  void skipExecutableGates(MutableArrayRef<WireIterator> wires, Layout& layout,
                           WalkDirection direction) {
    std::ignore = walkCircuitGraph(
        wires, direction, [&](const ReadyRange& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          for (const auto& [op, progs] : ready) {
            if (isa<BarrierOp>(op)) {
              released.emplace_back(op);
              continue;
            }

            const auto [hw0, hw1] =
                layout.getHardwareIndices(progs[0], progs[1]);

            if (arch->areAdjacent(hw0, hw1)) {
              released.emplace_back(op);
            }
          }

          // Stop, if there are no more ready AND executable gates.
          if (released.empty()) {
            return WalkResult::interrupt();
          }

          return WalkResult::advance();
        });
  }

  /**
   * @brief Route via SWAP insertion.
   * @details Iterates over a dynamically computed window of layers and uses A*
   * search to find a sequence of SWAPs that makes that layer executable.
   * Depending on the template parameter, this function only updates
   * (and hence modifies) the layout or also inserts the SWAPs into the IR.
   * @returns failure() if A* search isn't able to find a solution, the number
   * of SWAPs otherwise.
   */
  template <RoutingMode mode = RoutingMode::Cold>
  FailureOr<std::size_t> route(MutableArrayRef<WireIterator> wires,
                               const WalkDirection& direction, Layout& layout,
                               IRRewriter* rewriter = nullptr) {
    std::size_t nswaps{0};

    while (true) {
      skipExecutableGates(wires, layout, direction);

      const auto window = getWindow(wires, direction);
      if (window.empty()) {
        break;
      }

      if constexpr (mode == RoutingMode::Hot) {
        for (auto& it : wires) {
          if (it == std::default_sentinel) {
            --it;
          }
          --it;
        }
      }

      const auto swaps = search(window, layout);
      if (failed(swaps)) {
        return failure();
      }

      for (const auto& [hw0, hw1] : *swaps) {
        if constexpr (mode == RoutingMode::Hot) {
          const auto& [prog0, prog1] = layout.getProgramIndices(hw0, hw1);
          const auto& w0 = wires[prog0];
          const auto& w1 = wires[prog1];

          assert(!isa<SinkOp>(w0.operation()));
          assert(!isa<SinkOp>(w1.operation()));

          const auto in0 = w0.qubit();
          const auto in1 = w1.qubit();

          rewriter->setInsertionPointAfter(in0.getDefiningOp());
          auto swapOp = SWAPOp::create(*rewriter, in0.getLoc(), in0, in1);

          const auto out0 = swapOp.getQubit0Out();
          const auto out1 = swapOp.getQubit1Out();

          rewriter->replaceAllUsesExcept(in0, out1, swapOp);
          rewriter->replaceAllUsesExcept(in1, out0, swapOp);

          // Preserve program-indexed wire semantics.
          wires[prog0] = WireIterator(out1);
          wires[prog1] = WireIterator(out0);

          assert(isa<SWAPOp>(w0.operation()));
          assert(isa<SWAPOp>(w1.operation()));
        }
        layout.swap(hw0, hw1);
      }

      if constexpr (mode == RoutingMode::Hot) {
        for_each(wires, [](auto& it) { ++it; });
      }

      nswaps += swaps->size();
    }

    return nswaps;
  }

  std::shared_ptr<Architecture> arch;
};

} // namespace

std::unique_ptr<Pass> createMappingPass(std::shared_ptr<Architecture> arch,
                                        MappingPassOptions options) {
  return std::make_unique<MappingPass>(std::move(arch), options);
}

} // namespace mlir::qco
