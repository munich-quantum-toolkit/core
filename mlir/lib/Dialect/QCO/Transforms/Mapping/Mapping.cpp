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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Algorithms.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
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

namespace {

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using IndexType = size_t;
  using IndexPairType = std::pair<IndexType, IndexType>;
  using Window = SmallVector<IndexPairType>;
  using Neighbours = SmallVector<SmallVector<size_t, 4>>;

  enum class RoutingMode : std::uint8_t { Cold, Hot };

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
    static Layout identity(const size_t nqubits) {
      Layout layout(nqubits);
      for (size_t i = 0; i < nqubits; ++i) {
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
    static Layout random(const size_t nqubits, const size_t seed) {
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
              ((std::is_convertible_v<HwIndices, size_t>) && ...)
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
    [[nodiscard]] size_t nqubits() const { return programToHardware_.size(); }

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
    explicit Layout(const size_t nqubits)
        : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}
  };

  class [[nodiscard]] AugmentedDevice {
  public:
    AugmentedDevice() = default;

    AugmentedDevice(size_t nqubits, const Edges& coupling)
        : nqubits_(nqubits), dist_(findAllShortestPaths(nqubits, coupling)),
          coupling_(coupling), neighbours_(nqubits) {
      for (const auto& [u, v] : coupling_) {
        neighbours_[u].push_back(v);
      }
    }

    /**
     * @returns the device's number of qubits.
     */
    [[nodiscard]] size_t nqubits() const { return nqubits_; }

    /**
     * @returns true if @p u and @p v are adjacent.
     */
    [[nodiscard]] bool areAdjacent(size_t u, size_t v) const {
      return coupling_.contains(std::make_pair(u, v));
    }

    /**
     * @returns the length of the shortest path between @p u and @p v.
     */
    [[nodiscard]] size_t distanceBetween(size_t u, size_t v) const {
      if (dist_[u][v] == UINT64_MAX) {
        report_fatal_error("Failed to compute the distance between qubits " +
                           Twine(u) + " and " + Twine(v));
      }
      return dist_[u][v];
    }

    /**
     * @returns all neighbours of @p u.
     */
    [[nodiscard]] ArrayRef<size_t> neighboursOf(size_t u) const {
      return neighbours_[u];
    }

    /**
     * @returns the max degree (connectivity) of any qubit of the device.
     */
    [[nodiscard]] size_t maxDegree() const {
      size_t deg = 0;
      for (const auto& nbrs : neighbours_) {
        deg = std::max(deg, nbrs.size());
      }
      return deg;
    }

  private:
    size_t nqubits_{};
    Matrix dist_;
    Edges coupling_;
    Neighbours neighbours_;
  };

  struct [[nodiscard]] Trial {
    explicit Trial(Layout layout) : layout(std::move(layout)) {}

    Layout layout;
    size_t nswaps{};
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
    size_t depth;
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
         const AugmentedDevice& device, const Parameters& params)
        : layout(parent->layout), swap(swap), parent(parent),
          depth(parent->depth + 1), f(0) {
      layout.swap(swap.first, swap.second);
      f = g(params.alpha) + h(window, device, params); // NOLINT
    }

    /**
     * @returns true if the current SWAP sequence makes all gates in the front
     * executable.
     */
    [[nodiscard]] bool isGoal(const IndexPairType& front,
                              const AugmentedDevice& device) const {
      return device.areAdjacent(layout.getHardwareIndex(front.first),
                                layout.getHardwareIndex(front.second));
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
    [[nodiscard]] float h(const Window& window, const AugmentedDevice& device,
                          const Parameters& params) const {
      float costs{0};
      float decay{1.};

      for (const auto& [i, progs] : enumerate(window)) {
        const auto [prog0, prog1] = progs;
        const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
        const size_t nswaps = device.distanceBetween(hw0, hw1) - 1;
        costs += decay * static_cast<float>(nswaps);
        decay *= params.lambda;
      }
      return costs;
    }
  };

public:
  MappingPass() = default;
  explicit MappingPass(MappingPassOptions options) : MappingPassBase(options) {}
  explicit MappingPass(size_t nqubits, const Edges& coupling,
                       MappingPassOptions options = {})
      : MappingPassBase(options), device(nqubits, coupling) {}

protected:
  void runOnOperation() override {
    assert(alpha > 0 && "runOnOperation: expected alpha > 0");
    assert(niterations > 0 && "runOnOperation: expected niterations > 0");
    assert(ntrials > 0 && "runOnOperation: expected ntrials > 0");

    std::mt19937_64 rng{seed};
    IRRewriter rewriter(&getContext());

    ModuleOp m = getOperation();
    auto func = getEntryPoint(m);
    if (!func) {
      m.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    auto comp = getComputation(func);
    if (failed(comp)) {
      signalPassFailure();
      return;
    }

    if (comp->size() > device.nqubits()) {
      m.emitError() << "requires " + Twine(comp.value().size()) +
                           " qubits. However, the architecture only supports " +
                           Twine(device.nqubits()) + "qubits.";
      signalPassFailure();
      return;
    }

    // Create trials for initial layout refining. Currently, this includes
    // `ntrials` many random layouts.
    SmallVector<Trial> trials;
    trials.reserve(ntrials);
    for (size_t i = 0; i < ntrials; ++i) {
      trials.emplace_back(Layout::random(device.nqubits(), rng()));
    }

    // Execute each of the trials (possibly in parallel). Collect the results
    // and find the one with the fewest SWAPs on the final backwards pass.
    parallelForEach(&getContext(), trials, [&, this](Trial& trial) {
      if (const auto res = refineLayout(*comp, trial.layout); succeeded(res)) {
        trial.success = true;
        trial.nswaps = *res;
      }
    });

    Trial* best = findBestTrial(trials);
    if (best == nullptr) {
      func.emitError() << "failed to find the best layout trial";
      signalPassFailure();
      return;
    }

    // Perform placement and hot routing by inserting SWAPs into the IR.
    auto placedWires = place(func, best->layout, rewriter);
    const auto res = route<WireDirection::Forward, RoutingMode::Hot>(
        placedWires, best->layout, &rewriter);
    if (failed(res)) {
      func.emitError() << "failed to map the " << func.getName() << " function";
      signalPassFailure();
      return;
    }

    // Collect statistics.
    numSwaps += *res;

    // Fix SSA Dominance issues.
    for_each(func.getFunctionBody().getBlocks(),
             [](Block& b) { sortTopologically(&b); });
  }

private:
  /**
   * @brief Collect wires of the quantum computation before placement.
   * @details
   * The mapping pass currently assumes that the quantum computations consists
   * The required qubits of each tensor are extracted and inserted "in one go".
   *
   * @returns a vector of wire iterator, or failure() if any of the above
   * assumptions are violated.
   */
  static FailureOr<SmallVector<WireIterator>>
  getComputation(func::FuncOp func) {
    if (!func.getOps<AllocOp>().empty()) {
      func.emitError() << "must not contain qco.alloc operations";
      return failure();
    }

    SmallVector<WireIterator> wires;
    for (auto tensor : func.getOps<qtensor::AllocOp>()) {
      bool isInitPhase = true;
      TensorIterator it(tensor.getResult());
      for (; it != std::default_sentinel; ++it) {
        if (auto extract = dyn_cast<ExtractOp>(it.operation())) {
          if (!isInitPhase) {
            func.emitError() << "must extract and insert all qubits at once.";
            return failure();
          }
          wires.emplace_back(extract.getResult());
          continue;
        }

        if (isa<InsertOp>(it.operation())) {
          isInitPhase = false;
          continue;
        }
      }
    }
    return wires;
  }

  /**
   * @brief Perform placement by replacing dynamic with static qubits.
   * @details
   * Creates static qubits and replaces the extracted qubits with it.
   * Moreover, the function extends the computation with as many static qubits
   * as the architecture supports.
   * @returns a vector of wire iterators, where the i-th wire points at the i-th
   * static program qubit.
   */
  static SmallVector<WireIterator>
  place(func::FuncOp func, const Layout& layout, IRRewriter& rewriter) {
    SmallVector<StaticOp> staticOps;
    staticOps.reserve(layout.nqubits());

    // Create and save static qubit operations.
    rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
    for (size_t i = 0; i < layout.nqubits(); ++i) {
      const auto op = StaticOp::create(rewriter, func.getLoc(), i);
      staticOps.emplace_back(op);
      rewriter.setInsertionPointAfter(op);
    }

    // Replace extract ops and collect in program-qubit order.
    SmallVector<WireIterator> placedWires(layout.nqubits());

    size_t prog = 0UL;
    for (auto alloc : make_early_inc_range(func.getOps<qtensor::AllocOp>())) {
      TensorIterator it(alloc.getResult());
      while (it != std::default_sentinel) {
        // Get the operation and early increment to avoid issues after erasure.
        Operation* curr = it.operation();
        ++it;

        TypeSwitch<Operation*>(curr)
            .Case<ExtractOp>([&](auto op) {
              const auto hw = layout.getHardwareIndex(prog);
              const auto qubit = staticOps[hw].getQubit();

              rewriter.replaceAllUsesWith(op.getResult(), qubit);
              rewriter.replaceAllUsesWith(op.getOutTensor(), op.getTensor());
              rewriter.eraseOp(op);

              placedWires[prog] = WireIterator(qubit);
              ++prog;
            })
            .Case<InsertOp>([&](auto op) {
              rewriter.setInsertionPointAfter(op);
              SinkOp::create(rewriter, op.getLoc(), op.getScalar());
              rewriter.replaceAllUsesWith(op.getResult(), op.getDest());
              rewriter.eraseOp(op);
            })
            .Case<DeallocOp>([&](auto op) { rewriter.eraseOp(op); });
      }

      rewriter.eraseOp(alloc);
    }

    // Create sinks for remaining, unused, static qubits.
    rewriter.setInsertionPoint(func.getFunctionBody().back().getTerminator());
    for (; prog < layout.nqubits(); ++prog) {
      const auto hw = layout.getHardwareIndex(prog);
      const auto qubit = staticOps[hw].getQubit();
      placedWires[prog] = WireIterator(qubit);
      SinkOp::create(rewriter, func->getLoc(), qubit);
    }

    return placedWires;
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
  FailureOr<size_t> refineLayout(SmallVector<WireIterator> wires,
                                 Layout& layout) {
    size_t nswaps{0};
    for (size_t i = 0; i < niterations; ++i) {
      if (failed(route<WireDirection::Forward>(wires, layout))) {
        return failure();
      }

      const auto resB = route<WireDirection::Backward>(wires, layout);
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
    constexpr size_t cap = 25'000'000UL;

    const size_t b = device.maxDegree() * ((device.nqubits() + 1) / 2);
    const size_t budget = std::min(b * b * b, cap);

    const Parameters params{.alpha = alpha, .lambda = lambda};

    llvm::SpecificBumpPtrAllocator<Node> arena;
    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    // Early exit, if the root node is a goal node already.
    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(window.front(), device)) {
      return SmallVector<IndexPairType>{};
    }

    frontier.emplace(root);

    DenseMap<ArrayRef<IndexType>, size_t> bestDepth;
    DenseSet<IndexPairType> expansionSet;

    size_t i = 0;
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

      if (curr->isGoal(window.front(), device)) {
        SmallVector<IndexPairType> seq(curr->depth);
        size_t j = seq.size() - 1;
        for (Node* n = curr; n->parent != nullptr; n = n->parent) {
          seq[j] = n->swap;
          --j;
        }

        return seq;
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighbouring hardware qubits.

      expansionSet.clear();
      const auto& [q0, q1] = window.front();
      for (const auto prog : {q0, q1}) {
        for (const auto hw0 = curr->layout.getHardwareIndex(prog);
             const auto hw1 : device.neighboursOf(hw0)) {
          // Ensure consistent hashing/comparison.
          const IndexPairType swap = std::minmax(hw0, hw1);
          if (!expansionSet.insert(swap).second) {
            continue;
          }

          frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                             window, device, params));
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
  template <WireDirection Direction>
  static void skipQubitPairBlock(WireIterator& w0, WireIterator& w1) {
    using Traits = WireTraversalTraits<Direction>;

    WireIterator curr0(w0);
    WireIterator curr1(w1);
    while (true) {
      while (Traits::isActive(curr0)) {
        std::ranges::advance(curr0, Traits::stride());
      }

      if (curr0 == std::default_sentinel) {
        return;
      }

      while (Traits::isActive(curr1)) {
        std::ranges::advance(curr1, Traits::stride());
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
  template <WireDirection Direction>
  Window getWindow(ArrayRef<WireIterator> baseWires) {
    Window window;
    window.reserve(1 + nlookahead);

    SmallVector<WireIterator> wires(baseWires);
    std::ignore = walkProgramGraph<Direction>(
        wires, [&](const ReadyRange& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          for (const auto& [op, progs] : ready) {
            if (isa<BarrierOp>(op)) {
              released.emplace_back(op);
              continue;
            }

            const auto p0 = progs[0];
            const auto p1 = progs[1];
            window.emplace_back(p0, p1);
            if (window.size() == 1 + nlookahead) {
              return WalkResult::interrupt();
            }

            skipQubitPairBlock<Direction>(wires[p0], wires[p1]);
            released.emplace_back(wires[p0].operation());
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
  template <WireDirection Direction>
  void skipExecutableGates(MutableArrayRef<WireIterator> wires,
                           Layout& layout) {
    std::ignore = walkProgramGraph<Direction>(
        wires, [&](const ReadyRange& ready, ReleasedOps& released) {
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

            if (device.areAdjacent(hw0, hw1)) {
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
  template <WireDirection Direction, RoutingMode mode = RoutingMode::Cold>
  FailureOr<size_t> route(SmallVector<WireIterator>& wires, Layout& layout,
                          IRRewriter* rewriter = nullptr) {
    using Traits = WireTraversalTraits<Direction>;

    size_t nswaps{0};
    while (true) {
      skipExecutableGates<Direction>(wires, layout);

      const auto window = getWindow<Direction>(wires);
      if (window.empty()) {
        break;
      }

      if constexpr (mode == RoutingMode::Hot) {

        // At this point the wire iterators either point to
        // std::default_sentinel or a multi-qubit gate (including barriers) of
        // the current or subsequent layers. The former must be decremented
        // twice (sentinel -> sink -> unitary/static). For the latter we simply
        // must ensure the insertion point is before the multi-qubit gates.

        for (auto& it : wires) {
          std::ranges::advance(it, it == std::default_sentinel
                                       ? -2 * Traits::stride()
                                       : -Traits::stride());
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

        // After SWAP insertion, a wire is either untouched by the SWAP
        // insertion or pointing at a SWAP operation. If the former is the case,
        // incrementing the wire iterator will undo the previous decrement,
        // leaving it at the same position as before the SWAP insertion.
        // Otherwise, an increment will move the iterator to the multi-qubit op
        // of the current or subsequent layer or to a sink (and thus
        // std::default_sentinel).

        for_each(wires,
                 [](auto& it) { std::ranges::advance(it, Traits::stride()); });
      }

      nswaps += swaps->size();
    }

    return nswaps;
  }

  AugmentedDevice device;
};

} // namespace

std::unique_ptr<Pass> createMappingPass(size_t nqubits, const Edges& coupling,
                                        MappingPassOptions options) {
  return std::make_unique<MappingPass>(nqubits, coupling, options);
}

} // namespace mlir::qco
