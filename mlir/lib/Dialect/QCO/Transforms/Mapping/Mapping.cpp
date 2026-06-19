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
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/Graph.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Threading.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
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

    explicit AugmentedDevice(
        const llvm::DenseSet<std::pair<size_t, size_t>>& couplingSet)
        : coupling_(couplingSet), dist_(coupling_.getDistMatrix()) {}

    /**
     * @returns the device's number of qubits.
     */
    [[nodiscard]] size_t nqubits() const { return coupling_.getNumNodes(); }

    /**
     * @returns true if @p u and @p v are adjacent.
     */
    [[nodiscard]] bool areAdjacent(size_t u, size_t v) const {
      return dist_[u][v] == 1UL;
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
      return coupling_.getEdges(u);
    }

    /**
     * @returns the qubit identifiers.
     */
    [[nodiscard]] ArrayRef<size_t> qubits() const {
      return coupling_.getNodes();
    }

    /**
     * @returns the links of the device.
     */
    [[nodiscard]] llvm::DenseSet<std::pair<size_t, size_t>> links() const {
      return coupling_.getEdges();
    }

    /**
     * @returns the max degree (connectivity) of any qubit of the device.
     */
    [[nodiscard]] size_t maxDegree() const { return coupling_.getMaxDegree(); }

  private:
    Graph<GraphType::Undirected, size_t> coupling_;
    Matrix<size_t> dist_;
  };

  /// Statistics collected while routing.
  struct Statistics {
    size_t nswaps{0};
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
  /// Construct default mapping pass.
  MappingPass() = default;
  /// Construct default mapping pass with options.
  explicit MappingPass(MappingPassOptions options) : MappingPassBase(options) {}
  /// Construct mapping from coupling set.
  explicit MappingPass(
      const llvm::DenseSet<std::pair<size_t, size_t>>& couplingSet,
      MappingPassOptions options = {})
      : MappingPassBase(options), device(couplingSet) {}

protected:
  void runOnOperation() override {
    assert(alpha > 0 && "runOnOperation: expected alpha > 0");
    assert(niterations > 0 && "runOnOperation: expected niterations > 0");
    assert(ntrials > 0 && "runOnOperation: expected ntrials > 0");

    IRRewriter rewriter(&getContext());

    auto mod = getOperation();
    auto func = getEntryPoint(mod);
    if (!func) {
      mod.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    auto& body = func.getFunctionBody();

    auto wires = getComputation(func);
    if (failed(wires)) {
      signalPassFailure();
      return;
    }

    if (wires->size() > device.nqubits()) {
      func.emitError()
          << "requires " + Twine(wires.value().size()) +
                 " qubits. However, the architecture only supports " +
                 Twine(device.nqubits()) + "qubits.";
      signalPassFailure();
      return;
    }

    auto layout = generateLayout(*wires);
    if (failed(layout)) {
      func->emitError() << "failed to refine random initial layouts.";
      signalPassFailure();
    }

    wires = std::move(place(body, *layout, rewriter));

    Statistics s;

    const auto res = route<WireDirection::Forward, RoutingMode::Hot>(
        *wires, *layout, s, &rewriter);
    if (res.failed()) {
      func.emitError() << "failed to map the function";
      signalPassFailure();
      return;
    }

    func->dumpPretty();

    // Collect statistics.
    numSwaps += s.nswaps;

    // Fix SSA Dominance issues.
    llvm::for_each(func.getFunctionBody().getBlocks(),
                   [](Block& b) { sortTopologically(&b); });
  }

private:
  static scf::ForOp extend(scf::ForOp loop, ValueRange addons,
                           IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loop);

    const auto naddons = addons.size();

    SmallVector<Value> inits;
    llvm::append_range(inits, loop.getInits());
    llvm::append_range(inits, addons);

    auto newLoop = rewriter.create<scf::ForOp>(
        loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), inits);

    Block* loopBody = loop.getBody();
    Block* newLoopBody = newLoop.getBody();

    rewriter.mergeBlocks(
        loopBody, newLoopBody,
        newLoopBody->getArguments().take_front(loopBody->getNumArguments()));

    for (const auto [before, after] :
         llvm::zip_first(loop.getResults(), newLoop.getResults())) {
      rewriter.replaceAllUsesWith(before, after);
    }

    for (const auto [before, after] :
         llvm::zip_equal(addons, newLoop.getResults().take_back(naddons))) {
      rewriter.replaceAllUsesExcept(before, after, newLoop);
    }

    auto yield = cast<scf::YieldOp>(newLoopBody->getTerminator());

    SmallVector<Value> results;
    llvm::append_range(results, yield.getResults());
    llvm::append_range(results, newLoop.getRegionIterArgs().take_back(naddons));
    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, results);

    rewriter.eraseOp(loop);

    return newLoop;
  }

  /**
   * @brief Collect wires of the quantum computation before placement.
   * @details
   * The mapping pass currently assumes that the quantum computation allocates
   * all tensors at the start of the function. The required qubits are extracted
   * from these tensors and used for the computation. Finally, the qubits are
   * inserted back into the tensors at the end of the function.
   * Thus, a valid program has the following structure:
   *
   *    T ⨉ [qtensor::AllocOp]
   *  → N ⨉ [qtensor::ExtractOp]
   *  → (Computation)
   *  → N ⨉ [qtensor::InsertOp]
   *  → T ⨉ [qtensor::DeallocOp]
   *
   * @returns a vector of wire iterator, or failure() if any of the above
   * assumptions are violated.
   */
  static FailureOr<SmallVector<WireIterator>>
  getComputation(func::FuncOp func) {
    if (!func.getOps<AllocOp>().empty()) {
      return func.emitError() << "must not contain qco.alloc operations";
    }

    SmallVector<WireIterator> wires;
    for (auto tensor : func.getOps<qtensor::AllocOp>()) {
      bool isInitPhase = true;
      TensorIterator it(tensor.getResult());
      for (; it != std::default_sentinel; ++it) {
        if (auto extract = dyn_cast<ExtractOp>(it.operation())) {
          if (!isInitPhase) {
            return func.emitError()
                   << "must extract and insert all qubits at once.";
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
  static SmallVector<WireIterator> place(Region& body, const Layout& layout,
                                         IRRewriter& rewriter) {
    SmallVector<StaticOp> staticOps;
    staticOps.reserve(layout.nqubits());

    // Create and save static qubit operations.
    rewriter.setInsertionPointToStart(&body.front());
    for (size_t i = 0; i < layout.nqubits(); ++i) {
      const auto op = StaticOp::create(rewriter, body.getLoc(), i);
      staticOps.emplace_back(op);
      rewriter.setInsertionPointAfter(op);
    }

    // Replace extract ops and collect in program-qubit order.
    SmallVector<WireIterator> wires(layout.nqubits());

    size_t prog = 0UL;
    for (auto alloc : make_early_inc_range(body.getOps<qtensor::AllocOp>())) {
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

              wires[prog] = WireIterator(qubit);
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
    rewriter.setInsertionPoint(body.back().getTerminator());
    for (; prog < layout.nqubits(); ++prog) {
      const auto hw = layout.getHardwareIndex(prog);
      const auto qubit = staticOps[hw].getQubit();
      wires[prog] = WireIterator(qubit);
      SinkOp::create(rewriter, body.getLoc(), qubit);
    }

    // Finally, update the SCF operations such that they take all static qubits
    // as input. To handle recursively nested SCF operations, use a stack of
    // (region, mapping) pairs.

    SmallVector<std::pair<Region&, DenseSet<Value>>> stack;
    stack.emplace_back(body, DenseSet<Value>{});

    while (!stack.empty()) {
      auto [region, m] = stack.pop_back_val();

      for (Operation& op : llvm::make_early_inc_range(region.getOps())) {
        TypeSwitch<Operation*>(&op)
            .Case<StaticOp>([&](StaticOp op) { m.insert(op.getQubit()); })
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface& op) {
              for (const auto [pred, succ] :
                   llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
                m.insert(succ);
                m.erase(pred);
              }
            })
            .Case<scf::ForOp>([&](scf::ForOp loop) {
              assert(m.size() == layout.nqubits());

              DenseSet<Value> addons(m);
              llvm::for_each(loop.getInits(), [&](auto v) { addons.erase(v); });
              auto newLoop = extend(loop, to_vector(addons), rewriter);

              for (OpOperand& operand : newLoop.getInitsMutable()) {
                m.insert(newLoop.getTiedLoopResult(&operand));
                m.erase(operand.get());
              }

              stack.emplace_back(
                  newLoop.getRegion(),
                  DenseSet<Value>(newLoop.getRegionIterArgs().begin(),
                                  newLoop.getRegionIterArgs().end()));
            })
            .Case<ResetOp, MeasureOp>([&](auto op) {
              m.insert(op.getQubitOut());
              m.erase(op.getQubitIn());
            })
            .Case<AllocOp, qtensor::AllocOp>([&](auto) {
              llvm::reportFatalInternalError("unexpected dynamic qubit alloc");
            });
      }
    }

    return wires;
  }

  /// Execute `ntrials` many (parallel) initial layout refinement trials and
  /// return the heuristically best one.
  ///
  /// The function uses the SABRE Approach to improve the initial layout:
  /// Traverse the layers of the program from left-to-right-to-left and
  /// cold-route along the way. Repeat this procedure "niterations" times and
  /// finally find the trial with the fewest SWAPs on the final backwards pass
  /// and return the respective layout.
  FailureOr<Layout> generateLayout(ArrayRef<WireIterator> wires) {
    std::mt19937_64 rng{seed};

    struct Trial {
      Layout layout;
      Statistics stats{};
      bool success{false};
    };

    SmallVector<Trial> trials;
    trials.reserve(ntrials);
    for (size_t i = 0; i < ntrials; ++i) {
      trials.emplace_back(Layout::random(device.nqubits(), rng()));
    }

    parallelForEach(&getContext(), trials, [&, this](Trial& t) {
      SmallVector<WireIterator> local(wires);
      for (size_t i = 0; i < niterations; ++i) {
        if (route<WireDirection::Forward>(local, t.layout, t.stats).failed()) {
          return;
        }
        if (route<WireDirection::Backward>(local, t.layout, t.stats).failed()) {
          return;
        }
      }
      t.success = true;
    });

    Trial* best = nullptr;
    for (Trial& t : trials) {
      if (t.success &&
          (best == nullptr || best->stats.nswaps > t.stats.nswaps)) {
        best = &t;
      }
    }

    if (best == nullptr) {
      return failure();
    }

    return best->layout;
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
  FailureOr<SmallVector<IndexPairType>> search(const Window& window,
                                               const Layout& layout) {
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

      // If the currently visited node is a goal node, reconstruct the
      // sequence of SWAPs from this node to the root.

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

      if (curr0.operation() == nullptr) {
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
   * reached. Assumes that wires[i] = i-th program qubit. The size of the
   * window is 1 + nlookahead.
   * @returns window of layers.
   */
  template <WireDirection Direction>
  Window getWindow(ArrayRef<WireIterator> wires) {
    Window window;
    window.reserve(1 + nlookahead);

    SmallVector<WireIterator> local(wires);
    walkProgramGraph<Direction>(
        local, [&](const ReadyRange& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          for (const auto& [op, progs] : ready) {
            if (auto u = dyn_cast<UnitaryOpInterface>(op)) {
              const auto p0 = progs[0];
              const auto p1 = progs[1];
              window.emplace_back(p0, p1);
              if (window.size() == 1 + nlookahead) {
                return WalkResult::interrupt();
              }

              skipQubitPairBlock<Direction>(local[p0], local[p1]);
              released.emplace_back(u);
              return WalkResult::advance();
            }

            released.emplace_back(op);
            return WalkResult::advance();
          }

          return WalkResult::advance();
        });

    return window;
  }

  SmallVector<IndexPairType> restore(const Layout& from, const Layout& to) {
    SmallVector<IndexPairType> swaps;

    Layout curr(from);

    Graph<GraphType::Directed, size_t> g;
    const auto constructEdge = [&](size_t hwX, size_t hwY) {
      const auto prog = curr.getProgramIndex(hwX);

      const auto hwGoal = to.getHardwareIndex(prog);
      const auto distPre = device.distanceBetween(hwX, hwGoal);
      const auto distPost = device.distanceBetween(hwY, hwGoal);

      if (distPost < distPre) {
        llvm::dbgs() << "prog=" << prog << " edge=(" << hwX << ", " << hwY
                     << ")"
                     << " dist(pre)=" << distPre << " dist(post)=" << distPost
                     << '\n';
        g.addEdge(hwX, hwY);
      }
    };

    do {
      // Construct 'F' graph.
      g.clear();
      for (const auto& [hwA, hwB] : device.links()) {
        constructEdge(hwA, hwB);
        constructEdge(hwB, hwA);
      }

      // Find happy swap chain or unhappy swap.
      if (const auto cycle = g.findCycle(); cycle) {
        // Apply happy SWAP chain.
        for (size_t i = 0; i < cycle->size() - 1; ++i) {
          llvm::dbgs() << "happySWAP=(" << (*cycle)[i] << ", "
                       << (*cycle)[i + 1] << ")\n";
          curr.swap((*cycle)[i], (*cycle)[i + 1]);
          swaps.emplace_back((*cycle)[i], (*cycle)[i + 1]);
        }

        continue;
      }

      for (const auto e : g.getEdges()) {
        if (g.getDegree(e.second) == 0) {
          llvm::dbgs() << "unhappySWAP=(" << e.first << ", " << e.second
                       << ")\n";
          curr.swap(e.first, e.second);
          swaps.emplace_back(e.first, e.second);
          break;
        }
      }
    } while (!g.empty());

    return swaps;
  }

  /**
   * @brief Advance past all executable gates and recurse into nested regions,
   * if necessary.
   * @details Traverses the multi-qubit gates of the circuit until no more
   * executable gates are found.
   */
  template <WireDirection Direction, RoutingMode Mode>
  void advanceAndRecurse(MutableArrayRef<WireIterator> wires, Layout& layout,
                         Statistics& stats, IRRewriter* rewriter) {
    walkProgramGraph<Direction>(wires, [&](const ReadyRange& ready,
                                           ReleasedOps& released) {
      if (ready.empty()) {
        return WalkResult::advance();
      }

      for (const auto& [readyOp, progs] : ready) {
        TypeSwitch<Operation*>(readyOp)
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
              const auto [hw0, hw1] =
                  layout.getHardwareIndices(progs[0], progs[1]);

              if (device.areAdjacent(hw0, hw1)) {
                released.emplace_back(op);
              }
            })
            .template Case<BarrierOp>(
                [&](BarrierOp op) { released.emplace_back(op); })
            .template Case<scf::ForOp>([&](scf::ForOp op) {
              // TODO: Don't ignore result here.
              std::ignore = route<Direction, Mode>(op, layout, stats, rewriter);

              released.emplace_back(op);
            });
      }

      // Stop, if there are no more ready AND executable gates.
      if (released.empty()) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
  }

  /// Insert SWAP operations, exchanging two qubits, virtually (Mode::Cold) or
  /// into the IR (Mode::Hot). The function expects that the i-th wire points
  /// at the i-th program qubit and that each wire also points at the correct
  /// insertion point. The function preserves the program qubit order, by
  /// exchanging wires after a SWAP.
  template <RoutingMode Mode>
  static void insertSWAPs(ArrayRef<IndexPairType> swaps,
                          MutableArrayRef<WireIterator> wires, Layout& layout,
                          IRRewriter* rewriter) {
    for (const auto& [hw0, hw1] : swaps) {
      if constexpr (Mode == RoutingMode::Hot) {
        const auto& [prog0, prog1] = layout.getProgramIndices(hw0, hw1);
        const auto& w0 = wires[prog0];
        const auto& w1 = wires[prog1];

        const auto in0 = w0.qubit();
        const auto in1 = w1.qubit();

        rewriter->setInsertionPointAfterValue(in0); // Valid bc. Hot => Forward.
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
  }

  /**
   * @brief Route via SWAP insertion.
   * @details Iterates over a dynamically computed window of layers and uses
   * A* search to find a sequence of SWAPs that makes that layer executable.
   * Depending on the template parameter, this function only updates
   * (and hence modifies) the layout or also inserts the SWAPs into the IR.
   * @returns failure() if A* search isn't able to find a solution.
   */
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  LogicalResult route(SmallVector<WireIterator>& wires, Layout& layout,
                      Statistics& stats, IRRewriter* rewriter = nullptr) {
    while (true) {
      advanceAndRecurse<Direction, Mode>(wires, layout, stats, rewriter);

      const auto window = getWindow<Direction>(wires);
      if (window.empty()) {
        break;
      }

      const auto swaps = search(window, layout);
      if (failed(swaps)) {
        return failure();
      }

      if constexpr (Mode == RoutingMode::Hot) {

        // At this point the wire iterators either point to
        // std::default_sentinel or a multi-qubit gate (including barriers) of
        // the current or subsequent layers. The former must be decremented
        // twice (sentinel -> sink -> unitary/static). For the latter we
        // simply must ensure the insertion point is before the multi-qubit
        // gates.

        for (auto& it : wires) {
          std::ranges::advance(it, it == std::default_sentinel ? -2 : -1);
        }
      }

      insertSWAPs<Mode>(*swaps, wires, layout, rewriter);

      if constexpr (Mode == RoutingMode::Hot) {

        // After SWAP insertion, a wire is either untouched by the SWAP
        // insertion or pointing at a SWAP operation. If the former is the
        // case, incrementing the wire iterator will undo the previous
        // decrement, leaving it at the same position as before the SWAP
        // insertion. Otherwise, an increment will move the iterator to the
        // multi-qubit op of the current or subsequent layer or to a sink (and
        // thus std::default_sentinel).

        llvm::for_each(wires, [](auto& it) { std::ranges::advance(it, 1); });
      }

      stats.nswaps += swaps->size();
    }

    return success();
  }

  template <WireDirection Direction, RoutingMode Mode>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  LogicalResult route(scf::ForOp op, Layout& base, Statistics& stats,
                      IRRewriter* rewriter) {

    assert(llvm::all_of(op.getInitArgs(),
                        [](Value v) { return isa<QubitType>(v.getType()); }));

    // In the forward direction we start the block arguments of the loop body,
    // whereas in the backward direction we start at the yielded values.

    SmallVector<WireIterator> wires;
    if constexpr (Direction == WireDirection::Forward) {
      llvm::for_each(op.getRegionIterArgs(),
                     [&](Value v) { wires.emplace_back(v); });
    } else {
      auto yield = cast<scf::YieldOp>(op.getBody()->getTerminator());
      assert(yield != nullptr);
      llvm::for_each(yield.getResults(),
                     [&](Value v) { wires.emplace_back(v); });
    }

    Layout remote(base);
    if (route<Direction, Mode>(wires, remote, stats, rewriter).failed()) {
      return failure();
    }

    if constexpr (Mode == RoutingMode::Hot) {
      assert(llvm::all_of(
          wires, [](const auto& it) { return it == std::default_sentinel; }));
      llvm::for_each(wires, [](auto& it) { std::ranges::advance(it, -2); });
    }

    const auto swaps = restore(remote, base);
    insertSWAPs<Mode>(swaps, wires, remote, rewriter);
    stats.nswaps += swaps.size();

    if constexpr (Mode == RoutingMode::Hot) {
      sortTopologically(op.getBody());
    }

    return success();
  }

  AugmentedDevice device;
};

} // namespace

std::unique_ptr<Pass>
createMappingPass(const llvm::DenseSet<std::pair<size_t, size_t>>& couplingSet,
                  MappingPassOptions options) {
  return std::make_unique<MappingPass>(couplingSet, options);
}

} // namespace mlir::qco
