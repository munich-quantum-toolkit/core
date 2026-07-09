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
#include "mlir/Dialect/QCO/Utils/Layout.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Threading.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mapping-pass"

namespace mlir::qco {

using namespace mlir::qtensor;
using namespace mlir::utils;

#define GEN_PASS_DEF_MAPPINGPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using IndexPairType = std::pair<size_t, size_t>;
  using Window = SmallVector<IndexPairType>;
  using Wires = SmallVector<WireIterator>;

  enum class RoutingMode : bool { Cold, Hot };

  class AugmentedDevice {
  public:
    explicit AugmentedDevice(
        const llvm::DenseSet<std::pair<size_t, size_t>>& couplingSet)
        : coupling_(couplingSet), dist_(coupling_.getDistMatrix()) {}

    /// Return the device's number of qubits.
    [[nodiscard]] size_t nqubits() const { return coupling_.getNumNodes(); }

    /// Return true if two qubits are adjacent.
    [[nodiscard]] bool areAdjacent(size_t u, size_t v) const {
      return dist_[u][v] == 1UL;
    }

    /// Return the length of the shortest path between two qubits.
    [[nodiscard]] size_t distanceBetween(size_t u, size_t v) const {
      const auto dist = dist_[u][v];
      if (dist == UINT64_MAX) {
        report_fatal_error("Failed to compute the distance between qubits " +
                           Twine(u) + " and " + Twine(v));
      }
      return dist;
    }

    /// Return the qubit identifiers.
    [[nodiscard]] SmallVector<size_t> qubits() const {
      return coupling_.getNodes();
    }

    /// Return all neighbours of a qubit.
    [[nodiscard]] ArrayRef<size_t> neighboursOf(size_t u) const {
      return coupling_.getNeighbours(u);
    }

    /// Return the max degree (connectivity) of any qubit of the device.
    [[nodiscard]] size_t maxDegree() const { return coupling_.getMaxDegree(); }

  private:
    Graph coupling_;
    Graph::DistanceMatrix dist_;
  };

  struct WireInfos {
    /// Return the mapped wire index of a program index.
    [[nodiscard]] size_t lookupIndex(size_t prog) const {
      return programToIndex_.at(prog);
    }

    /// Return the mapped program index of a wire index.
    [[nodiscard]] size_t lookupProgram(size_t index) const {
      return indexToProgram_.at(index);
    }

    /// Bidirectionally map a wire index to a program index.
    /// Overwrites existing mappings.
    void map(size_t index, size_t prog) {
      indexToProgram_[index] = prog;
      programToIndex_[prog] = index;
    }

    /// Swap two program indices.
    void swap(size_t prog0, size_t prog1) {
      const auto i0 = lookupIndex(prog0);
      const auto i1 = lookupIndex(prog1);
      std::swap(programToIndex_[prog0], programToIndex_[prog1]);
      std::swap(indexToProgram_[i0], indexToProgram_[i1]);
    }

  private:
    /// Maps the i-th wire index to a program index.
    DenseMap<size_t, size_t> indexToProgram_;
    /// Maps a program index to the i-th wire index.
    DenseMap<size_t, size_t> programToIndex_;
  };

  /// Statistics collected while routing.
  struct Statistics {
    size_t nswaps{0};
  };

  /// Parameters influencing the behavior of the A* search algorithm.
  struct Parameters {
    float alpha;
    float lambda;
  };

  /// Utility-struct for routing functions.
  struct RoutingBundle {
    Wires wires;
    WireInfos infos;
    Layout layout;
  };

  /// Describes a node in the A* search graph.
  struct Node {
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

    /// Construct a root node with the given layout. Initialize the
    /// sequence with an empty vector and set the cost to zero.
    explicit Node(Layout layout)
        : layout(std::move(layout)), parent(nullptr), depth(0), f(0) {}

    /// Construct a non-root node from its parent node. Apply the given swap to
    /// the layout of the parent node.
    Node(Node* parent, const IndexPairType& swap, const Window& window,
         const AugmentedDevice& device, const Parameters& params)
        : layout(parent->layout), swap(swap), parent(parent),
          depth(parent->depth + 1), f(0) {
      layout.swap(swap.first, swap.second);
      f = g(params.alpha) + h(window, device, params); // NOLINT
    }

    /// Return true, if the current SWAP sequence makes all gates in the front
    /// executable.
    [[nodiscard]] bool isGoal(const IndexPairType& front,
                              const AugmentedDevice& device) const {
      const auto [hw0, hw1] =
          layout.getHardwareIndices(front.first, front.second);
      return device.areAdjacent(hw0, hw1);
    }

  private:
    /// Calculate the path cost for the A* search algorithm.
    /// The path costs are the weighted sum of the currently required SWAPs.
    [[nodiscard]] float g(const float alpha) const {
      return alpha * static_cast<float>(depth);
    }

    /// Calculate the heuristic cost for the A* search algorithm.
    ///
    /// Computes the minimal number of SWAPs required to route each gate in
    /// each layer. For each gate, this is determined by the shortest distance
    /// between its hardware qubits. Intuitively, this is the number of SWAPs
    /// that a naive router would insert to route the layers (with a constant
    /// layout).
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
      MappingPassOptions options)
      : MappingPassBase(options),
        device(std::make_shared<AugmentedDevice>(couplingSet)) {}

protected:
  void runOnOperation() override {
    assert(alpha > 0 && "expected alpha > 0");
    assert(niterations > 0 && "expected niterations > 0");
    assert(ntrials > 0 && "expected ntrials > 0");

    if (!device) {
      llvm::reportFatalUsageError("No device specified!");
    }

    IRRewriter rewriter(&getContext());

    auto mod = getOperation();
    auto func = getEntryPoint(mod);
    if (!func) {
      mod.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    auto comp = getComputation(func);
    if (failed(comp)) {
      signalPassFailure();
      return;
    }

    auto& body = func.getFunctionBody();
    auto& [wires, infos] = *comp;

    if (wires.size() > device->nqubits()) {
      func.emitError()
          << "requires " + Twine(wires.size()) +
                 " qubits. However, the architecture only supports " +
                 Twine(device->nqubits()) + "qubits.";
      signalPassFailure();
      return;
    }

    auto layout = generateLayout(wires, infos);
    if (failed(layout)) {
      func->emitError() << "failed to refine random initial layouts.";
      signalPassFailure();
      return;
    }

    std::tie(wires, infos) = std::move(place(body, *layout, rewriter));

    Statistics stats;
    RoutingBundle bundle{.wires = std::move(wires),
                         .infos = std::move(infos),
                         .layout = std::move(*layout)};

    const auto res = route<WireDirection::Forward, RoutingMode::Hot>(
        bundle, stats, &rewriter);
    if (res.failed()) {
      func.emitError() << "failed to map the function";
      signalPassFailure();
      return;
    }

    // Collect statistics.
    numSwaps += stats.nswaps;

    // Fix SSA Dominance issues.
    llvm::for_each(body.getBlocks(), [](Block& b) { sortTopologically(&b); });
  }

private:
  /// Extend the init arguments of an `scf::ForOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static scf::ForOp extend(scf::ForOp forOp, ValueRange addons,
                           IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);

    const auto naddons = addons.size();
    const auto res =
        forOp.replaceWithAdditionalIterOperands(rewriter, addons, true);
    assert(succeeded(res));

    auto newForOp = cast<scf::ForOp>(*res);
    for (const auto [oldUse, newResult] :
         llvm::zip_equal(addons, newForOp.getResults().take_back(naddons))) {
      rewriter.replaceAllUsesExcept(oldUse, newResult, newForOp);
    }
    return newForOp;
  }

  /// Return the wires of a dynamic computation.
  /// The mapping pass currently assumes that
  /// - there are no `qco.alloc` operation
  /// - there is an "extraction" and "insertion" phase, where the i-th extract
  ///   defines the i-th program qubit
  /// Thus, supported programs have the following structure:
  ///
  ///   T ⨉ [qtensor::AllocOp]
  /// → N ⨉ [qtensor::ExtractOp]
  /// → (Computation)
  /// → N ⨉ [qtensor::InsertOp]
  /// → T ⨉ [qtensor::DeallocOp]
  ///
  /// If any of the above assumptions are violated, the function returns
  /// failure.
  static FailureOr<std::pair<Wires, WireInfos>>
  getComputation(func::FuncOp func) {
    if (!func.getOps<AllocOp>().empty()) {
      return func.emitError() << "must not contain qco.alloc operations";
    }

    Wires wires;
    WireInfos infos;

    for (auto alloc : func.getOps<qtensor::AllocOp>()) {
      bool isInitPhase = true;
      TensorIterator it(alloc.getResult());
      for (; it != std::default_sentinel; ++it) {
        if (auto extract = dyn_cast<ExtractOp>(it.operation())) {
          if (!isInitPhase) {
            return func.emitError()
                   << "must extract and insert all qubits at once.";
          }

          const auto qubit = extract.getResult();
          const auto index = wires.size();

          wires.emplace_back(qubit);
          infos.map(index, index);

          continue;
        }

        if (isa<InsertOp>(it.operation())) {
          isInitPhase = false;
          continue;
        }
      }
    }

    return std::make_pair(wires, infos);
  }

  /// Perform placement by
  /// - initializing as many hardware qubits as the architecture supports
  /// - replacing dynamic with static qubits
  /// - extending the inputs of `scf::ForOp` to all hardware qubits.
  ///
  /// Analogously to the getComputation function, the i-th extract
  /// operation defines the i-th program qubit.
  static std::pair<Wires, WireInfos> place(Region& body, const Layout& layout,
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

    Wires wires;
    WireInfos infos;

    for (auto alloc :
         llvm::make_early_inc_range(body.getOps<qtensor::AllocOp>())) {
      TensorIterator it(alloc.getResult());
      while (it != std::default_sentinel) {
        // Get the operation and early increment to avoid issues after erasure.
        Operation* curr = it.operation();
        ++it;

        TypeSwitch<Operation*>(curr)
            .Case<ExtractOp>([&](auto op) {
              const auto prog = wires.size();
              const auto hw = layout.getHardwareIndex(prog);
              const auto qubit = staticOps[hw].getQubit();

              rewriter.replaceAllUsesWith(op.getResult(), qubit);
              rewriter.replaceAllUsesWith(op.getOutTensor(), op.getTensor());
              rewriter.eraseOp(op);

              wires.emplace_back(qubit);
              infos.map(prog, prog);
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
    for (size_t prog = wires.size(); prog < layout.nqubits(); ++prog) {
      const auto hw = layout.getHardwareIndex(prog);
      const auto qubit = staticOps[hw].getQubit();

      wires.emplace_back(qubit);
      infos.map(prog, prog);

      SinkOp::create(rewriter, body.getLoc(), qubit);
    }

    // Finally, update the SCF operations such that they take all static qubits
    // as input. To handle recursively nested SCF operations, use a stack of
    // (region, mapping) pairs.

    SmallVector<std::pair<Region&, DenseSet<Value>>> stack;
    stack.emplace_back(body, DenseSet<Value>{});

    while (!stack.empty()) {
      auto [region, qubits] = stack.pop_back_val();

      for (Operation& op : llvm::make_early_inc_range(region.getOps())) {
        TypeSwitch<Operation*>(&op)
            .Case<StaticOp>([&](StaticOp op) { qubits.insert(op.getQubit()); })
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface& op) {
              for (const auto [pred, succ] :
                   llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
                qubits.insert(succ);
                qubits.erase(pred);
              }
            })
            .Case<scf::ForOp>([&](scf::ForOp loop) {
              assert(qubits.size() == layout.nqubits());

              DenseSet<Value> addons(qubits);
              llvm::for_each(loop.getInits(), [&](auto v) { addons.erase(v); });
              auto newLoop = extend(loop, to_vector(addons), rewriter);

              for (OpOperand& operand : newLoop.getInitsMutable()) {
                qubits.insert(newLoop.getTiedLoopResult(&operand));
                qubits.erase(operand.get());
              }

              stack.emplace_back(
                  newLoop.getRegion(),
                  DenseSet<Value>(newLoop.getRegionIterArgs().begin(),
                                  newLoop.getRegionIterArgs().end()));
            })
            .Case<ResetOp, MeasureOp>([&](auto op) {
              qubits.insert(op.getQubitOut());
              qubits.erase(op.getQubitIn());
            })
            .Case<AllocOp, qtensor::AllocOp>([&](auto) {
              llvm::reportFatalInternalError("unexpected dynamic qubit alloc");
            });
      }
    }

    return {wires, infos};
  }

  /// Execute `ntrials` many (parallel) initial layout refinement trials and
  /// return the heuristically best one.
  ///
  /// The function uses the SABRE Approach to improve the initial layout:
  /// Traverse the layers of the program from left-to-right-to-left and
  /// cold-route along the way. Repeat this procedure "niterations" times and
  /// finally find the trial with the fewest SWAPs on the final backwards pass
  /// and return the respective layout.
  FailureOr<Layout> generateLayout(const Wires& wires, const WireInfos& infos) {
    std::mt19937_64 rng{seed};

    struct Trial {
      RoutingBundle bundle;
      Statistics stats{};
      bool success{false};
    };

    SmallVector<Trial, 0> trials;
    trials.reserve(ntrials);
    for (size_t i = 0; i < ntrials; ++i) {
      trials.emplace_back(
          RoutingBundle{.wires = wires,
                        .infos = infos,
                        .layout = Layout::random(device->nqubits(), rng())});
    }

    parallelForEach(&getContext(), trials, [&, this](Trial& t) {
      for (size_t i = 0; i < niterations; ++i) {
        if (route<WireDirection::Forward>(t.bundle, t.stats).failed()) {
          return;
        }
        t.stats.nswaps = 0;
        if (route<WireDirection::Backward>(t.bundle, t.stats).failed()) {
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

    return best->bundle.layout;
  }

  /// Perform A* search to find a sequence of SWAPs that makes all two-qubit ops
  /// inside the first layer executable.
  ///
  /// The iteration budget is b^{3} node expansions, i.e. roughly a depth-3
  /// search in a tree with branching factor b, where b is the product of the
  /// architecture's maximum qubit degree and the maximum number of two-qubit
  /// gates in any layer: `b = maxDegree × ⌈N/2⌉`. A hard cap prevents
  /// impractical runtimes on larger architectures.
  ///
  /// Returns `failure`, if the A* search fails.
  FailureOr<SmallVector<IndexPairType>> search(const Window& window,
                                               const Layout& layout) {
    constexpr size_t cap = 25'000'000UL;

    const size_t b = device->maxDegree() * ((device->nqubits() + 1) / 2);
    const size_t budget = std::min(b * b * b, cap);

    const Parameters params{.alpha = alpha, .lambda = lambda};

    llvm::SpecificBumpPtrAllocator<Node> arena;
    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    // Early exit, if the root node is a goal node already.
    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(window.front(), *device)) {
      return SmallVector<IndexPairType>{};
    }

    frontier.emplace(root);

    DenseMap<ArrayRef<size_t>, size_t> bestDepth;
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

      if (curr->isGoal(window.front(), *device)) {
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
             const auto hw1 : device->neighboursOf(hw0)) {
          // Ensure consistent hashing/comparison.
          const IndexPairType swap = std::minmax(hw0, hw1);
          if (!expansionSet.insert(swap).second) {
            continue;
          }

          frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                             window, *device, params));
        }
      }

      ++i;
    }

    return failure();
  }

  /// Return the sequence of SWAPs to move from one layout to another.
  /// Implements the 4-Approximation algorithm described in arXiv:1602.05150v3.
  SmallVector<IndexPairType> restore(const Layout& from, const Layout& to) {

    Layout curr(from);
    Graph f(device->qubits());
    SmallVector<IndexPairType> swaps;

    const auto shouldAddEdge = [&](size_t u, size_t v) {
      const auto prog = curr.getProgramIndex(u);
      const auto hwGoal = to.getHardwareIndex(prog);
      return device->distanceBetween(v, hwGoal) <
             device->distanceBetween(u, hwGoal);
    };

    while (true) {

      // Build F-graph: Add edges to F for each edge in the coupling graph.
      // Note that this assumes that the coupling graph is directed, but
      // symmetric (essentially: undirected).

      f.clearEdges();
      for (const auto u : device->qubits()) {
        for (const auto v : device->neighboursOf(u)) {
          if (shouldAddEdge(u, v)) {
            f.addEdge(u, v);
          }
        }
      }

      // Try to find a directed cycle in the F graph. If there is one,
      // we can apply a happy swap chain. Note that this happy swap chain
      // does not include the final back edge closing the cycle because the
      // first SWAP changes the token (the qubit) on the target, invalidating
      // the edge in F.

      if (const auto cycle = f.findCycle(); cycle) {
        for (size_t i = cycle->size() - 1; i > 0; --i) {
          curr.swap((*cycle)[i], (*cycle)[i - 1]);
          swaps.emplace_back((*cycle)[i], (*cycle)[i - 1]);
        }
        continue;
      }

      // Otherwise, search for an unhappy SWAP. That is, search for an edge (u,
      // v), where exchanging u and v, reduces u's distance to its target
      // location (by one) and increases v's distance from 0 (already at the
      // correct location) to one.

      bool found{false};
      for (const auto u : f.getNodes()) {
        for (const auto v : f.getNeighbours(u)) {
          if (f.getDegree(v) == 0) {
            curr.swap(u, v);
            swaps.emplace_back(u, v);
            found = true;
            break;
          }
        }

        if (found) {
          break;
        }
      }

      // If there are no happy or unhappy swaps anymore,
      // the final placement of every token is reached.

      if (!found) {
        break;
      }
    }

    return swaps;
  }

  /// Skip to the end of the two-qubit block for both wire iterators, where
  /// initially both must point at the same two-qubit operation.
  template <WireDirection Direction>
  static void skipQubitPairBlock(WireIterator& it0, WireIterator& it1) {
    using Traits = WireTraversalTraits<Direction>;

    // Traverses the pair of wire iterators in tandem until a two-qubit
    // operation is found. If the two-qubit operation is equivalent, continue.
    // Otherwise stop.

    std::array<WireIterator, 2> block{it0, it1};
    while (true) {
      for (auto& it : block) {
        while (Traits::isActive(it)) {
          std::ranges::advance(it, Traits::stride());

          if (it.operation() == nullptr) { // isa<Blockargument>
            return;
          }

          if (auto u = dyn_cast<UnitaryOpInterface>(it.operation());
              u && u.getNumQubits() > 1) {
            // Handle two-qubit barrier edge case explicitly.
            if (isa<BarrierOp>(u) && u.getNumQubits() != 2) {
              return;
            }
            // Otherwise stop for subsequent two-qubit unitary comparison.
            break;
          }
        }

        if (it == std::default_sentinel) {
          return;
        }
      }

      if (block[0].operation() != block[1].operation()) {
        return;
      }

      it0 = block[0];
      it1 = block[1];
    }
  }

  /// Return a window of layers with a maximum size of `1 + nlookahead`.
  template <WireDirection Direction>
  Window getWindow(Wires wires, const WireInfos& infos) {
    Window window;
    window.reserve(1 + nlookahead);

    walkProgramGraph<Direction>(
        wires, [&](const ReadyRange& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          for (const auto& [op, indices] : ready) {
            if (auto u = dyn_cast<UnitaryOpInterface>(op)) {
              const auto i0 = indices[0];
              const auto i1 = indices[1];

              const auto prog0 = infos.lookupProgram(i0);
              const auto prog1 = infos.lookupProgram(i1);

              window.emplace_back(prog0, prog1);
              if (window.size() == 1 + nlookahead) {
                return WalkResult::interrupt();
              }

              skipQubitPairBlock<Direction>(wires[i0], wires[i1]);
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

  /// Insert SWAP operations, exchanging two qubits, virtually
  /// (`RoutingMode::Cold`) or into the IR (`RoutingMode::Hot`). The function
  /// expects that each wire points at the correct insertion point.
  template <RoutingMode Mode>
  static void insertSWAPs(ArrayRef<IndexPairType> swaps, RoutingBundle& bundle,
                          Statistics& stats, IRRewriter* rewriter) {
    auto& [wires, infos, layout] = bundle;
    for (const auto& [hw0, hw1] : swaps) {
      if constexpr (Mode == RoutingMode::Hot) {
        const auto [prog0, prog1] = layout.getProgramIndices(hw0, hw1);

        const auto i0 = infos.lookupIndex(prog0);
        const auto i1 = infos.lookupIndex(prog1);

        auto& w0 = wires[i0];
        auto& w1 = wires[i1];

        const auto in0 = w0.qubit();
        const auto in1 = w1.qubit();

        rewriter->setInsertionPointAfterValue(in0); // Valid bc. Hot => Forward.
        auto swapOp = SWAPOp::create(*rewriter, in0.getLoc(), in0, in1);

        const auto out0 = swapOp.getQubit0Out();
        const auto out1 = swapOp.getQubit1Out();

        rewriter->replaceAllUsesExcept(in0, out1, swapOp);
        rewriter->replaceAllUsesExcept(in1, out0, swapOp);

        infos.swap(prog0, prog1);

        std::advance(w0, 1); // Move to SWAP.
        std::advance(w1, 1);
      }

      layout.swap(hw0, hw1);
    }

    stats.nswaps += swaps.size();
  }

  /// Advance past all executable gates and return operations with nested
  /// regions and the respective wire indices. Stops when no more executable
  /// gates are found. After the function returns, the wires point at the
  /// results of non-executable gates or operations with nested regions.
  template <WireDirection Direction>
  SmallVector<std::pair<Operation*, SmallVector<size_t>>>
  advance(Wires& wires, const WireInfos& infos, const Layout& layout) {
    SmallVector<std::pair<Operation*, SmallVector<size_t>>> stack;

    // Advance wires past all executable gates and push operations with
    // nested regions and the respective wire indices of their inputs onto the
    // result stack.

    walkProgramGraph<Direction>(wires, [&](const ReadyRange& ready,
                                           ReleasedOps& released) {
      if (ready.empty()) {
        return WalkResult::advance();
      }

      for (const auto& [readyOp, indices] : ready) {
        TypeSwitch<Operation*>(readyOp)
            .template Case<BarrierOp>(
                [&](BarrierOp op) { released.emplace_back(op); })
            .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
              const auto prog0 = infos.lookupProgram(indices[0]);
              const auto prog1 = infos.lookupProgram(indices[1]);
              const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
              if (device->areAdjacent(hw0, hw1)) {
                released.emplace_back(op);
              }
            })
            .template Case<scf::ForOp>(
                [&](scf::ForOp op) { stack.emplace_back(op, indices); });
      }

      if (released.empty()) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return stack;
  }

  /// Iterates over a dynamically computed window of layers and uses A* search
  /// to find a SWAP sequence that makes each layer executable. Depending on
  /// the template parameter, this function only updates the layout or also
  /// inserts the SWAPs into the IR. The function returns `failure` if A* is
  /// unable to find a solution.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  LogicalResult route(RoutingBundle& bundle, Statistics& stats,
                      IRRewriter* rewriter = nullptr) {
    using Traits = WireTraversalTraits<Direction>;

    auto& [wires, infos, layout] = bundle;

    while (true) {

      while (true) {
        const auto stack = advance<Direction>(wires, infos, layout);

        if (stack.empty()) {
          break;
        }

        // Continue with processing the nested regions recursively.

        for (const auto& [op, indices] : stack) {
          assert(isa<scf::ForOp>(op));
          auto forOp = cast<scf::ForOp>(op);

          RoutingBundle child{.layout = layout};

          // Map parent (results) to child values (iter args). Going forwards,
          // the recursive routing starts at block arguments, while the
          // backwards go starts at the yielded values.

          for (size_t i : indices) {
            const auto prog = infos.lookupProgram(i);
            const auto res = cast<OpResult>(wires[i].qubit());
            const auto arg = forOp.getTiedLoopRegionIterArg(res);
            const auto index = child.wires.size();

            if constexpr (Direction == WireDirection::Forward) {
              child.wires.emplace_back(arg);
              child.infos.map(index, prog);
            } else {
              const auto yield = forOp.getTiedLoopYieldedValue(arg)->get();
              child.wires.emplace_back(yield);
              child.infos.map(index, prog);
            }
          }

          const auto res = route<Direction, Mode>(child, stats, rewriter);
          if (failed(res)) {
            return failure();
          }

          const auto swaps = restore(child.layout, layout);

          if constexpr (Mode == RoutingMode::Hot) {

            // After routing the loop body, all iterators point to
            // std::default_sentinel. To move the iterators to the correct
            // qubit SSA values for the epilogue SWAPs, decrement each
            // twice: (sentinel → yield → unitary/block arg).

            llvm::for_each(child.wires, [](auto& it) { std::advance(it, -2); });
          }

          insertSWAPs<Mode>(swaps, child, stats, rewriter);

          if constexpr (Mode == RoutingMode::Hot) {
            sortTopologically(forOp.getBody());
          }

          // Finally, move past the operation with nested regions by
          // incrementing the respective global wires.

          llvm::for_each(indices, [&](size_t i) {
            std::advance(wires[i], Traits::stride());
          });
        }
      }

      const auto window = getWindow<Direction>(wires, infos);
      if (window.empty()) {
        break;
      }

      const auto swaps = search(window, layout);
      if (failed(swaps)) {
        return failure();
      }

      if constexpr (Mode == RoutingMode::Hot) {

        // At this point the wire iterators either point to
        // std::default_sentinel or a multi-qubit gate (incl. barriers) of
        // the current or subsequent layers. The former must be decremented
        // twice (sentinel → sink → unitary/static). For the latter, we
        // must ensure the insertion point is before the multi-qubit gates.

        for (auto& it : wires) {
          std::advance(it, it == std::default_sentinel ? -2 : -1);
        }
      }

      insertSWAPs<Mode>(*swaps, bundle, stats, rewriter);

      if constexpr (Mode == RoutingMode::Hot) {

        // After SWAP insertion, a wire is either untouched by the SWAP
        // insertion or pointing at a SWAP operation. If the former is the
        // case, incrementing the wire iterator will undo the previous
        // decrement, leaving it at the same position as before the SWAP
        // insertion. Otherwise, an increment will move the iterator to the
        // multi-qubit op of the current or subsequent layer or to a sink (and
        // thus std::default_sentinel).

        llvm::for_each(wires, [](auto& it) { std::advance(it, 1); });
      }
    }

    return success();
  }

  std::shared_ptr<AugmentedDevice> device;
};

} // namespace

std::unique_ptr<Pass>
createMappingPass(const llvm::DenseSet<std::pair<size_t, size_t>>& couplingSet,
                  MappingPassOptions options) {

  // Verify the assumption that the coupling set is symmetric:
  // For every edge (u, v) in the set, (v, u) must also be present.

  for (const auto& [u, v] : couplingSet) {
    if (u == v) {
      llvm::reportFatalUsageError("Found an invalid (u, u) edge.");
      return nullptr;
    }

    if (!couplingSet.contains({v, u})) {
      llvm::reportFatalUsageError("Expected symmetric coupling set: edge (" +
                                  Twine(u) + ", " + Twine(v) +
                                  ") exists but (" + Twine(v) + ", " +
                                  Twine(u) + ") does not.");
    }
  }

  return std::make_unique<MappingPass>(couplingSet, options);
}

} // namespace mlir::qco
