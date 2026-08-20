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

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
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
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
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
#include <mlir/IR/Region.h>
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
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
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
  using IndexPairType = std::pair<size_t, size_t>;
  using Window = SmallVector<IndexPairType>;
  using Wires = SmallVector<WireIterator>;

  enum class RoutingMode : bool { Cold, Hot };

  struct CompositeUnitary {
    /// The composite op (e.g. SCF).
    Operation* op = nullptr;
    /// Indices into a wire vector, where the order of indices has no meaning.
    SmallVector<size_t> indices;
  };

  struct WireInfos {
    /// Return the mapped wire index of a program index.
    [[nodiscard]] size_t lookupIndex(const size_t prog) const {
      assert(containsProgram(prog) && "program index is not mapped");
      return programToIndex_[prog];
    }

    /// Return the mapped program index of a wire index.
    [[nodiscard]] size_t lookupProgram(const size_t index) const {
      return indexToProgram_[index];
    }

    /// Bidirectionally map a wire index to a program index.
    /// Overwrites existing mappings.
    void insertOrUpdate(const size_t index, const size_t prog) {
      if (index >= indexToProgram_.size()) {
        indexToProgram_.resize(index + 1);
      }
      if (prog >= programToIndex_.size()) {
        programToIndex_.resize(prog + 1);
      }
      indexToProgram_[index] = prog;
      programToIndex_[prog] = index;
      programs_.insert(prog);
    }

    /// Return whether a program index has a corresponding wire.
    [[nodiscard]] bool containsProgram(const size_t prog) const {
      return programs_.contains(prog);
    }

    /// Swap two program indices.
    void swap(const size_t prog0, const size_t prog1) {
      const auto i0 = lookupIndex(prog0);
      const auto i1 = lookupIndex(prog1);
      std::swap(programToIndex_[prog0], programToIndex_[prog1]);
      std::swap(indexToProgram_[i0], indexToProgram_[i1]);
    }

    /// Return the number of index-wire mappings.
    [[nodiscard]] size_t size() const { return indexToProgram_.size(); }

  private:
    /// Maps the i-th wire index to a program index.
    SmallVector<size_t> indexToProgram_;
    /// Maps a program index to the i-th wire index.
    SmallVector<size_t> programToIndex_;
    /// Program indices that have corresponding wires.
    DenseSet<size_t> programs_;
  };

  struct TensorAllocation {
    qtensor::AllocOp allocation;
    SmallVector<Operation*> operations;
  };

  struct Computation {
    Wires wires;
    WireInfos infos;
    SmallVector<AllocOp> scalarAllocations;
    SmallVector<TensorAllocation> tensorAllocations;
    bool hasTwoQubitOperations{false};
  };

  /// Statistics collected while routing.
  struct Statistics {
    /// The number of inserted swaps.
    size_t nswaps{0};
    /// Merge another statistics object into this one.
    void merge(const Statistics& other) { nswaps += other.nswaps; }
  };

  /// Parameters influencing the behavior of the A* search algorithm.
  struct Parameters {
    /// The path weight.
    float alpha;
    /// The lookahead decay factor.
    float lambda;
  };

  /// Utility-struct for routing functions.
  struct RoutingBundle {
    Wires wires;
    WireInfos infos;
    Layout layout;

    struct Patch {
      std::optional<Layout> layout;
      std::optional<WireInfos> infos;
      std::optional<Wires> wires;
    };

    void applyPatch(Patch&& patch) {
      Patch p = std::move(patch);
      if (p.layout) {
        layout = std::move(*p.layout);
      }
      if (p.infos) {
        infos = std::move(*p.infos);
      }
      if (p.wires) {
        wires = std::move(*p.wires);
      }
    }
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
         const CompilerTarget& target, const Parameters& params)
        : layout(parent->layout), swap(swap), parent(parent),
          depth(parent->depth + 1), f(0) {
      layout.swap(swap.first, swap.second);
      f = g(params.alpha) + h(window, target, params); // NOLINT
    }

    /// Return true, if the current SWAP sequence makes all gates in the front
    /// executable.
    [[nodiscard]] bool isGoal(const IndexPairType& front,
                              const CompilerTarget& target) const {
      const auto [hw0, hw1] =
          layout.getHardwareIndices(front.first, front.second);
      return target.areAdjacent(hw0, hw1);
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
    [[nodiscard]] float h(const Window& window, const CompilerTarget& target,
                          const Parameters& params) const {
      float costs{0};
      float decay{1.};

      for (const auto& [i, progs] : enumerate(window)) {
        const auto [prog0, prog1] = progs;
        const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
        const size_t nswaps = target.distanceBetween(hw0, hw1) - 1;
        costs += decay * static_cast<float>(nswaps);
        decay *= params.lambda;
      }
      return costs;
    }
  };

  /// Describes the graph F of arXiv:1602.05150v3.
  struct FGraph {
    explicit FGraph(const CompilerTarget& target)
        : f_(llvm::to_vector(llvm::seq(target.numQubits()))),
          target_(&target) {};

    /// Build F-graph: Add edges to F for each edge in the coupling graph.
    /// Note that this assumes that the coupling graph is directed, but
    /// symmetric (essentially: undirected).
    void construct(const Layout& from, const Layout& to) {
      for (size_t u = 0; u < target_->numQubits(); ++u) {
        target_->forEachNeighbour(u, [&](const auto v) {
          if (shouldAddEdge(u, v, from, to)) {
            f_.addEdge(u, v);
          }
        });
      }
    }

    /// Try to find a directed cycle in the F graph. If there is one,
    /// we can apply a happy swap chain. Note that this happy swap chain
    /// does not include the final back edge closing the cycle because the
    /// first SWAP changes the token (the qubit) on the target, invalidating
    /// the edge in F.
    [[nodiscard]] std::optional<SmallVector<IndexPairType>>
    findHappySWAPChain() const {
      const auto optCycle = f_.findCycle();
      if (!optCycle) {
        return std::nullopt;
      }
      const auto& cycle = *optCycle;

      SmallVector<IndexPairType> swaps;
      for (size_t i = cycle.size() - 1; i > 0; --i) {
        swaps.emplace_back(cycle[i], cycle[i - 1]);
      }
      return swaps;
    }

    /// Find an unhappy SWAP. That is, find an edge (u, v), where exchanging u
    /// and v, reduces u's distance to its target location (by one) and
    /// increases v's distance from 0 (already at the correct location) to one.
    [[nodiscard]] std::optional<IndexPairType> findUnhappySWAP() const {
      for (const auto u : f_.getNodes()) {
        for (const auto v : f_.getNeighbours(u)) {
          if (f_.getDegree(v) == 0) {
            return {{u, v}};
          }
        }
      }

      return std::nullopt;
    }

    /// Reset the F graph for rebuilding.
    void reset() { f_.clearEdges(); }

  private:
    /// Return true, if moving the program qubit on hardware qubit u to hardware
    /// qubit v brings it closer to its destination hardware qubit.
    [[nodiscard]] bool shouldAddEdge(const size_t u, const size_t v,
                                     const Layout& from,
                                     const Layout& to) const {
      const auto dest = to.getHardwareIndex(from.getProgramIndex(u));
      return target_->distanceBetween(v, dest) <
             target_->distanceBetween(u, dest);
    }

    Graph f_;
    const CompilerTarget* target_;
  };

public:
  /// Construct default mapping pass.
  MappingPass() = default;

  /// Construct default mapping pass with options.
  explicit MappingPass(const MappingPassOptions& options)
      : MappingPassBase(options) {}

  /// Construct mapping for a compiler target.
  explicit MappingPass(const CompilerTarget& compilerTarget,
                       const MappingPassOptions& options)
      : MappingPassBase(options), target(compilerTarget) {}

protected:
  void runOnOperation() override {
    assert(alpha > 0 && "expected alpha > 0");
    assert(niterations > 0 && "expected niterations > 0");
    assert(ntrials > 0 && "expected ntrials > 0");

    if (!target) {
      llvm::reportFatalUsageError("No compiler target specified!");
    }

    IRRewriter rewriter(&getContext());

    auto mod = getOperation();
    auto func = mqt::getEntryPoint(mod);
    if (!func) {
      mod.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    auto comp = discoverComputation(func);
    if (failed(comp)) {
      signalPassFailure();
      return;
    }

    auto& body = func.getFunctionBody();
    auto& wires = comp->wires;
    auto& infos = comp->infos;

    if (wires.size() > target->numQubits()) {
      func.emitError()
          << "requires " + Twine(wires.size()) +
                 " qubits. However, the architecture only supports " +
                 Twine(target->numQubits()) + " qubits.";
      signalPassFailure();
      return;
    }

    auto layout = generateLayout(wires, infos);
    if (failed(layout)) {
      func->emitError() << "failed to refine random initial layouts.";
      signalPassFailure();
      return;
    }

    std::tie(wires, infos) = std::move(place(body, *layout, *comp, rewriter));

    RoutingBundle bundle{.wires = std::move(wires),
                         .infos = std::move(infos),
                         .layout = std::move(*layout)};

    const auto routeRes =
        route<WireDirection::Forward, RoutingMode::Hot>(bundle, &rewriter);
    if (failed(routeRes)) {
      func.emitError() << "failed to map the function";
      signalPassFailure();
      return;
    }

    // Collect statistics.
    const auto stats = *routeRes;
    numSwaps += stats.nswaps;

    // Fix SSA Dominance issues.
    llvm::for_each(body.getBlocks(), [](Block& b) { sortTopologically(&b); });
  }

private:
  /// Return the qubit values in `values`, preserving their relative order.
  static SmallVector<Value> getQubitValues(ValueRange values) {
    return to_vector(llvm::make_filter_range(
        values, [](Value value) { return isa<QubitType>(value.getType()); }));
  }

  /// Extend the init arguments of an `scf::ForOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static scf::ForOp extend(scf::ForOp forOp, ValueRange addons,
                           IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);

    const auto res =
        forOp.replaceWithAdditionalIterOperands(rewriter, addons, true);
    assert(succeeded(res));
    auto newForOp = cast<scf::ForOp>(*res);

    for (const auto [before, after] : llvm::zip_equal(
             addons, newForOp.getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newForOp);
    }
    return newForOp;
  }

  /// Extend the qubit arguments of an `IfOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static IfOp extend(IfOp ifOp, ValueRange addons, IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);

    auto newIfOp = ifOp.replaceWithAdditionalQubits(rewriter, addons);

    for (const auto [before, after] : llvm::zip_equal(
             addons, newIfOp->getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newIfOp);
    }

    return newIfOp;
  }

  /// Extend the target arguments of an `IndexSwitchOp` by adding a given range
  /// of additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static IndexSwitchOp extend(IndexSwitchOp switchOp, ValueRange addons,
                              IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(switchOp);

    auto newSwitchOp = switchOp.replaceWithAdditionalTargets(rewriter, addons);
    for (const auto [before, after] : llvm::zip_equal(
             addons, newSwitchOp.getLinearResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newSwitchOp);
    }
    return newSwitchOp;
  }

  /// Extend the arguments of an `scf::WhileOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static scf::WhileOp extend(scf::WhileOp whileOp, ValueRange addons,
                             IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(whileOp);

    Block* oldBefBlock = whileOp.getBeforeBody();
    Block* oldAftBlock = whileOp.getAfterBody();

    const auto oldBefNumArgs = oldBefBlock->getNumArguments();
    const auto oldAftNumArgs = oldAftBlock->getNumArguments();

    // Create a new while op at the same location as the old one with the
    // additional arguments.

    SmallVector<Value> newInits(whileOp.getInits());
    newInits.append(addons.begin(), addons.end());

    SmallVector<Type> newTypes(whileOp.getResultTypes());
    newTypes.append(addons.getTypes().begin(), addons.getTypes().end());

    auto newWhileOp =
        scf::WhileOp::create(rewriter, whileOp.getLoc(), newTypes, newInits);

    const SmallVector<Location> beforeLocs(newInits.size(), whileOp.getLoc());
    const SmallVector<Location> afterLocs(newTypes.size(), whileOp.getLoc());
    Block* newBefBlock =
        rewriter.createBlock(&newWhileOp.getBefore(), {},
                             ValueRange(newInits).getTypes(), beforeLocs);
    Block* newAftBlock =
        rewriter.createBlock(&newWhileOp.getAfter(), {}, newTypes, afterLocs);

    rewriter.mergeBlocks(oldBefBlock, newBefBlock,
                         newBefBlock->getArguments().take_front(oldBefNumArgs));
    rewriter.mergeBlocks(oldAftBlock, newAftBlock,
                         newAftBlock->getArguments().take_front(oldAftNumArgs));

    auto conditionOp = cast<scf::ConditionOp>(newBefBlock->getTerminator());
    rewriter.setInsertionPoint(conditionOp);

    // Replace the old condition operation with one that includes the new
    // "before" block arguments.

    SmallVector<Value> newConditionArgs(conditionOp.getArgs());
    llvm::append_range(newConditionArgs,
                       newBefBlock->getArguments().drop_front(oldBefNumArgs));

    scf::ConditionOp::create(rewriter, conditionOp.getLoc(),
                             conditionOp.getCondition(), newConditionArgs);
    rewriter.eraseOp(conditionOp);

    // Replace the old yield operation with one that includes the new "after"
    // block arguments.

    auto yieldOp = cast<scf::YieldOp>(newAftBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);

    SmallVector<Value> newYieldArgs(yieldOp.getResults());
    llvm::append_range(newYieldArgs,
                       newAftBlock->getArguments().drop_front(oldAftNumArgs));

    scf::YieldOp::create(rewriter, yieldOp.getLoc(), newYieldArgs);
    rewriter.eraseOp(yieldOp);

    // Finally, replace the old while operation with the new one.

    rewriter.replaceOp(
        whileOp, newWhileOp.getResults().take_front(whileOp.getNumResults()));

    for (const auto [before, after] : llvm::zip_equal(
             addons, newWhileOp->getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newWhileOp);
    }

    return newWhileOp;
  }

  /// Return the wires of a dynamic computation.
  /// Scalar `qco.alloc` operations define program qubits directly. For
  /// `qtensor` allocations, the mapping pass assumes an extraction and
  /// insertion phase where the i-th extract defines the i-th tensor-backed
  /// program qubit. Thus, supported tensor programs have the following
  /// structure:
  ///
  ///   T ⨉ [qtensor::AllocOp]
  /// → N ⨉ [qtensor::ExtractOp]
  /// → (Computation)
  /// → N ⨉ [qtensor::InsertOp]
  /// → T ⨉ [qtensor::DeallocOp]
  ///
  /// If any of the above assumptions are violated, the function returns
  /// failure.
  static FailureOr<Computation> discoverComputation(func::FuncOp func) {
    Computation computation;

    const auto discovery = func.walk([&](Operation* op) {
      if (auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
        if (isa<BarrierOp>(op)) {
          return WalkResult::advance();
        }
        if (unitary.getNumQubits() > 2) {
          unitary.emitError()
              << "cannot route an operation acting on "
              << unitary.getNumQubits()
              << " qubits; decompose it to one- and two-qubit operations "
                 "first";
          return WalkResult::interrupt();
        }
        computation.hasTwoQubitOperations |= unitary.getNumQubits() == 2;
      }

      if (!isa<AllocOp, qtensor::AllocOp>(op)) {
        return WalkResult::advance();
      }
      if (op->getParentRegion() == &func.getFunctionBody()) {
        TypeSwitch<Operation*>(op)
            .Case<AllocOp>([&](AllocOp alloc) {
              computation.scalarAllocations.emplace_back(alloc);
            })
            .Case<qtensor::AllocOp>([&](qtensor::AllocOp alloc) {
              computation.tensorAllocations.emplace_back(
                  TensorAllocation{.allocation = alloc});
            });
        return WalkResult::advance();
      }

      op->emitError()
          << "target mapping requires dynamic qubit allocations in the entry "
             "function body";
      return WalkResult::interrupt();
    });

    if (discovery.wasInterrupted()) {
      return failure();
    }

    for (auto alloc : computation.scalarAllocations) {
      const auto index = computation.wires.size();
      computation.wires.emplace_back(alloc.getResult());
      computation.infos.insertOrUpdate(index, index);
    }

    for (auto& tensor : computation.tensorAllocations) {
      bool isInitPhase = true;
      TensorIterator it(tensor.allocation.getResult());
      for (; it != std::default_sentinel; ++it) {
        Operation* const operation = it.operation();
        tensor.operations.emplace_back(operation);

        if (auto extract = dyn_cast<ExtractOp>(operation)) {
          if (!isInitPhase) {
            return func.emitError()
                   << "must extract and insert all qubits at once.";
          }

          const auto qubit = extract.getResult();
          const auto index = computation.wires.size();

          computation.wires.emplace_back(qubit);
          computation.infos.insertOrUpdate(index, index);

          continue;
        }

        if (isa<InsertOp>(operation)) {
          isInitPhase = false;
          continue;
        }
      }
    }

    return computation;
  }

  /// Perform placement by replacing dynamic qubits with static target sites
  /// and extending control-flow operations with target sites used for routing.
  ///
  /// Analogously to the discoverComputation function, the i-th extract
  /// operation defines the i-th program qubit.
  std::pair<Wires, WireInfos> place(Region& body, const Layout& layout,
                                    Computation& computation,
                                    IRRewriter& rewriter) {
    SmallVector<Value> staticQubits;
    staticQubits.reserve(target->numQubits());

    // Create and save static qubit operations.
    rewriter.setInsertionPointToStart(&body.front());
    for (size_t hw = 0; hw < layout.nqubits(); ++hw) {
      const auto site = target->siteForVertex(hw);
      auto op = StaticOp::create(rewriter, body.getLoc(), site);
      staticQubits.emplace_back(op.getQubit());
      rewriter.setInsertionPointAfter(op);
    }

    Wires wires;
    WireInfos infos;

    for (auto alloc : computation.scalarAllocations) {
      const auto prog = wires.size();
      const auto hw = layout.getHardwareIndex(prog);
      const auto qubit = staticQubits[hw];

      rewriter.replaceAllUsesWith(alloc.getResult(), qubit);
      rewriter.eraseOp(alloc);

      wires.emplace_back(qubit);
      infos.insertOrUpdate(prog, prog);
    }

    for (auto& tensor : computation.tensorAllocations) {
      for (Operation* const operation : tensor.operations) {
        TypeSwitch<Operation*>(operation)
            .Case<ExtractOp>([&](auto op) {
              const auto prog = wires.size();
              const auto hw = layout.getHardwareIndex(prog);
              const auto qubit = staticQubits[hw];

              rewriter.replaceAllUsesWith(op.getResult(), qubit);
              rewriter.replaceAllUsesWith(op.getOutTensor(), op.getTensor());
              rewriter.eraseOp(op);

              wires.emplace_back(qubit);
              infos.insertOrUpdate(prog, prog);
            })
            .Case<InsertOp>([&](auto op) {
              rewriter.setInsertionPointAfter(op);
              SinkOp::create(rewriter, op.getLoc(), op.getScalar());
              rewriter.replaceAllUsesWith(op.getResult(), op.getDest());
              rewriter.eraseOp(op);
            })
            .Case<DeallocOp>([&](auto op) { rewriter.eraseOp(op); });
      }

      rewriter.eraseOp(tensor.allocation);
    }

    // Create sinks for remaining, unused, static qubits.

    rewriter.setInsertionPoint(body.back().getTerminator());
    for (size_t prog = wires.size(); prog < layout.nqubits(); ++prog) {
      const auto hw = layout.getHardwareIndex(prog);
      const auto qubit = staticQubits[hw];

      wires.emplace_back(qubit);
      infos.insertOrUpdate(prog, prog);

      SinkOp::create(rewriter, body.getLoc(), qubit);
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
    if (!target->hasExplicitTopology()) {
      return Layout::fromMapping(
          llvm::to_vector(llvm::seq(target->numQubits())));
    }

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
                        .layout = Layout::random(target->numQubits(), rng())});
    }

    parallelForEach(&getContext(), trials, [&, this](Trial& t) {
      for (size_t i = 0; i < niterations; ++i) {
        const auto fwRouteRes = route<WireDirection::Forward>(t.bundle);
        if (failed(fwRouteRes)) {
          return;
        }

        assert(all_of(t.bundle.wires, [](const auto& it) {
          return it == std::default_sentinel;
        }));

        for_each(t.bundle.wires, [](auto& it) { std::advance(it, -1); });

        const auto bwRouteRes = route<WireDirection::Backward>(t.bundle);
        if (failed(bwRouteRes)) {
          return;
        }

        assert(all_of(t.bundle.wires, [](const auto& it) {
          return it == std::default_sentinel;
        }));

        for_each(t.bundle.wires, [](auto& it) { std::advance(it, 1); });

        t.stats = *bwRouteRes;
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
                                               const Layout& layout) const {
    constexpr size_t cap = 25'000'000UL;

    const size_t b = target->maxDegree() * ((target->numQubits() + 1) / 2);
    const size_t budget = std::min(b * b * b, cap);

    const Parameters params{.alpha = alpha, .lambda = lambda};

    llvm::SpecificBumpPtrAllocator<Node> arena;
    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    // Early exit, if the root node is a goal node already.
    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(window.front(), *target)) {
      return SmallVector<IndexPairType>{};
    }

    frontier.emplace(root);

    DenseMap<ArrayRef<size_t>, size_t> bestDepth;
    SmallVector<IndexPairType, 6> expansionSet;

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
        if (const auto otherDepth = it->getSecond();
            curr->depth >= otherDepth) {
          ++i;
          continue;
        }

        it->second = curr->depth;
      }

      // If the currently visited node is a goal node, reconstruct the
      // sequence of SWAPs from this node to the root.

      if (curr->isGoal(window.front(), *target)) {
        SmallVector<IndexPairType> seq(curr->depth);
        size_t j = seq.size() - 1;
        for (const Node* n = curr; n->parent != nullptr; n = n->parent) {
          seq[j] = n->swap;
          --j;
        }

        return seq;
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighboring hardware qubits.

      expansionSet.clear();
      for (const auto& [q0, q1] = window.front(); const auto prog : {q0, q1}) {
        const auto hw0 = curr->layout.getHardwareIndex(prog);
        target->forEachNeighbour(hw0, [&](const auto hw1) {
          // Ensure consistent hashing/comparison.
          const IndexPairType swap = std::minmax(hw0, hw1);
          if (is_contained(expansionSet, swap)) {
            return;
          }
          expansionSet.push_back(swap);

          frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                             window, *target, params));
        });
      }

      ++i;
    }

    return failure();
  }

  /// Return the SWAP sequence to move from one layout to another.
  /// Implements the 4-Approximation algorithm described in arXiv:1602.05150v3.
  [[nodiscard]] SmallVector<IndexPairType> restore(const Layout& from,
                                                   const Layout& to) const {
    Layout curr(from);
    FGraph f(*target);
    SmallVector<IndexPairType> swaps;

    while (true) {
      f.reset();
      f.construct(curr, to);

      if (const auto happy = f.findHappySWAPChain()) {
        for (const auto& swap : *happy) {
          swaps.emplace_back(swap);
          curr.swap(swap.first, swap.second);
        }
        continue;
      }

      // If there are no happy or unhappy swaps anymore,
      // the final placement of every token is reached.

      const auto unhappy = f.findUnhappySWAP();
      if (!unhappy) {
        break;
      }

      swaps.emplace_back(*unhappy);
      curr.swap(unhappy->first, unhappy->second);
    }

    assert(curr == to);

    return swaps;
  }

  /// Return a pair of SWAP sequences to transform two layouts into each other.
  /// Inspired by the 4-Approximation algorithm described in arXiv:1602.05150v3,
  /// with the key difference that the goal permutation is not static.
  [[nodiscard]] std::tuple<Layout, SmallVector<IndexPairType>,
                           SmallVector<IndexPairType>>
  converge(const Layout& lhs, const Layout& rhs) const {
    std::array layouts{Layout(lhs), Layout(rhs)};
    std::array graphs{FGraph(*target), FGraph(*target)};
    std::array<SmallVector<IndexPairType>, 2> swaps{};

    std::mt19937 gen(seed);
    std::uniform_int_distribution coin(0, 1);

    while (true) {
      size_t i = 0;
      for (; i < 2; ++i) {
        FGraph& f = graphs[i];

        f.reset();
        f.construct(layouts[i], layouts[(i + 1) % 2]);

        if (const auto happy = f.findHappySWAPChain()) {
          for (const auto& swap : *happy) {
            swaps[i].emplace_back(swap);
            layouts[i].swap(swap.first, swap.second);
          }
          break;
        }
      }

      // If we exit early from the loop, we've found a happy SWAP chain.
      if (i != 2) {
        continue;
      }

      // Otherwise, we randomly apply an unhappy SWAP to one of the layouts.
      // If there is no happy or unhappy swaps anymore, the final placement of
      // every token is reached.

      i = coin(gen);

      const auto unhappy = graphs[i].findUnhappySWAP();
      if (!unhappy) {
        break;
      }

      swaps[i].emplace_back(*unhappy);
      layouts[i].swap(unhappy->first, unhappy->second);
    }

    assert(layouts[0] == layouts[1]);

    return {layouts[0], std::move(swaps[0]), std::move(swaps[1])};
  }

  /// Compute a routing-friendly layout compromise between a range of layouts.
  /// Using the first layout of the range as an anchor, the function repeatedly
  /// nudges the current layout towards the next one using happy SWAP chains.
  /// Inspired by SABRE and to reduce ordering bias, the function performs an
  /// additional backward pass.
  template <typename Range>
  Layout driveby(Range layouts, const size_t niterations = 1) {
    assert(!layouts.empty() && "expected at least one layout");

    FGraph f(*target);
    Layout curr(*(layouts.begin()));

    // Nudge curr towards target by applying a happy SWAP chain.
    const auto merge = [&](const Layout& target) {
      f.reset();
      f.construct(curr, target);
      if (const auto happy = f.findHappySWAPChain()) {
        for (const auto& swap : *happy) {
          curr.swap(swap.first, swap.second);
        }
      }
    };

    // Perform multiple rounds of forward and backward drive-by's.
    for (size_t i = 0; i < niterations; ++i) {
      for_each(drop_begin(layouts), merge);
      for_each(drop_begin(reverse(layouts)), merge);
    }

    return curr;
  }

  /// Return a window of layers with a maximum size of `1 + nlookahead`.
  /// The function skips qubit-pair blocks.
  template <WireDirection Direction>
  Window getWindow(Wires wires, const WireInfos& infos) { // NOLINT
    Window window;
    window.reserve(1 + nlookahead);

    SmallVector<IndexPairType> layer;
    layer.reserve(nlookahead);

    enum class WalkMode : bool { Collect, SkipBlock };
    WalkMode mode = WalkMode::Collect;

    walkProgramGraph<Direction>(
        wires, [&](const ReadyMap& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          bool skipped = false;
          for (const auto& [op, indices] : ready) {
            if (!isa<BarrierOp>(op) && isa<UnitaryOpInterface>(op)) {
              const auto i0 = indices[0];
              const auto i1 = indices[1];
              const auto prog0 = infos.lookupProgram(i0);
              const auto prog1 = infos.lookupProgram(i1);
              const auto gate = std::minmax(prog0, prog1);

              if (mode == WalkMode::Collect) {

                // Collect gates in window and fill current layer vector.

                window.emplace_back(gate);
                if (window.size() == 1 + nlookahead) {
                  return WalkResult::interrupt();
                }

                layer.emplace_back(gate);
              } else {

                // Skip qubit-pair block: If the gate is contained in the
                // current layer release it. Otherwise, continue with the next
                // ready operation.

                if (is_contained(layer, gate)) {
                  released.emplace_back(op);
                  skipped = true;
                }

                continue;
              }
            }

            released.emplace_back(op);
          }

          if (mode == WalkMode::Collect) {
            mode = WalkMode::SkipBlock;
          } else if (mode == WalkMode::SkipBlock && !skipped) {
            mode = WalkMode::Collect;
            layer.clear();
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
      const auto [prog0, prog1] = layout.getProgramIndices(hw0, hw1);

      if constexpr (Mode == RoutingMode::Hot) {
        assert(infos.containsProgram(prog0) && infos.containsProgram(prog1) &&
               "expected the routing preview to materialize SWAP operands");
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
  SmallVector<CompositeUnitary> advance(Wires& wires, const WireInfos& infos,
                                        const Layout& layout) {
    DenseSet<Operation*> visited;
    SmallVector<CompositeUnitary> composites;

    // Advance wires past all executable gates and push composite unitaries and
    // the respective wire indices of their inputs onto the vector.

    walkProgramGraph<Direction>(wires, [&](const ReadyMap& ready,
                                           ReleasedOps& released) {
      if (ready.empty()) {
        return WalkResult::advance();
      }

      for (const auto& [op, indices] : ready) {
        if (isa<BarrierOp>(op)) {
          released.emplace_back(op);
          continue;
        }

        if (isa<UnitaryOpInterface>(op)) {
          const auto prog0 = infos.lookupProgram(indices[0]);
          const auto prog1 = infos.lookupProgram(indices[1]);
          if (const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
              target->areAdjacent(hw0, hw1)) {
            released.emplace_back(op);
          }
          continue;
        }

        if (op->getNumRegions() > 0 && visited.insert(op).second) {
          assert((isa<scf::ForOp, scf::WhileOp, IfOp, IndexSwitchOp>(op)));
          composites.emplace_back(op, indices);
          continue;
        }
      }

      if (released.empty()) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    // Preserve the block order when multiple independent composite operations
    // become ready at once. Hot routing threads every qubit through each
    // composite, so processing a later operation first could introduce a
    // use-before-definition for an earlier operation.
    llvm::sort(composites,
               [](const CompositeUnitary& lhs, const CompositeUnitary& rhs) {
                 assert(lhs.op->getBlock() == rhs.op->getBlock());
                 return lhs.op->isBeforeInBlock(rhs.op);
               });

    return composites;
  }

  /// Extends the composite unitary's operation to cover all target qubits by
  /// adding operands for indices not in the composite's index set. Returns a
  /// patch with the updated wire mapping which preserves the parent's wire
  /// infos and layout.
  RoutingBundle::Patch place(CompositeUnitary& composite,
                             const RoutingBundle& parent,
                             IRRewriter& rewriter) {
    DenseSet<size_t> included; // Already included indices.
    included.reserve(composite.indices.size());

    // Maps the i-th included index to its result number.
    DenseMap<size_t, size_t> indexToResultNum;
    indexToResultNum.reserve(composite.indices.size());

    for (const auto index : composite.indices) {
      const WireIterator& it = parent.wires[index];
      indexToResultNum.try_emplace(
          index, cast<OpResult>(it.qubit()).getResultNumber());
      included.insert(index);
    }

    const auto allIndices = to_vector(llvm::seq(target->numQubits()));

    const SmallVector<size_t> excluded(llvm::make_filter_range(
        allIndices, [&](const size_t i) { return !included.contains(i); }));

    const SmallVector<Value> addons(map_range(excluded, [&](const size_t i) {
      // Make sure the qubits point to an already processed operation.
      const auto& it = std::prev(
          parent.wires[i], parent.wires[i] == std::default_sentinel ? 2 : 1);
      return it.qubit();
    }));

    composite = CompositeUnitary{
        .op = TypeSwitch<Operation*, Operation*>(composite.op)
                  .Case<scf::ForOp, scf::WhileOp, IfOp, IndexSwitchOp>(
                      [&](auto cfOp) { return extend(cfOp, addons, rewriter); })
                  .Default([](Operation* op) {
                    report_fatal_error("place: unhandled op: " +
                                       op->getName().getStringRef());
                    return nullptr;
                  }),
        .indices = allIndices};

    const auto results = composite.op->getResults();

    Wires wires(allIndices.size());
    for (size_t index : included) {
      wires[index] = WireIterator(results[indexToResultNum.at(index)]);
    }
    for (const auto [index, res] :
         llvm::zip_equal(excluded, results.take_back(excluded.size()))) {
      wires[index] = WireIterator(res);
    }

    assert(llvm::all_of(wires, [&](WireIterator& it) {
      return it.operation() == composite.op;
    }));

    return RoutingBundle::Patch{.layout = std::nullopt,
                                .infos = std::nullopt,
                                .wires = std::move(wires)};
  }

  /// Return `values` with only the qubit entries realigned according to the
  /// given permutation of hardware indices.
  static SmallVector<Value> realignQubitValues(ValueRange values,
                                               ArrayRef<size_t> perm,
                                               const RoutingBundle& bundle) {
    // Map hardware indices to qubit values for the given bundle.
    DenseMap<size_t, Value> m(bundle.wires.size());
    for (size_t i = 0; i < bundle.wires.size(); ++i) {
      const auto prog = bundle.infos.lookupProgram(i);
      const auto hw = bundle.layout.getHardwareIndex(prog);
      m.try_emplace(hw, bundle.wires[i].qubit());
    }

    SmallVector<Value> realigned(values);
    size_t qubitIndex = 0;
    for (Value& value : realigned) {
      if (isa<QubitType>(value.getType())) {
        value = m.at(perm[qubitIndex++]);
      }
    }
    assert(qubitIndex == perm.size());
    return realigned;
  }

  /// Processes the composite unitary by routing the nested operation and
  /// inserting a SWAP appendix. Returns a pair of the patch to apply to the
  /// parent bundle and the accumulated statistics, or `failure` if routing
  /// fails.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  FailureOr<std::pair<RoutingBundle::Patch, Statistics>>
  dispatch(const CompositeUnitary& composite, const RoutingBundle& parent,
           IRRewriter* rewriter = nullptr) {
    const auto& [op, indices] = composite;

    SmallVector<size_t> permutation(indices.size());
    SmallVector<RoutingBundle, 0> children =
        TypeSwitch<Operation*, SmallVector<RoutingBundle, 0>>(op)
            .template Case<scf::ForOp, scf::WhileOp>([&](auto) {
              return SmallVector<RoutingBundle, 0>{
                  RoutingBundle{.layout = parent.layout}};
            })
            .template Case<IfOp>([&](IfOp) {
              return SmallVector<RoutingBundle, 0>(
                  2, RoutingBundle{.layout = parent.layout});
            })
            .template Case<IndexSwitchOp>([&](IndexSwitchOp switchOp) {
              return SmallVector<RoutingBundle, 0>(
                  switchOp.getNumRegions(),
                  RoutingBundle{.layout = parent.layout});
            });

    SmallVector<std::optional<size_t>> resultToQubitIndex(op->getNumResults());
    size_t numQubitResults = 0;
    for (const auto res : op->getResults()) {
      if (isa<QubitType>(res.getType())) {
        resultToQubitIndex[res.getResultNumber()] = numQubitResults++;
      }
    }
    assert(numQubitResults == indices.size());

    SmallVector<Value> whileBeforeQubits;
    SmallVector<Value> whileConditionQubits;
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      whileBeforeQubits = getQubitValues(whileOp.getBeforeArguments());
      whileConditionQubits = getQubitValues(
          cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator())
              .getArgs());
    }

    for (size_t i : indices) {
      const auto prog = parent.infos.lookupProgram(i);
      const auto hw = parent.layout.getHardwareIndex(prog);
      const auto res = cast<OpResult>(parent.wires[i].qubit());
      const auto resNum = res.getResultNumber();
      const auto qubitResNum = *resultToQubitIndex[resNum];

      const auto append = [&](RoutingBundle& child, Value arg, Value yielded) {
        child.infos.insertOrUpdate(child.infos.size(), prog);
        child.wires.emplace_back([&] -> Value {
          if constexpr (Direction == WireDirection::Forward) {
            return arg;
          } else {
            return yielded;
          }
        }());
      };

      TypeSwitch<Operation*>(op)
          .template Case<scf::ForOp>([&](scf::ForOp forOp) {
            const auto arg = forOp.getTiedLoopRegionIterArg(res);
            const auto yielded = forOp.getTiedLoopYieldedValue(arg)->get();
            append(children[0], arg, yielded);
          })
          .template Case<scf::WhileOp>([&](scf::WhileOp) {
            const auto arg = whileBeforeQubits[qubitResNum];
            const auto yielded = whileConditionQubits[qubitResNum];
            append(children[0], arg, yielded);
          })
          .template Case<IfOp>([&](IfOp ifOp) {
            OpOperand* const qubit = ifOp.getTiedQubit(res);
            const auto thenArg = ifOp.getTiedThenBlockArgument(qubit);
            const auto thenYielded =
                ifOp.getTiedThenYieldedValue(thenArg)->get();
            const auto elseArg = ifOp.getTiedElseBlockArgument(qubit);
            const auto elseYielded =
                ifOp.getTiedElseYieldedValue(elseArg)->get();

            append(children[0], thenArg, thenYielded);
            append(children[1], elseArg, elseYielded);
          })
          .template Case<IndexSwitchOp>([&](IndexSwitchOp switchOp) {
            OpOperand* const qubit = switchOp.getTiedTarget(res);
            const auto defaultArg = switchOp.getTiedDefaultBlockArgument(qubit);
            const auto defaultYielded =
                switchOp.getTiedDefaultYieldedValue(defaultArg)->get();
            append(children[0], defaultArg, defaultYielded);

            for (size_t r = 1; r < switchOp.getNumRegions(); ++r) {
              const auto arg = switchOp.getTiedCaseBlockArgument(qubit, r - 1);
              const auto yielded =
                  switchOp.getTiedCaseYieldedValue(arg, r - 1)->get();
              append(children[r], arg, yielded);
            }
          });

      permutation[qubitResNum] = hw;
    }

    // Route each child branch and prepare the wire iterators for
    // epilogue SWAP insertion, i.e., point each iterator at the final
    // qubit op (note: might be a measurement) before the yield.
    // TODO: Parallelize multiple children, if possible.

    Statistics totalStats;

    for (auto& child : children) {
      const auto stats = route<Direction, Mode>(child, rewriter);
      if (failed(stats)) {
        return failure();
      }

      totalStats.merge(*stats);

      if constexpr (Mode == RoutingMode::Hot) {
        for_each(child.wires, [](auto& it) { std::advance(it, -2); });
      }
    }

    // Exception: The layout of the "after" region depends on the final layout
    // of the before region. Thus, create / route the second child region /
    // bundle here.

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      children.emplace_back(RoutingBundle{.layout = children[0].layout});
      assert(children.size() == 2);

      const auto values = [&] -> ValueRange {
        if constexpr (Direction == WireDirection::Forward) {
          return whileOp.getAfterArguments();
        }
        return cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator())
            .getResults();
      }();

      for (auto [i, arg] : llvm::enumerate(getQubitValues(values))) {
        const auto hw = permutation[i];
        const auto prog = children[0].layout.getProgramIndex(hw);
        children[1].wires.emplace_back(arg);
        children[1].infos.insertOrUpdate(i, prog);
      }

      const auto stats = route<Direction, Mode>(children[1], rewriter);
      if (failed(stats)) {
        return failure();
      }

      totalStats.merge(*stats);

      if constexpr (Mode == RoutingMode::Hot) {
        for_each(children[1].wires, [](auto& it) { std::advance(it, -2); });
      }
    }

    // Find (insert) the epilogue SWAP sequence for (into) the child region
    // using the restore (scf::ForOp, scf::While), converge (IfOp), and vote
    // and restore (IndexSwitchOp) strategies.

    Layout exit =
        TypeSwitch<Operation*, Layout>(op)
            .Case<scf::ForOp>([&](scf::ForOp) {
              const auto swaps = restore(children[0].layout, parent.layout);
              insertSWAPs<Mode>(swaps, children[0], totalStats, rewriter);
              return parent.layout;
            })
            .template Case<scf::WhileOp>([&](scf::WhileOp) {
              const auto swaps = restore(children[1].layout, parent.layout);
              insertSWAPs<Mode>(swaps, children[1], totalStats, rewriter);
              // The scf::YieldOp is the terminator in the before region and
              // thus determines the final output layout.
              return children[0].layout;
            })
            .template Case<IfOp>([&](IfOp) {
              const auto [convergedLayout, fst, snd] =
                  converge(children[0].layout, children[1].layout);
              insertSWAPs<Mode>(fst, children[0], totalStats, rewriter);
              insertSWAPs<Mode>(snd, children[1], totalStats, rewriter);
              return convergedLayout;
            })
            .template Case<IndexSwitchOp>([&](IndexSwitchOp) {
              auto compromise = driveby(map_range(
                  children, [](const RoutingBundle& b) -> const Layout& {
                    return b.layout;
                  }));
              for (RoutingBundle& child : children) {
                const auto swaps = restore(child.layout, compromise);
                insertSWAPs<Mode>(swaps, child, totalStats, rewriter);
              }
              return compromise;
            });

    if constexpr (Mode == RoutingMode::Hot) {
      // Realign terminator values to ensure that i-th input qubit and the
      // i-th output qubit represent the equivalent hardware qubit. This is
      // redundant for scf::ForOp because its layout is restored, but handling
      // every supported region operation uniformly keeps this path simple.

      for (const auto& [region, child] :
           llvm::zip_equal(op->getRegions(), children)) {
        assert(region.hasOneBlock());

        Block* const block = &region.front();
        Operation* const terminator = block->getTerminator();

        rewriter->setInsertionPoint(terminator);
        TypeSwitch<Operation*>(terminator)
            .template Case<scf::YieldOp>([&](scf::YieldOp yieldOp) {
              rewriter->replaceOpWithNewOp<scf::YieldOp>(
                  yieldOp,
                  realignQubitValues(yieldOp.getResults(), permutation, child));
            })
            .template Case<scf::ConditionOp>([&](scf::ConditionOp condOp) {
              rewriter->replaceOpWithNewOp<scf::ConditionOp>(
                  condOp, condOp.getCondition(),
                  realignQubitValues(condOp.getArgs(), permutation, child));
            })
            .template Case<YieldOp>([&](YieldOp yieldOp) {
              rewriter->replaceOpWithNewOp<YieldOp>(
                  yieldOp,
                  realignQubitValues(yieldOp.getTargets(), permutation, child));
            });

        // Sort topologically to fix any occurring SSA dominance errors.

        sortTopologically(block);
      }
    }

    // If the operation is a scf::ForOp, where the parent.layout = child.layout,
    // we are done. Otherwise, propagate a patch with the final layout and
    // index-to-program mapping.

    if (isa<scf::ForOp>(op)) {
      return std::make_pair(RoutingBundle::Patch{}, totalStats);
    }

    RoutingBundle::Patch patch{.layout = std::nullopt, .infos = WireInfos{}};
    for (size_t i = 0; i < parent.wires.size(); ++i) {
      const auto oldProg = parent.infos.lookupProgram(i);
      const auto oldHw = parent.layout.getHardwareIndex(oldProg);
      const auto newProg = exit.getProgramIndex(oldHw);
      patch.infos->insertOrUpdate(i, newProg);
    }
    patch.layout = std::move(exit);

    return std::make_pair(std::move(patch), totalStats);
  }

  /// Iterates over a dynamically computed window of layers and uses A* search
  /// to find a SWAP sequence that makes each layer executable. Depending on
  /// the template parameter, this function only updates the layout or also
  /// inserts the SWAPs into the IR. Returns `FailureOr<Statistics>` containing
  /// the accumulated statistics on success, or `failure` if A* is unable to
  /// find a solution.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  FailureOr<Statistics> route(RoutingBundle& bundle,
                              IRRewriter* rewriter = nullptr) {
    auto& [wires, infos, layout] = bundle;

    Statistics stats;

    while (true) {
      while (true) {
        auto composites = advance<Direction>(wires, infos, layout);
        if (composites.empty()) {
          break;
        }

        for (auto& composite : composites) {
          if constexpr (Mode == RoutingMode::Hot) {
            auto patch = place(composite, bundle, *rewriter);
            bundle.applyPatch(std::move(patch));
          }

          auto res = dispatch<Direction, Mode>(composite, bundle, rewriter);
          if (failed(res)) {
            return failure();
          }

          bundle.applyPatch(std::move(res->first));
          stats.merge(res->second);

          // Once the composite is mapped, move past this op by incrementing the
          // respective wires.

          for_each(composite.indices, [&](size_t i) {
            std::advance(wires[i], WireTraversalTraits<Direction>::stride());
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
        // multi-qubit op of the current or subsequent layer or to a sink.

        for_each(wires, [](auto& it) { std::advance(it, 1); });
      }
    }

    return stats;
  }

  std::optional<CompilerTarget> target;
};

} // namespace

std::unique_ptr<Pass> createMappingPass(const CompilerTarget& target,
                                        MappingPassOptions options) {
  return std::make_unique<MappingPass>(target, options);
}

} // namespace mlir::qco
