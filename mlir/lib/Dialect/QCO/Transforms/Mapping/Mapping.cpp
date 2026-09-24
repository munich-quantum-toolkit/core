/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Mapping/Mapping.h"

#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/NativeSynthesis/NativeCost.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Drivers.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"
#include "mqt/Dialect/QCO/Utils/Layout.h"
#include "mqt/Dialect/QCO/Utils/Sorting.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Support/RandomSeed.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
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
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

using Wires = SmallVector<WireIterator>;

struct TensorAllocation {
  qtensor::AllocOp allocation;
  SmallVector<Operation*> operations;
};

struct Computation {
  Wires wires;
  SmallVector<AllocOp> scalarAllocations;
  SmallVector<TensorAllocation> tensorAllocations;
};

} // namespace

/// Check the structural input contract before traversing qubit wires.
static LogicalResult validateRoutingOperations(func::FuncOp func) {
  if (!llvm::hasSingleElement(func.getBody())) {
    return func.emitError("mapping requires a single-block entry function");
  }
  const auto result =
      func.walk([](Operation* operation) {
        if (isa<CallOpInterface>(operation) &&
            (llvm::any_of(operation->getOperandTypes(), isLinearQubitType) ||
             llvm::any_of(operation->getResultTypes(), isLinearQubitType))) {
          operation->emitError("inline calls that carry qubits before mapping");
          return WalkResult::interrupt();
        }
        if (operation->getNumRegions() == 0 &&
            !isa<QCODialect, qtensor::QTensorDialect, cbit::CBitDialect>(
                operation->getDialect()) &&
            !isMemoryEffectFree(operation)) {
          operation->emitError(
              "mapping supports classical side effects only through CBit "
              "operations; lower other side effects before mapping");
          return WalkResult::interrupt();
        }
        if (auto unitary = dyn_cast<UnitaryOpInterface>(operation);
            unitary && !isa<BarrierOp>(operation) &&
            unitary.getNumQubits() > 2) {
          unitary.emitError()
              << "cannot route an operation acting on "
              << unitary.getNumQubits()
              << " qubits; decompose it to one- and two-qubit operations first";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return result.wasInterrupted() ? failure() : success();
}

/// Discover the dynamic qubit roots of the entry function.
///
/// Scalar `qco.alloc` operations define program qubits directly. For
/// `qtensor` allocations, placement assumes an extraction and insertion phase
/// where the i-th extract defines the i-th tensor-backed program qubit. Thus,
/// supported tensor programs have the following structure:
///
///   T ⨉ [qtensor::AllocOp]
/// → N ⨉ [qtensor::ExtractOp]
/// → (Computation)
/// → N ⨉ [qtensor::InsertOp]
/// → T ⨉ [qtensor::DeallocOp]
///
/// If any of the above assumptions are violated, the function returns
/// failure without changing the IR.
static FailureOr<Computation> discoverComputation(func::FuncOp func) {
  Computation computation;

  for (Operation& op : func.getBody().front()) {
    TypeSwitch<Operation*>(&op)
        .Case([&](AllocOp alloc) {
          computation.scalarAllocations.emplace_back(alloc);
        })
        .Case([&](qtensor::AllocOp alloc) {
          computation.tensorAllocations.emplace_back(
              TensorAllocation{.allocation = alloc});
        });
  }

  for (auto alloc : computation.scalarAllocations) {
    computation.wires.emplace_back(alloc.getResult());
  }

  for (auto& tensor : computation.tensorAllocations) {
    bool isInitPhase = true;
    Value current = tensor.allocation.getResult();
    while (true) {
      Operation* operation = *current.getUsers().begin();
      tensor.operations.emplace_back(operation);

      if (auto extract = dyn_cast<ExtractOp>(operation)) {
        if (!isInitPhase) {
          return func.emitError() << "must extract and insert all qubits at "
                                     "once";
        }

        auto qubit = extract.getResult();

        computation.wires.emplace_back(qubit);
        current = extract.getOutTensor();
        continue;
      }

      if (auto insert = dyn_cast<InsertOp>(operation)) {
        isInitPhase = false;
        current = insert.getResult();
        continue;
      }
      if (isa<DeallocOp>(operation)) {
        break;
      }
      return operation->emitError(
          "mapping requires a flat qtensor extract/insert chain ending in "
          "deallocation; lower tensor control flow before mapping");
    }
  }

  return computation;
}

/// Check that the target has one site for every discovered program qubit.
static LogicalResult checkCapacity(func::FuncOp func,
                                   const CompilerTarget& target,
                                   const Computation& computation) {
  if (computation.wires.size() <= target.numSites()) {
    return success();
  }
  return func.emitError() << "requires " << computation.wires.size()
                          << " program qubits, but the target site count is "
                          << target.numSites();
}

/// Replace dynamic qubit roots with the target sites selected by `layout`.
///
/// Analogously to `discoverComputation`, the i-th extract operation defines
/// the i-th program qubit. The function assumes that discovery and capacity
/// checks succeeded.
static Wires applyPlacement(Region& body, const CompilerTarget& target,
                            const Layout& layout, Computation& computation,
                            IRRewriter& rewriter) {
  SmallVector<Value> staticQubits;
  staticQubits.reserve(layout.nHardwareQubits());

  rewriter.setInsertionPointToStart(&body.front());
  for (size_t hw = 0; hw < layout.nHardwareQubits(); ++hw) {
    auto op =
        StaticOp::create(rewriter, body.getLoc(), target.siteForVertex(hw));
    staticQubits.emplace_back(op.getQubit());
    rewriter.setInsertionPointAfter(op);
  }

  size_t prog = 0;

  for (auto alloc : computation.scalarAllocations) {
    auto qubit = staticQubits[layout.getHardwareIndex(prog++)];

    rewriter.replaceAllUsesWith(alloc.getResult(), qubit);
    rewriter.eraseOp(alloc);
  }

  for (auto& tensor : computation.tensorAllocations) {
    for (Operation* operation : tensor.operations) {
      TypeSwitch<Operation*>(operation)
          .Case([&](ExtractOp op) {
            auto qubit = staticQubits[layout.getHardwareIndex(prog++)];

            rewriter.replaceAllUsesWith(op.getResult(), qubit);
            rewriter.replaceAllUsesWith(op.getOutTensor(), op.getTensor());
            rewriter.eraseOp(op);
          })
          .Case([&](InsertOp op) {
            rewriter.setInsertionPointAfter(op);
            SinkOp::create(rewriter, op.getLoc(), op.getScalar());
            rewriter.replaceAllUsesWith(op.getResult(), op.getDest());
            rewriter.eraseOp(op);
          })
          .Case([&](DeallocOp op) { rewriter.eraseOp(op); });
    }

    rewriter.eraseOp(tensor.allocation);
  }

  rewriter.setInsertionPoint(body.back().getTerminator());
  for (; prog < layout.nHardwareQubits(); ++prog) {
    const auto hw = layout.getHardwareIndex(prog);
    auto qubit = staticQubits[hw];

    SinkOp::create(rewriter, body.getLoc(), qubit);
  }

  return map_to_vector(staticQubits,
                       [](Value qubit) { return WireIterator(qubit); });
}

/// Assign allocation slots to sites without traversing or expanding their uses.
static LogicalResult placeIndexedAllocations(func::FuncOp function,
                                             const CompilerTarget& target) {
  SmallVector<Operation*> allocations;
  llvm::DenseSet<CompilerTarget::SiteId> occupied;
  function.walk([&](StaticOp op) {
    occupied.insert(static_cast<CompilerTarget::SiteId>(op.getIndex()));
  });
  size_t required = occupied.size();
  for (Operation& operation : function.getBody().front()) {
    size_t width = 0;
    if (isa<AllocOp>(operation)) {
      width = 1;
    } else if (auto tensor = dyn_cast<qtensor::AllocOp>(operation)) {
      auto type = tensor.getResult().getType();
      if (!type.hasStaticShape()) {
        return tensor.emitError(
            "placement requires a statically sized qubit tensor");
      }
      width = static_cast<size_t>(type.getNumElements());
    } else {
      continue;
    }
    if (required > target.numSites() || width > target.numSites() - required) {
      return function.emitError()
             << "requires more program qubits than the target site count of "
             << target.numSites();
    }
    required += width;
    allocations.push_back(&operation);
  }
  IRRewriter rewriter(function.getContext());
  size_t vertex = 0;
  for (Operation* allocation : allocations) {
    rewriter.setInsertionPoint(allocation);
    const auto nextQubit = [&] {
      while (occupied.contains(target.siteForVertex(vertex))) {
        ++vertex;
      }
      return StaticOp::create(rewriter, allocation->getLoc(),
                              target.siteForVertex(vertex++));
    };
    if (isa<AllocOp>(allocation)) {
      auto qubit = nextQubit();
      qubit->setDiscardableAttrs(allocation->getDiscardableAttrDictionary());
      rewriter.replaceOp(allocation, qubit.getQubit());
      continue;
    }
    auto type = cast<RankedTensorType>(allocation->getResult(0).getType());
    SmallVector<Value> qubits;
    qubits.reserve(static_cast<size_t>(type.getNumElements()));
    for (int64_t index = 0; index < type.getNumElements(); ++index) {
      qubits.push_back(nextQubit().getQubit());
    }
    auto tensor = qtensor::FromElementsOp::create(
        rewriter, allocation->getLoc(), type, qubits);
    tensor->setDiscardableAttrs(allocation->getDiscardableAttrDictionary());
    rewriter.replaceOp(allocation, tensor.getResult());
  }
  return success();
}

namespace {

struct PlacementPass final
    : PassWrapper<PlacementPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlacementPass)

  explicit PlacementPass(const CompilerTarget& compilerTarget)
      : target(compilerTarget) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<QCODialect, qtensor::QTensorDialect>();
  }

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (failed(mqt::verifyQuantumAllocations(moduleOp))) {
      signalPassFailure();
      return;
    }

    auto func = mqt::getEntryPoint(moduleOp);
    if (!func) {
      moduleOp.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    const auto& environment = getAnalysis<TargetEnvironmentAnalysis>();
    if (environment && environment.environment().supportsIndexedQubits()) {
      if (failed(placeIndexedAllocations(func, target))) {
        signalPassFailure();
      }
      return;
    }

    auto computation = discoverComputation(func);
    if (failed(computation) ||
        failed(checkCapacity(func, target, *computation))) {
      signalPassFailure();
      return;
    }

    const auto layout = Layout::identity(computation->wires.size());
    IRRewriter rewriter(&getContext());
    applyPlacement(func.getFunctionBody(), target, layout, *computation,
                   rewriter);
  }

private:
  CompilerTarget target;
};

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using IndexPairType = std::pair<size_t, size_t>;
  using Window = SmallVector<IndexPairType>;
  using Score = std::pair<size_t, size_t>;

  /// Invocation data is prepared before trials, then borrowed read-only.
  struct RoutingContext {
    const CompilerTarget& target;
    uint64_t seed;
    std::unique_ptr<const NativeCostTable> nativeCosts;
    std::optional<size_t> nativeSwapCost;
  };

  enum class RoutingMode : bool { Cold, Hot };

  struct CompositeUnitary {
    /// The composite op (e.g. SCF).
    Operation* op = nullptr;
    /// Indices into a wire vector, where the order of indices has no meaning.
    SmallVector<size_t> indices;
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

  /// Wire slots are physical sites; layout alone tracks logical qubits.
  struct RoutingState {
    /// Create state from layout, enforcing wire[i] = i-th site.
    static RoutingState fromLayout(const Wires& roots, const Layout& layout,
                                   const RoutingContext* scoring = nullptr) {
      RoutingState state(Wires(layout.nHardwareQubits()), layout, scoring);
      for (auto [program, wire] : enumerate(roots)) {
        state.wires[layout.getHardwareIndex(program)] = wire;
      }
      return state;
    }

    /// Construct a routing state from a vector of wires and a layout.
    RoutingState(Wires wires, Layout layout,
                 const RoutingContext* scoring = nullptr)
        : wires(std::move(wires)), layout(std::move(layout)) {
      if (scoring != nullptr && scoring->nativeCosts) {
        costs.emplace(scoring->target, scoring->seed,
                      scoring->nativeCosts.get());
      }
    }

    Wires wires;
    Layout layout;
    std::optional<NativeCostTracker> costs;
  };

  /// Standalone search units and the signed first-step prefix adjustment.
  struct SwapCost {
    int64_t standalone = 1;
    int64_t prefixAdjustment = 0;
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
    Node* parent = nullptr;
    size_t depth = 0;
    int64_t cost = 0;
    float f = 0;

    /// Reuse layout capacity when starting a new search.
    void initializeRoot(const Layout& initialLayout) {
      layout = initialLayout;
      swap = {};
      parent = nullptr;
      depth = 0;
      cost = 0;
      f = 0;
    }

    /// Initialize a child from its parent while reusing layout capacity.
    void initializeChild(Node* nextParent, const IndexPairType& nextSwap,
                         const Window& window, const CompilerTarget& target,
                         const Parameters& params, SwapCost swapCost) {
      layout = nextParent->layout;
      swap = nextSwap;
      parent = nextParent;
      depth = parent->depth + 1;
      cost = parent->cost + swapCost.standalone + swapCost.prefixAdjustment;
      layout.swap(swap.first, swap.second);
      f = params.alpha * static_cast<float>(cost) +
          static_cast<float>(swapCost.standalone) * h(window, target, params);
    }

    /// Return true, if the current SWAP sequence makes all gates in the front
    /// executable.
    [[nodiscard]] bool isGoal(const IndexPairType& front,
                              const CompilerTarget& target) const {
      const auto [hw0, hw1] =
          layout.getHardwareIndices(front.first, front.second);
      return target.areAdjacent(hw0, hw1);
    }

    /// Return the sequence of SWAPs from the root to this node.
    [[nodiscard]] SmallVector<IndexPairType> swaps() const {
      SmallVector<IndexPairType> seq(depth);
      auto it = seq.rbegin();
      for (const Node* n = this; n->parent != nullptr; n = n->parent, ++it) {
        *it = n->swap;
      }
      return seq;
    }

  private:
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

      for (const auto& progs : window) {
        const auto [prog0, prog1] = progs;
        const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
        const size_t nswaps = target.distanceBetween(hw0, hw1) - 1;
        costs += decay * static_cast<float>(nswaps);
        decay *= params.lambda;
      }
      return costs;
    }
  };

  /// Memory arena for A* search nodes, enabling reuse across searches to reduce
  /// allocation overhead. Initializing a retained node reuses its layout.
  class Arena {
  public:
    /// Constructs an arena with a limited memory budget.
    /// The budget of nodes is derived as
    ///
    ///    `searchMemoryLimit / (sizeof(Node) + 2 * nsites * sizeof(size_t))`
    ///
    /// where the final summand accounts for the Node's layout member.
    explicit Arena(size_t nsites, size_t searchMemoryLimit)
        : budget(std::max<size_t>(
              1, searchMemoryLimit /
                     (sizeof(Node) + 2 * nsites * sizeof(size_t)))) {}

    /// Return a node slot to initialize, or nullptr when the arena is full.
    Node* allocate() {
      if (index >= budget) {
        return nullptr;
      }

      if (index == nodes.size()) {
        nodes.emplace_back();
      }
      return &nodes[index++];
    }

    /// Resets the arena for a new search. Retains allocated storage. Only the
    /// logical size (index) is reset to zero.
    void reset() { index = 0; }

  private:
    /// Storage for nodes. Uses deque for stable pointers across insertions.
    std::deque<Node> nodes;
    /// Maximum number of nodes permitted by the memory budget.
    size_t budget;
    /// Next available slot in nodes. Acts as the logical size counter.
    size_t index{0};
  };

  /// Describes the graph F of arXiv:1602.05150v3.
  struct TokenSwapGraph {
    explicit TokenSwapGraph(const CompilerTarget& target)
        : f_(llvm::to_vector(llvm::seq(target.numSites()))),
          target_(&target) {};

    /// Build F-graph: Add edges to F for each edge in the coupling graph.
    /// Note that this assumes that the coupling graph is directed, but
    /// symmetric (essentially: undirected).
    void construct(const Layout& from, const Layout& to) {
      for (size_t u = 0; u < target_->numSites(); ++u) {
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

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (!std::isfinite(alpha.getValue()) || alpha <= 0 || ntrials == 0) {
      moduleOp.emitError("mapping requires finite alpha > 0, niterations >= 0, "
                         "and ntrials > 0");
      signalPassFailure();
      return;
    }

    if (failed(mqt::verifyQuantumAllocations(moduleOp))) {
      signalPassFailure();
      return;
    }

    const auto& environment = getAnalysis<TargetEnvironmentAnalysis>();
    if (!environment) {
      moduleOp.emitError()
          << "place-and-route requires a valid mqt.target_env: "
          << environment.error();
      signalPassFailure();
      return;
    }

    RoutingContext routing{
        .target = environment.environment().target(),
        .seed = compilationSeed(moduleOp, seed),
    };
    if (routing.target.connectivityKind() !=
        CompilerTarget::Connectivity::Kind::Explicit) {
      moduleOp.emitError()
          << "place-and-route requires an explicit target topology";
      signalPassFailure();
      return;
    }

    auto func = mqt::getEntryPoint(moduleOp);
    if (!func) {
      moduleOp.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    if (failed(validateRoutingOperations(func))) {
      signalPassFailure();
      return;
    }

    auto computation = discoverComputation(func);
    if (failed(computation) ||
        failed(checkCapacity(func, routing.target, *computation))) {
      signalPassFailure();
      return;
    }

    auto& body = func.getFunctionBody();
    auto& wires = computation->wires;
    auto [layout, expectedScore] = generateLayout(wires, routing);

    IRRewriter rewriter(&getContext());
    wires =
        applyPlacement(body, routing.target, layout, *computation, rewriter);

    RoutingState state(std::move(wires), std::move(layout), &routing);

    Arena arena(routing.target.numSites(), searchMemoryLimit);
    const auto stats = route<WireDirection::Forward, RoutingMode::Hot>(
        state, arena, routing, &rewriter);

    assert((!expectedScore ||
            (state.costs ? state.costs->score() : std::nullopt)
                    .value_or(std::pair{std::numeric_limits<size_t>::max(),
                                        stats.nswaps}) == *expectedScore) &&
           "cold scoring and hot routing must agree");

    // Collect statistics.
    numSwaps += stats.nswaps;

    // Fix SSA dominance errors.
    reorderTopologically(body.front(), rewriter);
  }

private:
  /// Return the qubit values in `values`, preserving their relative order.
  static SmallVector<Value> getQubitValues(ValueRange values) {
    return llvm::filter_to_vector(
        values, [](Value value) { return isa<QubitType>(value.getType()); });
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

    for (auto [before, after] : llvm::zip_equal(
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

    for (auto [before, after] : llvm::zip_equal(
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
    for (auto [before, after] : llvm::zip_equal(
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

    for (auto [before, after] : llvm::zip_equal(
             addons, newWhileOp->getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newWhileOp);
    }

    return newWhileOp;
  }

  /// Thread the value before the next pending operation through the region.
  /// In particular, terminal measurements must remain after the region.
  static Value valueBeforeBoundary(WireIterator iterator, Operation* boundary) {
    --iterator;
    while (iterator.operation() != nullptr &&
           !iterator.operation()->isBeforeInBlock(boundary)) {
      --iterator;
    }
    return iterator.qubit();
  }

  /// Return an initial layout and whether identity needs no routing.
  ///
  /// Otherwise, place frequently interacting qubits near each other. Nested
  /// control flow has no single interaction frequency, so leave those programs
  /// to the identity and random starts.
  [[nodiscard]] std::optional<std::pair<Layout, bool>>
  generateGreedyLayout(Wires wires, const RoutingContext& routing) const {
    DenseMap<IndexPairType, size_t> weights;
    bool supported = true;
    walkProgramGraph<WireDirection::Forward>(
        MutableArrayRef(wires.data(), wires.size()),
        [&](const Frontier& frontier, ReleasedOps& released) {
          for (const auto& [op, indices] : frontier) {
            if (op->getNumRegions() != 0 && !isa<UnitaryOpInterface>(op)) {
              supported = false;
              return WalkResult::interrupt();
            }
            if (indices.size() == 2 && !isa<BarrierOp>(op) &&
                isa<UnitaryOpInterface>(op)) {
              ++weights[std::minmax(indices[0], indices[1])];
            }
            released.emplace_back(op);
          }
          return WalkResult::advance();
        });
    if (!supported) {
      return std::nullopt;
    }
    if (llvm::all_of(weights, [&](const auto& interaction) {
          const auto [a, b] = interaction.first;
          return routing.target.areAdjacent(a, b);
        })) {
      return std::pair{Layout::identity(routing.target.numSites()), true};
    }

    const size_t nprogram = wires.size();
    const size_t nhardware = routing.target.numSites();
    SmallVector<SmallVector<IndexPairType>> neighbours(nprogram);
    SmallVector<size_t> degree(nprogram, 0);
    SmallVector<size_t> attached(nprogram, 0);
    for (const auto& [pair, weight] : weights) {
      const auto [a, b] = pair;
      neighbours[a].emplace_back(b, weight);
      neighbours[b].emplace_back(a, weight);
      degree[a] += weight;
      degree[b] += weight;
    }

    /// Embed disjoint logical paths along one path through the target sites.
    /// ponytail: One hardware walk; add bounded backtracking only for measured
    /// missed embeddings.
    if (llvm::all_of(neighbours, [](const auto& adjacent) {
          return adjacent.size() <= 2;
        })) {
      SmallVector<size_t> order;
      SmallVector<bool> visited(nprogram, false);
      for (size_t start = 0; start < nprogram; ++start) {
        if (neighbours[start].size() > 1 || visited[start]) {
          continue;
        }
        size_t current = start;
        while (current != nprogram) {
          order.push_back(current);
          visited[current] = true;
          size_t next = nprogram;
          for (const auto& [partner, weight] : neighbours[current]) {
            if (!visited[partner]) {
              next = partner;
            }
          }
          current = next;
        }
      }
      if (order.size() == nprogram) {
        SmallVector<size_t> remaining(nhardware, 0);
        SmallVector<bool> usedHardware(nhardware, false);
        for (size_t hw = 0; hw < nhardware; ++hw) {
          routing.target.forEachNeighbour(hw, [&](size_t) { ++remaining[hw]; });
        }
        auto current = static_cast<size_t>(
            std::distance(remaining.begin(), llvm::min_element(remaining)));
        SmallVector<size_t> mapping(nhardware, nhardware);
        size_t placed = 0;
        while (current != nhardware && placed < nprogram) {
          mapping[order[placed++]] = current;
          usedHardware[current] = true;
          routing.target.forEachNeighbour(
              current, [&](size_t neighbour) { --remaining[neighbour]; });
          size_t next = nhardware;
          routing.target.forEachNeighbour(current, [&](size_t neighbour) {
            if (!usedHardware[neighbour] &&
                (remaining[neighbour] != 0 || placed + 1 == nprogram) &&
                (next == nhardware ||
                 std::tie(remaining[neighbour], neighbour) <
                     std::tie(remaining[next], next))) {
              next = neighbour;
            }
          });
          current = next;
        }
        if (placed == nprogram) {
          for (size_t hw = 0; hw < nhardware; ++hw) {
            if (!usedHardware[hw]) {
              mapping[placed++] = hw;
            }
          }
          return std::pair{Layout::fromMapping(mapping), false};
        }
      }
    }

    SmallVector<size_t> centrality(nhardware, 0);
    SmallVector<size_t> hardwareDegree(nhardware, 0);
    for (size_t hw = 0; hw < nhardware; ++hw) {
      for (size_t other = 0; other < nhardware; ++other) {
        centrality[hw] += routing.target.distanceBetween(hw, other);
      }
      routing.target.forEachNeighbour(hw,
                                      [&](size_t) { ++hardwareDegree[hw]; });
    }

    // The out-of-range hardware index marks an unplaced program qubit.
    SmallVector<size_t> mapping(nhardware, nhardware);
    SmallVector<bool> used(nhardware, false);
    for (size_t placed = 0; placed < nprogram; ++placed) {
      size_t prog = nprogram;
      for (size_t candidate = 0; candidate < nprogram; ++candidate) {
        if (mapping[candidate] == nhardware &&
            (prog == nprogram ||
             std::tie(attached[candidate], degree[candidate]) >
                 std::tie(attached[prog], degree[prog]))) {
          prog = candidate;
        }
      }

      size_t best = nhardware;
      size_t bestCost = 0;
      for (size_t hw = 0; hw < nhardware; ++hw) {
        if (used[hw]) {
          continue;
        }
        size_t cost = 0;
        for (const auto& [partner, weight] : neighbours[prog]) {
          if (mapping[partner] != nhardware) {
            cost +=
                weight * routing.target.distanceBetween(hw, mapping[partner]);
          }
        }
        if (best == nhardware ||
            std::tuple(cost, centrality[hw], nhardware - hardwareDegree[hw]) <
                std::tuple(bestCost, centrality[best],
                           nhardware - hardwareDegree[best])) {
          best = hw;
          bestCost = cost;
        }
      }
      mapping[prog] = best;
      used[best] = true;
      for (const auto& [partner, weight] : neighbours[prog]) {
        attached[partner] += weight;
      }
    }

    // Complete the permutation with unused sites for routing workspace.
    size_t prog = nprogram;
    for (size_t hw = 0; hw < nhardware; ++hw) {
      if (!used[hw]) {
        mapping[prog++] = hw;
      }
    }
    return std::pair{Layout::fromMapping(mapping), false};
  }

  /// Refine greedy, identity, and random starts with forward/backward routing.
  /// Score each candidate with a forward traversal, preserving its start
  /// layout.
  std::pair<Layout, std::optional<Score>>
  generateLayout(const Wires& wires, RoutingContext& routing) {
    const auto greedy = generateGreedyLayout(wires, routing);
    if (greedy && greedy->second) {
      return {greedy->first, std::nullopt};
    }

    if (const auto basis = routing.target.synthesisBasis();
        basis && basis->entangler &&
        routing.target.nativeOperationsKind() ==
            CompilerTarget::NativeOperations::Kind::Explicit) {
      routing.nativeCosts = NativeCostTable::precompute(
          mqt::getEntryPoint(getOperation()), *basis->entangler, routing.seed);
      routing.nativeSwapCost = uniformSwapCost(routing);
    }

    struct Trial {
      Layout layout;
      /// Synthesis available: (native-count, depth). Otherwise, (max(), swaps).
      Score score;
    };

    SmallVector<Trial, 0> trials;
    trials.reserve(ntrials);

    if (greedy) {
      trials.emplace_back(greedy->first);
    }

    if (trials.size() < ntrials) {
      trials.emplace_back(Layout::identity(routing.target.numSites()));

      auto rng = makeMt19937(routing.seed);
      for (size_t i = trials.size(); i < ntrials; ++i) {
        trials.emplace_back(Layout::random(routing.target.numSites(),
                                           routing.target.numSites(), rng()));
      }
    }

    assert(ntrials == trials.size());

    parallelForEach(&getContext(), trials, [&, this](Trial& t) {
      Arena arena(routing.target.numSites(), searchMemoryLimit);

      {
        auto state = RoutingState::fromLayout(wires, t.layout);
        for (size_t i = 0; i < niterations; ++i) {
          route<WireDirection::Forward>(state, arena, routing);
          route<WireDirection::Backward>(state, arena, routing);
        }
        t.layout = std::move(state.layout);
      }

      /// Refinement may permute wire cursors. Score from the original roots,
      /// preserving only the initial layout selected for final placement.
      auto state = RoutingState::fromLayout(wires, t.layout, &routing);

      const auto score = route<WireDirection::Forward>(state, arena, routing);
      const auto quality = state.costs ? state.costs->score() : std::nullopt;
      t.score = quality.value_or(
          std::pair{std::numeric_limits<size_t>::max(), score.nswaps});
    });

    Trial* const best = min_element(trials, [](const Trial& a, const Trial& b) {
      return a.score < b.score;
    });

    return {best->layout, best->score};
  }

  /// A uniform edge cost permits the existing distance heuristic to retain its
  /// units. Site-dependent SWAP costs need a weighted-distance heuristic.
  static std::optional<size_t> uniformSwapCost(const RoutingContext& routing) {
    NativeCostAnalysis analysis(routing.seed, routing.nativeCosts.get());
    std::optional<size_t> cost;
    for (const auto& [a, b] : routing.target.couplings()) {
      const auto next = analysis.swapCost(routing.target, std::array{a, b});
      if (!next || (cost && cost != next)) {
        return std::nullopt;
      }
      cost = next;
    }
    return cost;
  }

  /// Route the leading interaction with bounded A* node storage.
  /// Drain queued states at the limit, then use distance-reducing SWAPs.
  [[nodiscard]] SmallVector<IndexPairType>
  search(const Window& window, const Layout& layout, Arena& arena,
         const RoutingContext& routing,
         NativeCostTracker* costs = nullptr) const {
    const Parameters params{.alpha = alpha, .lambda = lambda};

    arena.reset();
    Node* root = arena.allocate();
    assert(root != nullptr);
    root->initializeRoot(layout);

    if (root->isGoal(window.front(), routing.target)) {
      return SmallVector<IndexPairType>{};
    }

    SmallVector<IndexPairType, 6> expansionSet;
    DenseMap<ArrayRef<size_t>, int64_t> bestCost;

    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;
    frontier.emplace(root);

    const int64_t standaloneCost =
        costs != nullptr && routing.nativeSwapCost
            ? static_cast<int64_t>(*routing.nativeSwapCost)
            : 1;
    while (!frontier.empty()) {
      Node* curr = frontier.top();
      frontier.pop();

      /// After the first edge, future costs depend only on the layout. Keep
      /// the least accumulated cost, including the one-time prefix adjustment.

      const auto [it, inserted] =
          bestCost.try_emplace(curr->layout.getProgramToHardware(), curr->cost);
      if (!inserted) {
        if (curr->cost >= it->getSecond()) {
          continue;
        }

        it->second = curr->cost;
      }

      // If the currently visited node is a goal node, reconstruct the
      // sequence of SWAPs from this node to the root.

      if (curr->isGoal(window.front(), routing.target)) {
        return curr->swaps();
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighboring hardware qubits.

      expansionSet.clear();
      for (const auto& [q0, q1] = window.front(); const auto prog : {q0, q1}) {
        const auto hw0 = curr->layout.getHardwareIndex(prog);
        routing.target.forEachNeighbour(hw0, [&](const auto hw1) {
          const IndexPairType swap = std::minmax(hw0, hw1); // Canonical SWAP.
          if (is_contained(expansionSet, swap)) {
            return;
          }

          if (Node* child = arena.allocate()) {
            SwapCost swapCost{.standalone = standaloneCost};
            if (curr->depth == 0 && costs != nullptr &&
                routing.nativeSwapCost) {
              swapCost.prefixAdjustment =
                  costs->swapCostAdjustment(hw0, hw1, *routing.nativeSwapCost);
            }
            child->initializeChild(curr, swap, window, routing.target, params,
                                   swapCost);
            expansionSet.push_back(swap);
            frontier.emplace(child);
          }
        });
      }
    }

    /// A connected target always permits a SWAP that brings the pair closer.
    /// ponytail: Greedy completion can cost later gates; increase the search
    /// budget when routing quality matters more than memory use.

    Node current;
    current.initializeRoot(layout);
    SmallVector<IndexPairType> swaps;

    const auto [q0, q1] = window.front();
    while (!current.isGoal(window.front(), routing.target)) {
      const auto [a, b] = current.layout.getHardwareIndices(q0, q1);
      const auto distance = routing.target.distanceBetween(a, b);

      std::optional<Node> best;
      for (const auto [from, to] : {IndexPairType{a, b}, IndexPairType{b, a}}) {
        routing.target.forEachNeighbour(from, [&](size_t next) {
          if (routing.target.distanceBetween(next, to) >= distance) {
            return;
          }

          const IndexPairType swap = std::minmax(from, next); // Canonical SWAP.

          Node candidate;
          candidate.initializeChild(&current, swap, window, routing.target,
                                    params, SwapCost{});
          if (!best || candidate.f < best->f) {
            best = std::move(candidate);
          }
        });
      }

      assert(best && "connected target must have a distance-reducing edge");
      swaps.push_back(best->swap);
      current.layout = std::move(best->layout);
    }

    return swaps;
  }

  /// Return the SWAP sequence to move from one layout to another.
  /// Implements the 4-Approximation algorithm described in arXiv:1602.05150v3.
  [[nodiscard]] SmallVector<IndexPairType>
  restore(const Layout& from, const Layout& to,
          const RoutingContext& routing) const {
    if (from == to) {
      return {};
    }
    Layout curr(from);
    TokenSwapGraph f(routing.target);
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
  converge(const Layout& lhs, const Layout& rhs,
           const RoutingContext& routing) {
    if (lhs == rhs) {
      return {lhs, {}, {}};
    }

    std::array layouts{Layout(lhs), Layout(rhs)};
    std::array graphs{
        TokenSwapGraph(routing.target),
        TokenSwapGraph(routing.target),
    };
    std::array<SmallVector<IndexPairType>, 2> swaps{};

    auto gen = makeMt19937(routing.seed);
    std::uniform_int_distribution coin(0, 1);

    while (true) {
      size_t i = 0;
      for (; i < 2; ++i) {
        TokenSwapGraph& f = graphs[i];

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
  Layout driveby(Range layouts, const RoutingContext& routing,
                 const size_t niterations = 1) {
    assert(!layouts.empty() && "expected at least one layout");

    TokenSwapGraph f(routing.target);
    Layout curr(*layouts.begin());

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

  template <WireDirection Direction>
  static bool precedes(Operation* a, Operation* b) {
    if constexpr (Direction == WireDirection::Forward) {
      return a->isBeforeInBlock(b);
    }
    return b->isBeforeInBlock(a);
  }

  /// Collect a routing lookahead window of up to `1 + nlookahead` ready
  /// two-qubit gates, while skipping qubit-pair blocks.
  template <WireDirection Direction>
  Window getWindow(Wires wires, const Layout& layout, Operation* boundary) {
    Window window;

    SmallVector<IndexPairType> prev;
    SmallVector<IndexPairType> next;

    walkProgramGraph<Direction>(
        MutableArrayRef(wires.data(), wires.size()),
        [&](const Frontier& frontier, ReleasedOps& released) {
          for (const auto& [op, indices] : frontier) {
            if (indices.size() == 1 &&
                (boundary == nullptr || precedes<Direction>(op, boundary))) {
              released.emplace_back(op);
            }
          }

          if (released.empty()) {
            for (const auto& [op, indices] : frontier) {
              if (boundary != nullptr && !precedes<Direction>(op, boundary)) {
                continue;
              }
              if (!isa<BarrierOp>(op) && isa<UnitaryOpInterface>(op)) {
                const auto i0 = indices[0];
                const auto i1 = indices[1];
                const auto prog0 = layout.getProgramIndex(i0);
                const auto prog1 = layout.getProgramIndex(i1);
                const IndexPairType gate = std::minmax(prog0, prog1);

                if (!is_contained(prev, gate)) {
                  window.emplace_back(gate);
                  if (window.size() - 1 == nlookahead) {
                    return WalkResult::interrupt();
                  }
                }
                next.emplace_back(gate);
              }

              released.emplace_back(op);
            }

            prev.swap(next);
            next.clear();
          }

          return WalkResult::advance();
        });

    return window;
  }

  /// Both modes leave cursors at the next operation on each physical wire.
  template <RoutingMode Mode>
  static void insertSWAPs(ArrayRef<IndexPairType> swaps, RoutingState& state,
                          Statistics& stats, IRRewriter* rewriter) {
    for (const auto& [a, b] : swaps) {
      if (state.costs) {
        state.costs->appendSwap(a, b);
      }

      if constexpr (Mode == RoutingMode::Hot) {
        auto in0 = std::prev(state.wires[a]).qubit();
        auto in1 = std::prev(state.wires[b]).qubit();
        rewriter->setInsertionPointAfterValue(in0);
        auto swap = SWAPOp::create(*rewriter, in0.getLoc(), in0, in1);
        rewriter->replaceAllUsesExcept(in0, swap.getQubit1Out(), swap);
        rewriter->replaceAllUsesExcept(in1, swap.getQubit0Out(), swap);
        state.wires[a] = std::next(WireIterator(swap.getQubit0Out()));
        state.wires[b] = std::next(WireIterator(swap.getQubit1Out()));
      } else {
        std::swap(state.wires[a], state.wires[b]);
      }

      state.layout.swap(a, b);
    }

    stats.nswaps += swaps.size();
  }

  /// Classify a consecutive measurement run from its end, so each suffix is
  /// visited once. The cache is valid only while the IR remains unchanged.
  static bool measurementNeedsRouting(MeasureOp measurement,
                                      DenseMap<Operation*, bool>& cache) {
    SmallVector<MeasureOp> measurements;
    bool needsRouting = false;
    WireIterator it(measurement.getQubitOut());
    for (; it != std::default_sentinel; ++it) {
      Operation* op = it.operation();
      if (const auto cached = cache.find(op); cached != cache.end()) {
        needsRouting = cached->second;
        break;
      }
      if (auto next = dyn_cast<MeasureOp>(op)) {
        measurements.push_back(next);
        continue;
      }
      /// Validated tensor insertion tails become sinks during placement.
      needsRouting = !isa<SinkOp, qtensor::InsertOp>(op);
      break;
    }

    Block* const block = measurement->getBlock();

    const auto addSlice = [](Operation* root, SetVector<Operation*>& worklist) {
      SetVector<Operation*> slice;
      ForwardSliceOptions options;
      options.inclusive = true;
      options.filter = [&](Operation* op) { return !worklist.contains(op); };
      getForwardSlice(root, &slice, options);
      worklist.insert(slice.begin(), slice.end());
    };

    SetVector<Operation*> worklist;

    DenseSet<TypedValue<cbit::RegisterType>> processed;

    size_t cursor = 0;
    const auto resultNeedsRouting = [&](MeasureOp next) {
      for_each(next.getResult().getUsers(),
               [&](Operation* user) { addSlice(user, worklist); });
      for (; cursor < worklist.size(); ++cursor) {
        Operation* op = worklist[cursor];

        /// Quantum consumers require the measurement to advance, independent
        /// of the selected target's permission to reuse measured qubits.

        if (any_of(op->getOperandTypes(),
                   [](auto type) { return isa<QubitType>(type); })) {
          return true;
        }

        // Captures and nested register accesses also constrain their enclosing
        // composite, whose placement threads every physical wire through it.

        if (op->getBlock() != block) {
          if (Operation* ancestor = block->findAncestorOpInBlock(*op);
              ancestor != nullptr) {
            addSlice(ancestor, worklist);
          }
        }

        const auto effects = getEffectsRecursively(op);
        if (!effects) {
          continue;
        }

        for (const auto& effect : *effects) {
          auto value = effect.getValue();
          auto reg = dyn_cast_if_present<TypedValue<cbit::RegisterType>>(value);
          if (!reg) {
            continue;
          }

          auto [it, inserted] = processed.insert(reg);
          if (!inserted) {
            continue;
          }

          for (Operation* user : reg.getUsers()) {
            Operation* ancestor = block->findAncestorOpInBlock(*user);
            if (user == op ||
                (ancestor && !measurement->isBeforeInBlock(ancestor))) {
              continue;
            }

            addSlice(user, worklist);
          }
        }
      }

      return false;
    };

    for (auto next : llvm::reverse(measurements)) {
      needsRouting = needsRouting || resultNeedsRouting(next);
      cache.try_emplace(next, needsRouting);
    }
    return needsRouting;
  }

  /// Advance past executable gates and return the first ready composite.
  /// Leave wires at non-executable gates, composites, terminal measurements,
  /// or sink-like operations. Backward traversal can exhaust block arguments.
  template <WireDirection Direction>
  std::optional<CompositeUnitary> advance(RoutingState& state,
                                          Operation* boundary,
                                          const RoutingContext& routing) {
    auto& wires = state.wires;
    std::optional<CompositeUnitary> composite;
    /// Advancement only moves iterators. Discard classifications before routing
    /// inserts SWAPs or replaces composites.
    DenseMap<Operation*, bool> measurementRouting;

    // The wire traversal does not follow classical dependencies. Defer a
    // composite until earlier routing work is complete, but let independent
    // composites pass terminal wires. Reverse block order for backward routing.

    const auto defer = [&wires, &measurementRouting](Operation* candidate) {
      return any_of(wires, [&](WireIterator& it) {
        if (it == std::default_sentinel) {
          return false;
        }

        Operation* op = it.operation();
        if (op == nullptr || op == candidate) {
          return false;
        }

        // A wire at its traversal boundary has no pending routing work.
        if (std::next(it, WireTraversalTraits<Direction>::stride()) ==
            std::default_sentinel) {
          return false;
        }
        if constexpr (Direction == WireDirection::Forward) {
          if (auto measurement = dyn_cast<MeasureOp>(op);
              measurement &&
              !measurementNeedsRouting(measurement, measurementRouting)) {
            return false;
          }
          return op->isBeforeInBlock(candidate);
        }

        return candidate->isBeforeInBlock(op);
      });
    };

    /// Keep the earliest ready region in block order. Hot placement threads
    /// every wire through it, so later regions must wait for its exit layout.

    walkProgramGraph<Direction>(wires, [&](const Frontier& frontier,
                                           ReleasedOps& released) {
      for (const auto& [op, indices] : frontier) {
        if (boundary != nullptr && precedes<Direction>(boundary, op)) {
          continue;
        }

        const auto release =
            TypeSwitch<Operation*, bool>(op)
                .Case([](BarrierOp&) { return true; })
                .Case([&](UnitaryOpInterface&) {
                  if (indices.size() == 1) {
                    return true;
                  }

                  return routing.target.areAdjacent(indices[0], indices[1]);
                })
                .Case([](ResetOp&) { return true; })
                .Case([&](MeasureOp& m) {
                  if (Direction == WireDirection::Backward) {
                    return true;
                  }

                  return measurementNeedsRouting(m, measurementRouting);
                })
                .template Case<AllocOp, StaticOp, qtensor::ExtractOp>(
                    [](auto&) { return Direction == WireDirection::Forward; })
                .template Case<SinkOp, qtensor::InsertOp, YieldOp, scf::YieldOp,
                               scf::ConditionOp>(
                    [](auto&) { return Direction == WireDirection::Backward; })
                .template Case<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>(
                    [&](auto& cf) {
                      if (!defer(cf) &&
                          (!composite ||
                           precedes<Direction>(op, composite->op))) {
                        composite.emplace(op, indices);
                      }
                      return false;
                    })
                .Default([&](auto) { return false; });

        if (release) {
          released.emplace_back(op);

          if (state.costs) {
            SmallVector<size_t, 2> vertices(indices.begin(), indices.end());
            /// Frontier indices are in traversal order, not operand order.
            if (auto gate = dyn_cast<UnitaryOpInterface>(op);
                gate && gate.isTwoQubit() &&
                wires[indices[0]].qubit() != gate.getOutputQubit(0)) {
              std::swap(vertices[0], vertices[1]);
            }
            state.costs->append(op, vertices);
          }
        }
      }

      if (released.empty()) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return composite;
  }

  /// Extends the composite unitary's operation to cover all target qubits by
  /// adding operands for sites outside the composite. Keeps the parent cursors
  /// at the corresponding physical results.
  void place(CompositeUnitary& composite, RoutingState& parent,
             IRRewriter& rewriter) {
    SmallVector<unsigned> resultNumbers(parent.wires.size());
    SmallVector<Value> addons;
    for (auto [site, wire] : enumerate(parent.wires)) {
      if (wire.operation() == composite.op) {
        resultNumbers[site] = cast<OpResult>(wire.qubit()).getResultNumber();
      } else {
        resultNumbers[site] = composite.op->getNumResults() + addons.size();
        addons.push_back(valueBeforeBoundary(wire, composite.op));
      }
    }
    composite.op =
        TypeSwitch<Operation*, Operation*>(composite.op)
            .Case<scf::ForOp, scf::WhileOp, IfOp, IndexSwitchOp>(
                [&](auto op) { return extend(op, addons, rewriter); });
    composite.indices = to_vector(llvm::seq(parent.wires.size()));
    for (auto [site, result] : enumerate(resultNumbers)) {
      parent.wires[site] = WireIterator(composite.op->getResult(result));
    }
  }

  /// Return `values` with only the qubit entries realigned according to the
  /// given permutation of hardware indices.
  static SmallVector<Value> realignQubitValues(ValueRange values,
                                               ArrayRef<size_t> perm,
                                               const RoutingState& bundle) {
    SmallVector<Value> realigned(values);
    size_t qubitIndex = 0;
    for (Value& value : realigned) {
      if (isa<QubitType>(value.getType())) {
        value = std::prev(bundle.wires[perm[qubitIndex++]]).qubit();
      }
    }
    assert(qubitIndex == perm.size());
    return realigned;
  }

  /// Destination of each physical slot after a layout change.
  static SmallVector<size_t> sitePermutation(const Layout& from,
                                             const Layout& to) {
    SmallVector<size_t> permutation(from.nHardwareQubits());
    for (size_t site = 0; site < permutation.size(); ++site) {
      permutation[site] = to.getHardwareIndex(from.getProgramIndex(site));
    }
    return permutation;
  }

  static void permuteWires(Wires& wires, ArrayRef<size_t> permutation) {
    Wires reordered(wires.size());
    for (size_t site = 0; site < wires.size(); ++site) {
      reordered[permutation[site]] = wires[site];
    }
    wires = std::move(reordered);
  }

  /// Capture uses before rebinding so permutation cycles are safe.
  static void realignQubitUses(ValueRange values, ArrayRef<size_t> sites,
                               ArrayRef<size_t> permutation,
                               IRRewriter& rewriter) {
    auto qubits = getQubitValues(values);
    SmallVector<Value> atSite(permutation.size());
    for (auto [site, qubit] : llvm::zip_equal(sites, qubits)) {
      atSite[site] = qubit;
    }
    SmallVector<std::pair<OpOperand*, Value>> replacements;
    for (auto [site, qubit] : llvm::zip_equal(sites, qubits)) {
      replacements.emplace_back(&*qubit.getUses().begin(),
                                atSite[permutation[site]]);
    }
    for (auto [use, value] : replacements) {
      rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(value); });
    }
  }

  /// Values carried by a supported region terminator.
  static ValueRange yieldedValues(Block& block) {
    return TypeSwitch<Operation*, ValueRange>(block.getTerminator())
        .Case([](scf::YieldOp op) { return op.getResults(); })
        .Case([](scf::ConditionOp op) { return op.getArgs(); })
        .Case([](YieldOp op) { return op.getTargets(); });
  }

  /// Construct child states, route their bodies, reconcile layouts, then
  /// publish the physical result order to the parent.
  template <WireDirection Direction, RoutingMode Mode>
  Statistics routeComposite(const CompositeUnitary& composite,
                            RoutingState& parent, Arena& arena,
                            const RoutingContext& routing,
                            IRRewriter* rewriter) {
    const auto& [op, indices] = composite;
    if (parent.costs) {
      parent.costs->flush();
    }

    SmallVector<size_t> resultSites(op->getNumResults());
    for (size_t site : indices) {
      resultSites[cast<OpResult>(parent.wires[site].qubit())
                      .getResultNumber()] = site;
    }

    SmallVector<size_t> sites;
    for (auto result : op->getResults()) {
      if (isa<QubitType>(result.getType())) {
        sites.push_back(resultSites[result.getResultNumber()]);
      }
    }
    assert(sites.size() == indices.size());

    Statistics totalStats;

    SmallVector<RoutingState, 0> children;
    children.reserve(op->getNumRegions());

    for (auto [index, region] : enumerate(op->getRegions())) {
      auto& child =
          children.emplace_back(Wires(routing.target.numSites()), parent.layout,
                                parent.costs ? &routing : nullptr);

      auto roots = getQubitValues(Direction == WireDirection::Forward
                                      ? region.front().getArguments()
                                      : yieldedValues(region.front()));
      for (auto [site, root] : zip_equal(sites, roots)) {
        child.wires[site] = WireIterator(root);
      }

      if (isa<scf::WhileOp>(op) && index == 1) {
        /// The before-region exit places the after region and loop results.
        child.layout = children[0].layout;
        const auto permutation = sitePermutation(parent.layout, child.layout);
        if constexpr (Mode == RoutingMode::Hot) {
          realignQubitUses(region.front().getArguments(), sites, permutation,
                           *rewriter);
        } else {
          permuteWires(child.wires, permutation);
        }
      }

      totalStats.merge(route<Direction, Mode>(child, arena, routing, rewriter));
    }

    Layout exit =
        TypeSwitch<Operation*, Layout>(op)
            .Case([&](scf::ForOp) {
              insertSWAPs<Mode>(
                  restore(children[0].layout, parent.layout, routing),
                  children[0], totalStats, rewriter);
              return parent.layout;
            })
            .Case([&](scf::WhileOp) {
              insertSWAPs<Mode>(
                  restore(children[1].layout, parent.layout, routing),
                  children[1], totalStats, rewriter);
              return children[0].layout;
            })
            .Case([&](IfOp) {
              const auto [layout, first, second] =
                  converge(children[0].layout, children[1].layout, routing);
              insertSWAPs<Mode>(first, children[0], totalStats, rewriter);
              insertSWAPs<Mode>(second, children[1], totalStats, rewriter);
              return layout;
            })
            .Case([&](IndexSwitchOp) {
              auto layout = driveby(
                  map_range(children,
                            [](const RoutingState& child) -> const Layout& {
                              return child.layout;
                            }),
                  routing);
              for (auto& child : children) {
                insertSWAPs<Mode>(restore(child.layout, layout, routing), child,
                                  totalStats, rewriter);
              }
              return layout;
            });

    for (auto [region, child] : zip_equal(op->getRegions(), children)) {
      if (parent.costs) {
        parent.costs->merge(*child.costs);
      }

      if constexpr (Mode == RoutingMode::Hot) {
        auto* terminator = region.front().getTerminator();
        auto values =
            realignQubitValues(yieldedValues(region.front()), sites, child);

        rewriter->modifyOpInPlace(terminator, [&] {
          if (auto condition = dyn_cast<scf::ConditionOp>(terminator)) {
            condition.getArgsMutable().assign(values);
          } else {
            terminator->setOperands(values);
          }
        });

        reorderTopologically(region.front(), *rewriter);
      }
    }

    const auto permutation = sitePermutation(parent.layout, exit);
    if constexpr (Mode == RoutingMode::Hot) {
      realignQubitUses(op->getResults(), sites, permutation, *rewriter);
    } else {
      permuteWires(parent.wires, permutation);
    }
    parent.layout = std::move(exit);
    return totalStats;
  }

  /// Regions fence every traversal, independent of native-cost availability.
  template <WireDirection Direction>
  static Operation* nextRoutingBoundary(Operation* op) {
    for (; op != nullptr; op = Direction == WireDirection::Forward
                                   ? op->getNextNode()
                                   : op->getPrevNode()) {
      if (isa<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>(op) &&
          any_of(op->getResultTypes(),
                 [](Type type) { return isa<QubitType>(type); })) {
        return op;
      }
    }
    return nullptr;
  }

  /// Advance executable operations, route ready regions, or search for SWAPs
  /// that release the next interaction. Finish terminal measurements last.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  Statistics route(RoutingState& state, Arena& arena,
                   const RoutingContext& routing,
                   IRRewriter* rewriter = nullptr) {
    Operation* boundary = nullptr;
    for (auto& wire : state.wires) {
      if (wire != std::default_sentinel) {
        auto* block = wire.qubit().getParentBlock();
        boundary = nextRoutingBoundary<Direction>(
            Direction == WireDirection::Forward ? &block->front()
                                                : &block->back());
        break;
      }
    }

    Statistics stats;
    while (true) {
      auto composite = advance<Direction>(state, boundary, routing);
      if (composite) {
        assert(composite->op == boundary);
        boundary = nextRoutingBoundary<Direction>(
            Direction == WireDirection::Forward ? boundary->getNextNode()
                                                : boundary->getPrevNode());

        if constexpr (Mode == RoutingMode::Hot) {
          place(*composite, state, *rewriter);
        }

        stats.merge(routeComposite<Direction, Mode>(*composite, state, arena,
                                                    routing, rewriter));
        for (auto& wire : state.wires) {
          if (wire != std::default_sentinel &&
              wire.operation() == composite->op) {
            std::ranges::advance(wire,
                                 WireTraversalTraits<Direction>::stride());
          }
        }
        continue;
      }

      const auto window =
          getWindow<Direction>(state.wires, state.layout, boundary);
      if (window.empty()) {
        break;
      }

      const auto swaps = search(window, state.layout, arena, routing,
                                state.costs ? &*state.costs : nullptr);
      insertSWAPs<Mode>(swaps, state, stats, rewriter);
    }

    if constexpr (Direction == WireDirection::Forward) {
      for (auto [site, wire] : enumerate(state.wires)) {
        while (wire != std::default_sentinel &&
               isa_and_nonnull<MeasureOp>(wire.operation())) {
          if (state.costs) {
            const std::array vertex{site};
            state.costs->append(wire.operation(), vertex);
          }
          ++wire;
        }
      }
    }

    return stats;
  }
};

} // namespace

std::unique_ptr<Pass> createPlacementPass(const CompilerTarget& target) {
  return std::make_unique<PlacementPass>(target);
}

} // namespace mlir::qco
