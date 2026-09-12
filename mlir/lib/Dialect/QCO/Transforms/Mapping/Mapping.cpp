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
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Drivers.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"
#include "mqt/Dialect/QCO/Utils/Layout.h"
#include "mqt/Dialect/QCO/Utils/Sorting.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
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
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
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
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

using Wires = SmallVector<WireIterator>;

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
  /// Callers preserve a one-to-one mapping and append wire indices densely.
  void insertOrUpdate(const size_t index, const size_t prog) {
    if (index >= indexToProgram_.size()) {
      indexToProgram_.resize(index + 1);
    }
    if (prog >= programToIndex_.size()) {
      programToIndex_.resize(prog + 1);
    }
    indexToProgram_[index] = prog;
    programToIndex_[prog] = index;
  }

  /// Return whether a program index has a corresponding wire.
  [[nodiscard]] bool containsProgram(const size_t prog) const {
    return prog < programToIndex_.size() &&
           indexToProgram_[programToIndex_[prog]] == prog;
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
    const auto index = computation.wires.size();
    computation.wires.emplace_back(alloc.getResult());
    computation.infos.insertOrUpdate(index, index);
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
        const auto index = computation.wires.size();

        computation.wires.emplace_back(qubit);
        computation.infos.insertOrUpdate(index, index);
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
static std::pair<Wires, WireInfos>
applyPlacement(Region& body, const CompilerTarget& target, const Layout& layout,
               Computation& computation, IRRewriter& rewriter) {
  SmallVector<Value> staticQubits;
  staticQubits.reserve(layout.nHardwareQubits());

  rewriter.setInsertionPointToStart(&body.front());
  for (size_t hw = 0; hw < layout.nHardwareQubits(); ++hw) {
    auto op =
        StaticOp::create(rewriter, body.getLoc(), target.siteForVertex(hw));
    staticQubits.emplace_back(op.getQubit());
    rewriter.setInsertionPointAfter(op);
  }

  Wires wires;
  WireInfos infos;

  for (auto alloc : computation.scalarAllocations) {
    const auto prog = wires.size();
    auto qubit = staticQubits[layout.getHardwareIndex(prog)];

    rewriter.replaceAllUsesWith(alloc.getResult(), qubit);
    rewriter.eraseOp(alloc);

    wires.emplace_back(qubit);
    infos.insertOrUpdate(prog, prog);
  }

  for (auto& tensor : computation.tensorAllocations) {
    for (Operation* operation : tensor.operations) {
      TypeSwitch<Operation*>(operation)
          .Case([&](ExtractOp op) {
            const auto prog = wires.size();
            auto qubit = staticQubits[layout.getHardwareIndex(prog)];

            rewriter.replaceAllUsesWith(op.getResult(), qubit);
            rewriter.replaceAllUsesWith(op.getOutTensor(), op.getTensor());
            rewriter.eraseOp(op);

            wires.emplace_back(qubit);
            infos.insertOrUpdate(prog, prog);
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
  for (size_t prog = wires.size(); prog < layout.nHardwareQubits(); ++prog) {
    const auto hw = layout.getHardwareIndex(prog);
    auto qubit = staticQubits[hw];

    wires.emplace_back(qubit);
    infos.insertOrUpdate(prog, prog);
    SinkOp::create(rewriter, body.getLoc(), qubit);
  }

  return {wires, infos};
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

  /// State shared by traversal and routing.
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

  /// Describes the graph F of arXiv:1602.05150v3.
  struct FGraph {
    explicit FGraph(const CompilerTarget& target)
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
    if (!std::isfinite(alpha.getValue()) || alpha <= 0 || niterations == 0 ||
        ntrials == 0) {
      moduleOp.emitError("mapping requires finite alpha > 0, niterations > 0, "
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
    target = &environment.environment().target();

    if (target->connectivityKind() !=
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
        failed(checkCapacity(func, *target, *computation))) {
      signalPassFailure();
      return;
    }

    auto& body = func.getFunctionBody();
    auto& wires = computation->wires;
    auto& infos = computation->infos;
    auto layout = generateLayout(wires, infos);
    if (failed(layout)) {
      func.emitError() << "failed to refine random initial layouts";
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    std::tie(wires, infos) = std::move(
        applyPlacement(body, *target, *layout, *computation, rewriter));

    RoutingBundle bundle{
        .wires = std::move(wires),
        .infos = std::move(infos),
        .layout = std::move(*layout),
    };

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

  /// Return the wire value before a composite, even if advancement passed it.
  static Value valueBeforeBoundary(WireIterator iterator, Operation* boundary) {
    if (iterator == std::default_sentinel) {
      --iterator;
    }
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
  generateGreedyLayout(Wires wires, const WireInfos& infos) const {
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
              ++weights[std::minmax(infos.lookupProgram(indices[0]),
                                    infos.lookupProgram(indices[1]))];
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
          return target->areAdjacent(a, b);
        })) {
      return std::pair{Layout::identity(target->numSites()), true};
    }

    const size_t nprogram = infos.size();
    const size_t nhardware = target->numSites();
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
          target->forEachNeighbour(hw, [&](size_t) { ++remaining[hw]; });
        }
        auto current = static_cast<size_t>(
            std::distance(remaining.begin(), llvm::min_element(remaining)));
        SmallVector<size_t> mapping(nhardware, nhardware);
        size_t placed = 0;
        while (current != nhardware && placed < nprogram) {
          mapping[order[placed++]] = current;
          usedHardware[current] = true;
          target->forEachNeighbour(
              current, [&](size_t neighbour) { --remaining[neighbour]; });
          size_t next = nhardware;
          target->forEachNeighbour(current, [&](size_t neighbour) {
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
        centrality[hw] += target->distanceBetween(hw, other);
      }
      target->forEachNeighbour(hw, [&](size_t) { ++hardwareDegree[hw]; });
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
            cost += weight * target->distanceBetween(hw, mapping[partner]);
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

  /// Refine identity, random, and greedy starts with forward/backward routing.
  /// Keep the raw greedy start too: refinement can worsen forward routing.
  /// Score each candidate with a forward traversal, preserving its start
  /// layout.
  FailureOr<Layout> generateLayout(const Wires& wires, const WireInfos& infos) {
    const auto greedy = generateGreedyLayout(wires, infos);
    if (greedy && greedy->second) {
      return greedy->first;
    }
    std::mt19937_64 rng{seed};

    struct Trial {
      RoutingBundle bundle;
      size_t iterations;
      Statistics stats{};
      bool success{false};
    };

    SmallVector<Trial, 0> trials;
    trials.reserve(ntrials + 2);
    for (size_t i = 0; i < ntrials; ++i) {
      trials.emplace_back(
          RoutingBundle{
              .wires = wires,
              .infos = infos,
              .layout = i == 0 ? Layout::identity(target->numSites())
                               : Layout::random(target->numSites(),
                                                target->numSites(), rng()),
          },
          niterations);
    }
    if (greedy) {
      trials.emplace_back(
          RoutingBundle{
              .wires = wires,
              .infos = infos,
              .layout = greedy->first,
          },
          niterations);
      trials.emplace_back(
          RoutingBundle{
              .wires = wires,
              .infos = infos,
              .layout = greedy->first,
          },
          0);
    }

    parallelForEach(&getContext(), trials, [&, this](Trial& t) {
      for (size_t i = 0; i < t.iterations; ++i) {
        const auto fwRouteRes = route<WireDirection::Forward>(t.bundle);
        if (failed(fwRouteRes)) {
          return;
        }

        const auto bwRouteRes = route<WireDirection::Backward>(t.bundle);
        if (failed(bwRouteRes)) {
          return;
        }
      }
      auto scoringBundle = t.bundle;
      const auto score = route<WireDirection::Forward>(scoringBundle);
      if (failed(score)) {
        return;
      }
      t.stats = *score;
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

  /// Route the leading interaction with bounded A* node storage.
  /// Drain queued states at the limit, then use distance-reducing SWAPs.
  [[nodiscard]] SmallVector<IndexPairType> search(const Window& window,
                                                  const Layout& layout) const {
    /// Estimate retained node and layout storage; keep at least the root.
    const size_t nodeBytes =
        sizeof(Node) + 2 * target->numSites() * sizeof(size_t);
    const size_t nodeBudget =
        std::max<size_t>(1, searchMemoryLimit.getValue() / nodeBytes);

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

    size_t nodes = 1;
    while (!frontier.empty()) {
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
          if (nodes >= nodeBudget || is_contained(expansionSet, swap)) {
            return;
          }
          expansionSet.push_back(swap);

          frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                             window, *target, params));
          ++nodes;
        });
      }
    }

    /// A connected target always permits a SWAP that brings the pair closer.
    /// ponytail: Greedy completion can cost later gates; increase the search
    /// budget when routing quality matters more than memory use.
    SmallVector<IndexPairType> swaps;
    Node current(layout);
    const auto [q0, q1] = window.front();
    while (!current.isGoal(window.front(), *target)) {
      const auto [a, b] = current.layout.getHardwareIndices(q0, q1);
      const auto distance = target->distanceBetween(a, b);
      std::optional<Node> best;
      for (const auto [from, to] : {IndexPairType{a, b}, IndexPairType{b, a}}) {
        target->forEachNeighbour(from, [&](size_t next) {
          if (target->distanceBetween(next, to) >= distance) {
            return;
          }
          Node candidate(&current, std::minmax(from, next), window, *target,
                         params);
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
  [[nodiscard]] SmallVector<IndexPairType> restore(const Layout& from,
                                                   const Layout& to) const {
    if (from == to) {
      return {};
    }
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
    if (lhs == rhs) {
      return {lhs, {}, {}};
    }
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

  /// Collect a routing lookahead window of up to `1 + nlookahead` ready
  /// two-qubit gates, while skipping qubit-pair blocks.
  template <WireDirection Direction>
  Window getWindow(Wires wires, const WireInfos& infos) {
    Window window;
    window.reserve(1 + nlookahead);

    SmallVector<IndexPairType> prev;
    SmallVector<IndexPairType> next;

    walkProgramGraph<Direction>(
        MutableArrayRef(wires.data(), wires.size()),
        [&](const Frontier& frontier, ReleasedOps& released) {
          for (const auto& [op, indices] : frontier) {
            if (indices.size() == 1) {
              released.emplace_back(op);
            }
          }

          if (released.empty()) {
            for (const auto& [op, indices] : frontier) {
              if (!isa<BarrierOp>(op) && isa<UnitaryOpInterface>(op)) {
                const auto i0 = indices[0];
                const auto i1 = indices[1];
                const auto prog0 = infos.lookupProgram(i0);
                const auto prog1 = infos.lookupProgram(i1);
                const IndexPairType gate = std::minmax(prog0, prog1);

                if (!is_contained(prev, gate)) {
                  window.emplace_back(gate);
                  if (window.size() == 1 + nlookahead) {
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

        auto in0 = w0.qubit();
        auto in1 = w1.qubit();

        rewriter->setInsertionPointAfterValue(in0); // Valid bc. Hot → Forward.
        auto swapOp = SWAPOp::create(*rewriter, in0.getLoc(), in0, in1);

        auto out0 = swapOp.getQubit0Out();
        auto out1 = swapOp.getQubit1Out();

        rewriter->replaceAllUsesExcept(in0, out1, swapOp);
        rewriter->replaceAllUsesExcept(in1, out0, swapOp);

        infos.swap(prog0, prog1);

        std::ranges::advance(w0, 1); // Move to SWAP.
        std::ranges::advance(w1, 1);
      }

      layout.swap(hw0, hw1);
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
      needsRouting = !isa<SinkOp>(op);
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

  /// Advance past executable gates and return ready composite operations.
  /// Leave wires at non-executable gates, composites, terminal measurements,
  /// or sink-like operations. Backward traversal can exhaust block arguments.
  template <WireDirection Direction>
  SmallVector<CompositeUnitary> advance(Wires& wires, const WireInfos& infos,
                                        const Layout& layout) {
    DenseSet<Operation*> visited;
    SmallVector<CompositeUnitary> composites;
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

    // Advance wires past all executable gates and push composite unitaries
    // and the respective wire indices of their inputs onto the vector.

    walkProgramGraph<Direction>(wires, [&](const Frontier& frontier,
                                           ReleasedOps& released) {
      for (const auto& [op, indices] : frontier) {
        const auto release =
            TypeSwitch<Operation*, bool>(op)
                .Case([](BarrierOp&) { return true; })
                .Case([&](UnitaryOpInterface&) {
                  if (indices.size() == 1) {
                    return true;
                  }

                  const auto prog0 = infos.lookupProgram(indices[0]);
                  const auto prog1 = infos.lookupProgram(indices[1]);
                  const auto [hw0, hw1] =
                      layout.getHardwareIndices(prog0, prog1);
                  return target->areAdjacent(hw0, hw1);
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
                .template Case<SinkOp, MeasureOp, qtensor::InsertOp, YieldOp,
                               scf::YieldOp, scf::ConditionOp>(
                    [](auto&) { return Direction == WireDirection::Backward; })
                .template Case<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>(
                    [&](auto& cf) {
                      if (!defer(cf) && visited.insert(op).second) {
                        composites.emplace_back(op, indices);
                      }
                      return false;
                    })
                .Default([&](auto) { return false; });

        if (release) {
          released.emplace_back(op);
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
  /// adding operands for indices not in the composite's index set. Updates
  /// the parent's wires while preserving its wire information and layout.
  void place(CompositeUnitary& composite, RoutingBundle& parent,
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

    const auto allIndices = to_vector(llvm::seq(target->numSites()));

    const SmallVector<size_t> excluded(llvm::make_filter_range(
        allIndices, [&](const size_t i) { return !included.contains(i); }));

    const SmallVector<Value> addons(map_range(excluded, [&](const size_t i) {
      return valueBeforeBoundary(parent.wires[i], composite.op);
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
        .indices = allIndices,
    };

    auto results = composite.op->getResults();

    Wires wires(allIndices.size());
    for (size_t index : included) {
      wires[index] = WireIterator(results[indexToResultNum.at(index)]);
    }
    for (auto [index, res] :
         llvm::zip_equal(excluded, results.take_back(excluded.size()))) {
      wires[index] = WireIterator(res);
    }

    assert(llvm::all_of(wires, [&](WireIterator& it) {
      return it.operation() == composite.op;
    }));

    parent.wires = std::move(wires);
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
  /// inserting epilogue SWAPs. Updates the parent bundle and returns the
  /// accumulated statistics, or `failure` if routing fails.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  FailureOr<Statistics> dispatch(const CompositeUnitary& composite,
                                 RoutingBundle& parent,
                                 IRRewriter* rewriter = nullptr) {
    const auto& [op, indices] = composite;

    SmallVector<size_t> permutation(indices.size());
    SmallVector<RoutingBundle, 0> children =
        TypeSwitch<Operation*, SmallVector<RoutingBundle, 0>>(op)
            .template Case<scf::ForOp, scf::WhileOp>([&](auto) {
              return SmallVector<RoutingBundle, 0>{
                  RoutingBundle{.layout = parent.layout},
              };
            })
            .Case([&](IfOp) {
              return SmallVector<RoutingBundle, 0>(
                  2, RoutingBundle{.layout = parent.layout});
            })
            .Case([&](IndexSwitchOp switchOp) {
              return SmallVector<RoutingBundle, 0>(
                  switchOp.getNumRegions(),
                  RoutingBundle{.layout = parent.layout});
            });

    SmallVector<std::optional<size_t>> resultToQubitIndex(op->getNumResults());
    size_t numQubitResults = 0;
    for (auto res : op->getResults()) {
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
      auto res = cast<OpResult>(parent.wires[i].qubit());
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
          .Case([&](scf::ForOp forOp) {
            auto arg = forOp.getTiedLoopRegionIterArg(res);
            auto yielded = forOp.getTiedLoopYieldedValue(arg)->get();
            append(children[0], arg, yielded);
          })
          .Case([&](scf::WhileOp) {
            auto arg = whileBeforeQubits[qubitResNum];
            auto yielded = whileConditionQubits[qubitResNum];
            append(children[0], arg, yielded);
          })
          .Case([&](IfOp ifOp) {
            OpOperand* const qubit = ifOp.getTiedQubit(res);
            auto thenArg = ifOp.getTiedThenBlockArgument(qubit);
            auto thenYielded = ifOp.getTiedThenYieldedValue(thenArg)->get();
            auto elseArg = ifOp.getTiedElseBlockArgument(qubit);
            auto elseYielded = ifOp.getTiedElseYieldedValue(elseArg)->get();

            append(children[0], thenArg, thenYielded);
            append(children[1], elseArg, elseYielded);
          })
          .Case([&](IndexSwitchOp switchOp) {
            OpOperand* const qubit = switchOp.getTiedTarget(res);
            auto defaultArg = switchOp.getTiedDefaultBlockArgument(qubit);
            auto defaultYielded =
                switchOp.getTiedDefaultYieldedValue(defaultArg)->get();
            append(children[0], defaultArg, defaultYielded);

            for (size_t r = 1; r < switchOp.getNumRegions(); ++r) {
              auto arg = switchOp.getTiedCaseBlockArgument(qubit, r - 1);
              auto yielded =
                  switchOp.getTiedCaseYieldedValue(arg, r - 1)->get();
              append(children[r], arg, yielded);
            }
          });

      permutation[qubitResNum] = hw;
    }

    // Route each child branch and prepare the wire iterators for
    // epilogue SWAP insertion, i.e., point each iterator at the final
    // qubit op (note: might be a measurement) before the yield.

    Statistics totalStats;

    for (auto& child : children) {
      const auto stats = route<Direction, Mode>(child, rewriter);
      if (failed(stats)) {
        return failure();
      }

      totalStats.merge(*stats);

      if constexpr (Mode == RoutingMode::Hot) {
        for_each(child.wires, [](auto& it) { std::ranges::advance(it, -1); });
      }
    }

    // Exception: The layout of the "after" region depends on the final layout
    // of the before region. Thus, create / route the second child region /
    // bundle here.

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      children.emplace_back(RoutingBundle{.layout = children[0].layout});
      assert(children.size() == 2);

      auto values = [&] -> ValueRange {
        if constexpr (Direction == WireDirection::Forward) {
          return whileOp.getAfterArguments();
        }
        Operation* const terminator = whileOp.getAfterBody()->getTerminator();
        return cast<scf::YieldOp>(terminator).getResults();
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
        for_each(children[1].wires,
                 [](auto& it) { std::ranges::advance(it, -1); });
      }
    }

    // Find (insert) the epilogue SWAP sequence for (into) the child region
    // using the restore (scf::ForOp, scf::While), converge (IfOp), and vote
    // and restore (IndexSwitchOp) strategies.

    Layout exit =
        TypeSwitch<Operation*, Layout>(op)
            .Case([&](scf::ForOp) {
              const auto swaps = restore(children[0].layout, parent.layout);
              insertSWAPs<Mode>(swaps, children[0], totalStats, rewriter);
              return parent.layout;
            })
            .Case([&](scf::WhileOp) {
              const auto swaps = restore(children[1].layout, parent.layout);
              insertSWAPs<Mode>(swaps, children[1], totalStats, rewriter);
              // The scf::YieldOp is the terminator in the before region and
              // thus determines the final output layout.
              return children[0].layout;
            })
            .Case([&](IfOp) {
              const auto [convergedLayout, fst, snd] =
                  converge(children[0].layout, children[1].layout);
              insertSWAPs<Mode>(fst, children[0], totalStats, rewriter);
              insertSWAPs<Mode>(snd, children[1], totalStats, rewriter);
              return convergedLayout;
            })
            .Case([&](IndexSwitchOp) {
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
            .Case([&](scf::YieldOp yieldOp) {
              rewriter->replaceOpWithNewOp<scf::YieldOp>(
                  yieldOp,
                  realignQubitValues(yieldOp.getResults(), permutation, child));
            })
            .Case([&](scf::ConditionOp condOp) {
              rewriter->replaceOpWithNewOp<scf::ConditionOp>(
                  condOp, condOp.getCondition(),
                  realignQubitValues(condOp.getArgs(), permutation, child));
            })
            .Case([&](YieldOp yieldOp) {
              rewriter->replaceOpWithNewOp<YieldOp>(
                  yieldOp,
                  realignQubitValues(yieldOp.getTargets(), permutation, child));
            });

        // Sort topologically to fix any occurring SSA dominance errors.

        // Fix SSA dominance errors.
        reorderTopologically(*block, *rewriter);
      }
    }

    // If the operation is a scf::ForOp, where the parent.layout =
    // child.layout, we are done. Otherwise, propagate a patch with the final
    // layout and index-to-program mapping.

    if (isa<scf::ForOp>(op)) {
      return totalStats;
    }

    WireInfos updatedInfos;
    for (size_t i = 0; i < parent.wires.size(); ++i) {
      const auto oldProg = parent.infos.lookupProgram(i);
      const auto oldHw = parent.layout.getHardwareIndex(oldProg);
      const auto newProg = exit.getProgramIndex(oldHw);
      updatedInfos.insertOrUpdate(i, newProg);
    }
    parent.infos = std::move(updatedInfos);
    parent.layout = std::move(exit);
    return totalStats;
  }

  /// Iterates over a dynamically computed window of layers and uses A* search
  /// to find a SWAP sequence that makes each layer executable. Depending on
  /// the template parameter, this function only updates the layout or also
  /// inserts the SWAPs into the IR. Returns the accumulated statistics, or
  /// failure if routing a nested operation fails.
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
            place(composite, bundle, *rewriter);
          }

          auto res = dispatch<Direction, Mode>(composite, bundle, rewriter);
          if (failed(res)) {
            return failure();
          }

          stats.merge(*res);

          // Once the composite is mapped, move past this op by incrementing
          // the respective wires.

          for_each(composite.indices, [&](size_t i) {
            std::ranges::advance(wires[i],
                                 WireTraversalTraits<Direction>::stride());
          });
        }
      }

      const auto window = getWindow<Direction>(wires, infos);
      if (window.empty()) {
        break;
      }

      const auto swaps = search(window, layout);

      if constexpr (Mode == RoutingMode::Hot) {

        // At this point the wire iterators point to sink-like operations
        // (e.g. SinkOp, YieldOp), measurements, or two-qubit gate of the
        // subsequent layer. Decrementing once ensures that the wire iterators
        // point at the input qubits of those operations.

        for_each(wires, [](auto& it) { std::ranges::advance(it, -1); });
      }

      insertSWAPs<Mode>(swaps, bundle, stats, rewriter);

      if constexpr (Mode == RoutingMode::Hot) {

        // After SWAP insertion, a wire is either untouched by the SWAP
        // insertion or pointing at a SWAP operation. If the former is the
        // case, incrementing the wire iterator will undo the previous
        // decrement, leaving it at the same position as before the SWAP
        // insertion. Otherwise, an increment will move the iterator past the
        // inserted SWAP operation.

        for_each(wires, [](auto& it) { std::ranges::advance(it, 1); });
      }
    }

    return stats;
  }

  const CompilerTarget* target = nullptr;
};

} // namespace

std::unique_ptr<Pass> createPlacementPass(const CompilerTarget& target) {
  return std::make_unique<PlacementPass>(target);
}

} // namespace mlir::qco
