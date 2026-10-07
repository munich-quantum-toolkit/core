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
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <llvm/Support/ErrorHandling.h>
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

/// Widen this alias when targets need more than 65,535 sites.
using QubitIndex = uint16_t;
using QubitIndexPair = std::pair<QubitIndex, QubitIndex>;

struct TensorAllocation {
  qtensor::AllocOp allocation;
  SmallVector<Operation*> operations;
};

struct Computation {
  Wires wires;
  SmallVector<AllocOp> scalarAllocations;
  SmallVector<TensorAllocation> tensorAllocations;
};

/// Consume source labels while the existing placement loop replaces roots.
class LayoutRecorder {
public:
  LayoutRecorder(func::FuncOp func, const CompilerTarget& target)
      : module_(func->getParentOfType<ModuleOp>()) {
    if (auto count =
            module_->getAttrOfType<IntegerAttr>(mqt::kSourceQubitCountAttr)) {
      layout_.emplace();
      layout_->initial.assign(target.numSites(), -1);
      layout_->inputCount = count.getInt();
      layout_->sites.emplace(target.siteIds().begin(), target.siteIds().end());
      valid_ = count.getInt() >= 0 &&
               std::cmp_less_equal(count.getInt(), target.numSites());
    }
  }

  void record(Operation* root, std::optional<int64_t> slot, size_t vertex) {
    if (!layout_) {
      return;
    }
    auto indices =
        root->getAttrOfType<DenseI64ArrayAttr>(mqt::kSourceQubitIndicesAttr);
    if (!indices || !slot || *slot < 0 || *slot >= indices.size() ||
        vertex >= layout_->initial.size()) {
      valid_ = false;
      return;
    }
    const auto source = indices[*slot];
    if (source < 0 || source >= layout_->inputCount ||
        layout_->initial[source] != -1) {
      valid_ = false;
      return;
    }
    layout_->initial[source] = static_cast<int64_t>(vertex);
  }

  LogicalResult finish() {
    module_->removeAttr(mqt::kSourceQubitCountAttr);
    if (!valid_) {
      return module_.emitError(
          "input qubit identity was lost during placement");
    }
    if (!layout_) {
      return success();
    }
    std::vector<bool> used(layout_->initial.size(), false);
    for (const auto vertex : layout_->initial) {
      if (vertex >= 0) {
        if (used[vertex]) {
          return module_.emitError("input qubits share a physical site");
        }
        used[vertex] = true;
      }
    }
    size_t next = 0;
    for (auto& vertex : layout_->initial) {
      if (vertex < 0) {
        while (used[next]) {
          ++next;
        }
        vertex = static_cast<int64_t>(next);
        used[next] = true;
      }
    }
    module_->setAttr("mqt.layout", layout_->toAttr(module_.getContext()));
    return success();
  }

private:
  ModuleOp module_;
  std::optional<mqt::QubitLayout> layout_;
  bool valid_ = true;
};

} // namespace

/// Check the structural input contract before traversing qubit wires.
static LogicalResult validateRoutingOperations(func::FuncOp func) {
  if (!llvm::hasSingleElement(func.getBody())) {
    return func.emitError("mapping requires a single-block entry function");
  }

  const auto result =
      func.walk([](Operation* op) {
        if (isa<CallOpInterface>(op) &&
            (llvm::any_of(op->getOperandTypes(), isLinearQubitType) ||
             llvm::any_of(op->getResultTypes(), isLinearQubitType))) {
          op->emitError("inline calls that carry qubits before mapping");
          return WalkResult::interrupt();
        }

        if (op->getNumRegions() == 0 &&
            !isa<QCODialect, qtensor::QTensorDialect, cbit::CBitDialect>(
                op->getDialect()) &&
            !isMemoryEffectFree(op)) {
          op->emitError(
              "mapping supports classical side effects only through CBit "
              "operations; lower other side effects before mapping");
          return WalkResult::interrupt();
        }

        if (auto unitary = dyn_cast<UnitaryOpInterface>(op);
            unitary && !isa<BarrierOp>(op) && unitary.getNumQubits() > 2) {
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

  for (Operation& op : func.getOps()) {
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
  if (target.numSites() > std::numeric_limits<QubitIndex>::max()) {
    return func.emitError()
           << "target site count exceeds mapping index capacity ("
           << +std::numeric_limits<QubitIndex>::max() << ")";
  }

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
static FailureOr<Wires> applyPlacement(Region& body,
                                       const CompilerTarget& target,
                                       const Layout<QubitIndex>& layout,
                                       const Computation& computation,
                                       IRRewriter& rewriter) {
  SmallVector<Value> staticQubits;
  const auto nhardware = layout.nHardwareQubits();
  staticQubits.reserve(nhardware);

  rewriter.setInsertionPointToStart(&body.front());
  for (size_t hw = 0; hw < nhardware; ++hw) {
    auto op =
        StaticOp::create(rewriter, body.getLoc(), target.siteForVertex(hw));
    staticQubits.emplace_back(op.getQubit());
    rewriter.setInsertionPointAfter(op);
  }

  size_t prog = 0;

  LayoutRecorder recorder(cast<func::FuncOp>(body.getParentOp()), target);
  for (auto alloc : computation.scalarAllocations) {
    const auto vertex = layout.getHardwareIndex(prog++);
    recorder.record(alloc, 0, vertex);
    auto qubit = staticQubits[vertex];

    rewriter.replaceAllUsesWith(alloc.getResult(), qubit);
    rewriter.eraseOp(alloc);
  }

  for (const auto& tensor : computation.tensorAllocations) {
    for (Operation* operation : tensor.operations) {
      TypeSwitch<Operation*>(operation)
          .Case([&](ExtractOp op) {
            const auto vertex = layout.getHardwareIndex(prog++);
            recorder.record(tensor.allocation,
                            getConstantIntValue(op.getIndex()), vertex);
            auto qubit = staticQubits[vertex];

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
  for (; prog < nhardware; ++prog) {
    const auto hw = layout.getHardwareIndex(prog);
    auto qubit = staticQubits[hw];

    SinkOp::create(rewriter, body.getLoc(), qubit);
  }

  if (failed(recorder.finish())) {
    return failure();
  }

  return map_to_vector(staticQubits,
                       [](Value qubit) { return WireIterator(qubit); });
}

/// Assign allocation slots to sites without traversing or expanding their uses.
static LogicalResult placeIndexedAllocations(func::FuncOp func,
                                             const CompilerTarget& target,
                                             IRRewriter& rewriter) {
  SmallVector<Operation*> allocations;
  size_t required = 0;
  for (Operation& op : func.getOps()) {
    size_t width = 0;
    if (isa<AllocOp>(op)) {
      width = 1;
    } else if (auto tensor = dyn_cast<qtensor::AllocOp>(op)) {
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
      return func.emitError()
             << "requires more program qubits than the target site count of "
             << target.numSites();
    }
    required += width;
    allocations.push_back(&op);
  }

  size_t vertex = 0;
  LayoutRecorder recorder(func, target);
  for (Operation* alloc : allocations) {
    rewriter.setInsertionPoint(alloc);
    int64_t slot = 0;
    const auto nextQubit = [&] {
      recorder.record(alloc, slot++, vertex);
      return StaticOp::create(rewriter, alloc->getLoc(),
                              target.siteForVertex(vertex++));
    };
    if (isa<AllocOp>(alloc)) {
      auto qubit = nextQubit();
      alloc->removeAttr(mqt::kSourceQubitIndicesAttr);
      qubit->setDiscardableAttrs(alloc->getDiscardableAttrDictionary());
      rewriter.replaceOp(alloc, qubit.getQubit());
      continue;
    }
    auto type = cast<RankedTensorType>(alloc->getResult(0).getType());
    SmallVector<Value> qubits;
    qubits.reserve(static_cast<size_t>(type.getNumElements()));
    for (int64_t index = 0; index < type.getNumElements(); ++index) {
      qubits.push_back(nextQubit().getQubit());
    }
    auto tensor = qtensor::FromElementsOp::create(rewriter, alloc->getLoc(),
                                                  type, qubits);
    alloc->removeAttr(mqt::kSourceQubitIndicesAttr);
    tensor->setDiscardableAttrs(alloc->getDiscardableAttrDictionary());
    rewriter.replaceOp(alloc, tensor.getResult());
  }

  return recorder.finish();
}

/// Returns true, if a function has not been placed before.
static bool needsPlacement(func::FuncOp func) {
  if (!func.getOps<StaticOp>().empty()) {
    return false;
  }

  auto moduleOp = func->getParentOfType<ModuleOp>();
  return moduleOp->hasAttr(mqt::kSourceQubitCountAttr) ||
         !func.getOps<AllocOp>().empty() ||
         !func.getOps<qtensor::AllocOp>().empty();
}

namespace {
struct PlacementPass final
    : PassWrapper<PlacementPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlacementPass)

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

    if (!needsPlacement(func)) {
      return;
    }

    const auto& targetAnalysis = getAnalysis<TargetEnvironmentAnalysis>();
    if (!targetAnalysis) {
      moduleOp.emitError() << "expected a valid mqt.target_env: "
                           << targetAnalysis.error();
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    const auto& target = targetAnalysis.environment().target();
    if (targetAnalysis.environment().supportsIndexedQubits()) {
      if (failed(placeIndexedAllocations(func, target, rewriter))) {
        signalPassFailure();
      }
      return;
    }

    const auto computation = discoverComputation(func);
    if (failed(computation) ||
        failed(checkCapacity(func, target, *computation))) {
      signalPassFailure();
      return;
    }

    const auto layout = Layout<QubitIndex>::identity(computation->wires.size());
    if (failed(applyPlacement(func.getFunctionBody(), target, layout,
                              *computation, rewriter))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace {

/// Return the shortest-path distance between a gate's program qubits after
/// applying the given layout.
///
/// The returned distance is measured on the target connectivity graph.
[[nodiscard]] static size_t distance(const QubitIndexPair& gate,
                                     const Layout<QubitIndex>& layout,
                                     const CompilerTarget& target) {
  const auto [hw0, hw1] = layout.getHardwareIndices(gate.first, gate.second);
  return target.distanceBetween(hw0, hw1);
}

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using Score = std::pair<size_t, size_t>;

  enum class RoutingMode : bool { Cold, Hot };
  enum class RoutingPolicy : bool { Serial, Wave };

  /// Invocation data is prepared before trials, then borrowed read-only.
  struct Environment {
    const CompilerTarget& target;
    uint64_t seed;
    std::unique_ptr<const NativeCostTable> nativeCosts;
    std::optional<size_t> nativeSwapCost;

    void prepareNativeCosts(Operation* root) {
      const auto& basis = target.synthesisBasis();
      if (!basis || !basis->entangler ||
          target.nativeOperationsKind() !=
              CompilerTarget::NativeOperations::Kind::Explicit) {
        return;
      }
      nativeCosts = NativeCostTable::precompute(root, *basis->entangler, seed);

      /// Uniform SWAP costs keep the distance heuristic in native-gate units.
      NativeCostAnalysis analysis(seed, nativeCosts.get());
      for (const auto& [a, b] : target.couplings()) {
        const auto next = analysis.swapCost(target, std::array{a, b});
        if (!next || (nativeSwapCost && nativeSwapCost != next)) {
          nativeSwapCost = std::nullopt;
          break;
        }
        nativeSwapCost = next;
      }
    }
  };

  /// Describes a control-flow operation that acts on qubits.
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
    static RoutingState fromLayout(const Wires& roots,
                                   const Layout<QubitIndex>& layout,
                                   const Environment& env) {
      RoutingState state(Wires(layout.nHardwareQubits()), layout, env);
      for (auto [program, wire] : enumerate(roots)) {
        state.wires[layout.getHardwareIndex(program)] = wire;
      }
      return state;
    }

    /// Construct a routing state from a vector of wires and a layout.
    RoutingState(Wires wires, Layout<QubitIndex> layout, const Environment& env)
        : wires(std::move(wires)), layout(std::move(layout)) {
      if (env.nativeCosts) {
        costs.emplace(env.target, env.seed, env.nativeCosts.get());
      }
    }

    Wires wires;
    Layout<QubitIndex> layout;
    std::optional<NativeCostTracker> costs;
  };

  /// Describes a SWAP and its associated costs.
  struct SwapCandidate {
    /// The hardware indices on which the SWAP acts.
    QubitIndexPair indices;
    /// The local (uniform) costs of a SWAP.
    size_t standalone;
    /// Signed first-step prefix adjustment.
    int64_t prefix;
  };

  struct LayoutTrial {
    Layout<QubitIndex> layout;
    /// The routing policy to test.
    RoutingPolicy policy{RoutingPolicy::Serial};
    /// Synthesis available: (native-count, depth). Otherwise, (max(), swaps).
    Score score;
  };

  /// A fast, flat data structure designed specifically for layer-by-layer
  /// iteration, where the elements are stored sequentially in a single
  /// contiguous buffer.
  struct Horizon {
    /// Start a new and empty layer.
    void next() { offsets_.emplace_back(storage_.size()); }

    /// Append a single element to the current active layer.
    void push(const QubitIndexPair& gate) {
      assert(offsets_.size() > 1 && "No active layer. Call next() first.");
      storage_.emplace_back(gate);
      offsets_.back() = storage_.size();
    }

    /// Returns a slice view of a specific layer.
    [[nodiscard]] ArrayRef<QubitIndexPair> get(size_t i) const {
      assert(i < nlayers() && "layer index out of bounds");
      const auto start = offsets_[i];
      const auto end = offsets_[i + 1];
      return {storage_.data() + start, end - start};
    }

    /// Returns a mutable slice view of a specific layer.
    [[nodiscard]] MutableArrayRef<QubitIndexPair> get(size_t i) {
      assert(i < nlayers() && "layer index out of bounds");
      const auto start = offsets_[i];
      const auto end = offsets_[i + 1];
      return {storage_.data() + start, end - start};
    }

    /// Return the number of elements stored across all layers.
    [[nodiscard]] size_t size() const { return storage_.size(); }

    /// Returns the total number of layers.
    [[nodiscard]] size_t nlayers() const { return offsets_.size() - 1; }

    /// Returns true, if there are no elements in the horizon.
    [[nodiscard]] bool empty() const { return storage_.empty(); }

  private:
    /// Flat contiguous array storing all elements across the layers.
    SmallVector<QubitIndexPair> storage_;
    /// Offsets marking the start index of each layer, where [l]
    /// defines the start of the layer l and [l + 1] its end.
    SmallVector<size_t, 8> offsets_ = {0};
  };

  /// Interpret the horizon as a serial sequence of gates.
  struct SerialPolicy {
    explicit SerialPolicy(Horizon horizon, const Layout<QubitIndex>& layout,
                          const CompilerTarget& target)
        : horizon_(std::move(horizon)) {
      for (size_t i = 0; i < horizon_.nlayers(); ++i) {
        llvm::stable_sort(horizon_.get(i), [&](const auto& lhs,
                                               const auto& rhs) {
          return distance(lhs, layout, target) < distance(rhs, layout, target);
        });
      }
    }

    /// Estimates routing cost by summing distance-based SWAP counts over the
    /// lookahead horizon with exponential decay.
    ///
    /// Computes the minimal number of SWAPs required to route each gate in
    /// each layer. For each gate, this is determined by the shortest distance
    /// between its hardware qubits. Intuitively, this is the number of SWAPs
    /// that a naive router would insert (with a constant layout).
    [[nodiscard]] float heuristic(const Layout<QubitIndex>& layout,
                                  const CompilerTarget& target,
                                  const Parameters& params) const {
      float costs{0};
      float weight{1.};
      for (size_t i = 0; i < horizon_.nlayers(); ++i) {
        for (const auto& gate : horizon_.get(i)) {
          const auto nswaps = distance(gate, layout, target) - 1;
          costs += weight * static_cast<float>(nswaps);
          weight *= params.lambda; // Each gate increases the weight.
        }
      }
      return costs;
    }

    /// Return the to be routed gate, i.e., the objective.
    [[nodiscard]] QubitIndexPair objective() const {
      return horizon_.get(0).front();
    }

  private:
    Horizon horizon_;
  };

  /// Interpret the horizon as a wave of ripples starting from a vertex gate.
  struct WavePolicy {
    explicit WavePolicy(Horizon horizon, const Layout<QubitIndex>& layout,
                        const CompilerTarget& target)
        : horizon_(std::move(horizon)) {
      objective_ = *llvm::min_element(horizon_.get(0), [&](const auto& lhs,
                                                           const auto& rhs) {
        return distance(lhs, layout, target) < distance(rhs, layout, target);
      });
    }

    [[nodiscard]] float heuristic(const Layout<QubitIndex>& layout,
                                  const CompilerTarget& target,
                                  const Parameters& params) const {
      float weight{1.};
      float costs{
          static_cast<float>(distance(objective(), layout, target) - 1)};

      for (size_t i = 0; i < horizon_.nlayers(); ++i) {
        weight *= params.lambda;
        const auto layer =
            i == 0 ? horizon_.get(0).drop_front() : horizon_.get(i);
        for (const auto& gate : layer) {
          const auto nswaps = distance(gate, layout, target) - 1;
          costs += weight * static_cast<float>(nswaps);
        }
      }

      return costs;
    }

    /// Return the to be routed gate, i.e., the objective.
    [[nodiscard]] QubitIndexPair objective() const { return objective_; }

  private:
    Horizon horizon_;
    QubitIndexPair objective_;
  };

  /// Describes a node in the A* search graph.
  struct Node {
    Layout<QubitIndex> layout;
    QubitIndexPair swap;
    Node* parent = nullptr;
    int64_t cost = 0;
    size_t depth = 0;
    float f = 0;

    /// Reuse layout capacity when starting a new search.
    void initializeRoot(const Layout<QubitIndex>& initialLayout) {
      layout = initialLayout;
      swap = {};
      parent = nullptr;
      depth = 0;
      cost = 0;
      f = 0;
    }

    /// Initialize a child from its parent using a templated heuristic functor.
    template <class Policy>
    void initializeChild(Node* nextParent, const SwapCandidate& candidate,
                         const Policy& policy, const CompilerTarget& target,
                         const Parameters& params) {
      layout = nextParent->layout;
      layout.swap(candidate.indices.first, candidate.indices.second);

      depth = nextParent->depth + 1;
      cost = nextParent->cost + static_cast<int64_t>(candidate.standalone) +
             candidate.prefix;

      const float g = params.alpha * static_cast<float>(cost);
      const float h = static_cast<float>(candidate.standalone) *
                      policy.heuristic(layout, target, params);
      f = g + h;

      swap = candidate.indices;
      parent = nextParent;
    }

    /// Return true, if the current SWAP sequence makes all gates in the front
    /// executable.
    [[nodiscard]] bool isGoal(const QubitIndexPair& front,
                              const CompilerTarget& target) const {
      const auto [hw0, hw1] =
          layout.getHardwareIndices(front.first, front.second);
      return target.areAdjacent(hw0, hw1);
    }

    /// Return true, if the current node is the root node.
    [[nodiscard]] bool isRoot() const { return parent == nullptr; }

    /// Return the sequence of SWAPs from the root to this node.
    [[nodiscard]] SmallVector<QubitIndexPair> swaps() const {
      SmallVector<QubitIndexPair> seq(depth);
      auto it = seq.rbegin();
      for (const Node* n = this; n->parent != nullptr; n = n->parent, ++it) {
        *it = n->swap;
      }
      return seq;
    }
  };

  /// A deduplicated priority queue for A* search nodes.
  struct SearchFrontier {
    /// Push a node onto the frontier.
    void push(Node* node) {
      auto*& incumbent = best[node->layout.getProgramToHardware()];
      if (incumbent == nullptr || node->cost < incumbent->cost) {
        incumbent = node;
        queue.push(node);
      }
    }

    /// Pop a node from the frontier.
    [[nodiscard]] Node* pop() {
      while (!queue.empty()) {
        Node* node = queue.top();
        queue.pop();

        const auto key = node->layout.getProgramToHardware();

        // If the node matches the entry in the best map, it's valid.
        // Otherwise, the node was superseded by a cheaper state. Thus, drop it.

        if (best.lookup(key) == node) {
          return node;
        }
      }

      return nullptr;
    }

  private:
    struct CompareNodePointer {
      bool operator()(const Node* lhs, const Node* rhs) const {
        return lhs->f > rhs->f;
      }
    };

    /// Priority queue of node pointers managed by the caller.
    llvm::PriorityQueue<Node*, std::vector<Node*>, CompareNodePointer> queue;
    /// Maps a layout to the node that reached it using the lowest cost.
    DenseMap<ArrayRef<QubitIndex>, Node*> best;
  };

  /// Memory arena for A* search nodes, enabling reuse across searches to reduce
  /// allocation overhead. Initializing a retained node reuses its layout.
  struct Arena {
    /// Constructs an arena with a limited memory budget.
    /// The budget of nodes is derived as
    ///
    ///    `searchMemoryLimit / (sizeof(Node) + 2 * nsites *
    ///    sizeof(QubitIndex))`
    ///
    /// where the final summand accounts for the Node's layout member.
    explicit Arena(size_t nsites, size_t searchMemoryLimit)
        : budget(std::max<size_t>(
              1, searchMemoryLimit /
                     (sizeof(Node) + 2 * nsites * sizeof(QubitIndex)))) {}

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
    void construct(const Layout<QubitIndex>& from,
                   const Layout<QubitIndex>& to) {
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
    [[nodiscard]] std::optional<SmallVector<QubitIndexPair>>
    findHappySWAPChain() const {
      const auto optCycle = f_.findCycle();
      if (!optCycle) {
        return std::nullopt;
      }
      const auto& cycle = *optCycle;

      SmallVector<QubitIndexPair> swaps;
      for (size_t i = cycle.size() - 1; i > 0; --i) {
        swaps.emplace_back(cycle[i], cycle[i - 1]);
      }
      return swaps;
    }

    /// Find an unhappy SWAP. That is, find an edge (u, v), where exchanging u
    /// and v, reduces u's distance to its target location (by one) and
    /// increases v's distance from 0 (already at the correct location) to one.
    [[nodiscard]] std::optional<QubitIndexPair> findUnhappySWAP() const {
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
    [[nodiscard]] bool shouldAddEdge(const QubitIndex u, const QubitIndex v,
                                     const Layout<QubitIndex>& from,
                                     const Layout<QubitIndex>& to) const {
      const auto dest = to.getHardwareIndex(from.getProgramIndex(u));
      return target_->distanceBetween(v, dest) <
             target_->distanceBetween(u, dest);
    }

    Graph f_;
    const CompilerTarget* target_;
  };

  /// A stateful iterator over routing boundaries in a block.
  ///
  /// A routing boundary is a control flow operation (IfOp, IndexSwitchOp,
  /// scf::ForOp, scf::WhileOp) that produces qubit types. These boundaries
  /// delimit regions where qubit routing must account for control flow.
  template <WireDirection Direction> struct Boundary {
    /// Initialize the routing boundary.
    explicit Boundary(Block& block) {
      if constexpr (Direction == WireDirection::Forward) {
        boundary_ = findNextBoundary(&block.front());
      } else {
        boundary_ = findNextBoundary(&block.back());
      }
    }

    /// Advance to the next routing boundary.
    void setNextBoundary() { boundary_ = findNextBoundary(next(boundary_)); }

    /// Return the current boundary operation, or nullptr if exhausted.
    [[nodiscard]] Operation* operation() const { return boundary_; }

  private:
    /// Starting from op, walk the IR in block-order until a boundary operation
    /// is discovered or the block is exhausted.
    Operation* findNextBoundary(Operation* op) {
      for (; op != nullptr; op = next(op)) {
        if (isa<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>(op) &&
            any_of(op->getResultTypes(),
                   [](Type type) { return isa<QubitType>(type); })) {
          return op;
        }
      }
      return nullptr;
    }

    /// Return the next (or previous) operation in block-order.
    static Operation* next(Operation* op) {
      return Direction == WireDirection::Forward ? op->getNextNode()
                                                 : op->getPrevNode();
    }

    Operation* boundary_;
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

    const auto& targetAnalysis = getAnalysis<TargetEnvironmentAnalysis>();
    if (!targetAnalysis) {
      moduleOp.emitError() << "expected a valid mqt.target_env: "
                           << targetAnalysis.error();
      signalPassFailure();
      return;
    }

    const auto& target = targetAnalysis.environment().target();
    if (target.connectivityKind() !=
        CompilerTarget::Connectivity::Kind::Explicit) {
      moduleOp.emitError() << "expected an explicit target topology";
      signalPassFailure();
      return;
    }

    auto func = mqt::getEntryPoint(moduleOp);
    if (!func) {
      moduleOp.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    if (!needsPlacement(func)) {
      return;
    }

    if (failed(validateRoutingOperations(func))) {
      signalPassFailure();
      return;
    }

    auto computation = discoverComputation(func);
    if (failed(computation) ||
        failed(checkCapacity(func, target, *computation))) {
      signalPassFailure();
      return;
    }

    Environment env{.target = target, .seed = compilationSeed(moduleOp, seed)};
    const auto [layout, policyTag, expectedScore] =
        generateLayout(computation->wires, func, env);

    IRRewriter rewriter(&getContext());
    Arena arena(target.numSites(), searchMemoryLimit);
    auto wires = applyPlacement(func.getFunctionBody(), target, layout,
                                *computation, rewriter);
    if (failed(wires)) {
      signalPassFailure();
      return;
    }

    Statistics stats;
    RoutingState state(std::move(*wires), layout, env);
    switch (policyTag) {
    case RoutingPolicy::Serial:
      stats = route<WireDirection::Forward, SerialPolicy, RoutingMode::Hot>(
          func.getFunctionBody(), state, arena, env, &rewriter);
      break;
    case RoutingPolicy::Wave:
      stats = route<WireDirection::Forward, WavePolicy, RoutingMode::Hot>(
          func.getFunctionBody(), state, arena, env, &rewriter);
      break;
    default:
      llvm_unreachable("unknown routing policy");
    }

    assert(((state.costs ? state.costs->score() : std::nullopt)
                .value_or(std::pair{std::numeric_limits<size_t>::max(),
                                    stats.nswaps}) == expectedScore) &&
           "cold scoring and hot routing must agree");

    if (auto attr = moduleOp->getAttrOfType<DictionaryAttr>("mqt.layout")) {
      const auto permutation = sitePermutation(layout, state.layout);
      const SmallVector<int64_t> routing(permutation.begin(),
                                         permutation.end());
      NamedAttrList fields(attr);
      fields.set("routing", rewriter.getDenseI64ArrayAttr(routing));
      moduleOp->setAttr("mqt.layout", fields.getDictionary(&getContext()));
    }

    // Collect statistics.
    numSwaps += stats.nswaps;

    // Fix SSA dominance errors.
    reorderTopologically(func.getFunctionBody().front(), rewriter);
  }

private:
  /// Return true, if a precedes b (or b precedes a for backwards iteration).
  template <WireDirection Direction>
  static bool precedes(Operation* a, Operation* b) {
    if constexpr (Direction == WireDirection::Forward) {
      return a->isBeforeInBlock(b);
    }
    return b->isBeforeInBlock(a);
  }

  /// Return values carried by a supported region terminator.
  static ValueRange yieldedValues(Block& block) {
    return TypeSwitch<Operation*, ValueRange>(block.getTerminator())
        .Case([](scf::YieldOp op) { return op.getResults(); })
        .Case([](scf::ConditionOp op) { return op.getArgs(); })
        .Case([](YieldOp op) { return op.getTargets(); });
  }

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

  /// Return an initial layout and whether identity needs no routing.
  ///
  /// Otherwise, place frequently interacting qubits near each other. Nested
  /// control flow has no single interaction frequency, so leave those programs
  /// to the identity and random starts.
  [[nodiscard]] std::optional<std::pair<Layout<QubitIndex>, bool>>
  generateGreedyLayout(Wires wires, const Environment& env) const {
    DenseMap<QubitIndexPair, size_t> weights;
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
          return env.target.areAdjacent(a, b);
        })) {
      return std::pair{Layout<QubitIndex>::identity(env.target.numSites()),
                       true};
    }

    const size_t nprogram = wires.size();
    const size_t nhardware = env.target.numSites();
    SmallVector<SmallVector<std::pair<size_t, size_t>>> neighbours(nprogram);
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
          env.target.forEachNeighbour(hw, [&](size_t) { ++remaining[hw]; });
        }
        auto current = static_cast<size_t>(
            std::distance(remaining.begin(), llvm::min_element(remaining)));
        SmallVector<QubitIndex> mapping(nhardware,
                                        static_cast<QubitIndex>(nhardware));
        size_t placed = 0;
        while (current != nhardware && placed < nprogram) {
          mapping[order[placed++]] = static_cast<QubitIndex>(current);
          usedHardware[current] = true;
          env.target.forEachNeighbour(
              current, [&](size_t neighbour) { --remaining[neighbour]; });
          size_t next = nhardware;
          env.target.forEachNeighbour(current, [&](size_t neighbour) {
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
              mapping[placed++] = static_cast<QubitIndex>(hw);
            }
          }
          return std::pair{Layout<QubitIndex>::fromMapping(mapping), true};
        }
      }
    }

    SmallVector<size_t> centrality(nhardware, 0);
    SmallVector<size_t> hardwareDegree(nhardware, 0);
    for (size_t hw = 0; hw < nhardware; ++hw) {
      for (size_t other = 0; other < nhardware; ++other) {
        centrality[hw] += env.target.distanceBetween(hw, other);
      }
      env.target.forEachNeighbour(hw, [&](size_t) { ++hardwareDegree[hw]; });
    }

    // The out-of-range hardware index marks an unplaced program qubit.
    SmallVector<QubitIndex> mapping(nhardware,
                                    static_cast<QubitIndex>(nhardware));
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
            cost += weight * env.target.distanceBetween(hw, mapping[partner]);
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
      mapping[prog] = static_cast<QubitIndex>(best);
      used[best] = true;
      for (const auto& [partner, weight] : neighbours[prog]) {
        attached[partner] += weight;
      }
    }

    // Complete the permutation with unused sites for routing workspace.
    size_t prog = nprogram;
    for (size_t hw = 0; hw < nhardware; ++hw) {
      if (!used[hw]) {
        mapping[prog++] = static_cast<QubitIndex>(hw);
      }
    }
    return std::pair{Layout<QubitIndex>::fromMapping(mapping), false};
  }

  template <class Policy>
  void refineLayout(LayoutTrial& trial, const Wires& wires, Region& region,
                    Environment& env) {
    Arena arena(env.target.numSites(), searchMemoryLimit);

    {
      auto state = RoutingState::fromLayout(wires, trial.layout, env);
      for (size_t i = 0; i < niterations; ++i) {
        route<WireDirection::Forward, Policy>(region, state, arena, env);
        route<WireDirection::Backward, Policy>(region, state, arena, env);
      }

      trial.layout = std::move(state.layout);
    }

    auto state = RoutingState::fromLayout(wires, trial.layout, env);
    const auto stats =
        route<WireDirection::Forward, Policy>(region, state, arena, env);

    if (state.costs && state.costs->score()) {
      trial.score = *state.costs->score();
    } else {
      trial.score =
          std::make_pair(std::numeric_limits<size_t>::max(), stats.nswaps);
    }
  }

  /// Refine greedy, identity, and random starts with forward/backward routing.
  /// Score each candidate with a forward traversal, preserving its start
  /// layout.
  LayoutTrial generateLayout(const Wires& wires, func::FuncOp func,
                             Environment& env) {
    const auto greedy = generateGreedyLayout(wires, env);
    if (greedy && greedy->second) {
      return {.layout = greedy->first,
              .score = std::make_pair(std::numeric_limits<size_t>::max(), 0)};
    }

    env.prepareNativeCosts(func);

    const size_t nlayouts = (ntrials + 1) / 2;
    SmallVector<Layout<QubitIndex>, 0> layouts;
    layouts.reserve(nlayouts);

    if (greedy) {
      layouts.emplace_back(greedy->first);
    }

    auto rng = makeMt19937(env.seed);
    if (layouts.size() < nlayouts) {
      layouts.emplace_back(Layout<QubitIndex>::identity(env.target.numSites()));
      while (layouts.size() < nlayouts) {
        layouts.emplace_back(Layout<QubitIndex>::random(
            env.target.numSites(), env.target.numSites(), rng()));
      }
    }

    SmallVector<LayoutTrial, 0> trials;
    trials.reserve(ntrials);
    for (auto& layout : layouts) {
      trials.emplace_back(layout, RoutingPolicy::Serial);
      if (trials.size() < ntrials) {
        trials.emplace_back(layout, RoutingPolicy::Wave);
      }
    }

    assert(ntrials == trials.size());

    parallelForEach(&getContext(), trials, [&, this](LayoutTrial& trial) {
      switch (trial.policy) {
      case RoutingPolicy::Serial:
        refineLayout<SerialPolicy>(trial, wires, func.getFunctionBody(), env);
        break;
      case RoutingPolicy::Wave:
        refineLayout<WavePolicy>(trial, wires, func.getFunctionBody(), env);
        break;
      }
    });

    return *llvm::min_element(trials,
                              [](const LayoutTrial& a, const LayoutTrial& b) {
                                return a.score < b.score;
                              });
  }

  /// Route the objective interaction with bounded A* node storage and templated
  /// routing policy. Drain queued states at the limit, then use
  /// distance-reducing SWAPs.
  template <class Policy>
  [[nodiscard]] SmallVector<QubitIndexPair>
  search(const Policy& policy, RoutingState& state, Arena& arena,
         const Environment& env) const {
    const Parameters params{.alpha = alpha, .lambda = lambda};

    arena.reset();
    Node* root = arena.allocate();
    assert(root != nullptr && "expected root allocation to succeed");

    root->initializeRoot(state.layout);
    if (root->isGoal(policy.objective(), env.target)) {
      return SmallVector<QubitIndexPair>{};
    }

    SearchFrontier frontier;
    frontier.push(root);

    Node* curr = frontier.pop();
    for (; curr != nullptr; curr = frontier.pop()) {

      // If the currently visited node is a goal node, reconstruct the
      // sequence of SWAPs from this node to the root.

      if (curr->isGoal(policy.objective(), env.target)) {
        return curr->swaps();
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighboring hardware qubits.

      llvm::SmallDenseSet<QubitIndexPair, 8> seen;
      for (const auto& [q0, q1] = policy.objective();
           const auto prog : {q0, q1}) {
        const auto hw0 = curr->layout.getHardwareIndex(prog);
        env.target.forEachNeighbour(hw0, [&](const QubitIndex hw1) {
          const QubitIndexPair indices(std::minmax(hw0, hw1));
          if (seen.contains(indices)) {
            return;
          }

          const auto standalone = env.nativeSwapCost.value_or(1L);
          const auto prefix =
              curr->isRoot() && state.costs && env.nativeSwapCost.has_value()
                  ? state.costs->swapCostAdjustment(indices.first,
                                                    indices.second, standalone)
                  : 0;

          if (Node* child = arena.allocate()) {
            const SwapCandidate candidate = {
                .indices = indices,
                .standalone = standalone,
                .prefix = prefix,
            };

            child->initializeChild(curr, candidate, policy, env.target, params);
            seen.insert(indices);
            frontier.push(child);
          }
        });
      }
    }

    /// Fallback to shortest-path swapping, if the budget is exhausted.
    /// Greedy completion can cost later gates. Thus, increase the search
    /// budget when routing quality matters more than memory use.

    const auto [prog0, prog1] = policy.objective();
    const auto [hw0, hw1] = state.layout.getHardwareIndices(prog0, prog1);
    const auto path = env.target.shortestPathBetween(hw0, hw1);

    SmallVector<QubitIndexPair> swaps;
    for (size_t i = 0; i < path.size() - 2; ++i) {
      swaps.emplace_back(path[i], path[i + 1]);
    }

    return swaps;
  }

  /// Route the leading interaction with bounded A* node storage and templated
  /// heuristic. Drain queued states at the limit, then use distance-reducing
  /// SWAPs. Return the SWAP sequence to move from one layout to another.
  /// Implements the 4-Approximation algorithm described in arXiv:1602.05150v3.
  [[nodiscard]] SmallVector<QubitIndexPair>
  restore(const Layout<QubitIndex>& from, const Layout<QubitIndex>& to,
          const Environment& env) const {
    if (from == to) {
      return {};
    }

    Layout curr(from);
    TokenSwapGraph f(env.target);
    SmallVector<QubitIndexPair> swaps;

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
  [[nodiscard]] std::tuple<Layout<QubitIndex>, SmallVector<QubitIndexPair>,
                           SmallVector<QubitIndexPair>>
  converge(const Layout<QubitIndex>& lhs, const Layout<QubitIndex>& rhs,
           const Environment& env) {
    if (lhs == rhs) {
      return {lhs, {}, {}};
    }

    std::array layouts{Layout(lhs), Layout(rhs)};
    std::array graphs{
        TokenSwapGraph(env.target),
        TokenSwapGraph(env.target),
    };
    std::array<SmallVector<QubitIndexPair>, 2> swaps{};

    auto gen = makeMt19937(env.seed);
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
  Layout<QubitIndex> driveby(Range layouts, const Environment& env,
                             const size_t niterations = 1) {
    assert(!layouts.empty() && "expected at least one layout");

    TokenSwapGraph f(env.target);
    Layout curr(*layouts.begin());

    // Nudge curr towards target by applying a happy SWAP chain.
    const auto merge = [&](const Layout<QubitIndex>& target) {
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
  Horizon getHorizon(Wires wires, const Layout<QubitIndex>& layout,
                     const Boundary<Direction>& boundary) {
    SmallVector<QubitIndexPair> prev;
    SmallVector<QubitIndexPair> next;

    Horizon horizon;
    walkProgramGraph<Direction>(
        MutableArrayRef(wires.data(), wires.size()),
        [&](const Frontier& frontier, ReleasedOps& released) {
          for (const auto& [op, indices] : frontier) {
            if (indices.size() == 1 &&
                (boundary.operation() == nullptr ||
                 precedes<Direction>(op, boundary.operation()))) {
              released.emplace_back(op);
            }
          }

          if (released.empty()) {
            horizon.next();
            for (const auto& [op, indices] : frontier) {
              if (boundary.operation() != nullptr &&
                  !precedes<Direction>(op, boundary.operation())) {
                continue;
              }

              if (!isa<BarrierOp>(op) && isa<UnitaryOpInterface>(op)) {
                const auto i0 = indices[0];
                const auto i1 = indices[1];
                const auto prog0 = layout.getProgramIndex(i0);
                const auto prog1 = layout.getProgramIndex(i1);
                const QubitIndexPair gate = std::minmax(prog0, prog1);

                if (!is_contained(prev, gate)) {
                  horizon.push(gate);
                  if (horizon.size() == 1 + nlookahead) {
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

    return horizon;
  }

  /// Both modes leave cursors at the next operation on each physical wire.
  template <RoutingMode Mode>
  static void insertSWAPs(ArrayRef<QubitIndexPair> swaps, RoutingState& state,
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
                                          const Boundary<Direction>& boundary,
                                          const Environment& env) {

    // Unexecutable gates are not released and may thus be revisited multiple
    // times while advancing. Stale gates are immediately skipped to avoid the
    // overhead of isa-checks and layout lookups.
    llvm::SmallPtrSet<Operation*, 16> stale;

    /// Advancement only moves iterators. Discard classifications before routing
    /// inserts SWAPs or replaces composites.
    DenseMap<Operation*, bool> measurementRouting;

    // The wire traversal does not follow classical dependencies. Defer a
    // composite until earlier routing work is complete, but let independent
    // composites pass terminal wires. Reverse block order for backward routing.

    const auto defer = [&](Operation* candidate) {
      return llvm::any_of(state.wires, [&](WireIterator& it) {
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

    std::optional<CompositeUnitary> composite;
    walkProgramGraph<Direction>(state.wires, [&](const Frontier& frontier,
                                                 ReleasedOps& released) {
      for (const auto& [op, indices] : frontier) {
        if (stale.contains(op) ||
            (boundary.operation() != nullptr &&
             precedes<Direction>(boundary.operation(), op))) {
          continue;
        }

        const auto release =
            TypeSwitch<Operation*, bool>(op)
                .Case([](BarrierOp) { return true; })
                .Case([&](UnitaryOpInterface) {
                  if (indices.size() == 1 ||
                      env.target.areAdjacent(indices[0], indices[1])) {
                    return true;
                  }
                  stale.insert(op);
                  return false;
                })
                .Case([](ResetOp) { return true; })
                .Case([&](MeasureOp m) {
                  if constexpr (Direction == WireDirection::Backward) {
                    return true;
                  }
                  return measurementNeedsRouting(m, measurementRouting);
                })
                .template Case<AllocOp, StaticOp, qtensor::ExtractOp>(
                    [](auto) { return Direction == WireDirection::Forward; })
                .template Case<SinkOp, qtensor::InsertOp, YieldOp, scf::YieldOp,
                               scf::ConditionOp>(
                    [](auto) { return Direction == WireDirection::Backward; })
                .template Case<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>(
                    [&](auto cf) {
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
                state.wires[indices[0]].qubit() != gate.getOutputQubit(0)) {
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
    SmallVector<Value> addons;
    SmallVector<unsigned> resultNumbers(parent.wires.size());

    for (size_t site = 0; site < parent.wires.size(); ++site) {
      WireIterator it(parent.wires[site]);
      if (it.operation() == composite.op) {
        resultNumbers[site] = cast<OpResult>(it.qubit()).getResultNumber();
      } else {
        resultNumbers[site] = composite.op->getNumResults() + addons.size();

        /// Thread the value before the next pending operation through the
        /// region. Particulary, terminal measurements must remain after the
        /// region.

        --it;
        while (it.operation() != nullptr &&
               !it.operation()->isBeforeInBlock(composite.op)) {
          --it;
        }

        addons.push_back(it.qubit());
      }
    }

    composite = CompositeUnitary{
        .op = TypeSwitch<Operation*, Operation*>(composite.op)
                  .Case<scf::ForOp, scf::WhileOp, IfOp, IndexSwitchOp>(
                      [&](auto op) { return extend(op, addons, rewriter); }),
        .indices = to_vector(llvm::seq(parent.wires.size()))};

    for (auto [site, result] : enumerate(resultNumbers)) {
      parent.wires[site] = WireIterator(composite.op->getResult(result));
    }
  }

  /// Return `values` with only the qubit entries realigned according to the
  /// given permutation of hardware indices.
  static SmallVector<Value> realignQubitValues(ValueRange values,
                                               ArrayRef<QubitIndex> perm,
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
  static SmallVector<QubitIndex> sitePermutation(const Layout<QubitIndex>& from,
                                                 const Layout<QubitIndex>& to) {
    SmallVector<QubitIndex> permutation(from.nHardwareQubits());
    for (size_t site = 0; site < permutation.size(); ++site) {
      permutation[site] = to.getHardwareIndex(from.getProgramIndex(site));
    }
    return permutation;
  }

  static void permuteWires(Wires& wires, ArrayRef<QubitIndex> permutation) {
    Wires reordered(wires.size());
    for (size_t site = 0; site < wires.size(); ++site) {
      reordered[permutation[site]] = wires[site];
    }
    wires = std::move(reordered);
  }

  /// Capture uses before rebinding so permutation cycles are safe.
  static void realignQubitUses(ValueRange values, ArrayRef<QubitIndex> sites,
                               ArrayRef<QubitIndex> permutation,
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

  /// Construct child states, route their bodies, reconcile layouts, then
  /// publish the physical result order to the parent.
  template <WireDirection Direction, class Policy, RoutingMode Mode>
  Statistics routeComposite(const CompositeUnitary& composite,
                            RoutingState& parent, Arena& arena,
                            const Environment& env, IRRewriter* rewriter) {
    const auto& [op, indices] = composite;
    if (parent.costs) {
      parent.costs->flush();
    }

    SmallVector<QubitIndex> resultSites(op->getNumResults());
    for (size_t site : indices) {
      resultSites[cast<OpResult>(parent.wires[site].qubit())
                      .getResultNumber()] = site;
    }

    SmallVector<QubitIndex> sites;
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
      auto& child = children.emplace_back(Wires(env.target.numSites()),
                                          parent.layout, env);

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

      totalStats.merge(
          route<Direction, Policy, Mode>(region, child, arena, env, rewriter));
    }

    Layout<QubitIndex> exit =
        TypeSwitch<Operation*, Layout<QubitIndex>>(op)
            .Case([&](scf::ForOp) {
              insertSWAPs<Mode>(restore(children[0].layout, parent.layout, env),
                                children[0], totalStats, rewriter);
              return parent.layout;
            })
            .Case([&](scf::WhileOp) {
              insertSWAPs<Mode>(restore(children[1].layout, parent.layout, env),
                                children[1], totalStats, rewriter);
              return children[0].layout;
            })
            .Case([&](IfOp) {
              const auto [layout, first, second] =
                  converge(children[0].layout, children[1].layout, env);
              insertSWAPs<Mode>(first, children[0], totalStats, rewriter);
              insertSWAPs<Mode>(second, children[1], totalStats, rewriter);
              return layout;
            })
            .Case([&](IndexSwitchOp) {
              auto layout = driveby(map_range(children,
                                              [](const RoutingState& child)
                                                  -> const Layout<QubitIndex>& {
                                                return child.layout;
                                              }),
                                    env);
              for (auto& child : children) {
                insertSWAPs<Mode>(restore(child.layout, layout, env), child,
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

    // Propagate the exit layout to the parent state and advance past the
    // composite for the parent's wires.

    parent.layout = std::move(exit);
    for (auto& wire : parent.wires) {
      if (wire != std::default_sentinel && wire.operation() == composite.op) {
        std::ranges::advance(wire, WireTraversalTraits<Direction>::stride());
      }
    }

    return totalStats;
  }

  /// Advance executable operations, route ready regions, or search for SWAPs
  /// that release the next interaction. Finish terminal measurements last.
  template <WireDirection Direction, class Policy,
            RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  Statistics route(Region& region, RoutingState& state, Arena& arena,
                   const Environment& env, IRRewriter* rewriter = nullptr) {
    Statistics stats;
    Boundary<Direction> boundary(region.front());

    if (state.costs) {
      state.costs->reset(Direction);
    }

    while (true) {
      auto composite = advance<Direction>(state, boundary, env);
      if (composite) {
        assert(composite->op == boundary.operation());
        boundary.setNextBoundary();

        if constexpr (Mode == RoutingMode::Hot) {
          place(*composite, state, *rewriter);
        }

        stats.merge(routeComposite<Direction, Policy, Mode>(
            *composite, state, arena, env, rewriter));
      } else {
        const auto horizon =
            getHorizon<Direction>(state.wires, state.layout, boundary);
        if (horizon.empty()) {
          break;
        }

        const Policy policy(horizon, state.layout, env.target);
        const auto swaps = search(policy, state, arena, env);
        insertSWAPs<Mode>(swaps, state, stats, rewriter);
      }
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

std::unique_ptr<Pass> createPlacementPass() {
  return std::make_unique<PlacementPass>();
}

} // namespace mlir::qco
