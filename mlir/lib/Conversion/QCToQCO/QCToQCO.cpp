/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQCO/QCToQCO.h"

#include "mlir/Conversion/GateTable.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstddef>
#include <utility>

namespace mlir {

using namespace qco;
using namespace qc;

#define GEN_PASS_DEF_QCTOQCO
#include "mlir/Conversion/QCToQCO/QCToQCO.h.inc"

namespace {
/**
 * @brief State object for tracking qubit value flow during conversion
 *
 * @details
 * This struct maintains the mapping between QC dialect qubits (which use
 * reference semantics) and their corresponding QCO dialect qubit values
 * (which use value semantics). As the conversion progresses, each QC
 * qubit reference is mapped to its latest QCO SSA value.
 *
 * The key insight is that QC operations modify qubits in-place:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * qc.h %q : !qc.qubit        // modifies %q in-place
 * qc.x %q : !qc.qubit        // modifies %q in-place
 * ```
 *
 * While QCO operations consume inputs and produce new outputs:
 * ```mlir
 * %q0 = qco.alloc : !qco.qubit
 * %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit   // %q0 consumed, %q1 produced
 * %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit   // %q1 consumed, %q2 produced
 * ```
 *
 * The qubitMap tracks that the QC qubit %q corresponds to:
 * - %q0 after allocation
 * - %q1 after the H gate
 * - %q2 after the X gate
 */
struct LoweringState {
  struct ModifierFrame {
    /// QC qubits yielded from the current modifier region, in yield order.
    SmallVector<Value> yieldOrder;

    /// Latest QCO SSA values for QC qubits that are remapped inside the
    /// modifier region.
    llvm::DenseMap<Value, Value> currentQubits;
  };

  /// Per-region map from QC qubit references to latest QCO SSA values.
  ///
  /// @details Keys are `Operation::getParentRegion()` for ops being converted
  /// (typically a `func.func` body, or a `qc.ctrl` / `qc.inv` region).
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> qubitMap;

  /// Stack of active modifier regions (`qc.ctrl` / `qc.inv`).
  SmallVector<ModifierFrame> modifierFrames;
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which tracks the mapping from reference-semantics QC qubits
 * to value-semantics QCO qubits across multiple pattern applications.
 *
 * This stateful approach is necessary because the conversion needs to:
 * 1. Track which QCO value corresponds to each QC qubit reference
 * 2. Update these mappings as operations transform qubits
 * 3. Share this information across different conversion patterns
 *
 * @tparam OpType The QC operation type to convert
 */
template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter,
                              MLIRContext* context, LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, context), state_(state) {}

  /// Returns the shared lowering state object
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};
} // namespace

/**
 * @brief Helper function to look up the latest QCO qubit value for a given QC
 * qubit reference
 *
 * @param qubitMap The mapping from QC qubits to QCO qubits for the current
 * region
 * @param qcQubit The QC qubit reference to look up
 * @return The latest QCO qubit value corresponding to the given QC qubit
 * reference
 */
[[nodiscard]] static Value
lookupMappedQubit(llvm::DenseMap<Value, Value>& qubitMap, Value qcQubit) {
  auto it = qubitMap.find(qcQubit);
  assert(it != qubitMap.end() && "QC qubit not found");
  return it->second;
}

/** @brief Returns whether lowering currently processes a modifier body. */
[[nodiscard]] static bool isInsideModifier(const LoweringState& state) {
  return !state.modifierFrames.empty();
}

/** @brief Returns the active modifier frame. */
[[nodiscard]] static LoweringState::ModifierFrame&
currentModifierFrame(LoweringState& state) {
  assert(isInsideModifier(state) && "expected active modifier frame");
  return state.modifierFrames.back();
}

/** @brief Finds the nearest region-local qubit map containing @p qcQubit. */
[[nodiscard]] static llvm::DenseMap<Value, Value>*
findMappedQubitMap(LoweringState& state, Operation* anchor, Value qcQubit) {
  for (Region* current = anchor->getParentRegion(); current != nullptr;
       current = current->getParentRegion()) {
    auto mapIt = state.qubitMap.find(current);
    if (mapIt != state.qubitMap.end() && mapIt->second.contains(qcQubit)) {
      return &mapIt->second;
    }
  }
  return nullptr;
}

/** @brief Resolves the latest QCO SSA value for a QC qubit reference. */
[[nodiscard]] static Value lookupMappedQubit(LoweringState& state,
                                             Operation* anchor, Value qcQubit) {
  if (isInsideModifier(state)) {
    auto& frame = currentModifierFrame(state);
    if (auto it = frame.currentQubits.find(qcQubit);
        it != frame.currentQubits.end()) {
      return it->second;
    }
  }

  auto* qubitMap = findMappedQubitMap(state, anchor, qcQubit);
  assert(qubitMap != nullptr && "QC qubit not found");
  return lookupMappedQubit(*qubitMap, qcQubit);
}

/** @brief Updates the latest QCO SSA value for a QC qubit reference. */
static void assignMappedQubit(LoweringState& state, Operation* anchor,
                              Value qcQubit, Value qcoQubit) {
  if (isInsideModifier(state)) {
    auto& frame = currentModifierFrame(state);
    if (auto it = frame.currentQubits.find(qcQubit);
        it != frame.currentQubits.end()) {
      it->second = qcoQubit;
      return;
    }
  }

  if (auto* qubitMap = findMappedQubitMap(state, anchor, qcQubit)) {
    (*qubitMap)[qcQubit] = qcoQubit;
    return;
  }

  state.qubitMap[anchor->getParentRegion()][qcQubit] = qcoQubit;
}

/** @brief Resolves a range of QC qubits to their latest QCO values. */
template <typename Range>
[[nodiscard]] static SmallVector<Value>
resolveMappedQubits(LoweringState& state, Operation* anchor,
                    const Range& qcQubits) {
  return llvm::to_vector(llvm::map_range(qcQubits, [&](Value qcQubit) {
    return lookupMappedQubit(state, anchor, qcQubit);
  }));
}

/** @brief Updates mappings for matching QC and QCO qubit ranges. */
template <typename QcRange, typename QcoRange>
static void assignMappedQubits(LoweringState& state, Operation* anchor,
                               const QcRange& qcQubits,
                               const QcoRange& qcoQubits) {
  for (auto [qcQubit, qcoQubit] : llvm::zip_equal(qcQubits, qcoQubits)) {
    assignMappedQubit(state, anchor, qcQubit, qcoQubit);
  }
}

/** @brief Pushes a new modifier frame seeded with aliased target values. */
static void pushModifierFrame(LoweringState& state, ValueRange qcTargets,
                              ValueRange qcoTargets) {
  auto& [yieldOrder, currentQubits] = state.modifierFrames.emplace_back();
  llvm::append_range(yieldOrder, qcTargets);
  for (auto [qcTarget, qcoTarget] : llvm::zip_equal(qcTargets, qcoTargets)) {
    currentQubits.try_emplace(qcTarget, qcoTarget);
  }
}

/** @brief Pops the active modifier frame after lowering its yield. */
static void popModifierFrame(LoweringState& state) {
  assert(isInsideModifier(state) && "expected active modifier frame");
  state.modifierFrames.pop_back();
}

/** @brief Adds entry block aliases for modifier target values. */
template <typename OpType>
[[nodiscard]] static ValueRange addModifierAliases(OpType op,
                                                   const size_t numTargets,
                                                   PatternRewriter& rewriter) {
  auto& entryBlock = op.getRegion().front();
  const auto opLoc = op.getLoc();
  const auto qubitType = qco::QubitType::get(op.getContext());
  rewriter.modifyOpInPlace(op, [&] {
    for (size_t i = 0; i < numTargets; ++i) {
      entryBlock.addArgument(qubitType, opLoc);
    }
  });
  return entryBlock.getArguments().take_back(numTargets);
}

namespace {

/**
 * @brief Converts func.return and sinks remaining live qubits.
 *
 * @details
 * QC uses reference semantics and does not enforce linear typing for qubits.
 * After conversion, QCO requires that every qubit SSA value is consumed
 * exactly once. For allocations (including static qubits), the sink is
 * `qco.sink`. This pattern inserts `qco.sink` operations for all
 * still-live qubits tracked in the lowering state right before the return.
 */
struct ConvertFuncReturnOp final : StatefulOpConversionPattern<func::ReturnOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    Region* funcRegion = op->getParentRegion();
    auto& map = state.qubitMap[funcRegion];

    // Build return values from qubitMap and collect live qubit information.
    // A qubit from the current scope is considered alive if it is returned from
    // the function. Otherwise, it is considered dead.
    llvm::SmallVector<Value> returnValues;
    returnValues.reserve(op.getNumOperands());
    llvm::DenseSet<Value> liveQubits;
    for (auto [qcOperand, adaptorOperand] :
         llvm::zip_equal(op.getOperands(), adaptor.getOperands())) {
      if (auto it = map.find(qcOperand); it != map.end()) {
        auto latest = it->second;
        returnValues.emplace_back(latest);
        liveQubits.insert(latest);
      } else {
        returnValues.emplace_back(adaptorOperand);
      }
    }

    // Deallocate dead qubit values
    for (Value qcoQubit : llvm::make_second_range(map)) {
      if (!liveQubits.contains(qcoQubit)) {
        SinkOp::create(rewriter, op.getLoc(), qcoQubit);
      }
    }
    state.qubitMap.erase(funcRegion);

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, returnValues);
    return success();
  }
};

/**
 * @brief Type converter for QC-to-QCO conversion
 *
 * @details
 * Handles type conversion between the QC and QCO dialects.
 * The primary conversion is from !qc.qubit to !qco.qubit, which
 * represents the semantic shift from reference types to value types.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
class QCToQCOTypeConverter final : public TypeConverter {
public:
  explicit QCToQCOTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert QC qubit references to QCO qubit values
    addConversion([ctx](qc::QubitType /*type*/) -> Type {
      return qco::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Converts qc.alloc to qco.alloc
 *
 * @details
 * Allocates a new qubit and establishes the initial mapping in the state.
 * Both dialects initialize qubits to the |0⟩ state.
 *
 * Register metadata (name, size, index) is preserved during conversion,
 * allowing the QCO representation to maintain register information for
 * debugging and visualization.
 *
 * Example transformation:
 * ```mlir
 * %q = qc.alloc("q", 3, 0) : !qc.qubit
 * // becomes:
 * %q0 = qco.alloc("q", 3, 0) : !qco.qubit
 * ```
 */
struct ConvertQCAllocOp final : StatefulOpConversionPattern<qc::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getResult();

    // Create the qco.alloc operation with preserved register metadata
    auto qcoOp = rewriter.replaceOpWithNewOp<qco::AllocOp>(
        op, op.getRegisterNameAttr(), op.getRegisterSizeAttr(),
        op.getRegisterIndexAttr());

    auto qcoQubit = qcoOp.getResult();

    // Establish initial mapping: this QC qubit reference now corresponds
    // to this QCO SSA value
    assignMappedQubit(state, operation, qcQubit, qcoQubit);

    return success();
  }
};

/**
 * @brief Converts qc.dealloc to qco.sink
 *
 * @details
 * Deallocates a qubit by looking up its latest QCO value and creating
 * a corresponding qco.sink operation. The mapping is removed from
 * the state as the qubit is no longer in use.
 *
 * Example transformation:
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * // becomes (where %q maps to %q_final):
 * qco.sink %q_final : !qco.qubit
 * ```
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<qc::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto* region = operation->getParentRegion();
    auto& qubitMap = state.qubitMap[region];
    Value qcQubit = op.getQubit();
    Value qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create the sink operation
    rewriter.replaceOpWithNewOp<SinkOp>(op, qcoQubit);

    // Remove from state as qubit is no longer in use
    qubitMap.erase(qcQubit);

    return success();
  }
};

/**
 * @brief Converts qc.static to qco.static
 *
 * @details
 * Static qubits represent references to hardware-mapped or fixed-position
 * qubits identified by an index. This conversion creates the corresponding
 * qco.static operation and establishes the mapping.
 *
 * Example transformation:
 * ```mlir
 * %q = qc.static 0 : !qc.qubit
 * // becomes:
 * %q0 = qco.static 0 : !qco.qubit
 * ```
 */
struct ConvertQCStaticOp final : StatefulOpConversionPattern<qc::StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    Value qcQubit = op.getQubit();

    auto qcoOp = rewriter.replaceOpWithNewOp<qco::StaticOp>(op, op.getIndex());
    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubit());

    return success();
  }
};

/**
 * @brief Converts qc.measure to qco.measure
 *
 * @details
 * Measurement is a key operation where the semantic difference is visible:
 * - QC: Measures in-place, returning only the classical bit
 * - QCO: Consumes input qubit, returns both output qubit and classical bit
 *
 * The conversion looks up the latest QCO value for the QC qubit,
 * performs the measurement, updates the mapping with the output qubit,
 * and returns the classical bit result.
 *
 * Register metadata (name, size, index) for output recording is preserved
 * during conversion.
 *
 * Example transformation:
 * ```mlir
 * %c = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
 * // becomes (where %q maps to %q_in):
 * %q_out, %c = qco.measure("c", 2, 0) %q_in : !qco.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<qc::MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    Value qcQubit = op.getQubit();
    Value qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create qco.measure (returns both output qubit and bit result)
    auto qcoOp = qco::MeasureOp::create(
        rewriter, op.getLoc(), qcoQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    // Update mapping: the QC qubit now corresponds to the output qubit
    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    // Replace the QC operation's bit result with the QCO bit result
    rewriter.replaceOp(op, qcoOp.getResult());

    return success();
  }
};

/**
 * @brief Converts qc.reset to qco.reset
 *
 * @details
 * Reset operations force a qubit to the |0⟩ state. The semantic difference:
 * - QC: Resets in-place (no result value)
 * - QCO: Consumes input qubit, returns reset output qubit
 *
 * The conversion looks up the latest QCO value, performs the reset,
 * and updates the mapping with the output qubit. The QC operation
 * is erased as it has no results to replace.
 *
 * Example transformation:
 * ```mlir
 * qc.reset %q : !qc.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQCResetOp final : StatefulOpConversionPattern<qc::ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::ResetOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    Value qcQubit = op.getQubit();
    Value qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create qco.reset (consumes input, produces output)
    auto qcoOp = qco::ResetOp::create(rewriter, op.getLoc(), qcoQubit);

    // Update mapping: the QC qubit now corresponds to the reset output
    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    // Erase the old (it has no results to replace)
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename QCOpType, typename QCOOpType, std::size_t NumTargets,
          std::size_t NumParams>
struct ConvertQCGateToQCO final : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  template <std::size_t... ParamIndices>
  static auto createGate(ConversionPatternRewriter& rewriter, Location loc,
                         const llvm::SmallVector<Value>& qcoTargets,
                         QCOpType op,
                         std::index_sequence<ParamIndices...> /*params*/) {
    if constexpr (NumTargets == 0) {
      return QCOOpType::create(rewriter, loc, op.getParameter(ParamIndices)...);
    } else if constexpr (NumTargets == 1) {
      return QCOOpType::create(rewriter, loc, qcoTargets[0],
                               op.getParameter(ParamIndices)...);
    } else {
      return QCOOpType::create(rewriter, loc, qcoTargets[0], qcoTargets[1],
                               op.getParameter(ParamIndices)...);
    }
  }

  LogicalResult
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();

    llvm::SmallVector<Value> qcTargets;
    qcTargets.reserve(NumTargets);
    if constexpr (NumTargets == 0) {
      // No target qubits.
    } else if constexpr (NumTargets == 1) {
      qcTargets.push_back(op.getQubitIn());
    } else {
      qcTargets.push_back(op.getQubit0In());
      qcTargets.push_back(op.getQubit1In());
    }

    llvm::SmallVector<Value> qcoTargets;
    qcoTargets.reserve(NumTargets);
    for (auto qcQubit : qcTargets) {
      qcoTargets.push_back(lookupMappedQubit(state, operation, qcQubit));
    }

    auto qcoOp = createGate(rewriter, op.getLoc(), qcoTargets, op,
                            std::make_index_sequence<NumParams>{});

    if constexpr (NumTargets == 0) {
      // No qubit mapping updates required.
    } else if constexpr (NumTargets == 1) {
      assignMappedQubit(state, operation, qcTargets[0], qcoOp.getQubitOut());
    } else {
      assignMappedQubit(state, operation, qcTargets[0], qcoOp.getQubit0Out());
      assignMappedQubit(state, operation, qcTargets[1], qcoOp.getQubit1Out());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.barrier to qco.barrier
 *
 * @par Example:
 * ```mlir
 * qc.barrier %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit,
 * !qco.qubit
 * ```
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<qc::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    const auto qcQubits = llvm::to_vector(op.getQubits());
    auto qcoQubits = resolveMappedQubits(state, operation, qcQubits);

    // Create qco.barrier
    auto qcoOp = qco::BarrierOp::create(rewriter, op.getLoc(), qcoQubits);

    assignMappedQubits(state, operation, qcQubits, qcoOp.getQubitsOut());

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.ctrl to qco.ctrl
 *
 * @par Example:
 * ```mlir
 * qc.ctrl(%q0) {
 *   qc.x %q1 : !qc.qubit
 * } : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%a_in = %q1_in) {
 *   %a_res = qco.x %a_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a_res
 * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
 * ```
 */
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<qc::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::CtrlOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    const auto numTargets = op.getNumTargets();
    const auto qcControls = op.getControls();
    const auto qcTargets = op.getTargets();
    auto qcoControls = resolveMappedQubits(state, operation, qcControls);
    auto qcoTargets = resolveMappedQubits(state, operation, qcTargets);

    // Create qco.ctrl
    auto qcoOp =
        qco::CtrlOp::create(rewriter, op.getLoc(), qcoControls, qcoTargets);

    assignMappedQubits(state, operation, qcControls, qcoOp.getControlsOut());
    assignMappedQubits(state, operation, qcTargets, qcoOp.getTargetsOut());

    // Clone body region from QC to QCO
    auto& dstRegion = qcoOp.getRegion();
    rewriter.cloneRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Create block arguments for QCO targets
    auto& entryBlock = dstRegion.front();
    assert(entryBlock.getNumArguments() == 0 &&
           "QC ctrl region unexpectedly has entry block arguments");
    pushModifierFrame(state, qcTargets,
                      addModifierAliases(qcoOp, numTargets, rewriter));

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.inv to qco.inv
 *
 * @par Example:
 * ```mlir
 * qc.inv {
 *   qc.s %q0 : !qc.qubit
 * } : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out = qco.inv (%a0_in = %q0_in) {
 *   %a0_res = qco.s %a0_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a0_res
 * } : {!qco.qubit} -> {!qco.qubit}
 * ```
 */
struct ConvertQCInvOp final : StatefulOpConversionPattern<qc::InvOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::InvOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    const auto numTargets = op.getNumTargets();
    const auto qcTargets = op.getTargets();
    auto qcoTargets = resolveMappedQubits(state, operation, qcTargets);

    // Create qco.inv
    auto qcoOp = qco::InvOp::create(rewriter, op.getLoc(), qcoTargets);

    assignMappedQubits(state, operation, qcTargets, qcoOp.getQubitsOut());

    // Clone body region from QC to QCO
    auto& dstRegion = qcoOp.getRegion();
    rewriter.cloneRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Create block arguments for target qubits and seed the nested frame.
    auto& entryBlock = dstRegion.front();
    assert(entryBlock.getNumArguments() == 0 &&
           "QC inv region unexpectedly has entry block arguments");
    pushModifierFrame(state, qcTargets,
                      addModifierAliases(qcoOp, numTargets, rewriter));

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.yield to qco.yield
 *
 * @par Example:
 * ```mlir
 * qc.yield
 * ```
 * is converted to
 * ```mlir
 * qco.yield %targets
 * ```
 */
struct ConvertQCYieldOp final : StatefulOpConversionPattern<qc::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& frame = currentModifierFrame(state);
    auto targets = resolveMappedQubits(state, operation, frame.yieldOrder);
    rewriter.replaceOpWithNewOp<qco::YieldOp>(op, targets);
    popModifierFrame(state);
    return success();
  }
};

/**
 * @brief Pass implementation for QC-to-QCO conversion
 *
 * @details
 * This pass converts QC dialect operations (reference semantics) to QCO dialect
 * operations (value semantics). The conversion is essential for enabling
 * optimization passes that rely on SSA form and explicit dataflow analysis.
 *
 * The pass operates in several phases:
 * 1. Type conversion: !qc.qubit -> !qco.qubit
 * 2. Operation conversion: Each QC op is converted to its QCO equivalent
 * 3. State tracking: A LoweringState maintains qubit value mappings
 * 4. Function/control-flow adaptation: Function signatures and control flow are
 * updated to use QCO types
 *
 * The conversion maintains semantic equivalence while transforming the
 * representation from imperative (mutation-based) to functional (SSA-based).
 */
struct QCToQCO final : impl::QCToQCOBase<QCToQCO> {
  using QCToQCOBase::QCToQCOBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    // Create state object to track qubit value flow
    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCToQCOTypeConverter typeConverter(context);

    // Configure conversion target: QC illegal, QCO legal
    target.addIllegalDialect<QCDialect>();
    target.addLegalDialect<QCODialect>();

    // Register operation conversion patterns with state tracking.
    patterns.add<ConvertQCAllocOp, ConvertQCDeallocOp, ConvertQCStaticOp,
                 ConvertQCMeasureOp, ConvertQCResetOp, ConvertQCBarrierOp,
                 ConvertQCCtrlOp, ConvertQCInvOp, ConvertQCYieldOp>(
        typeConverter, context, &state);

    // Not part of the central gate table (no Jeff/QIR lowering).
    patterns.add<ConvertQCGateToQCO<qc::GPhaseOp, qco::GPhaseOp, 0, 1>>(
        typeConverter, context, &state);

#define MQT_ADD_QC_TO_QCO_GATE(KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, \
                               JEFF_OP, JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME,   \
                               JEFF_PPR, QIR_KIND, QIR_FN)                     \
  patterns.add<ConvertQCGateToQCO<QC_OP, QCO_OP, (TARGETS), (PARAMS)>>(        \
      typeConverter, context, &state);
    MQT_GATE_TABLE(MQT_ADD_QC_TO_QCO_GATE)
#undef MQT_ADD_QC_TO_QCO_GATE

    // Conversion of qc types in func.func signatures
    // Note: This currently has limitations with signature changes
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Conversion of qc types in func.return
    //
    // Note: `func.return` may already be type-legal even though we still need
    // to insert sink operations (`qco.sink`) for dead qubit values. Therefore,
    // we mark it illegal as long as the qubit map of the region is not empty.
    patterns.add<ConvertFuncReturnOp>(typeConverter, context, &state);
    target.addDynamicallyLegalOp<func::ReturnOp>([&](const func::ReturnOp op) {
      if (!typeConverter.isLegal(op)) {
        return false;
      }
      const auto it = state.qubitMap.find(op->getParentRegion());
      return it == state.qubitMap.end() || it->second.empty();
    });

    // Conversion of qc types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // Conversion of qc types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
