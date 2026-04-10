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

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
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
 * @brief Information about a qubit
 */
struct QubitInfo {
  /// Register the qubit belongs to
  Value reg;
  /// Index of the qubit within its register
  Value index;
};

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

    SmallVector<Value> memrefs;
    /// Latest QCO SSA values for QC qubits that are remapped inside the
    /// modifier region.
    llvm::DenseMap<Value, Value> currentQubits;

    llvm::DenseMap<Value, Value> currentRegisters;
  };

  /// Per-region map from original QC qubit reference to its latest QCO SSA
  /// value.
  ///
  /// @details Keys are `Operation::getParentRegion()` for ops being converted
  /// (typically a `func.func` body or a modifier region).
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> qubitMap;

  /// Per-region map from original QC register to its latest QTensor SSA value
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> tensorMap;

  /// Per-region map from original QC qubit reference to its register
  /// information
  llvm::DenseMap<Region*, llvm::DenseMap<Value, QubitInfo>> qubitInfoMap;

  llvm::DenseMap<Region*, llvm::DenseMap<Value, llvm::SetVector<Value>>>
      extractedQubits;

  llvm::DenseMap<Operation*, llvm::SetVector<Value>> regionQubitMap;

  llvm::DenseMap<Operation*, llvm::SetVector<Value>> regionRegisterMap;

  /// Stack of active modifier regions
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

/**
 * @brief Finds the nearest region-local map containing @p reference and
 * returns the pair containing the map and a mutable reference to the value in
 * the map.
 */
[[nodiscard]] static std::pair<llvm::DenseMap<Value, Value>*, Value*>
findRegionLocalMap(llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>>& map,
                   Operation* anchor, Value reference) {
  for (auto* current = anchor->getParentRegion(); current != nullptr;
       current = current->getParentRegion()) {
    if (auto it = map.find(current); it != map.end()) {
      auto& regionMap = it->second;
      if (auto valueIt = regionMap.find(reference);
          valueIt != regionMap.end()) {
        return {&regionMap, &valueIt->second};
      }
      return {&regionMap, nullptr};
    }
  }
  return {nullptr, nullptr};
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

  const auto& [qubitMap, qubitValue] =
      findRegionLocalMap(state.qubitMap, anchor, qcQubit);
  assert(qubitMap != nullptr && qubitValue != nullptr && "QC qubit not found");
  return *qubitValue;
}

/** @brief Resolves the latest QTensor SSA value for a QC register. */
[[nodiscard]] static Value lookupMappedTensor(LoweringState& state,
                                              Operation* anchor, Value memref) {

  if (isInsideModifier(state)) {
    auto& frame = currentModifierFrame(state);
    if (auto it = frame.currentRegisters.find(memref);
        it != frame.currentRegisters.end()) {
      return it->second;
    }
  }
  const auto& [tensorMap, tensorValue] =
      findRegionLocalMap(state.tensorMap, anchor, memref);
  assert(tensorMap != nullptr && tensorValue != nullptr &&
         "QC register not found");
  return *tensorValue;
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
    frame.currentQubits.try_emplace(qcQubit, qcoQubit);
    return;
  }

  auto [qubitMap, qubitValue] =
      findRegionLocalMap(state.qubitMap, anchor, qcQubit);
  if (qubitValue != nullptr) {
    *qubitValue = qcoQubit;
    return;
  }
  if (qubitMap != nullptr) {
    (*qubitMap)[qcQubit] = qcoQubit;
    return;
  }

  state.qubitMap[anchor->getParentRegion()][qcQubit] = qcoQubit;
}

/** @brief Updates the latest QTensor SSA value for a QC register. */
static void assignMappedTensor(LoweringState& state, Operation* anchor,
                               Value memref, Value tensor) {
  if (isInsideModifier(state)) {
    auto& frame = currentModifierFrame(state);
    if (auto it = frame.currentRegisters.find(memref);
        it != frame.currentRegisters.end()) {
      it->second = tensor;
      return;
    }
  }

  auto [tensorMap, tensorValue] =
      findRegionLocalMap(state.tensorMap, anchor, memref);

  if (tensorValue != nullptr) {
    *tensorValue = tensor;
    return;
  }
  if (tensorMap != nullptr) {
    (*tensorMap)[memref] = tensor;
    return;
  }
  state.tensorMap[anchor->getParentRegion()][memref] = tensor;
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

/** @brief Resolves a range of QC memrefs to their latest QCO tensor values. */
template <typename Range>
[[nodiscard]] static SmallVector<Value>
resolveMappedTensors(LoweringState& state, Operation* anchor,
                     const Range& memrefs) {
  return llvm::to_vector(llvm::map_range(memrefs, [&](Value memref) {
    return lookupMappedTensor(state, anchor, memref);
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

/** @brief Updates mappings for matching QC memref and QCO tensor ranges. */
template <typename QcRange, typename QcoRange>
static void assignMappedTensors(LoweringState& state, Operation* anchor,
                                const QcRange& memrefs,
                                const QcoRange& tensors) {
  for (auto [memref, tensor] : llvm::zip_equal(memrefs, tensors)) {
    assignMappedTensor(state, anchor, memref, tensor);
  }
}

/** @brief Pushes a new modifier frame seeded with aliased target values. */
static void pushModifierFrame(LoweringState& state, ValueRange qcTargets,
                              ValueRange qcoTargets) {
  auto& [yieldOrder, registers, currentQubits, currentRegisters] =
      state.modifierFrames.emplace_back();
  llvm::append_range(yieldOrder, qcTargets);
  for (auto [qcTarget, qcoTarget] : llvm::zip_equal(qcTargets, qcoTargets)) {
    currentQubits.try_emplace(qcTarget, qcoTarget);
  }
}

/** @brief Pushes a new modifier frame seeded with aliased target values. */
static void pushModifierFrameTemp(LoweringState& state, ValueRange qcTargets,
                                  ValueRange memrefs, ValueRange qcoTargets,
                                  ValueRange tensors) {
  auto& [yieldOrder, registers, currentQubits, currentRegisters] =
      state.modifierFrames.emplace_back();
  llvm::append_range(yieldOrder, qcTargets);
  for (auto [qcTarget, qcoTarget] : llvm::zip_equal(qcTargets, qcoTargets)) {
    currentQubits.try_emplace(qcTarget, qcoTarget);
  }
  llvm::append_range(registers, memrefs);
  for (auto [memref, tensor] : llvm::zip_equal(memrefs, tensors)) {
    currentRegisters.try_emplace(memref, tensor);
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

/**
 * @brief Helper function to check whether the type is a qc qubit type or a
 * container that holds qc qubit types
 *
 * @param type The type that is checked
 * @return Whether it is a qc type or not
 */
static bool isQCQubitType(Type type) {
  if (llvm::isa<qc::QubitType>(type)) {
    return true;
  }
  auto memref = dyn_cast<MemRefType>(type);
  return memref && llvm::isa<qc::QubitType>(memref.getElementType());
}

/**
 * @brief Recursively collects all the QC qubit references used by an
 * operation and store them in map
 *
 * @param Operation The operation that is currently traversed
 * @param state The lowering state
 * @param ctx The MLIRContext of the current program
 * @return llvm::Setvector<Value> The set of unique QC qubit references
 */
static std::pair<llvm::SetVector<Value>, llvm::SetVector<Value>>
collectUniqueQubits(Operation* op, LoweringState* state) {
  // Get the regions of the current operation
  const auto& regions = op->getRegions();
  bool needChange = false;
  SetVector<Value> uniqueQubits;
  SetVector<Value> uniqueRegisters;
  auto& regionQubitMap = state->regionQubitMap[op];
  auto& regionRegisterMap = state->regionRegisterMap[op];
  for (auto& region : regions) {

    auto& qubitInfoMap = state->qubitInfoMap[&region];
    // Skip empty regions e.g. empty else region of an If operation
    if (region.empty()) {
      continue;
    }
    // Check that the region has only one block
    assert(region.hasOneBlock() && "Expected single-block region");

    // Collect qubits from the blockarguments
    for (auto arg : region.front().getArguments()) {
      if (isQCQubitType(arg.getType())) {
        // TODO
        // uniqueQubits.insert(arg);
      }
    }
    for (auto& operation : region.front().getOperations()) {
      if (op->getDialect()->getNamespace() == "scf") {
        if (operation.getNumRegions() > 0) {
          auto [qubits, registers] = collectUniqueQubits(&operation, state);
          regionQubitMap.set_union(qubits);
          regionRegisterMap.set_union(registers);
        }
        if (auto loadOp = dyn_cast<memref::LoadOp>(operation)) {
          QubitInfo info{.reg = loadOp.getMemRef(),
                         .index = loadOp.getIndices()[0]};
          qubitInfoMap.try_emplace(loadOp.getResult(), info);
          regionRegisterMap.insert(loadOp.getMemRef());
          needChange = true;
          continue;
        }

        for (const auto& operand : operation.getOperands()) {
          if (llvm::isa<qc::QubitType>(operand.getType())) {
            if (!qubitInfoMap.contains(operand)) {
              regionQubitMap.insert(operand);
            }
          }
          if (auto memref = dyn_cast<MemRefType>(operand.getType())) {
            if (llvm::isa<qc::QubitType>(memref.getElementType())) {
              regionRegisterMap.insert(operand);
            }
          }
        }
      } else {
        if (operation.getNumRegions() > 0) {
          collectUniqueQubits(&operation, state);
        }
      }
    }
    // Mark scf operations that need to be changed afterwards
  }
  if (needChange || op->getDialect()->getNamespace() == "scf") {
    op->setAttr("needChange", StringAttr::get(op->getContext(), "yes"));
  }

  return {regionQubitMap, regionRegisterMap};
}

namespace {

/**
 * @brief Converts func.return and sinks remaining live qubits.
 *
 * @details
 * QC uses reference semantics and does not enforce linear typing for
 * qubits. After conversion, QCO requires that every qubit SSA value is
 * consumed exactly once. For allocations (including static qubits), the
 * sink is `qco.sink`. This pattern inserts `qco.sink` operations for all
 * still-live qubits tracked in the lowering state right before the return.
 */
struct ConvertFuncReturnOp final : StatefulOpConversionPattern<func::ReturnOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* funcRegion = op->getParentRegion();
    auto& map = state.qubitMap[funcRegion];

    // Build return values from qubitMap and collect live qubit information.
    // A qubit from the current scope is considered alive if it is returned
    // from the function. Otherwise, it is considered dead.
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
    for (auto qcoQubit : llvm::make_second_range(map)) {
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
 * @brief Converts memref.alloc to qtensor.alloc
 *
 * @par Example:
 * ```mlir
 * %memref = memref.alloc(%c3) : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!llvm::isa<qc::QubitType>(op.getType().getElementType())) {
      return failure();
    }

    auto shape = op.getType().getShape();
    if (shape.size() != 1) {
      return failure();
    }

    Value qtensor;
    if (shape[0] == ShapedType::kDynamic) {
      qtensor = rewriter.replaceOpWithNewOp<qtensor::AllocOp>(
          op, adaptor.getDynamicSizes()[0]);
    } else {
      auto size =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), shape[0]);
      qtensor =
          rewriter.replaceOpWithNewOp<qtensor::AllocOp>(op, size.getResult());
    }

    auto& state = getState();
    auto memref = op.getResult();
    assignMappedTensor(state, qtensor.getDefiningOp(), memref, qtensor);

    return success();
  }
};

/**
 * @brief Converts memref.load to qtensor.extract
 *
 * @par Example:
 * ```mlir
 * %q = memref.load %memref[%c0] : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %tensor_out, %q = qtensor.extract %tensor_in[%c0]: tensor<3x!qco.qubit>
 * ```
 */
struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto memref = op.getMemref();
    if (!llvm::isa<qc::QubitType>(memref.getType().getElementType())) {
      return failure();
    }

    auto& state = getState();
    auto& qubitInfoMap = state.qubitInfoMap;
    auto* operation = op.getOperation();

    // Look up latest QTensor value for this QC register
    auto qtensor = lookupMappedTensor(state, operation, memref);

    auto index = adaptor.getIndices()[0];
    auto extract =
        qtensor::ExtractOp::create(rewriter, op.getLoc(), qtensor, index);

    auto qcQubit = op.getResult();
    auto qcoQubit = extract.getResult();

    assignMappedQubit(state, operation, qcQubit, qcoQubit);
    assignMappedTensor(state, operation, memref, extract.getOutTensor());

    QubitInfo info{.reg = memref, .index = index};
    auto* parentRegion = operation->getParentRegion();
    if (auto it = qubitInfoMap.find(parentRegion); it != qubitInfoMap.end()) {
      it->second[qcQubit] = info;
    } else {
      qubitInfoMap[parentRegion][qcQubit] = info;
    }
    state.extractedQubits[parentRegion][memref].insert(qcQubit);
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts memref.dealloc to qtensor.dealloc
 *
 * @details
 * Before deallocating the tensor, all qubits are inserted back into it at
 * their original location.
 *
 * @par Example:
 * ```mlir
 * memref.dealloc %memref : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %t1 = qtensor.insert %q0 into %t0[%c0] : tensor<3x!qco.qubit>
 * %t2 = qtensor.insert %q1 into %t1[%c1] : tensor<3x!qco.qubit>
 * %t3 = qtensor.insert %q2 into %t2[%c2] : tensor<3x!qco.qubit>
 * qtensor.dealloc %t3 : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto memref = op.getMemref();
    if (!llvm::isa<qc::QubitType>(memref.getType().getElementType())) {
      return failure();
    }

    auto& state = getState();
    auto& qubitMap = state.qubitMap[op->getParentRegion()];
    auto& tensorMap = state.tensorMap[op->getParentRegion()];
    auto& qubitInfoMap = state.qubitInfoMap[op->getParentRegion()];

    // Look up latest QTensor value for this QC register
    auto qtensor = lookupMappedTensor(state, op.getOperation(), memref);

    // Filter out qubits belonging to this tensor
    for (auto it = qubitMap.begin(); it != qubitMap.end();) {
      auto current = it++;
      auto qcQubit = current->first;
      auto qcoQubit = current->second;

      auto infoIt = qubitInfoMap.find(qcQubit);
      if (infoIt == qubitInfoMap.end()) {
        continue;
      }

      auto& [reg, index] = infoIt->second;
      if (reg != memref) {
        continue;
      }

      qtensor = qtensor::InsertOp::create(rewriter, op.getLoc(), qcoQubit,
                                          qtensor, index)
                    .getResult();
      qubitMap.erase(current);
      qubitInfoMap.erase(infoIt);
    }
    tensorMap.erase(memref);

    rewriter.replaceOpWithNewOp<qtensor::DeallocOp>(op, qtensor);
    return success();
  }
};

/**
 * @brief Converts qc.alloc to qco.alloc
 *
 * @par Example:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = qco.alloc : !qco.qubit
 * ```
 */
struct ConvertQCAllocOp final : StatefulOpConversionPattern<qc::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto qcQubit = op.getResult();

    // Create the qco.alloc operation
    auto qcoOp = rewriter.replaceOpWithNewOp<qco::AllocOp>(op);

    auto qcoQubit = qcoOp.getResult();
    assignMappedQubit(state, qcoOp, qcQubit, qcoQubit);

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
    auto& qubitMap = state.qubitMap[op->getParentRegion()];
    auto* operation = op.getOperation();

    auto qcQubit = op.getQubit();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

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
    auto qcQubit = op.getQubit();

    auto qcoOp = rewriter.replaceOpWithNewOp<qco::StaticOp>(op, op.getIndex());
    assignMappedQubit(state, qcoOp, qcQubit, qcoOp.getQubit());

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
    auto qcQubit = op.getQubit();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

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
    auto qcQubit = op.getQubit();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create qco.reset (consumes input, produces output)
    auto qcoOp = qco::ResetOp::create(rewriter, op.getLoc(), qcoQubit);

    // Update mapping: the QC qubit now corresponds to the reset output
    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    // Erase the old (it has no results to replace)
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a zero-target, one-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.gphase(%theta)
 * ```
 * is converted to
 * ```mlir
 * qco.gphase(%theta)
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCZeroTargetOneParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    QCOOpType::create(rewriter, op.getLoc(), op.getParameter(0));

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a one-target, zero-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.x %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.x %q_in : !qco.qubit -> !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCOneTargetZeroParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getQubitIn();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit);

    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a one-target, one-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.rx(%theta) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.rx(%theta) %q_in : !qco.qubit -> !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCOneTargetOneParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getQubitIn();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp =
        QCOOpType::create(rewriter, op.getLoc(), qcoQubit, op.getParameter(0));

    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a one-target, two-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.r(%theta, %phi) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.r(%theta, %phi) %q_in : !qco.qubit -> !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCOneTargetTwoParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getQubitIn();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit,
                                   op.getParameter(0), op.getParameter(1));

    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a one-target, three-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.u(%theta, %phi, %lambda) %q_in : !qco.qubit -> !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCOneTargetThreeParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit = op.getQubitIn();
    auto qcoQubit = lookupMappedQubit(state, operation, qcQubit);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp =
        QCOOpType::create(rewriter, op.getLoc(), qcoQubit, op.getParameter(0),
                          op.getParameter(1), op.getParameter(2));

    assignMappedQubit(state, operation, qcQubit, qcoOp.getQubitOut());

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a two-target, zero-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.swap %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCTwoTargetZeroParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit0 = op.getQubit0In();
    auto qcQubit1 = op.getQubit1In();
    auto qcoQubit0 = lookupMappedQubit(state, operation, qcQubit0);
    auto qcoQubit1 = lookupMappedQubit(state, operation, qcQubit1);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit0, qcoQubit1);

    assignMappedQubit(state, operation, qcQubit0, qcoOp.getQubit0Out());
    assignMappedQubit(state, operation, qcQubit1, qcoOp.getQubit1Out());

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a two-target, one-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.rxx(%theta) %q0_in, %q1_in : !qco.qubit,
 * !qco.qubit
 * -> !qco.qubit, !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCTwoTargetOneParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit0 = op.getQubit0In();
    auto qcQubit1 = op.getQubit1In();
    auto qcoQubit0 = lookupMappedQubit(state, operation, qcQubit0);
    auto qcoQubit1 = lookupMappedQubit(state, operation, qcQubit1);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit0, qcoQubit1,
                                   op.getParameter(0));

    assignMappedQubit(state, operation, qcQubit0, qcoOp.getQubit0Out());
    assignMappedQubit(state, operation, qcQubit1, qcoOp.getQubit1Out());

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts a two-target, two-parameter QC gate to QCO
 *
 * @tparam QCOpType The operation type of the QC gate
 * @tparam QCOOpType The operation type of the QCO gate
 *
 * @par Example:
 * ```mlir
 * qc.xx_minus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.xx_minus_yy(%theta, %beta) %q0_in, %q1_in :
 * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCTwoTargetTwoParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto* operation = op.getOperation();
    auto qcQubit0 = op.getQubit0In();
    auto qcQubit1 = op.getQubit1In();
    auto qcoQubit0 = lookupMappedQubit(state, operation, qcQubit0);
    auto qcoQubit1 = lookupMappedQubit(state, operation, qcQubit1);

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit0, qcoQubit1,
                                   op.getParameter(0), op.getParameter(1));

    assignMappedQubit(state, operation, qcQubit0, qcoOp.getQubit0Out());
    assignMappedQubit(state, operation, qcQubit1, qcoOp.getQubit1Out());

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
 * %q_out:2 = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<qc::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto qcQubits = op.getQubits();
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

struct ConvertSCFYieldOp final : StatefulOpConversionPattern<scf::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& frame = currentModifierFrame(state);
    auto* operation = op.getOperation();
    auto* region = op->getParentRegion()->getParentOp();
    auto& registerMap = state.regionRegisterMap[region];
    auto& qubitMap = state.regionQubitMap[region];
    auto& memrefs = frame.memrefs;

    for (auto memref : memrefs) {
      auto& extractedQubits =
          state.extractedQubits[op->getParentRegion()][memref];
      // Insert all extracted qubits before extracting
      for (auto qubit : extractedQubits) {
        auto tensor = lookupMappedTensor(state, operation, memref);
        auto qcoQubit = lookupMappedQubit(state, operation, qubit);
        auto index = state.qubitInfoMap[op->getParentRegion()][qubit].index;
        auto insertOp = qtensor::InsertOp::create(rewriter, op.getLoc(),
                                                  qcoQubit, tensor, index);

        assignMappedTensor(state, operation, memref, insertOp.getResult());
      }
    }
    SmallVector<Value> targets;
    targets.reserve(registerMap.size() + qubitMap.size());
    targets.append(resolveMappedTensors(state, operation, frame.memrefs));
    targets.append(resolveMappedQubits(state, operation, frame.yieldOrder));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, targets);
    popModifierFrame(state);

    return success();
  }
};

struct ConvertSCFConditionOp final
    : StatefulOpConversionPattern<scf::ConditionOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& frame = currentModifierFrame(state);
    auto* operation = op.getOperation();
    auto* region = op->getParentRegion()->getParentOp();
    auto& registerMap = state.regionRegisterMap[region];
    auto& qubitMap = state.regionQubitMap[region];
    auto& memrefs = frame.memrefs;

    for (auto memref : memrefs) {
      auto& extractedQubits =
          state.extractedQubits[op->getParentRegion()][memref];
      // Insert all extracted qubits before extracting
      for (auto qubit : extractedQubits) {
        llvm::outs() << "hi\n";
        auto tensor = lookupMappedTensor(state, operation, memref);
        auto qcoQubit = lookupMappedQubit(state, operation, qubit);
        auto index = state.qubitInfoMap[op->getParentRegion()][qubit].index;
        auto insertOp = qtensor::InsertOp::create(rewriter, op.getLoc(),
                                                  qcoQubit, tensor, index);
        assignMappedTensor(state, operation, memref, insertOp.getResult());
      }
    }
    SmallVector<Value> targets;
    targets.reserve(registerMap.size() + qubitMap.size());
    targets.append(resolveMappedTensors(state, operation, frame.memrefs));
    targets.append(resolveMappedQubits(state, operation, frame.yieldOrder));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, op.getCondition(),
                                                  targets);
    popModifierFrame(state);

    return success();
  }
};

struct ConvertSCFForOp final : StatefulOpConversionPattern<scf::ForOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& registerMap = state.regionRegisterMap[op];
    auto& qubitMap = state.regionQubitMap[op];

    SmallVector<Value> qcTargets;
    qcTargets.reserve(registerMap.size() + qubitMap.size());
    qcTargets.append(registerMap.begin(), registerMap.end());
    qcTargets.append(qubitMap.begin(), qubitMap.end());

    auto tensors = resolveMappedTensors(state, operation, registerMap);
    auto qcoQubits = resolveMappedQubits(state, operation, qubitMap);

    SmallVector<Value> qcoTargets;
    qcoTargets.reserve(tensors.size() + qcoQubits.size());
    qcoTargets.append(tensors.begin(), tensors.end());
    qcoTargets.append(qcoQubits.begin(), qcoQubits.end());

    auto newFor =
        scf::ForOp::create(rewriter, op.getLoc(), op.getLowerBound(),
                           op.getUpperBound(), op.getStep(), qcoTargets);
    assignMappedTensors(state, op.getOperation(), registerMap,
                        newFor.getResults().take_front(tensors.size()));
    assignMappedQubits(state, op.getOperation(), qubitMap,
                       newFor->getResults().take_back(qcoQubits.size()));

    auto& srcBlock = op.getRegion().front();
    auto& dstBlock = newFor.getRegion().front();

    dstBlock.getOperations().splice(dstBlock.end(), srcBlock.getOperations());
    rewriter.replaceAllUsesWith(op.getInductionVar(), newFor.getInductionVar());

    SmallVector<Value, 8> qubitVector(qubitMap.begin(), qubitMap.end());
    SmallVector<Value, 8> registers(registerMap.begin(), registerMap.end());

    pushModifierFrameTemp(
        state, qubitVector, registers,
        dstBlock.getArguments().drop_front(1).take_back(qcoQubits.size()),
        dstBlock.getArguments().drop_front(1).drop_back(qcoQubits.size()));

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertSCFWhileOp final : StatefulOpConversionPattern<scf::WhileOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* operation = op.getOperation();
    auto& registerMap = state.regionRegisterMap[op];
    auto& qubitMap = state.regionQubitMap[op];

    SmallVector<Value> qcTargets;
    qcTargets.reserve(registerMap.size() + qubitMap.size());
    qcTargets.append(registerMap.begin(), registerMap.end());
    qcTargets.append(qubitMap.begin(), qubitMap.end());

    auto tensors = resolveMappedTensors(state, operation, registerMap);
    auto qcoQubits = resolveMappedQubits(state, operation, qubitMap);

    SmallVector<Value> qcoTargets;
    qcoTargets.reserve(tensors.size() + qcoQubits.size());
    qcoTargets.append(tensors.begin(), tensors.end());
    qcoTargets.append(qcoQubits.begin(), qcoQubits.end());

    auto newWhileOp = scf::WhileOp::create(
        rewriter, op.getLoc(), TypeRange(qcoTargets), ValueRange(qcoTargets));
    assignMappedTensors(state, op.getOperation(), registerMap,
                        newWhileOp.getResults().take_front(tensors.size()));
    assignMappedQubits(state, op.getOperation(), qubitMap,
                       newWhileOp->getResults().take_back(qcoQubits.size()));

    auto& newBeforeRegion = newWhileOp.getBefore();
    auto& newAfterRegion = newWhileOp.getAfter();

    const SmallVector<Location> locs(qcoTargets.size(), op->getLoc());
    // Create the new blocks
    auto* newBeforeBlock = rewriter.createBlock(
        &newBeforeRegion, {}, newWhileOp->getResultTypes(), locs);
    auto* newAfterBlock = rewriter.createBlock(
        &newAfterRegion, {}, newWhileOp->getResultTypes(), locs);
    newBeforeBlock->getOperations().splice(newBeforeBlock->end(),
                                           op.getBeforeBody()->getOperations());
    newAfterBlock->getOperations().splice(newAfterBlock->end(),
                                          op.getAfterBody()->getOperations());

    SmallVector<Value, 8> qubitVector(qubitMap.begin(), qubitMap.end());
    SmallVector<Value, 8> registers(registerMap.begin(), registerMap.end());

    pushModifierFrameTemp(
        state, qubitVector, registers,
        newAfterBlock->getArguments().take_back(qcoQubits.size()),
        newAfterBlock->getArguments().take_front(tensors.size()));
    pushModifierFrameTemp(
        state, qubitVector, registers,
        newBeforeBlock->getArguments().take_back(qcoQubits.size()),
        newBeforeBlock->getArguments().take_front(tensors.size()));

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Pass implementation for QC-to-QCO conversion
 *
 * @details
 * This pass converts QC dialect operations (reference semantics) to QCO
 * dialect operations (value semantics). The conversion is essential for
 * enabling optimization passes that rely on SSA form and explicit dataflow
 * analysis.
 *
 * The pass operates in several phases:
 * 1. Type conversion: !qc.qubit -> !qco.qubit
 * 2. Operation conversion: Each QC op is converted to its QCO equivalent
 * 3. State tracking: A LoweringState maintains qubit value mappings
 * 4. Function/control-flow adaptation: Function signatures and control flow
 * are updated to use QCO types
 *
 * The conversion maintains semantic equivalence while transforming the
 * representation from imperative (mutation-based) to functional
 * (SSA-based).
 */
struct QCToQCO final : impl::QCToQCOBase<QCToQCO> {
  using QCToQCOBase::QCToQCOBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    // Create state object to track qubit value flow
    LoweringState state;
    collectUniqueQubits(module, &state);
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCToQCOTypeConverter typeConverter(context);

    // Configure conversion target
    target.addIllegalDialect<QCDialect>();
    target.addLegalDialect<QCODialect, arith::ArithDialect,
                           qtensor::QTensorDialect>();

    target.addDynamicallyLegalDialect<memref::MemRefDialect>([](Operation* op) {
      auto isQubitMemref = [](Type t) {
        auto mt = llvm::dyn_cast<MemRefType>(t);
        return mt && llvm::isa<qc::QubitType>(mt.getElementType());
      };
      return llvm::none_of(op->getOperandTypes(), isQubitMemref) &&
             llvm::none_of(op->getResultTypes(), isQubitMemref);
    });
    //  target.addIllegalDialect<scf::SCFDialect>();
    target.addDynamicallyLegalDialect<scf::SCFDialect>([](Operation* op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::YieldOp>(
        [](Operation* op) { return op->getNumOperands() != 0; });
    target.addDynamicallyLegalOp<scf::ConditionOp>(
        [](Operation* op) { return op->getNumOperands() > 1; });
    // Register operation conversion patterns with state tracking
    patterns.add<
        ConvertSCFForOp, ConvertSCFYieldOp, ConvertSCFWhileOp,
        ConvertSCFConditionOp, ConvertMemRefAllocOp, ConvertMemRefLoadOp,
        ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCDeallocOp,
        ConvertQCStaticOp, ConvertQCMeasureOp, ConvertQCResetOp,
        ConvertQCZeroTargetOneParameterToQCO<qc::GPhaseOp, qco::GPhaseOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::IdOp, qco::IdOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::XOp, qco::XOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::YOp, qco::YOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::ZOp, qco::ZOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::HOp, qco::HOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::SOp, qco::SOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::SdgOp, qco::SdgOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::TOp, qco::TOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::TdgOp, qco::TdgOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::SXOp, qco::SXOp>,
        ConvertQCOneTargetZeroParameterToQCO<qc::SXdgOp, qco::SXdgOp>,
        ConvertQCOneTargetOneParameterToQCO<qc::RXOp, qco::RXOp>,
        ConvertQCOneTargetOneParameterToQCO<qc::RYOp, qco::RYOp>,
        ConvertQCOneTargetOneParameterToQCO<qc::RZOp, qco::RZOp>,
        ConvertQCOneTargetOneParameterToQCO<qc::POp, qco::POp>,
        ConvertQCOneTargetTwoParameterToQCO<qc::ROp, qco::ROp>,
        ConvertQCOneTargetTwoParameterToQCO<qc::U2Op, qco::U2Op>,
        ConvertQCOneTargetThreeParameterToQCO<qc::UOp, qco::UOp>,
        ConvertQCTwoTargetZeroParameterToQCO<qc::SWAPOp, qco::SWAPOp>,
        ConvertQCTwoTargetZeroParameterToQCO<qc::iSWAPOp, qco::iSWAPOp>,
        ConvertQCTwoTargetZeroParameterToQCO<qc::DCXOp, qco::DCXOp>,
        ConvertQCTwoTargetZeroParameterToQCO<qc::ECROp, qco::ECROp>,
        ConvertQCTwoTargetOneParameterToQCO<qc::RXXOp, qco::RXXOp>,
        ConvertQCTwoTargetOneParameterToQCO<qc::RYYOp, qco::RYYOp>,
        ConvertQCTwoTargetOneParameterToQCO<qc::RZXOp, qco::RZXOp>,
        ConvertQCTwoTargetOneParameterToQCO<qc::RZZOp, qco::RZZOp>,
        ConvertQCTwoTargetTwoParameterToQCO<qc::XXPlusYYOp, qco::XXPlusYYOp>,
        ConvertQCTwoTargetTwoParameterToQCO<qc::XXMinusYYOp, qco::XXMinusYYOp>,
        ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCInvOp, ConvertQCYieldOp>(
        typeConverter, context, &state);

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
    // Note: `func.return` may already be type-legal even though we still
    // need to insert sink operations (`qco.sink`) for dead qubit values.
    // Therefore, we mark it illegal as long as the qubit map of the region
    // is not empty.
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
