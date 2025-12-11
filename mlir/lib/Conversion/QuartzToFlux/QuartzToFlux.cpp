/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h"

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mlir {
using namespace flux;
using namespace quartz;

#define GEN_PASS_DEF_QUARTZTOFLUX
#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h.inc"

namespace {

/**
 * @brief State object for tracking qubit value flow during conversion
 *
 * @details
 * This struct maintains the mapping between Quartz dialect qubits (which use
 * reference semantics) and their corresponding Flux dialect qubit values
 * (which use value semantics). As the conversion progresses, each Quartz
 * qubit reference is mapped to its latest Flux SSA value.
 *
 * The key insight is that Quartz operations modify qubits in-place:
 * ```mlir
 * %q = quartz.alloc : !quartz.qubit
 * quartz.h %q : !quartz.qubit        // modifies %q in-place
 * quartz.x %q : !quartz.qubit        // modifies %q in-place
 * ```
 *
 * While Flux operations consume inputs and produce new outputs:
 * ```mlir
 * %q0 = flux.alloc : !flux.qubit
 * %q1 = flux.h %q0 : !flux.qubit -> !flux.qubit   // %q0 consumed, %q1 produced
 * %q2 = flux.x %q1 : !flux.qubit -> !flux.qubit   // %q1 consumed, %q2 produced
 * ```
 *
 * The qubitMap tracks that the Quartz qubit %q corresponds to:
 * - %q0 after allocation
 * - %q1 after the H gate
 * - %q2 after the X gate
 */
struct LoweringState {
  /// Map from original Quartz qubit references to their latest Flux SSA values
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> qubitMap;
  /// Map each initial op to its refQubits.
  llvm::DenseMap<Operation*, llvm::SetVector<Value>> regionMap;

  /// Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, SmallVector<Value>> targetsIn;
  DenseMap<int64_t, SmallVector<Value>> targetsOut;
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which tracks the mapping from reference-semantics Quartz qubits
 * to value-semantics Flux qubits across multiple pattern applications.
 *
 * This stateful approach is necessary because the conversion needs to:
 * 1. Track which Flux value corresponds to each Quartz qubit reference
 * 2. Update these mappings as operations transform qubits
 * 3. Share this information across different conversion patterns
 *
 * @tparam OpType The Quartz operation type to convert
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

llvm::SetVector<Value> collectRegionQubits(Operation* op, LoweringState* state,
                                           MLIRContext* ctx) {

  // get the regions of the current operation
  const auto regions = op->getRegions();
  SetVector<Value> uniqueQubits;
  for (auto& region : regions) {
    // skip empty regions e.g. empty else region of an If operation
    if (region.empty()) {
      continue;
    }
    // iterate over all operations inside the region
    // currently assumes that each region only has one block
    for (auto& operation : region.front().getOperations()) {
      // check if the operation has an region, if yes recursively collect the
      // qubits
      if (operation.getNumRegions() > 0) {
        auto qubits = collectRegionQubits(&operation, state, ctx);
        for (auto qubit : qubits) {
          uniqueQubits.insert(qubit);
        }
      }
      // collect qubits form the operands
      for (auto operand : operation.getOperands()) {
        if (operand.getType() == quartz::QubitType::get(ctx)) {
          uniqueQubits.insert(operand);
        }
      }
      // collect qubits from the results
      for (auto result : operation.getResults()) {
        if (result.getType() == quartz::QubitType::get(ctx)) {
          uniqueQubits.insert(result);
        }
      }
      if ((llvm::isa<scf::YieldOp>(operation) ||
           llvm::isa<scf::ConditionOp>(operation)) &&
          uniqueQubits.size() > 0) {
        operation.setAttr("needChange", StringAttr::get(ctx, "yes"));
      }
    }
  }
  if (!uniqueQubits.empty() &&
      (llvm::isa<scf::IfOp>(op) || (llvm::isa<scf::ForOp>(op)) ||
       llvm::isa<scf::WhileOp>(op))) {
    state->regionMap[op] = uniqueQubits;
    // mark operations that need to be changed afterwards
    op->setAttr("needChange", StringAttr::get(ctx, "yes"));
  }
  return uniqueQubits;
}

/**
 * @brief Converts a zero-target, one-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertZeroTargetOneParameter(QuartzOpType& op,
                                            ConversionPatternRewriter& rewriter,
                                            LoweringState& state) {
  const auto inCtrlOp = state.inCtrlOp;

  rewriter.create<FluxOpType>(op.getLoc(), op.getParameter(0));

  // Update the state
  if (inCtrlOp != 0) {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut;
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, zero-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertOneTargetZeroParameter(QuartzOpType& op,
                                            ConversionPatternRewriter& rewriter,
                                            LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubit
  const auto& quartzQubit = op.getQubitIn();
  Value fluxQubit;
  if (inCtrlOp == 0) {
    fluxQubit = qubitMap[quartzQubit];
  } else {
    fluxQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp = rewriter.create<FluxOpType>(op.getLoc(), fluxQubit);

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit] = fluxOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({fluxOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, one-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertOneTargetOneParameter(QuartzOpType& op,
                                           ConversionPatternRewriter& rewriter,
                                           LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubit
  const auto& quartzQubit = op.getQubitIn();
  Value fluxQubit;
  if (inCtrlOp == 0) {
    fluxQubit = qubitMap[quartzQubit];
  } else {
    fluxQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp =
      rewriter.create<FluxOpType>(op.getLoc(), fluxQubit, op.getParameter(0));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit] = fluxOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({fluxOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, two-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertOneTargetTwoParameter(QuartzOpType& op,
                                           ConversionPatternRewriter& rewriter,
                                           LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubit
  const auto& quartzQubit = op.getQubitIn();
  Value fluxQubit;
  if (inCtrlOp == 0) {
    fluxQubit = qubitMap[quartzQubit];
  } else {
    fluxQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp = rewriter.create<FluxOpType>(
      op.getLoc(), fluxQubit, op.getParameter(0), op.getParameter(1));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit] = fluxOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({fluxOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, three-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult
convertOneTargetThreeParameter(QuartzOpType& op,
                               ConversionPatternRewriter& rewriter,
                               LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubit
  const auto& quartzQubit = op.getQubitIn();
  Value fluxQubit;
  if (inCtrlOp == 0) {
    fluxQubit = qubitMap[quartzQubit];
  } else {
    fluxQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp =
      rewriter.create<FluxOpType>(op.getLoc(), fluxQubit, op.getParameter(0),
                                  op.getParameter(1), op.getParameter(2));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit] = fluxOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({fluxOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertTwoTargetZeroParameter(QuartzOpType& op,
                                            ConversionPatternRewriter& rewriter,
                                            LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubits
  const auto& quartzQubit0 = op.getQubit0In();
  const auto& quartzQubit1 = op.getQubit1In();
  Value fluxQubit0;
  Value fluxQubit1;
  if (inCtrlOp == 0) {
    fluxQubit0 = qubitMap[quartzQubit0];
    fluxQubit1 = qubitMap[quartzQubit1];
  } else {
    const auto& targetsIn = state.targetsIn[inCtrlOp];
    fluxQubit0 = targetsIn[0];
    fluxQubit1 = targetsIn[1];
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp =
      rewriter.create<FluxOpType>(op.getLoc(), fluxQubit0, fluxQubit1);

  // Update state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit0] = fluxOp.getQubit0Out();
    qubitMap[quartzQubit1] = fluxOp.getQubit1Out();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut(
        {fluxOp.getQubit0Out(), fluxOp.getQubit1Out()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a two-target, one-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertTwoTargetOneParameter(QuartzOpType& op,
                                           ConversionPatternRewriter& rewriter,
                                           LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubits
  const auto& quartzQubit0 = op.getQubit0In();
  const auto& quartzQubit1 = op.getQubit1In();
  Value fluxQubit0;
  Value fluxQubit1;
  if (inCtrlOp == 0) {
    fluxQubit0 = qubitMap[quartzQubit0];
    fluxQubit1 = qubitMap[quartzQubit1];
  } else {
    const auto& targetsIn = state.targetsIn[inCtrlOp];
    fluxQubit0 = targetsIn[0];
    fluxQubit1 = targetsIn[1];
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp = rewriter.create<FluxOpType>(op.getLoc(), fluxQubit0, fluxQubit1,
                                            op.getParameter(0));

  // Update state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit0] = fluxOp.getQubit0Out();
    qubitMap[quartzQubit1] = fluxOp.getQubit1Out();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut(
        {fluxOp.getQubit0Out(), fluxOp.getQubit1Out()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a two-target, two-parameter Quartz operation to Flux
 *
 * @tparam FluxOpType The operation type of the Flux operation
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename FluxOpType, typename QuartzOpType>
LogicalResult convertTwoTargetTwoParameter(QuartzOpType& op,
                                           ConversionPatternRewriter& rewriter,
                                           LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest Flux qubits
  const auto& quartzQubit0 = op.getQubit0In();
  const auto& quartzQubit1 = op.getQubit1In();
  Value fluxQubit0;
  Value fluxQubit1;
  if (inCtrlOp == 0) {
    fluxQubit0 = qubitMap[quartzQubit0];
    fluxQubit1 = qubitMap[quartzQubit1];
  } else {
    const auto& targetsIn = state.targetsIn[inCtrlOp];
    fluxQubit0 = targetsIn[0];
    fluxQubit1 = targetsIn[1];
  }

  // Create the Flux operation (consumes input, produces output)
  auto fluxOp =
      rewriter.create<FluxOpType>(op.getLoc(), fluxQubit0, fluxQubit1,
                                  op.getParameter(0), op.getParameter(1));

  // Update state map
  if (inCtrlOp == 0) {
    qubitMap[quartzQubit0] = fluxOp.getQubit0Out();
    qubitMap[quartzQubit1] = fluxOp.getQubit1Out();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut(
        {fluxOp.getQubit0Out(), fluxOp.getQubit1Out()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

} // namespace

/**
 * @brief Type converter for Quartz-to-Flux conversion
 *
 * @details
 * Handles type conversion between the Quartz and Flux dialects.
 * The primary conversion is from !quartz.qubit to !flux.qubit, which
 * represents the semantic shift from reference types to value types.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
class QuartzToFluxTypeConverter final : public TypeConverter {
public:
  explicit QuartzToFluxTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert Quartz qubit references to Flux qubit values
    addConversion([ctx](quartz::QubitType /*type*/) -> Type {
      return flux::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Converts quartz.alloc to flux.alloc
 *
 * @details
 * Allocates a new qubit and establishes the initial mapping in the state.
 * Both dialects initialize qubits to the |0⟩ state.
 *
 * Register metadata (name, size, index) is preserved during conversion,
 * allowing the Flux representation to maintain register information for
 * debugging and visualization.
 *
 * Example transformation:
 * ```mlir
 * %q = quartz.alloc("q", 3, 0) : !quartz.qubit
 * // becomes:
 * %q0 = flux.alloc("q", 3, 0) : !flux.qubit
 * ```
 */
struct ConvertQuartzAllocOp final
    : StatefulOpConversionPattern<quartz::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& quartzQubit = op.getResult();

    // Create the flux.alloc operation with preserved register metadata
    auto fluxOp = rewriter.replaceOpWithNewOp<flux::AllocOp>(
        op, op.getRegisterNameAttr(), op.getRegisterSizeAttr(),
        op.getRegisterIndexAttr());

    const auto& fluxQubit = fluxOp.getResult();

    // Establish initial mapping: this Quartz qubit reference now corresponds
    // to this Flux SSA value
    qubitMap.try_emplace(quartzQubit, fluxQubit);

    return success();
  }
};

/**
 * @brief Converts quartz.dealloc to flux.dealloc
 *
 * @details
 * Deallocates a qubit by looking up its latest Flux value and creating
 * a corresponding flux.dealloc operation. The mapping is removed from
 * the state as the qubit is no longer in use.
 *
 * Example transformation:
 * ```mlir
 * quartz.dealloc %q : !quartz.qubit
 * // becomes (where %q maps to %q_final):
 * flux.dealloc %q_final : !flux.qubit
 * ```
 */
struct ConvertQuartzDeallocOp final
    : StatefulOpConversionPattern<quartz::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& quartzQubit = op.getQubit();

    // Look up the latest Flux value for this Quartz qubit
    const auto& fluxQubit = qubitMap[quartzQubit];

    // Create the dealloc operation
    rewriter.replaceOpWithNewOp<flux::DeallocOp>(op, fluxQubit);

    // Remove from state as qubit is no longer in use
    qubitMap.erase(quartzQubit);

    return success();
  }
};

/**
 * @brief Converts quartz.static to flux.static
 *
 * @details
 * Static qubits represent references to hardware-mapped or fixed-position
 * qubits identified by an index. This conversion creates the corresponding
 * flux.static operation and establishes the mapping.
 *
 * Example transformation:
 * ```mlir
 * %q = quartz.static 0 : !quartz.qubit
 * // becomes:
 * %q0 = flux.static 0 : !flux.qubit
 * ```
 */
struct ConvertQuartzStaticOp final
    : StatefulOpConversionPattern<quartz::StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& quartzQubit = op.getQubit();

    // Create new flux.static operation with the same index
    auto fluxOp = rewriter.create<flux::StaticOp>(op.getLoc(), op.getIndex());

    // Collect Flux qubit SSA value
    const auto& fluxQubit = fluxOp.getQubit();

    // Establish mapping from Quartz reference to Flux value
    qubitMap[quartzQubit] = fluxQubit;

    // Replace the old operation result with the new result
    rewriter.replaceOp(op, fluxQubit);

    return success();
  }
};

/**
 * @brief Converts quartz.measure to flux.measure
 *
 * @details
 * Measurement is a key operation where the semantic difference is visible:
 * - Quartz: Measures in-place, returning only the classical bit
 * - Flux: Consumes input qubit, returns both output qubit and classical bit
 *
 * The conversion looks up the latest Flux value for the Quartz qubit,
 * performs the measurement, updates the mapping with the output qubit,
 * and returns the classical bit result.
 *
 * Register metadata (name, size, index) for output recording is preserved
 * during conversion.
 *
 * Example transformation:
 * ```mlir
 * %c = quartz.measure("c", 2, 0) %q : !quartz.qubit -> i1
 * // becomes (where %q maps to %q_in):
 * %q_out, %c = flux.measure("c", 2, 0) %q_in : !flux.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQuartzMeasureOp final
    : StatefulOpConversionPattern<quartz::MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& quartzQubit = op.getQubit();

    // Get the latest Flux qubit value from the state map
    const auto& fluxQubit = qubitMap[quartzQubit];

    // Create flux.measure (returns both output qubit and bit result)
    auto fluxOp = rewriter.create<flux::MeasureOp>(
        op.getLoc(), fluxQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    const auto& outFluxQubit = fluxOp.getQubitOut();
    const auto& newBit = fluxOp.getResult();

    // Update mapping: the Quartz qubit now corresponds to the output qubit
    qubitMap[quartzQubit] = outFluxQubit;

    // Replace the Quartz operation's bit result with the Flux bit result
    rewriter.replaceOp(op, newBit);

    return success();
  }
};

/**
 * @brief Converts quartz.reset to flux.reset
 *
 * @details
 * Reset operations force a qubit to the |0⟩ state. The semantic difference:
 * - Quartz: Resets in-place (no result value)
 * - Flux: Consumes input qubit, returns reset output qubit
 *
 * The conversion looks up the latest Flux value, performs the reset,
 * and updates the mapping with the output qubit. The Quartz operation
 * is erased as it has no results to replace.
 *
 * Example transformation:
 * ```mlir
 * quartz.reset %q : !quartz.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = flux.reset %q_in : !flux.qubit -> !flux.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQuartzResetOp final
    : StatefulOpConversionPattern<quartz::ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::ResetOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& quartzQubit = op.getQubit();

    // Get the latest Flux qubit value from the state map
    const auto& fluxQubit = qubitMap[quartzQubit];

    // Create flux.reset (consumes input, produces output)
    auto fluxOp = rewriter.create<flux::ResetOp>(op.getLoc(), fluxQubit);

    // Update mapping: the Quartz qubit now corresponds to the reset output
    qubitMap[quartzQubit] = fluxOp.getQubitOut();

    // Erase the old (it has no results to replace)
    rewriter.eraseOp(op);

    return success();
  }
};

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM)                                                    \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * flux.OP_NAME(%PARAM)                                                      \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertZeroTargetOneParameter<flux::OP_CLASS>(op, rewriter,       \
                                                           getState());        \
    }                                                                          \
  };

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME %q : !quartz.qubit                                         \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME %q_in : !flux.qubit -> !flux.qubit                  \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<flux::OP_CLASS>(op, rewriter,       \
                                                           getState());        \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, id)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXOp, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM) %q : !quartz.qubit                                 \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME(%PARAM) %q_in : !flux.qubit -> !flux.qubit          \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<flux::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM1, %PARAM2) %q : !quartz.qubit                       \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME(%PARAM1, %PARAM2) %q_in : !flux.qubit ->            \
   * !flux.qubit                                                               \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetTwoParameter<flux::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q : !quartz.qubit              \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q_in : !flux.qubit      \
   * -> !flux.qubit                                                            \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetThreeParameter<flux::OP_CLASS>(op, rewriter,      \
                                                            getState());       \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME %q0, %q1 : !quartz.qubit, !quartz.qubit                    \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = flux.OP_NAME %q0_in, %q1_in : !flux.qubit, !flux.qubit \
   * -> !flux.qubit, !flux.qubit                                               \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetZeroParameter<flux::OP_CLASS>(op, rewriter,       \
                                                           getState());        \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM) %q0, %q1 : !quartz.qubit, !quartz.qubit            \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = flux.OP_NAME(%PARAM) %q0_in, %q1_in : !flux.qubit,     \
   * !flux.qubit -> !flux.qubit, !flux.qubit                                   \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetOneParameter<flux::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME to flux.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM1, %PARAM2) %q0, %q1 : !quartz.qubit, !quartz.qubit  \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = flux.OP_NAME(%PARAM1, %PARAM2) %q0_in, %q1_in :        \
   * !flux.qubit, !flux.qubit -> !flux.qubit, !flux.qubit                      \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS final                                         \
      : StatefulOpConversionPattern<quartz::OP_CLASS> {                        \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(quartz::OP_CLASS op, OpAdaptor /*adaptor*/,                \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetTwoParameter<flux::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

/**
 * @brief Converts quartz.barrier to flux.barrier
 *
 * @par Example:
 * ```mlir
 * quartz.barrier %q0, %q1 : !quartz.qubit, !quartz.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = flux.barrier %q0_in, %q1_in : !flux.qubit, !flux.qubit ->
 * !flux.qubit, !flux.qubit
 * ```
 */
struct ConvertQuartzBarrierOp final
    : StatefulOpConversionPattern<quartz::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& qubitMap = state.qubitMap[op->getParentRegion()];

    // Get Flux qubits from state map
    const auto& quartzQubits = op.getQubits();
    SmallVector<Value> fluxQubits;
    fluxQubits.reserve(quartzQubits.size());
    for (const auto& quartzQubit : quartzQubits) {
      fluxQubits.push_back(qubitMap[quartzQubit]);
    }

    // Create flux.barrier
    auto fluxOp = rewriter.create<flux::BarrierOp>(op.getLoc(), fluxQubits);

    // Update state map
    for (const auto& [quartzQubit, fluxQubitOut] :
         llvm::zip(quartzQubits, fluxOp.getQubitsOut())) {
      qubitMap[quartzQubit] = fluxQubitOut;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts quartz.ctrl to flux.ctrl
 *
 * @par Example:
 * ```mlir
 * quartz.ctrl(%q0) {
 *   quartz.x %q1
 *   quartz.yield
 * }
 * ```
 * is converted to
 * ```mlir
 * %controls_out, %targets_out = flux.ctrl(%q0_in) %q1_in {
 *   %q1_res = flux.x %q1_in : !flux.qubit -> !flux.qubit
 *   flux.yield %q1_res
 * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
 * ```
 */
struct ConvertQuartzCtrlOp final : StatefulOpConversionPattern<quartz::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::CtrlOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& qubitMap = state.qubitMap[op->getParentRegion()];

    // Get Flux controls from state map
    const auto& quartzControls = op.getControls();
    SmallVector<Value> fluxControls;
    fluxControls.reserve(quartzControls.size());
    for (const auto& quartzControl : quartzControls) {
      fluxControls.push_back(qubitMap[quartzControl]);
    }

    // Get Flux targets from state map
    const auto numTargets = op.getNumTargets();
    SmallVector<Value> fluxTargets;
    fluxTargets.reserve(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      const auto& quartzTarget = op.getTarget(i);
      const auto& fluxTarget = qubitMap[quartzTarget];
      fluxTargets.push_back(fluxTarget);
    }

    // Create flux.ctrl
    auto fluxOp =
        rewriter.create<flux::CtrlOp>(op.getLoc(), fluxControls, fluxTargets);

    // Update state map if this is a top-level CtrlOp
    // Nested CtrlOps are managed via the targetsIn and targetsOut maps
    if (state.inCtrlOp == 0) {
      for (const auto& [quartzControl, fluxControl] :
           llvm::zip(quartzControls, fluxOp.getControlsOut())) {
        qubitMap[quartzControl] = fluxControl;
      }
      const auto& targetsOut = fluxOp.getTargetsOut();
      for (size_t i = 0; i < numTargets; ++i) {
        const auto& quartzTarget = op.getTarget(i);
        qubitMap[quartzTarget] = targetsOut[i];
      }
    }

    // Update modifier information
    state.inCtrlOp++;
    state.targetsIn.try_emplace(state.inCtrlOp, fluxTargets);

    // Clone body region from Quartz to Flux
    auto& dstRegion = fluxOp.getBody();
    rewriter.cloneRegionBefore(op.getBody(), dstRegion, dstRegion.end());

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts quartz.yield to flux.yield
 *
 * @par Example:
 * ```mlir
 * quartz.yield
 * ```
 * is converted to
 * ```mlir
 * flux.yield %targets
 * ```
 */
struct ConvertQuartzYieldOp final
    : StatefulOpConversionPattern<quartz::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto& targets = state.targetsOut[state.inCtrlOp];
    rewriter.replaceOpWithNewOp<flux::YieldOp>(op, targets);
    state.targetsOut.erase(state.inCtrlOp);
    state.inCtrlOp--;
    return success();
  }
};

struct ConvertQuartzScfIfOp final : StatefulOpConversionPattern<scf::IfOp> {
  using StatefulOpConversionPattern<scf::IfOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto quartzQubits = getState().regionMap[op];
    SmallVector<Value> values;
    values.reserve(quartzQubits.size());
    for (auto qubit : quartzQubits) {
      values.push_back(qubit);
    }
    // create result typerange
    auto const optType = flux::QubitType::get(rewriter.getContext());
    SmallVector<Type> resultTypes;
    resultTypes.assign(quartzQubits.size(), optType);

    // create new if operation
    auto newIf = rewriter.create<scf::IfOp>(
        op->getLoc(), TypeRange{resultTypes}, op.getCondition(), true);
    auto& thenRegion = newIf.getThenRegion();
    auto& elseRegion = newIf.getElseRegion();
    // move the regions of the old operations inside the new operation
    rewriter.inlineRegionBefore(op.getThenRegion(), thenRegion,
                                thenRegion.end());
    rewriter.eraseBlock(&thenRegion.front());

    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), elseRegion,
                                  elseRegion.end());
      rewriter.eraseBlock(&elseRegion.front());
    } else {
      rewriter.setInsertionPointToEnd(&elseRegion.front());
      const auto elseYield =
          rewriter.create<scf::YieldOp>(op->getLoc(), values);
      elseYield->setAttr("needChange",
                         StringAttr::get(rewriter.getContext(), "yes"));
    }

    auto& thenRegionQubitMap = getState().qubitMap[&thenRegion];
    auto& elseRegionQubitMap = getState().qubitMap[&elseRegion];
    for (const auto& refQubit : quartzQubits) {
      thenRegionQubitMap.try_emplace(
          refQubit, getState().qubitMap[op->getParentRegion()][refQubit]);
      elseRegionQubitMap.try_emplace(
          refQubit, getState().qubitMap[op->getParentRegion()][refQubit]);
    }
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    for (size_t i = 0; i < newIf->getResults().size(); i++) {
      qubitMap[quartzQubits[i]] = newIf->getResult(i);
    }

    rewriter.eraseOp(op);
    return success();
  }
};
struct ConvertQuartzScfWhileOp final
    : StatefulOpConversionPattern<scf::WhileOp> {
  using StatefulOpConversionPattern<scf::WhileOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    auto refQubits = getState().regionMap[op];

    SmallVector<Value> values;
    values.reserve(refQubits.size());
    for (auto qubit : refQubits) {
      values.push_back(qubit);
    }
    SmallVector<Value> optQubits;
    SmallVector<Type> types(refQubits.size(),
                            flux::QubitType::get(rewriter.getContext()));
    for (auto [refQubit, optQubit] : qubitMap) {
      optQubits.push_back(optQubit);
    }
    auto newWhileOp = rewriter.create<scf::WhileOp>(
        op.getLoc(), TypeRange(types), ValueRange(optQubits));
    auto& newBeforeRegion = newWhileOp.getBefore();
    auto& newAfterRegion = newWhileOp.getAfter();
    SmallVector<Location> locs(refQubits.size(), op->getLoc());
    auto* newBeforeBlock =
        rewriter.createBlock(&newBeforeRegion, {}, types, locs);
    auto* newAfterBlock =
        rewriter.createBlock(&newAfterRegion, {}, types, locs);

    newBeforeBlock->getOperations().splice(newBeforeBlock->end(),
                                           op.getBeforeBody()->getOperations());
    newAfterBlock->getOperations().splice(newAfterBlock->end(),
                                          op.getAfterBody()->getOperations());
    auto& newBeforeRegionMap = getState().qubitMap[&newWhileOp.getBefore()];
    auto& newAfterRegionMap = getState().qubitMap[&newWhileOp.getAfter()];
    for (size_t i = 0; i < refQubits.size(); i++) {
      newBeforeRegionMap.try_emplace(refQubits[i],
                                     newWhileOp.getBeforeArguments()[i]);
    }
    for (size_t i = 0; i < refQubits.size(); i++) {
      newAfterRegionMap.try_emplace(refQubits[i],
                                    newWhileOp.getAfterArguments()[i]);
    }

    for (size_t i = 0; i < newWhileOp->getResults().size(); i++) {
      qubitMap[refQubits[i]] = newWhileOp->getResult(i);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuartzScfForOp final : StatefulOpConversionPattern<scf::ForOp> {
  using StatefulOpConversionPattern<scf::ForOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    auto& qubitMap = getState().qubitMap[op->getParentRegion()];

    auto refQubits = getState().regionMap[op];
    SmallVector<Value> values;
    values.reserve(refQubits.size());
    for (auto qubit : refQubits) {
      values.push_back(qubit);
    }

    SmallVector<Value> optQubits;
    for (auto [quartQubit, fluxQubit] : qubitMap) {
      optQubits.push_back(fluxQubit);
    }
    // Create a new for-loop with flux qubits as iter_args
    auto newFor = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), ValueRange(optQubits));
    auto& srcBlock = op.getRegion().front();
    auto& dstBlock = newFor.getRegion().front();
    dstBlock.getOperations().splice(dstBlock.end(), srcBlock.getOperations());

    auto& newRegion = newFor.getRegion();

    auto& regionQubitMap = getState().qubitMap[&newRegion];


    for (const auto& [refQubit, optQubit] :
         llvm::zip_equal(refQubits, newFor.getRegionIterArgs())) {
      regionQubitMap.try_emplace(refQubit, optQubit);
    }

    auto& map = getState().qubitMap[op->getParentRegion()];
    for (size_t i = 0; i < newFor->getResults().size(); i++) {
      map[refQubits[i]] = newFor->getResult(i);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuartzScfYieldOp final
    : StatefulOpConversionPattern<scf::YieldOp> {
  using StatefulOpConversionPattern<scf::YieldOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    const auto& qubitMap = getState().qubitMap[op->getParentRegion()];

    SmallVector<Value> optQubits;
    for (auto [refQubit, optQubit] : qubitMap) {
      optQubits.push_back(optQubit);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, optQubits);
    return success();
  }
};
struct ConvertQuartzScfCondtionOp final
    : StatefulOpConversionPattern<scf::ConditionOp> {
  using StatefulOpConversionPattern<
      scf::ConditionOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    const auto& qubitMap = getState().qubitMap[op->getParentRegion()];

    SmallVector<Value> optQubits;
    for (auto [refQubit, optQubit] : qubitMap) {
      optQubits.push_back(optQubit);
    }
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, op.getCondition(),
                                                  optQubits);

    return success();
  }
};

/**
 * @brief Pass implementation for Quartz-to-Flux conversion
 *
 * @details
 * This pass converts Quartz dialect operations (reference
 * semantics) to Flux dialect operations (value semantics).
 * The conversion is essential for enabling optimization
 * passes that rely on SSA form and explicit dataflow
 * analysis.
 *
 * The pass operates in several phases:
 * 1. Type conversion: !quartz.qubit -> !flux.qubit
 * 2. Operation conversion: Each Quartz op is converted to
 * its Flux equivalent
 * 3. State tracking: A LoweringState maintains qubit value
 * mappings
 * 4. Function/control-flow adaptation: Function signatures
 * and control flow are updated to use Flux types
 *
 * The conversion maintains semantic equivalence while
 * transforming the representation from imperative
 * (mutation-based) to functional (SSA-based).
 */
struct QuartzToFlux final : impl::QuartzToFluxBase<QuartzToFlux> {
  using QuartzToFluxBase::QuartzToFluxBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    // Create state object to track qubit value flow
    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QuartzToFluxTypeConverter typeConverter(context);

    collectRegionQubits(module, &state, context);
    // Configure conversion target: Quartz illegal, Flux
    // legal
    target.addIllegalDialect<QuartzDialect>();
    target.addLegalDialect<FluxDialect>();

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::WhileOp>([&](scf::WhileOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::ConditionOp>([&](scf::ConditionOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    // Register operation conversion patterns with state
    // tracking
    patterns.add<
        ConvertQuartzAllocOp, ConvertQuartzDeallocOp, ConvertQuartzStaticOp,
        ConvertQuartzMeasureOp, ConvertQuartzResetOp, ConvertQuartzGPhaseOp,
        ConvertQuartzIdOp, ConvertQuartzXOp, ConvertQuartzYOp, ConvertQuartzZOp,
        ConvertQuartzHOp, ConvertQuartzSOp, ConvertQuartzSdgOp,
        ConvertQuartzTOp, ConvertQuartzTdgOp, ConvertQuartzSXOp,
        ConvertQuartzSXdgOp, ConvertQuartzRXOp, ConvertQuartzRYOp,
        ConvertQuartzRZOp, ConvertQuartzPOp, ConvertQuartzROp,
        ConvertQuartzU2Op, ConvertQuartzUOp, ConvertQuartzSWAPOp,
        ConvertQuartziSWAPOp, ConvertQuartzDCXOp, ConvertQuartzECROp,
        ConvertQuartzRXXOp, ConvertQuartzRYYOp, ConvertQuartzRZXOp,
        ConvertQuartzRZZOp, ConvertQuartzXXPlusYYOp, ConvertQuartzXXMinusYYOp,
        ConvertQuartzBarrierOp, ConvertQuartzCtrlOp, ConvertQuartzYieldOp,
        ConvertQuartzScfIfOp, ConvertQuartzScfYieldOp, ConvertQuartzScfWhileOp,
        ConvertQuartzScfCondtionOp, ConvertQuartzScfForOp>(typeConverter,
                                                           context, &state);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
