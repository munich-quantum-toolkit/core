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

#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/IRMapping.h>
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
  llvm::DenseMap<Value, Value> qubitMap;
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
  using OpConversionPattern<OpType>::OpConversionPattern;

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
    const auto& quartzQubit = op.getResult();

    // Create the flux.alloc operation with preserved register metadata
    auto fluxOp = rewriter.replaceOpWithNewOp<flux::AllocOp>(
        op, op.getRegisterNameAttr(), op.getRegisterSizeAttr(),
        op.getRegisterIndexAttr());

    const auto& fluxQubit = fluxOp.getResult();

    // Establish initial mapping: this Quartz qubit reference now corresponds
    // to this Flux SSA value
    getState().qubitMap.try_emplace(quartzQubit, fluxQubit);

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
    const auto& quartzQubit = op.getQubit();

    // Look up the latest Flux value for this Quartz qubit
    const auto& fluxQubit = getState().qubitMap[quartzQubit];

    // Create the dealloc operation
    rewriter.replaceOpWithNewOp<flux::DeallocOp>(op, fluxQubit);

    // Remove from state as qubit is no longer in use
    getState().qubitMap.erase(quartzQubit);

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
    // Prepare result type
    const auto& qubitType = flux::QubitType::get(rewriter.getContext());

    // Create new flux.static operation with the same index
    auto fluxOp =
        rewriter.create<flux::StaticOp>(op.getLoc(), qubitType, op.getIndex());

    // Collect Quartz and Flux SSA values
    const auto& quartzQubit = op.getQubit();
    const auto& fluxQubit = fluxOp.getQubit();

    // Establish mapping from Quartz reference to Flux value
    getState().qubitMap[quartzQubit] = fluxQubit;

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

    const auto& quartzQubit = op.getQubit();

    // Get the latest Flux qubit value from the state map
    const auto& fluxQubit = getState().qubitMap[quartzQubit];

    // Create flux.measure operation (returns both output qubit and bit result)
    auto fluxOp = rewriter.create<flux::MeasureOp>(
        op.getLoc(), fluxQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    const auto outFluxQubit = fluxOp.getQubitOut();
    const auto newBit = fluxOp.getResult();

    // Update mapping: the Quartz qubit now corresponds to the output qubit
    getState().qubitMap[quartzQubit] = outFluxQubit;

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

    // Prepare result type
    const auto& qubitType = flux::QubitType::get(rewriter.getContext());

    const auto& quartzQubit = op.getQubit();

    // Get the latest Flux qubit value from the state map
    const auto& fluxQubit = getState().qubitMap[quartzQubit];

    // Create flux.reset operation (consumes input, produces output)
    auto fluxOp =
        rewriter.create<flux::ResetOp>(op.getLoc(), qubitType, fluxQubit);

    auto outFluxQubit = fluxOp.getQubitOut();

    // Update mapping: the Quartz qubit now corresponds to the reset output
    getState().qubitMap[quartzQubit] = outFluxQubit;

    // Erase the old operation (it has no results to replace)
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts quartz.x to flux.x
 *
 * @details
 * Example transformation:
 * ```mlir
 * quartz.x %q : !quartz.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = flux.x %q_in : !flux.qubit -> !flux.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQuartzXOp final : StatefulOpConversionPattern<quartz::XOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::XOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& quartzQubit = op.getQubit();

    auto inRegion = dyn_cast<flux::CtrlOp>(op->getParentOp()) != nullptr;

    // Get the latest Flux qubit
    Value fluxQubitIn;
    if (inRegion) {
      fluxQubitIn = rewriter.getRemappedValue(quartzQubit);
    } else {
      fluxQubitIn = getState().qubitMap[quartzQubit];
    }

    // Create flux.x operation (consumes input, produces output)
    auto fluxOp = rewriter.create<flux::XOp>(op.getLoc(), fluxQubitIn);

    // Update state map
    if (!inRegion) {
      getState().qubitMap[quartzQubit] = fluxOp.getQubitOut();
    }

    // Replace the Quartz operation with the Flux operation
    rewriter.replaceOp(op, fluxOp.getResult());

    return success();
  }
};

/**
 * @brief Converts quartz.rx to flux.rx
 *
 * @details
 * Example transformation:
 * ```mlir
 * quartz.rx(%theta) %q : !quartz.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = flux.rx(%theta) %q_in : !flux.qubit -> !flux.qubit
 * // state updated: %q now maps to %q_out
 */
struct ConvertQuartzRXOp final : StatefulOpConversionPattern<quartz::RXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::RXOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& quartzQubit = op.getQubit();

    // Get the latest Flux qubit value from the state map
    const auto fluxQubit = getState().qubitMap[quartzQubit];

    const auto& theta = op.getThetaAttr();
    const auto& thetaDyn = op.getThetaDyn();

    // Create flux.rx operation (consumes input, produces output)
    auto fluxOp =
        rewriter.create<flux::RXOp>(op.getLoc(), fluxQubit, theta, thetaDyn);

    // Update state map: the Quartz qubit now corresponds to the output qubit
    getState().qubitMap[quartzQubit] = fluxOp.getQubitOut();

    // Replace the Quartz operation with the Flux operation
    rewriter.replaceOp(op, fluxOp.getResult());

    return success();
  }
};

/**
 * @brief Converts quartz.u2 to flux.u2
 *
 * @details
 * Example transformation:
 * ```mlir
 * quartz.u2(%phi, %lambda) %q : !quartz.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = flux.u2(%phi, %lambda) %q_in : !flux.qubit -> !flux.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQuartzU2Op final : StatefulOpConversionPattern<quartz::U2Op> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::U2Op op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& quartzQubit = op.getQubit();

    // Get the latest Flux qubit value from the state map
    const auto fluxQubit = getState().qubitMap[quartzQubit];

    const auto& phi = op.getPhiAttr();
    const auto& phiDyn = op.getPhiDyn();

    const auto& lambda = op.getLambdaAttr();
    const auto& lambdaDyn = op.getLambdaDyn();

    // Create flux.u2 operation (consumes input, produces output)
    auto fluxOp = rewriter.create<flux::U2Op>(op.getLoc(), fluxQubit, phi,
                                              phiDyn, lambda, lambdaDyn);

    // Update state map: the Quartz qubit now corresponds to the output qubit
    getState().qubitMap[quartzQubit] = fluxOp.getQubitOut();

    // Replace the Quartz operation with the Flux operation
    rewriter.replaceOp(op, fluxOp.getResult());

    return success();
  }
};

/**
 * @brief Converts quartz.swap to flux.swap
 *
 * @details
 * Example transformation:
 * ```mlir
 * quartz.swap %q0, %q1 : !quartz.qubit, !quartz.qubit
 * // becomes (where {%q0, %q1} map to {%q0_in, %q1_in}):
 * %q0_out, %q1_out = flux.swap %q0_in, %q1_in : !flux.qubit, !flux.qubit ->
 * !flux.qubit, !flux.qubit
 * // state updated: {%q0, %q1} now map to {%q0_out, %q1_out}
 * ```
 */
struct ConvertQuartzSWAPOp final : StatefulOpConversionPattern<quartz::SWAPOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::SWAPOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& quartzQubit0 = op.getQubit0();
    const auto& quartzQubit1 = op.getQubit1();

    // Get the latest Flux qubit values from the state map
    const auto& fluxQubit0 = getState().qubitMap[quartzQubit0];
    const auto& fluxQubit1 = getState().qubitMap[quartzQubit1];

    // Create flux.swap operation (consumes input, produces output)
    auto fluxOp =
        rewriter.create<flux::SWAPOp>(op.getLoc(), fluxQubit0, fluxQubit1);

    // Update state map: the Quartz qubit now corresponds to the output qubit
    getState().qubitMap[quartzQubit0] = fluxOp.getQubit0Out();
    getState().qubitMap[quartzQubit1] = fluxOp.getQubit1Out();

    // Replace the Quartz operation with the Flux operation
    rewriter.replaceOp(op, fluxOp.getOperands());

    return success();
  }
};

struct ConvertQuartzCtrlOp final : StatefulOpConversionPattern<quartz::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Get Flux controls from state map
    const auto& quartzControls = op.getControls();
    SmallVector<Value> fluxControls;
    fluxControls.reserve(quartzControls.size());
    for (const auto& quartzControl : quartzControls) {
      fluxControls.push_back(getState().qubitMap[quartzControl]);
    }

    // Get Flux targets from state map
    SmallVector<Value> fluxTargets;
    fluxTargets.reserve(op.getNumTargets());
    for (auto i = 0; i < op.getNumTargets(); ++i) {
      const auto& quartzTarget = op.getTarget(i);
      fluxTargets.push_back(getState().qubitMap[quartzTarget]);
    }

    // Create flux.ctrl operation
    auto fluxOp =
        rewriter.create<flux::CtrlOp>(op.getLoc(), fluxControls, fluxTargets);

    // Clone the body region from Quartz to Flux
    IRMapping regionMap;
    for (auto i = 0; i < op.getNumTargets(); ++i) {
      regionMap.map(op.getTarget(i), fluxTargets[i]);
    }
    rewriter.cloneRegionBefore(op.getBody(), fluxOp.getBody(),
                               fluxOp.getBody().end(), regionMap);

    // Update state map
    for (auto i = 0; i < op.getNumPosControls(); ++i) {
      const auto& quartzControl = quartzControls[i];
      getState().qubitMap[quartzControl] = fluxOp.getControlsOut()[i];
    }
    for (auto i = 0; i < op.getNumTargets(); ++i) {
      const auto& quartzTarget = op.getTarget(i);
      getState().qubitMap[quartzTarget] = fluxOp.getTargetsOut()[i];
    }

    // Replace the Quartz operation with the Flux operation
    rewriter.replaceOp(op, fluxOp.getOperands());

    return success();
  }
};

struct ConvertQuartzYieldOp final
    : StatefulOpConversionPattern<quartz::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(quartz::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto ctrlOp = dyn_cast<flux::CtrlOp>(op->getParentOp());
    auto unitaryOp = ctrlOp.getBodyUnitary();

    SmallVector<Value> targets;
    targets.reserve(unitaryOp.getNumTargets());
    for (auto i = 0; i < unitaryOp.getNumTargets(); ++i) {
      auto yield = rewriter.getRemappedValue(unitaryOp.getOutputTarget(i));
      targets.push_back(yield);
    }

    rewriter.replaceOpWithNewOp<flux::YieldOp>(op, targets);

    return success();
  }
};

/**
 * @brief Pass implementation for Quartz-to-Flux conversion
 *
 * @details
 * This pass converts Quartz dialect operations (reference semantics) to
 * Flux dialect operations (value semantics). The conversion is essential
 * for enabling optimization passes that rely on SSA form and explicit
 * dataflow analysis.
 *
 * The pass operates in several phases:
 * 1. Type conversion: !quartz.qubit -> !flux.qubit
 * 2. Operation conversion: Each Quartz op is converted to its Flux equivalent
 * 3. State tracking: A LoweringState maintains qubit value mappings
 * 4. Function/control-flow adaptation: Function signatures and control flow
 *    are updated to use Flux types
 *
 * The conversion maintains semantic equivalence while transforming the
 * representation from imperative (mutation-based) to functional (SSA-based).
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

    // Configure conversion target: Quartz illegal, Flux legal
    target.addIllegalDialect<QuartzDialect>();
    target.addLegalDialect<FluxDialect>();

    // Register operation conversion patterns with state tracking
    patterns.add<ConvertQuartzAllocOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzDeallocOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzStaticOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzMeasureOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzResetOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzXOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzRXOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzU2Op>(typeConverter, context, &state);
    patterns.add<ConvertQuartzSWAPOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzCtrlOp>(typeConverter, context, &state);
    patterns.add<ConvertQuartzYieldOp>(typeConverter, context, &state);

    // Conversion of quartz types in func.func signatures
    // Note: This currently has limitations with signature changes
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Conversion of quartz types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Conversion of quartz types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // Conversion of quartz types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mlir
