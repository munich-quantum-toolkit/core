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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
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
  /// Map from original QC qubit references to their latest QCO SSA values
  llvm::DenseMap<Value, Value> qubitMap;

  /// Modifier information
  int64_t inNestedRegion = 0;
  DenseMap<int64_t, SmallVector<Value>> targetsIn;
  DenseMap<int64_t, SmallVector<Value>> targetsOut;
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
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getResult();

    // Create the qco.alloc operation with preserved register metadata
    auto qcoOp = rewriter.replaceOpWithNewOp<qco::AllocOp>(
        op, op.getRegisterNameAttr(), op.getRegisterSizeAttr(),
        op.getRegisterIndexAttr());

    auto qcoQubit = qcoOp.getResult();

    // Establish initial mapping: this QC qubit reference now corresponds
    // to this QCO SSA value
    qubitMap.try_emplace(qcQubit, qcoQubit);

    return success();
  }
};

/**
 * @brief Converts qc.dealloc to qco.dealloc
 *
 * @details
 * Deallocates a qubit by looking up its latest QCO value and creating
 * a corresponding qco.dealloc operation. The mapping is removed from
 * the state as the qubit is no longer in use.
 *
 * Example transformation:
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * // becomes (where %q maps to %q_final):
 * qco.dealloc %q_final : !qco.qubit
 * ```
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<qc::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getQubit();

    // Look up the latest QCO value for this QC qubit
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    auto qcoQubit = qubitMap[qcQubit];

    // Create the dealloc operation
    rewriter.replaceOpWithNewOp<qco::DeallocOp>(op, qcoQubit);

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
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getQubit();

    // Create new qco.static operation with the same index
    auto qcoOp = qco::StaticOp::create(rewriter, op.getLoc(), op.getIndex());

    // Collect QCO qubit SSA value
    auto qcoQubit = qcoOp.getQubit();

    // Establish mapping from QC reference to QCO value
    qubitMap[qcQubit] = qcoQubit;

    // Replace the old operation result with the new result
    rewriter.replaceOp(op, qcoQubit);

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
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getQubit();

    // Get the latest QCO qubit value from the state map
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    auto qcoQubit = qubitMap[qcQubit];

    // Create qco.measure (returns both output qubit and bit result)
    auto qcoOp = qco::MeasureOp::create(
        rewriter, op.getLoc(), qcoQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    auto outQcoQubit = qcoOp.getQubitOut();
    auto newBit = qcoOp.getResult();

    // Update mapping: the QC qubit now corresponds to the output qubit
    qubitMap[qcQubit] = outQcoQubit;

    // Replace the QC operation's bit result with the QCO bit result
    rewriter.replaceOp(op, newBit);

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
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getQubit();

    // Get the latest QCO qubit value from the state map
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    auto qcoQubit = qubitMap[qcQubit];

    // Create qco.reset (consumes input, produces output)
    auto qcoOp = qco::ResetOp::create(rewriter, op.getLoc(), qcoQubit);

    // Update mapping: the QC qubit now corresponds to the reset output
    qubitMap[qcQubit] = qcoOp.getQubitOut();

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    const auto inNestedRegion = state.inNestedRegion;

    QCOOpType::create(rewriter, op.getLoc(), op.getParameter(0));

    // Update the state
    if (inNestedRegion != 0) {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut;
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubit
    auto qcQubit = op.getQubitIn();
    Value qcoQubit;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubit = qubitMap[qcQubit];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 1 &&
             "Invalid number of input targets");
      qcoQubit = state.targetsIn[inNestedRegion].front();
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit);

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit] = qcoOp.getQubitOut();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubit
    auto qcQubit = op.getQubitIn();
    Value qcoQubit;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubit = qubitMap[qcQubit];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 1 &&
             "Invalid number of input targets");
      qcoQubit = state.targetsIn[inNestedRegion].front();
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp =
        QCOOpType::create(rewriter, op.getLoc(), qcoQubit, op.getParameter(0));

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit] = qcoOp.getQubitOut();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubit
    auto qcQubit = op.getQubitIn();
    Value qcoQubit;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubit = qubitMap[qcQubit];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 1 &&
             "Invalid number of input targets");
      qcoQubit = state.targetsIn[inNestedRegion].front();
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit,
                                   op.getParameter(0), op.getParameter(1));

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit] = qcoOp.getQubitOut();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubit
    auto qcQubit = op.getQubitIn();
    Value qcoQubit;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubit = qubitMap[qcQubit];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 1 &&
             "Invalid number of input targets");
      qcoQubit = state.targetsIn[inNestedRegion].front();
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp =
        QCOOpType::create(rewriter, op.getLoc(), qcoQubit, op.getParameter(0),
                          op.getParameter(1), op.getParameter(2));

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit] = qcoOp.getQubitOut();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubits
    auto qcQubit0 = op.getQubit0In();
    auto qcQubit1 = op.getQubit1In();
    Value qcoQubit0;
    Value qcoQubit1;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit0) && "QC qubit not found");
      assert(qubitMap.contains(qcQubit1) && "QC qubit not found");
      qcoQubit0 = qubitMap[qcQubit0];
      qcoQubit1 = qubitMap[qcQubit1];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 2 &&
             "Invalid number of input targets");
      const auto& targetsIn = state.targetsIn[inNestedRegion];
      qcoQubit0 = targetsIn[0];
      qcoQubit1 = targetsIn[1];
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit0, qcoQubit1);

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit0] = qcoOp.getQubit0Out();
      qubitMap[qcQubit1] = qcoOp.getQubit1Out();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut(
          {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
 * %q0_out, %q1_out = qco.rxx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 */
template <typename QCOpType, typename QCOOpType>
struct ConvertQCTwoTargetOneParameterToQCO final
    : StatefulOpConversionPattern<QCOpType> {
  using StatefulOpConversionPattern<QCOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubits
    auto qcQubit0 = op.getQubit0In();
    auto qcQubit1 = op.getQubit1In();
    Value qcoQubit0;
    Value qcoQubit1;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit0) && "QC qubit not found");
      assert(qubitMap.contains(qcQubit1) && "QC qubit not found");
      qcoQubit0 = qubitMap[qcQubit0];
      qcoQubit1 = qubitMap[qcQubit1];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 2 &&
             "Invalid number of input targets");
      const auto& targetsIn = state.targetsIn[inNestedRegion];
      qcoQubit0 = targetsIn[0];
      qcoQubit1 = targetsIn[1];
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit0, qcoQubit1,
                                   op.getParameter(0));

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit0] = qcoOp.getQubit0Out();
      qubitMap[qcQubit1] = qcoOp.getQubit1Out();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut(
          {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
    }

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
  matchAndRewrite(QCOpType op, typename QCOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    auto& qubitMap = state.qubitMap;
    const auto inNestedRegion = state.inNestedRegion;

    // Get the latest QCO qubits
    auto qcQubit0 = op.getQubit0In();
    auto qcQubit1 = op.getQubit1In();
    Value qcoQubit0;
    Value qcoQubit1;
    if (inNestedRegion == 0) {
      assert(qubitMap.contains(qcQubit0) && "QC qubit not found");
      assert(qubitMap.contains(qcQubit1) && "QC qubit not found");
      qcoQubit0 = qubitMap[qcQubit0];
      qcoQubit1 = qubitMap[qcQubit1];
    } else {
      assert(state.targetsIn[inNestedRegion].size() == 2 &&
             "Invalid number of input targets");
      const auto& targetsIn = state.targetsIn[inNestedRegion];
      qcoQubit0 = targetsIn[0];
      qcoQubit1 = targetsIn[1];
    }

    // Create the QCO operation (consumes input, produces output)
    auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), qcoQubit0, qcoQubit1,
                                   op.getParameter(0), op.getParameter(1));

    // Update the state map
    if (inNestedRegion == 0) {
      qubitMap[qcQubit0] = qcoOp.getQubit0Out();
      qubitMap[qcQubit1] = qcoOp.getQubit1Out();
    } else {
      state.targetsIn.erase(inNestedRegion);
      const SmallVector<Value> targetsOut(
          {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()});
      state.targetsOut.try_emplace(inNestedRegion, targetsOut);
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
    auto& qubitMap = state.qubitMap;

    // Get QCO qubits from state map
    auto qcQubits = op.getQubits();
    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(qcQubits.size());
    for (const auto& qcQubit : qcQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap[qcQubit]);
    }

    // Create qco.barrier
    auto qcoOp = qco::BarrierOp::create(rewriter, op.getLoc(), qcoQubits);

    // Update the state map
    for (const auto& [qcQubit, qcoQubitOut] :
         llvm::zip(qcQubits, qcoOp.getQubitsOut())) {
      qubitMap[qcQubit] = qcoQubitOut;
    }

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
    auto& qubitMap = state.qubitMap;

    // Get QCO controls from state map
    auto qcControls = op.getControls();
    SmallVector<Value> qcoControls;
    qcoControls.reserve(qcControls.size());
    for (const auto& qcControl : qcControls) {
      assert(qubitMap.contains(qcControl) && "QC qubit not found");
      qcoControls.push_back(qubitMap[qcControl]);
    }

    // Get QCO targets from state map
    const auto numTargets = op.getNumTargets();
    SmallVector<Value> qcoTargets;
    qcoTargets.reserve(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      auto qcTarget = op.getTarget(i);
      assert(qubitMap.contains(qcTarget) && "QC qubit not found");
      auto qcoTarget = qubitMap[qcTarget];
      qcoTargets.push_back(qcoTarget);
    }

    // Create qco.ctrl
    auto qcoOp =
        qco::CtrlOp::create(rewriter, op.getLoc(), qcoControls, qcoTargets);

    // Update the state map if this is a top-level CtrlOp
    // Nested CtrlOps are managed via the targetsIn and targetsOut maps
    if (state.inNestedRegion == 0) {
      for (const auto& [qcControl, qcoControl] :
           llvm::zip(qcControls, qcoOp.getControlsOut())) {
        qubitMap[qcControl] = qcoControl;
      }
      auto qcoTargetsOut = qcoOp.getTargetsOut();
      for (size_t i = 0; i < numTargets; ++i) {
        auto qcTarget = op.getTarget(i);
        qubitMap[qcTarget] = qcoTargetsOut[i];
      }
    }

    // Update modifier information
    state.inNestedRegion++;

    // Clone body region from QC to QCO
    auto& dstRegion = qcoOp.getRegion();
    rewriter.cloneRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Create block arguments for QCO targets
    auto& entryBlock = dstRegion.front();
    assert(entryBlock.getNumArguments() == 0 &&
           "QC ctrl region unexpectedly has entry block arguments");
    SmallVector<Value> qcoTargetAliases;
    qcoTargetAliases.reserve(numTargets);
    const auto qubitType = qco::QubitType::get(qcoOp.getContext());
    const auto opLoc = op.getLoc();
    rewriter.modifyOpInPlace(qcoOp, [&] {
      for (auto i = 0UL; i < numTargets; i++) {
        qcoTargetAliases.emplace_back(entryBlock.addArgument(qubitType, opLoc));
      }
    });
    state.targetsIn[state.inNestedRegion] = std::move(qcoTargetAliases);

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
    auto& [qubitMap, inNestedRegion, targetsIn, targetsOut] = getState();

    // Get QCO targets from state map
    const auto numTargets = op.getNumTargets();
    SmallVector<Value> qcoTargets;
    if (inNestedRegion == 0) {
      qcoTargets.reserve(numTargets);
      for (size_t i = 0; i < numTargets; ++i) {
        auto qcTarget = op.getTarget(i);
        assert(qubitMap.contains(qcTarget) && "QC qubit not found");
        qcoTargets.emplace_back(qubitMap[qcTarget]);
      }
    } else {
      assert(targetsIn[inNestedRegion].size() == numTargets &&
             "Invalid number of input targets");
      qcoTargets = targetsIn[inNestedRegion];
    }

    // Create qco.inv
    auto qcoOp = qco::InvOp::create(rewriter, op.getLoc(), qcoTargets);

    // Update state map
    if (inNestedRegion == 0) {
      const auto qubitsOut = qcoOp.getQubitsOut();
      for (size_t i = 0; i < numTargets; ++i) {
        auto qcTarget = op.getTarget(i);
        qubitMap[qcTarget] = qubitsOut[i];
      }
    } else {
      targetsIn.erase(inNestedRegion);
      targetsOut.try_emplace(inNestedRegion, qcoOp.getQubitsOut());
    }

    // Update modifier information
    inNestedRegion++;

    // Clone body region from QC to QCO
    auto& dstRegion = qcoOp.getRegion();
    rewriter.cloneRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Create block arguments for target qubits and store them in
    // `state.targetsIn`.
    auto& entryBlock = dstRegion.front();
    assert(entryBlock.getNumArguments() == 0 &&
           "QC inv region unexpectedly has entry block arguments");
    SmallVector<Value> qcoTargetAliases;
    qcoTargetAliases.reserve(numTargets);
    const auto qubitType = qco::QubitType::get(qcoOp.getContext());
    const auto opLoc = op.getLoc();
    rewriter.modifyOpInPlace(qcoOp, [&] {
      for (auto i = 0UL; i < numTargets; i++) {
        qcoTargetAliases.emplace_back(entryBlock.addArgument(qubitType, opLoc));
      }
    });
    targetsIn[inNestedRegion] = std::move(qcoTargetAliases);

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
    const auto& targets = state.targetsOut[state.inNestedRegion];
    rewriter.replaceOpWithNewOp<qco::YieldOp>(op, targets);
    state.targetsOut.erase(state.inNestedRegion);
    state.inNestedRegion--;
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

    // Register operation conversion patterns with state tracking
    patterns.add<
        ConvertQCAllocOp, ConvertQCDeallocOp, ConvertQCStaticOp,
        ConvertQCMeasureOp, ConvertQCResetOp,
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
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

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

} // namespace mlir
