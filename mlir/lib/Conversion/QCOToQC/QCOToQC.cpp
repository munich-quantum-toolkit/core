/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToQC/QCOToQC.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstdint>
#include <utility>

namespace mlir {

using namespace qco;
using namespace qc;

namespace {
/** @brief Qubit addressing mode */
enum class QubitAddressingMode : std::uint8_t {
  Unknown, //!< The addressing mode is not known.
  Static,  //!< The module uses static qubit allocation.
  Dynamic  //!< The module uses dynamic qubit allocation.
};

/**
 * @brief State object for tracking qubit addressing mode.
 *
 * @details
 * Used to track whether a function uses static or dynamic qubit allocation.
 * This is used to determine whether to convert `qco.sink` to `qc.dealloc` (for
 * dynamic qubits) or simply erase it (for static qubits). This is also used to
 * catch cases of mixed addressing being used, which is not supported.
 */
struct LoweringState {
  /// The qubit addressing mode used in the module
  QubitAddressingMode mode = QubitAddressingMode::Unknown;
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which is used to track the addressing mode of the module.
 * @tparam OpType The QCO operation type to be converted.
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

static void inlineRegion(Region& source, Region& target, unsigned int start,
                         unsigned int numArgs, ValueRange replacementArgs,
                         ConversionPatternRewriter& rewriter) {
  rewriter.inlineRegionBefore(source, target, target.end());
  auto& block = target.front();

  for (auto [arg, operand] : llvm::zip_equal(
           block.getArguments().drop_front(start), replacementArgs)) {
    arg.replaceAllUsesWith(operand);
  }
  block.eraseArguments(start, numArgs);
}

#define GEN_PASS_DEF_QCOTOQC
#include "mlir/Conversion/QCOToQC/QCOToQC.h.inc"

namespace {

/**
 * @brief Type converter for QCO-to-QC conversion
 *
 * @details
 * Handles type conversion between the QCO and QC dialects.
 * The primary conversion is from !qco.qubit to !qc.qubit, which
 * represents the semantic shift from value types to reference types.
 *
 * Qubit tensor types preserve their shape during conversion: a statically
 * shaped `tensor<Nx!qco.qubit>` becomes `memref<Nx!qc.qubit>`, while a
 * dynamically shaped `tensor<?x!qco.qubit>` becomes `memref<?x!qc.qubit>`.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
class QCOToQCTypeConverter final : public TypeConverter {
public:
  explicit QCOToQCTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert QCO qubit values to QC qubit references
    addConversion([ctx](qco::QubitType /*type*/) -> Type {
      return qc::QubitType::get(ctx);
    });

    addConversion([ctx](RankedTensorType type) -> Type {
      if (llvm::isa<qco::QubitType>(type.getElementType())) {
        return MemRefType::get(type.getShape(), qc::QubitType::get(ctx));
      }
      return type;
    });
  }
};

/**
 * @brief Converts qtensor.alloc to memref.alloc
 *
 * @par Example:
 * ```mlir
 * %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * %memref = memref.alloc(%c3) : memref<3x!qc.qubit>
 * ```
 */
struct ConvertQTensorAllocOp final : OpConversionPattern<qtensor::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto qubitType = qc::QubitType::get(op.getContext());
    auto tensorType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto memrefType = MemRefType::get(tensorType.getShape(), qubitType);

    if (tensorType.hasStaticShape()) {
      // Static size: no dynamic size operand needed
      rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);
    } else {
      // Dynamic size: forward the runtime size operand
      rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType,
                                                   op.getSize());
    }
    return success();
  }
};

/**
 * @brief Converts qtensor.extract to memref.load
 *
 * @par Example:
 * ```mlir
 * %tensor_out, %q = qtensor.extract %tensor_in[%c0]: tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * %q = memref.load %memref[%c0] : memref<3x!qc.qubit>
 * ```
 */
struct ConvertQTensorExtractOp final : OpConversionPattern<qtensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                       adaptor.getTensor(), adaptor.getIndex());
    rewriter.replaceOp(op, {adaptor.getTensor(), load.getResult()});
    return success();
  }
};

/**
 * @brief Removes qtensor.insert operations
 */
struct ConvertQTensorInsertOp final : OpConversionPattern<qtensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getDest());
    return success();
  }
};

/**
 * @brief Converts qtensor.dealloc to memref.dealloc
 *
 * @par Example:
 * ```mlir
 * qtensor.dealloc %tensor : tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * memref.dealloc %memref : memref<3x!qc.qubit>
 * ```
 */
struct ConvertQTensorDeallocOp final : OpConversionPattern<qtensor::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, adaptor.getTensor());
    return success();
  }
};

/**
 * @brief Converts qco.alloc to qc.alloc
 *
 * @par Example:
 * ```mlir
 * %q = qco.alloc : !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 */
struct ConvertQCOAllocOp final : StatefulOpConversionPattern<qco::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMode = getState().mode;
    if (qubitMode == QubitAddressingMode::Unknown) {
      qubitMode = QubitAddressingMode::Dynamic;
    }
    assert(qubitMode != QubitAddressingMode::Static &&
           "Static qubits cannot be mixed with dynamic qubits");

    // Create qc.alloc
    rewriter.replaceOpWithNewOp<qc::AllocOp>(op);

    return success();
  }
};

/**
 * @brief Converts qco.sink to qc.dealloc.
 *
 * @details
 * In QCO, qubits have value/linear semantics and must be consumed explicitly
 * (via `qco.sink`). In QC, qubits have reference semantics; for dynamic qubits
 * we materialize this end-of-lifetime as `qc.dealloc`. Static qubits do not
 * need explicit deallocation, so we simply erase the `qco.sink` operation.
 *
 * The OpAdaptor automatically provides the type-converted qubit operand
 * (`!qc.qubit` instead of `!qco.qubit`), so we simply pass it through to the
 * new operation when needed.
 *
 * Example transformation:
 * ```mlir
 * qco.sink %q_qco : !qco.qubit
 * // becomes:
 * qc.dealloc %q_qc : !qc.qubit
 * ```
 */
struct ConvertQCOSinkOp final : StatefulOpConversionPattern<SinkOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto mode = getState().mode;
    assert(mode != QubitAddressingMode::Unknown &&
           "Sinks cannot exist without allocations");

    if (mode == QubitAddressingMode::Static) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<DeallocOp>(op, adaptor.getQubit());
    return success();
  }
};

/**
 * @brief Converts qco.static to qc.static
 *
 * @details
 * Static qubits represent references to hardware-mapped or fixed-position
 * qubits identified by an index. The conversion preserves the index attribute
 * and creates the corresponding qc.static operation.
 *
 * Example transformation:
 * ```mlir
 * %q0 = qco.static 0 : !qco.qubit
 * // becomes:
 * %q = qc.static 0 : !qc.qubit
 * ```
 */
struct ConvertQCOStaticOp final : StatefulOpConversionPattern<qco::StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMode = getState().mode;
    if (qubitMode == QubitAddressingMode::Unknown) {
      qubitMode = QubitAddressingMode::Static;
    }
    assert(qubitMode != QubitAddressingMode::Dynamic &&
           "Dynamic qubits cannot be mixed with static qubits");

    // Create qc.static with the same index
    rewriter.replaceOpWithNewOp<qc::StaticOp>(op, op.getIndex());
    return success();
  }
};

/**
 * @brief Converts qco.measure to qc.measure
 *
 * @details
 * Measurement demonstrates the key semantic difference between the dialects:
 * - QCO (value semantics): Consumes input qubit, returns both output qubit
 *   and classical bit result
 * - QC (reference semantics): Measures qubit in-place, returns only the
 *   classical bit result
 *
 * The OpAdaptor provides the input qubit already converted to !qc.qubit.
 * Since QC operations are in-place, we return the same qubit reference
 * alongside the measurement bit. MLIR's conversion infrastructure automatically
 * routes subsequent uses of the QCO output qubit to this QC reference.
 *
 * Register metadata (name, size, index) for output recording is preserved
 * during conversion.
 *
 * Example transformation:
 * ```mlir
 * %q_out, %c = qco.measure("c", 2, 0) %q_in : !qco.qubit
 * // becomes:
 * %c = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
 * // %q_out uses are replaced with %q (the adaptor-converted input)
 * ```
 */
struct ConvertQCOMeasureOp final : OpConversionPattern<qco::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create qc.measure (in-place operation, returns only bit)
    // Preserve register metadata for output recording
    auto qcOp = qc::MeasureOp::create(
        rewriter, op.getLoc(), qcQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    auto measureBit = qcOp.getResult();

    // Replace both results: qubit output → same qc reference, bit → new bit
    rewriter.replaceOp(op, {qcQubit, measureBit});

    return success();
  }
};

/**
 * @brief Converts qco.reset to qc.reset
 *
 * @details
 * Reset operations force a qubit to the |0⟩ state:
 * - QCO (value semantics): Consumes input qubit, returns reset output qubit
 * - QC (reference semantics): Resets qubit in-place, no result value
 *
 * The OpAdaptor provides the input qubit already converted to !qc.qubit.
 * Since QC's reset is in-place, we return the same qubit reference.
 * MLIR's conversion infrastructure automatically routes subsequent uses of
 * the QCO output qubit to this QC reference.
 *
 * Example transformation:
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
 * // becomes:
 * qc.reset %q : !qc.qubit
 * // %q_out uses are replaced with %q (the adaptor-converted input)
 * ```
 */
struct ConvertQCOResetOp final : OpConversionPattern<qco::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create qc.reset (in-place operation, no result)
    qc::ResetOp::create(rewriter, op.getLoc(), qcQubit);

    // Replace the output qubit with the same qc reference
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

/**
 * @brief Converts a zero-target, one-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * qco.gphase(%theta)
 * ```
 * is converted to
 * ```mlir
 * qc.gphase(%theta)
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOZeroTargetOneParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    QCOpType::create(rewriter, op.getLoc(), op.getParameter(0));
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts a one-target, zero-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.x %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.x %q : !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOOneTargetZeroParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit);

    // Replace the output qubit with the same QC reference
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

/**
 * @brief Converts a one-target, one-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.rx(%theta) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.rx(%theta) %q : !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOOneTargetOneParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit, op.getParameter(0));

    // Replace the output qubit with the same QC reference
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

/**
 * @brief Converts a one-target, two-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.r(%theta, %phi) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.r(%theta, %phi) %q : !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOOneTargetTwoParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit, op.getParameter(0),
                     op.getParameter(1));

    // Replace the output qubit with the same QC reference
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

/**
 * @brief Converts a one-target, three-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.u(%theta, %phi, %lambda) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOOneTargetThreeParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit, op.getParameter(0),
                     op.getParameter(1), op.getParameter(2));

    // Replace the output qubit with the same QC reference
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

/**
 * @brief Converts a two-target, zero-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.swap %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOTwoTargetZeroParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubits
    auto qcQubit0 = adaptor.getQubit0In();
    auto qcQubit1 = adaptor.getQubit1In();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit0, qcQubit1);

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, {qcQubit0, qcQubit1});

    return success();
  }
};

/**
 * @brief Converts a two-target, one-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.rxx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOTwoTargetOneParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubits
    auto qcQubit0 = adaptor.getQubit0In();
    auto qcQubit1 = adaptor.getQubit1In();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit0, qcQubit1,
                     op.getParameter(0));

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, {qcQubit0, qcQubit1});

    return success();
  }
};

/**
 * @brief Converts a two-target, two-parameter QCO gate to QC
 *
 * @tparam QCOOpType The operation type of the QCO gate
 * @tparam QCOpType The operation type of the QC gate
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.xx_minus_yy(%theta, %beta) %q0_in, %q1_in :
 * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.xx_minus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 */
template <typename QCOOpType, typename QCOpType>
struct ConvertQCOTwoTargetTwoParameterToQC final
    : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubits
    auto qcQubit0 = adaptor.getQubit0In();
    auto qcQubit1 = adaptor.getQubit1In();

    // Create the QC operation (in-place, no result)
    QCOpType::create(rewriter, op.getLoc(), qcQubit0, qcQubit1,
                     op.getParameter(0), op.getParameter(1));

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, {qcQubit0, qcQubit1});

    return success();
  }
};

/**
 * @brief Converts qco.barrier to qc.barrier
 *
 * @par Example:
 * ```mlir
 * %q_out:2 = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit,
 * !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.barrier %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 */
struct ConvertQCOBarrierOp final : OpConversionPattern<qco::BarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted qubits
    auto qcQubits = adaptor.getQubitsIn();

    // Create qc.barrier operation
    qc::BarrierOp::create(rewriter, op.getLoc(), qcQubits);

    // Replace the output qubits with the same qc references
    rewriter.replaceOp(op, qcQubits);

    return success();
  }
};

/**
 * @brief Converts qco.ctrl to qc.ctrl
 *
 * @par Example:
 * ```mlir
 * %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%a_in = %q1_in) {
 *   %a_res = qco.x %a_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a_res
 * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
 * ```
 * is converted to
 * ```mlir
 * qc.ctrl(%q0) {
 *   qc.x %q1 : !qc.qubit
 * } : !qc.qubit
 * ```
 */
struct ConvertQCOCtrlOp final : OpConversionPattern<qco::CtrlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Get QC controls
    auto qcControls = adaptor.getControlsIn();

    // Create qc.ctrl operation
    auto qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), qcControls);

    // Inline the region and replace the blockarguments
    inlineRegion(op.getRegion(), qcOp.getRegion(), 0,
                 adaptor.getTargetsIn().size(), adaptor.getTargetsIn(),
                 rewriter);

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, adaptor.getOperands());

    return success();
  }
};

/**
 * @brief Converts qco.inv to qc.inv
 *
 * @par Example:
 * ```mlir
 * %q0_out = qco.inv (%a_in = %q0_in) {
 *   %a_res = qco.s %a_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a_res
 * } : {!qco.qubit} -> {!qco.qubit}
 * ```
 * is converted to
 * ```mlir
 * qc.inv {
 *   qc.s %q0 : !qc.qubit
 * }
 * ```
 */
struct ConvertQCOInvOp final : OpConversionPattern<qco::InvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::InvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create qc.inv operation
    auto qcOp = qc::InvOp::create(rewriter, op.getLoc());

    // Inline the region and replace the blockarguments
    inlineRegion(op.getRegion(), qcOp.getRegion(), 0,
                 adaptor.getOperands().size(), adaptor.getQubitsIn(), rewriter);

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, adaptor.getOperands());

    return success();
  }
};

/**
 * @brief Converts qco.yield to qc.yield
 *
 * @par Example:
 * ```mlir
 * qco.yield %targets
 * ```
 * is converted to
 * ```mlir
 * qc.yield
 * ```
 */
struct ConvertQCOYieldOp final : OpConversionPattern<qco::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (llvm::isa<scf::IfOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    } else {
      rewriter.replaceOpWithNewOp<qc::YieldOp>(op);
    }

    return success();
  }
};

struct ConvertQCOSCFForOp final : OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto newFor = scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), ValueRange{});
    // Erase default block
    rewriter.eraseBlock(&newFor.getRegion().front());

    // Inline the region and replace the blockarguments
    inlineRegion(op.getRegion(), newFor.getRegion(), 1,
                 adaptor.getInitArgs().size(), adaptor.getInitArgs(), rewriter);

    rewriter.replaceOp(op, adaptor.getInitArgs());

    return success();
  }
};

struct ConvertQCOSCFWhileOp final : OpConversionPattern<scf::WhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto newWhileOp =
        scf::WhileOp::create(rewriter, op->getLoc(), TypeRange{}, ValueRange{});

    // Inline the regions and replace the blockarguments
    inlineRegion(op.getBefore(), newWhileOp.getBefore(), 0,
                 adaptor.getInits().size(), adaptor.getInits(), rewriter);
    inlineRegion(op.getAfter(), newWhileOp.getAfter(), 0,
                 adaptor.getInits().size(), adaptor.getInits(), rewriter);

    rewriter.replaceOp(op, adaptor.getInits());

    return success();
  }
};

struct ConvertQCOSCFConditionOp final : OpConversionPattern<scf::ConditionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, op.getCondition(),
                                                  ValueRange{});

    return success();
  }
};

struct ConvertQCOSCFYieldOp final : OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    return success();
  }
};

struct ConvertQCOIfOp final : OpConversionPattern<qco::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create the new if operation
    auto newIf = scf::IfOp::create(rewriter, op.getLoc(), TypeRange{},
                                   op.getCondition(), false);
    auto& newThenRegion = newIf.getThenRegion();
    auto& oldElseRegion = op.getElseRegion();
    // Erase the default empty then block
    rewriter.eraseBlock(&newThenRegion.front());

    // Inline the region and replace the blockarguments
    inlineRegion(op.getThenRegion(), newThenRegion, 0,
                 adaptor.getOperands().size() - 1,
                 adaptor.getOperands().drop_front(1), rewriter);

    // Inline the else block if it has more than just the yield operation
    if (oldElseRegion.front().getOperations().size() > 1) {
      inlineRegion(oldElseRegion, newIf.getElseRegion(), 0,
                   adaptor.getOperands().size() - 1,
                   adaptor.getOperands().drop_front(1), rewriter);
    }

    // Replace the qco results with the input qc values except the condition
    rewriter.replaceOp(op, adaptor.getOperands().drop_front(1));

    return success();
  }
};

/**
 * @brief Pass implementation for QCO-to-QC conversion
 *
 * @details
 * This pass converts QCO dialect operations (value semantics) to
 * QC dialect operations (reference semantics). The conversion is useful
 * for lowering optimized SSA-form code back to a hardware-oriented
 * representation suitable for backend code generation.
 *
 * The conversion leverages MLIR's built-in type conversion infrastructure:
 * The TypeConverter handles !qco.qubit → !qc.qubit transformations,
 * and the OpAdaptor automatically provides type-converted operands to each
 * conversion pattern. This eliminates the need for manual state tracking.
 *
 * Key semantic transformation:
 * - QCO operations form explicit SSA chains where each operation consumes
 *   inputs and produces new outputs
 * - QC operations modify qubits in-place using references
 * - The conversion maps each QCO SSA chain to a single QC reference,
 *   with MLIR's conversion framework automatically handling the plumbing
 *
 * The pass operates through:
 * 1. Type conversion: !qco.qubit → !qc.qubit
 * 2. Operation conversion: Each QCO op converted to its QC equivalent
 * 3. Automatic operand mapping: OpAdaptors provide converted operands
 * 4. Function/control-flow adaptation: Signatures updated to use QC types
 */
struct QCOToQC final : impl::QCOToQCBase<QCOToQC> {
  using QCOToQCBase::QCOToQCBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    // Create state object to track the qubit addressing mode
    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCOToQCTypeConverter typeConverter(context);

    // Configure conversion target
    target.addIllegalDialect<QCODialect, qtensor::QTensorDialect>();
    target.addLegalDialect<QCDialect, memref::MemRefDialect>();

    // Fix later
    target.addDynamicallyLegalDialect<scf::SCFDialect>(
        [](Operation* op) { return op->getResults().size() == 0; });
    target.addDynamicallyLegalOp<scf::YieldOp>(
        [&](Operation* op) { return op->getNumOperands() == 0; });
    target.addDynamicallyLegalOp<scf::ConditionOp>(
        [&](Operation* op) { return op->getNumOperands() == 1; });

    // Register operation conversion patterns that do not need state tracking
    patterns.add<
        ConvertQTensorAllocOp, ConvertQTensorExtractOp, ConvertQTensorInsertOp,
        ConvertQTensorDeallocOp, ConvertQCOMeasureOp, ConvertQCOResetOp,
        ConvertQCOZeroTargetOneParameterToQC<qco::GPhaseOp, qc::GPhaseOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::XOp, qc::XOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::YOp, qc::YOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::ZOp, qc::ZOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::HOp, qc::HOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::SOp, qc::SOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::SdgOp, qc::SdgOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::TOp, qc::TOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::TdgOp, qc::TdgOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::SXOp, qc::SXOp>,
        ConvertQCOOneTargetZeroParameterToQC<qco::SXdgOp, qc::SXdgOp>,
        ConvertQCOOneTargetOneParameterToQC<qco::RXOp, qc::RXOp>,
        ConvertQCOOneTargetOneParameterToQC<qco::RYOp, qc::RYOp>,
        ConvertQCOOneTargetOneParameterToQC<qco::RZOp, qc::RZOp>,
        ConvertQCOOneTargetOneParameterToQC<qco::POp, qc::POp>,
        ConvertQCOOneTargetTwoParameterToQC<qco::ROp, qc::ROp>,
        ConvertQCOOneTargetTwoParameterToQC<qco::U2Op, qc::U2Op>,
        ConvertQCOOneTargetThreeParameterToQC<qco::UOp, qc::UOp>,
        ConvertQCOTwoTargetZeroParameterToQC<qco::SWAPOp, qc::SWAPOp>,
        ConvertQCOTwoTargetZeroParameterToQC<qco::iSWAPOp, qc::iSWAPOp>,
        ConvertQCOTwoTargetZeroParameterToQC<qco::DCXOp, qc::DCXOp>,
        ConvertQCOTwoTargetZeroParameterToQC<qco::ECROp, qc::ECROp>,
        ConvertQCOTwoTargetOneParameterToQC<qco::RXXOp, qc::RXXOp>,
        ConvertQCOTwoTargetOneParameterToQC<qco::RYYOp, qc::RYYOp>,
        ConvertQCOTwoTargetOneParameterToQC<qco::RZXOp, qc::RZXOp>,
        ConvertQCOTwoTargetOneParameterToQC<qco::RZZOp, qc::RZZOp>,
        ConvertQCOTwoTargetTwoParameterToQC<qco::XXPlusYYOp, qc::XXPlusYYOp>,
        ConvertQCOTwoTargetTwoParameterToQC<qco::XXMinusYYOp, qc::XXMinusYYOp>,
        ConvertQCOBarrierOp, ConvertQCOCtrlOp, ConvertQCOInvOp,
        ConvertQCOYieldOp, ConvertQCOIfOp, ConvertQCOSCFWhileOp,
        ConvertQCOSCFConditionOp, ConvertQCOSCFYieldOp, ConvertQCOSCFForOp>(
        typeConverter, context);

    // Register operation conversion patterns that need state tracking
    patterns.add<ConvertQCOAllocOp, ConvertQCOStaticOp, ConvertQCOSinkOp>(
        typeConverter, context, &state);

    // Conversion of qco types in func.func signatures
    // Note: This currently has limitations with signature changes
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Conversion of qco types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Conversion of qco types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // Conversion of qco types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
