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
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mlir {
using namespace qco;
using namespace qc;

#define GEN_PASS_DEF_QCOTOQC
#include "mlir/Conversion/QCOToQC/QCOToQC.h.inc"

/**
 * @brief Converts a zero-target, one-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType>
static LogicalResult
convertZeroTargetOneParameter(QCOOpType& op,
                              ConversionPatternRewriter& rewriter) {
  rewriter.create<QCOpType>(op.getLoc(), op.getParameter(0));
  rewriter.eraseOp(op);
  return success();
}

/**
 * @brief Converts a one-target, zero-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetZeroParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& qcQubit = adaptor.getQubitIn();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit);

  // Replace the output qubit with the same QC reference
  rewriter.replaceOp(op, qcQubit);

  return success();
}

/**
 * @brief Converts a one-target, one-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetOneParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& qcQubit = adaptor.getQubitIn();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit, op.getParameter(0));

  // Replace the output qubit with the same QC reference
  rewriter.replaceOp(op, qcQubit);

  return success();
}

/**
 * @brief Converts a one-target, two-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetTwoParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& qcQubit = adaptor.getQubitIn();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit, op.getParameter(0),
                            op.getParameter(1));

  // Replace the output qubit with the same QC reference
  rewriter.replaceOp(op, qcQubit);

  return success();
}

/**
 * @brief Converts a one-target, three-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetThreeParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                               ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& qcQubit = adaptor.getQubitIn();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit, op.getParameter(0),
                            op.getParameter(1), op.getParameter(2));

  // Replace the output qubit with the same QC reference
  rewriter.replaceOp(op, qcQubit);

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertTwoTargetZeroParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubits
  const auto& qcQubit0 = adaptor.getQubit0In();
  const auto& qcQubit1 = adaptor.getQubit1In();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit0, qcQubit1);

  // Replace the output qubits with the same QC references
  rewriter.replaceOp(op, {qcQubit0, qcQubit1});

  return success();
}

/**
 * @brief Converts a two-target, one-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertTwoTargetOneParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubits
  const auto& qcQubit0 = adaptor.getQubit0In();
  const auto& qcQubit1 = adaptor.getQubit1In();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit0, qcQubit1,
                            op.getParameter(0));

  // Replace the output qubits with the same QC references
  rewriter.replaceOp(op, {qcQubit0, qcQubit1});

  return success();
}

/**
 * @brief Converts a two-target, two-parameter QCO operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor of the QCO operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertTwoTargetTwoParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubits
  const auto& qcQubit0 = adaptor.getQubit0In();
  const auto& qcQubit1 = adaptor.getQubit1In();

  // Create the QC operation (in-place, no result)
  rewriter.create<QCOpType>(op.getLoc(), qcQubit0, qcQubit1, op.getParameter(0),
                            op.getParameter(1));

  // Replace the output qubits with the same QC references
  rewriter.replaceOp(op, {qcQubit0, qcQubit1});

  return success();
}

/**
 * @brief Type converter for QCO-to-QC conversion
 *
 * @details
 * Handles type conversion between the QCO and QC dialects.
 * The primary conversion is from !qco.qubit to !qc.qubit, which
 * represents the semantic shift from value types to reference types.
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
  }
};

/**
 * @brief Converts qco.alloc to qc.alloc
 *
 * @details
 * Allocates a new qubit initialized to the |0⟩ state. Register metadata
 * (name, size, index) is preserved during conversion.
 *
 * The conversion is straightforward: the QCO allocation produces an SSA
 * value, while the QC allocation produces a reference. MLIR's type
 * conversion system automatically handles the semantic shift.
 *
 * Example transformation:
 * ```mlir
 * %q0 = qco.alloc("q", 3, 0) : !qco.qubit
 * // becomes:
 * %q = qc.alloc("q", 3, 0) : !qc.qubit
 * ```
 */
struct ConvertQCOAllocOp final : OpConversionPattern<qco::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Create qc.alloc with preserved register metadata
    rewriter.replaceOpWithNewOp<qc::AllocOp>(op, op.getRegisterNameAttr(),
                                             op.getRegisterSizeAttr(),
                                             op.getRegisterIndexAttr());

    return success();
  }
};

/**
 * @brief Converts qco.dealloc to qc.dealloc
 *
 * @details
 * Deallocates a qubit, releasing its resources. The OpAdaptor automatically
 * provides the type-converted qubit operand (!qc.qubit instead of
 * !qco.qubit), so we simply pass it through to the new operation.
 *
 * Example transformation:
 * ```mlir
 * qco.dealloc %q_qco : !qco.qubit
 * // becomes:
 * qc.dealloc %q_qc : !qc.qubit
 * ```
 */
struct ConvertQCODeallocOp final : OpConversionPattern<qco::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted qubit
    rewriter.replaceOpWithNewOp<qc::DeallocOp>(op, adaptor.getQubit());
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
struct ConvertQCOStaticOp final : OpConversionPattern<qco::StaticOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
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
    const auto& qcQubit = adaptor.getQubitIn();

    // Create qc.measure (in-place operation, returns only bit)
    // Preserve register metadata for output recording
    auto qcOp = rewriter.create<qc::MeasureOp>(
        op.getLoc(), qcQubit, op.getRegisterNameAttr(),
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
    const auto& qcQubit = adaptor.getQubitIn();

    // Create qc.reset (in-place operation, no result)
    rewriter.create<qc::ResetOp>(op.getLoc(), qcQubit);

    // Replace the output qubit with the same qc reference
    rewriter.replaceOp(op, qcQubit);

    return success();
  }
};

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  /**                                                                          \
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qco.OP_NAME(%PARAM)                                                       \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM)                                                        \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor /*adaptor*/,                   \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertZeroTargetOneParameter<qc::OP_CLASS>(op, rewriter);        \
    }                                                                          \
  };

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME %q_in : !qco.qubit -> !qco.qubit                     \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME %q : !qc.qubit                                                 \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<qc::OP_CLASS>(op, adaptor,          \
                                                         rewriter);            \
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
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM) %q_in : !qco.qubit -> !qco.qubit             \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM) %q : !qc.qubit                                         \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<qc::OP_CLASS>(op, adaptor,           \
                                                        rewriter);             \
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
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM1, %PARAM2) %q_in : !qco.qubit ->              \
   * !qco.qubit                                                                \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM1, %PARAM2) %q : !qc.qubit                               \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetTwoParameter<qc::OP_CLASS>(op, adaptor,           \
                                                        rewriter);             \
    }                                                                          \
  };

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  /**                                                                          \
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q_in : !qco.qubit        \
   * -> !qco.qubit                                                             \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q : !qc.qubit                      \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetThreeParameter<qc::OP_CLASS>(op, adaptor,         \
                                                          rewriter);           \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME %q0_in, %q1_in : !qco.qubit, !qco.qubit    \
   * -> !qco.qubit, !qco.qubit                                                 \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME %q0, %q1 : !qc.qubit, !qc.qubit                                \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetZeroParameter<qc::OP_CLASS>(op, adaptor,          \
                                                         rewriter);            \
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
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME(%PARAM) %q0_in, %q1_in : !qco.qubit,       \
   * !qco.qubit -> !qco.qubit, !qco.qubit                                      \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM) %q0, %q1 : !qc.qubit, !qc.qubit                        \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetOneParameter<qc::OP_CLASS>(op, adaptor,           \
                                                        rewriter);             \
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
   * @brief Converts qco.OP_NAME to qc.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME(%PARAM1, %PARAM2) %q0_in, %q1_in :         \
   * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit                          \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM1, %PARAM2) %q0, %q1 : !qc.qubit, !qc.qubit              \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQCO##OP_CLASS final : OpConversionPattern<qco::OP_CLASS> {     \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS op, OpAdaptor adaptor,                       \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetTwoParameter<qc::OP_CLASS>(op, adaptor,           \
                                                        rewriter);             \
    }                                                                          \
  };

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

/**
 * @brief Converts qco.barrier to qc.barrier
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
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
    const auto& qcQubits = adaptor.getQubitsIn();

    // Create qc.barrier operation
    rewriter.create<qc::BarrierOp>(op.getLoc(), qcQubits);

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
 * %controls_out, %targets_out = qco.ctrl({%q0_in}, {%q1_in}) {
 *   %q1_res = qco.x %q1_in : !qco.qubit -> !qco.qubit
 *   qco.yield %q1_res
 * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
 * ```
 * is converted to
 * ```mlir
 * qc.ctrl(%q0) {
 *   qc.x %q1 : !qc.qubit
 * }
 * ```
 */
struct ConvertQCOCtrlOp final : OpConversionPattern<qco::CtrlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Get QC controls
    const auto& qcControls = adaptor.getControlsIn();

    // Create qc.ctrl operation
    auto qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), qcControls);

    // Clone body region from QCO to QC
    auto& dstRegion = qcOp.getRegion();
    rewriter.cloneRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Remove all block arguments in the cloned region
    rewriter.modifyOpInPlace(qcOp, [&] {
      auto& entryBlock = dstRegion.front();
      const auto numArgs = entryBlock.getNumArguments();
      assert(
          adaptor.getTargetsIn().size() == numArgs &&
          +"qco.ctrl: entry block args must match number of target operands");

      // 1. Replace uses (Must be done BEFORE erasing)
      // We iterate 0..N using indices since the block args are still stable
      // here.
      for (auto i = 0UL; i < numArgs; ++i) {
        entryBlock.getArgument(i).replaceAllUsesWith(adaptor.getTargetsIn()[i]);
      }

      // 2. Erase all block arguments
      // Now that they have no uses, we can safely wipe them.
      // We use a bulk erase for efficiency (start index 0, count N).
      if (numArgs > 0) {
        entryBlock.eraseArguments(0, numArgs);
      }
    });

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
    rewriter.replaceOpWithNewOp<qc::YieldOp>(op);
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

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCOToQCTypeConverter typeConverter(context);

    // Configure conversion target: QCO illegal, QC legal
    target.addIllegalDialect<QCODialect>();
    target.addLegalDialect<QCDialect>();

    // Register operation conversion patterns
    // Note: No state tracking needed - OpAdaptors handle type conversion
    patterns.add<ConvertQCOAllocOp, ConvertQCODeallocOp, ConvertQCOStaticOp,
                 ConvertQCOMeasureOp, ConvertQCOResetOp, ConvertQCOGPhaseOp,
                 ConvertQCOIdOp, ConvertQCOXOp, ConvertQCOYOp, ConvertQCOZOp,
                 ConvertQCOHOp, ConvertQCOSOp, ConvertQCOSdgOp, ConvertQCOTOp,
                 ConvertQCOTdgOp, ConvertQCOSXOp, ConvertQCOSXdgOp,
                 ConvertQCORXOp, ConvertQCORYOp, ConvertQCORZOp, ConvertQCOPOp,
                 ConvertQCOROp, ConvertQCOU2Op, ConvertQCOUOp, ConvertQCOSWAPOp,
                 ConvertQCOiSWAPOp, ConvertQCODCXOp, ConvertQCOECROp,
                 ConvertQCORXXOp, ConvertQCORYYOp, ConvertQCORZXOp,
                 ConvertQCORZZOp, ConvertQCOXXPlusYYOp, ConvertQCOXXMinusYYOp,
                 ConvertQCOBarrierOp, ConvertQCOCtrlOp, ConvertQCOYieldOp>(
        typeConverter, context);

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

} // namespace mlir
