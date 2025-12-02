/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/FluxToQuartz/FluxToQuartz.h"

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

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

namespace {}

namespace mlir {
using namespace flux;
using namespace quartz;

#define GEN_PASS_DEF_FLUXTOQUARTZ
#include "mlir/Conversion/FluxToQuartz/FluxToQuartz.h.inc"

namespace {

/**
 * @brief Converts a zero-target, one-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType>
LogicalResult
convertZeroTargetOneParameter(FluxOpType& op,
                              ConversionPatternRewriter& rewriter) {
  rewriter.create<QuartzOpType>(op.getLoc(), op.getOperand());
  rewriter.eraseOp(op);
  return success();
}

/**
 * @brief Converts a one-target, zero-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertOneTargetZeroParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& quartzQubit = adaptor.getQubitIn();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit);

  // Replace the output qubit with the same Quartz reference
  rewriter.replaceOp(op, quartzQubit);

  return success();
}

/**
 * @brief Converts a one-target, one-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertOneTargetOneParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& quartzQubit = adaptor.getQubitIn();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit, op.getOperand(1));

  // Replace the output qubit with the same Quartz reference
  rewriter.replaceOp(op, quartzQubit);

  return success();
}

/**
 * @brief Converts a one-target, two-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertOneTargetTwoParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& quartzQubit = adaptor.getQubitIn();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit, op.getOperand(1),
                                op.getOperand(2));

  // Replace the output qubit with the same Quartz reference
  rewriter.replaceOp(op, quartzQubit);

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertOneTargetThreeParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                               ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubit
  const auto& quartzQubit = adaptor.getQubitIn();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit, op.getOperand(1),
                                op.getOperand(2), op.getOperand(3));

  // Replace the output qubit with the same Quartz reference
  rewriter.replaceOp(op, quartzQubit);

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertTwoTargetZeroParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubits
  const auto& quartzQubit0 = adaptor.getQubit0In();
  const auto& quartzQubit1 = adaptor.getQubit1In();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit0, quartzQubit1);

  // Replace the output qubits with the same Quartz references
  rewriter.replaceOp(op, {quartzQubit0, quartzQubit1});

  return success();
}

/**
 * @brief Converts a two-target, one-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertTwoTargetOneParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubits
  const auto& quartzQubit0 = adaptor.getQubit0In();
  const auto& quartzQubit1 = adaptor.getQubit1In();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit0, quartzQubit1,
                                op.getOperand(2));

  // Replace the output qubits with the same Quartz references
  rewriter.replaceOp(op, {quartzQubit0, quartzQubit1});

  return success();
}

/**
 * @brief Converts a two-target, two-parameter Flux operation to Quartz
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam FluxOpType The operation type of the Flux operation
 * @param FluxOpAdaptorType The OpAdaptor type of the Flux operation
 * @param op The Flux operation instance to convert
 * @param adaptor The OpAdaptor of the Flux operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename FluxOpType,
          typename FluxOpAdaptorType>
LogicalResult
convertTwoTargetTwoParameter(FluxOpType& op, FluxOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  // OpAdaptor provides the already type-converted input qubits
  const auto& quartzQubit0 = adaptor.getQubit0In();
  const auto& quartzQubit1 = adaptor.getQubit1In();

  // Create the Quartz operation (in-place, no result)
  rewriter.create<QuartzOpType>(op.getLoc(), quartzQubit0, quartzQubit1,
                                op.getOperand(2), op.getOperand(3));

  // Replace the output qubits with the same Quartz references
  rewriter.replaceOp(op, {quartzQubit0, quartzQubit1});

  return success();
}

} // namespace

/**
 * @brief Type converter for Flux-to-Quartz conversion
 *
 * @details
 * Handles type conversion between the Flux and Quartz dialects.
 * The primary conversion is from !flux.qubit to !quartz.qubit, which
 * represents the semantic shift from value types to reference types.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
class FluxToQuartzTypeConverter final : public TypeConverter {
public:
  explicit FluxToQuartzTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert Flux qubit values to Quartz qubit references
    addConversion([ctx](flux::QubitType /*type*/) -> Type {
      return quartz::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Converts flux.alloc to quartz.alloc
 *
 * @details
 * Allocates a new qubit initialized to the |0⟩ state. Register metadata
 * (name, size, index) is preserved during conversion.
 *
 * The conversion is straightforward: the Flux allocation produces an SSA
 * value, while the Quartz allocation produces a reference. MLIR's type
 * conversion system automatically handles the semantic shift.
 *
 * Example transformation:
 * ```mlir
 * %q0 = flux.alloc("q", 3, 0) : !flux.qubit
 * // becomes:
 * %q = quartz.alloc("q", 3, 0) : !quartz.qubit
 * ```
 */
struct ConvertFluxAllocOp final : OpConversionPattern<flux::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Create quartz.alloc with preserved register metadata
    rewriter.replaceOpWithNewOp<quartz::AllocOp>(op, op.getRegisterNameAttr(),
                                                 op.getRegisterSizeAttr(),
                                                 op.getRegisterIndexAttr());

    return success();
  }
};

/**
 * @brief Converts flux.dealloc to quartz.dealloc
 *
 * @details
 * Deallocates a qubit, releasing its resources. The OpAdaptor automatically
 * provides the type-converted qubit operand (!quartz.qubit instead of
 * !flux.qubit), so we simply pass it through to the new operation.
 *
 * Example transformation:
 * ```mlir
 * flux.dealloc %q_flux : !flux.qubit
 * // becomes:
 * quartz.dealloc %q_quartz : !quartz.qubit
 * ```
 */
struct ConvertFluxDeallocOp final : OpConversionPattern<flux::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted qubit
    rewriter.replaceOpWithNewOp<quartz::DeallocOp>(op, adaptor.getQubit());
    return success();
  }
};

/**
 * @brief Converts flux.static to quartz.static
 *
 * @details
 * Static qubits represent references to hardware-mapped or fixed-position
 * qubits identified by an index. The conversion preserves the index attribute
 * and creates the corresponding quartz.static operation.
 *
 * Example transformation:
 * ```mlir
 * %q0 = flux.static 0 : !flux.qubit
 * // becomes:
 * %q = quartz.static 0 : !quartz.qubit
 * ```
 */
struct ConvertFluxStaticOp final : OpConversionPattern<flux::StaticOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Create quartz.static with the same index
    rewriter.replaceOpWithNewOp<quartz::StaticOp>(op, op.getIndex());
    return success();
  }
};

/**
 * @brief Converts flux.measure to quartz.measure
 *
 * @details
 * Measurement demonstrates the key semantic difference between the dialects:
 * - Flux (value semantics): Consumes input qubit, returns both output qubit
 *   and classical bit result
 * - Quartz (reference semantics): Measures qubit in-place, returns only the
 *   classical bit result
 *
 * The OpAdaptor provides the input qubit already converted to !quartz.qubit.
 * Since Quartz operations are in-place, we return the same qubit reference
 * alongside the measurement bit. MLIR's conversion infrastructure automatically
 * routes subsequent uses of the Flux output qubit to this Quartz reference.
 *
 * Register metadata (name, size, index) for output recording is preserved
 * during conversion.
 *
 * Example transformation:
 * ```mlir
 * %q_out, %c = flux.measure("c", 2, 0) %q_in : !flux.qubit
 * // becomes:
 * %c = quartz.measure("c", 2, 0) %q : !quartz.qubit -> i1
 * // %q_out uses are replaced with %q (the adaptor-converted input)
 * ```
 */
struct ConvertFluxMeasureOp final : OpConversionPattern<flux::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    const auto& quartzQubit = adaptor.getQubitIn();

    // Create quartz.measure (in-place operation, returns only bit)
    // Preserve register metadata for output recording
    auto quartzOp = rewriter.create<quartz::MeasureOp>(
        op.getLoc(), quartzQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    auto measureBit = quartzOp.getResult();

    // Replace both results: qubit output → same quartz reference, bit → new bit
    rewriter.replaceOp(op, {quartzQubit, measureBit});

    return success();
  }
};

/**
 * @brief Converts flux.reset to quartz.reset
 *
 * @details
 * Reset operations force a qubit to the |0⟩ state:
 * - Flux (value semantics): Consumes input qubit, returns reset output qubit
 * - Quartz (reference semantics): Resets qubit in-place, no result value
 *
 * The OpAdaptor provides the input qubit already converted to !quartz.qubit.
 * Since Quartz's reset is in-place, we return the same qubit reference.
 * MLIR's conversion infrastructure automatically routes subsequent uses of
 * the Flux output qubit to this Quartz reference.
 *
 * Example transformation:
 * ```mlir
 * %q_out = flux.reset %q_in : !flux.qubit -> !flux.qubit
 * // becomes:
 * quartz.reset %q : !quartz.qubit
 * // %q_out uses are replaced with %q (the adaptor-converted input)
 * ```
 */
struct ConvertFluxResetOp final : OpConversionPattern<flux::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    const auto& quartzQubit = adaptor.getQubitIn();

    // Create quartz.reset (in-place operation, no result)
    rewriter.create<quartz::ResetOp>(op.getLoc(), quartzQubit);

    // Replace the output qubit with the same quartz reference
    rewriter.replaceOp(op, quartzQubit);

    return success();
  }
};

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  /**                                                                          \
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * flux.OP_NAME(%PARAM)                                                      \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM)                                                    \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor /*adaptor*/,                  \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertZeroTargetOneParameter<quartz::OP_CLASS>(op, rewriter);    \
    }                                                                          \
  };

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME %q_in : !flux.qubit -> !flux.qubit                  \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME %q : !quartz.qubit                                         \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<quartz::OP_CLASS>(op, adaptor,      \
                                                             rewriter);        \
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
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME(%PARAM) %q_in : !flux.qubit -> !flux.qubit          \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM) %q : !quartz.qubit                                 \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<quartz::OP_CLASS>(op, adaptor,       \
                                                            rewriter);         \
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
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME(%PARAM1, %PARAM2) %q_in : !flux.qubit ->            \
   * !flux.qubit                                                               \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM1, %PARAM2) %q : !quartz.qubit                       \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetTwoParameter<quartz::OP_CLASS>(op, adaptor,       \
                                                            rewriter);         \
    }                                                                          \
  };

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  /**                                                                          \
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q_out = flux.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q_in : !flux.qubit      \
   * -> !flux.qubit                                                            \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q : !quartz.qubit              \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetThreeParameter<quartz::OP_CLASS>(op, adaptor,     \
                                                              rewriter);       \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = flux.OP_NAME %q0_in, %q1_in : !flux.qubit, !flux.qubit \
   * -> !flux.qubit, !flux.qubit                                               \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME %q0, %q1 : !quartz.qubit, !quartz.qubit                    \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetZeroParameter<quartz::OP_CLASS>(op, adaptor,      \
                                                             rewriter);        \
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
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = flux.OP_NAME(%PARAM) %q0_in, %q1_in : !flux.qubit,     \
   * !flux.qubit -> !flux.qubit, !flux.qubit                                   \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM) %q0, %q1 : !quartz.qubit, !quartz.qubit            \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetOneParameter<quartz::OP_CLASS>(op, adaptor,       \
                                                            rewriter);         \
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
   * @brief Converts flux.OP_NAME to quartz.OP_NAME                            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = flux.OP_NAME(%PARAM1, %PARAM2) %q0_in, %q1_in :        \
   * !flux.qubit, !flux.qubit -> !flux.qubit, !flux.qubit                      \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM1, %PARAM2) %q0, %q1 : !quartz.qubit, !quartz.qubit  \
   * ```                                                                       \
   */                                                                          \
  struct ConvertFlux##OP_CLASS final : OpConversionPattern<flux::OP_CLASS> {   \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(flux::OP_CLASS op, OpAdaptor adaptor,                      \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetTwoParameter<quartz::OP_CLASS>(op, adaptor,       \
                                                            rewriter);         \
    }                                                                          \
  };

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

/**
 * @brief Converts quartz.ctrl to flux.ctrl
 *
 * @par Example:
 * ```mlir
 * %controls_out, %targets_out = flux.ctrl({%q0_in}, {%q1_in}) {
 *   %q1_res = flux.x %q1_in : !flux.qubit -> !flux.qubit
 *   flux.yield %q1_res
 * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
 * ```
 * is converted to
 * ```mlir
 * quartz.ctrl(%q0) {
 *   quartz.x %q1
 *   quartz.yield
 * }
 * ```
 */
struct ConvertFluxCtrlOp final : OpConversionPattern<flux::CtrlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Get Quartz controls
    const auto& quartzControls = adaptor.getControlsIn();

    // Create quartz.ctrl operation
    auto fluxOp = rewriter.create<quartz::CtrlOp>(op.getLoc(), quartzControls);

    // Clone body region from Flux to Quartz
    auto& dstRegion = fluxOp.getBody();
    rewriter.cloneRegionBefore(op.getBody(), dstRegion, dstRegion.end());

    // Replace the output qubits with the same quartz references
    rewriter.replaceOp(op, adaptor.getOperands());

    return success();
  }
};

/**
 * @brief Converts flux.yield to quartz.yield
 *
 * @par Example:
 * ```mlir
 * flux.yield %targets
 * ```
 * is converted to
 * ```mlir
 * quartz.yield
 * ```
 */
struct ConvertFluxYieldOp final : OpConversionPattern<flux::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(flux::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<quartz::YieldOp>(op);
    return success();
  }
};

/**
 * @brief Pass implementation for Flux-to-Quartz conversion
 *
 * @details
 * This pass converts Flux dialect operations (value semantics) to
 * Quartz dialect operations (reference semantics). The conversion is useful
 * for lowering optimized SSA-form code back to a hardware-oriented
 * representation suitable for backend code generation.
 *
 * The conversion leverages MLIR's built-in type conversion infrastructure:
 * The TypeConverter handles !flux.qubit → !quartz.qubit transformations,
 * and the OpAdaptor automatically provides type-converted operands to each
 * conversion pattern. This eliminates the need for manual state tracking.
 *
 * Key semantic transformation:
 * - Flux operations form explicit SSA chains where each operation consumes
 *   inputs and produces new outputs
 * - Quartz operations modify qubits in-place using references
 * - The conversion maps each Flux SSA chain to a single Quartz reference,
 *   with MLIR's conversion framework automatically handling the plumbing
 *
 * The pass operates through:
 * 1. Type conversion: !flux.qubit → !quartz.qubit
 * 2. Operation conversion: Each Flux op converted to its Quartz equivalent
 * 3. Automatic operand mapping: OpAdaptors provide converted operands
 * 4. Function/control-flow adaptation: Signatures updated to use Quartz types
 */
struct FluxToQuartz final : impl::FluxToQuartzBase<FluxToQuartz> {
  using FluxToQuartzBase::FluxToQuartzBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    FluxToQuartzTypeConverter typeConverter(context);

    // Configure conversion target: Flux illegal, Quartz legal
    target.addIllegalDialect<FluxDialect>();
    target.addLegalDialect<QuartzDialect>();

    // Register operation conversion patterns
    // Note: No state tracking needed - OpAdaptors handle type conversion
    patterns
        .add<ConvertFluxAllocOp, ConvertFluxDeallocOp, ConvertFluxStaticOp,
             ConvertFluxMeasureOp, ConvertFluxResetOp, ConvertFluxGPhaseOp,
             ConvertFluxIdOp, ConvertFluxXOp, ConvertFluxYOp, ConvertFluxZOp,
             ConvertFluxHOp, ConvertFluxSOp, ConvertFluxSdgOp, ConvertFluxTOp,
             ConvertFluxTdgOp, ConvertFluxSXOp, ConvertFluxSXdgOp,
             ConvertFluxRXOp, ConvertFluxRYOp, ConvertFluxRZOp, ConvertFluxPOp,
             ConvertFluxROp, ConvertFluxU2Op, ConvertFluxUOp, ConvertFluxSWAPOp,
             ConvertFluxiSWAPOp, ConvertFluxDCXOp, ConvertFluxECROp,
             ConvertFluxRXXOp, ConvertFluxRYYOp, ConvertFluxRZXOp,
             ConvertFluxRZZOp, ConvertFluxXXPlusYYOp, ConvertFluxXXMinusYYOp,
             ConvertFluxCtrlOp, ConvertFluxYieldOp>(typeConverter, context);

    // Conversion of flux types in func.func signatures
    // Note: This currently has limitations with signature changes
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Conversion of flux types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Conversion of flux types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // Conversion of flux types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mlir
