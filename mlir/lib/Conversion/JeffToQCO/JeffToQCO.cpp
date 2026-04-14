/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <jeff/Conversion/JeffToNative/JeffToNative.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstddef>
#include <utility>

namespace mlir {

using namespace qco;

#define GEN_PASS_DEF_JEFFTOQCO
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h.inc"

/**
 * @brief Creates a modified QCO operation from a Jeff operation
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param controls The control qubits of the operation
 * @param targets The target qubits of the operation
 * @param lambda A lambda function that creates the inner QCO operation and
 * returns its results
 */
template <typename JeffOpType>
static void createModified(
    JeffOpType& op, ConversionPatternRewriter& rewriter, ValueRange controls,
    ValueRange targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> lambda) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0) {
    CtrlOp ctrlOp;
    if (!op.getIsAdjoint()) {
      ctrlOp = CtrlOp::create(rewriter, loc, controls, targets, lambda);
    } else {
      ctrlOp = CtrlOp::create(
          rewriter, loc, controls, targets,
          [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
            auto invOp = InvOp::create(rewriter, loc, ctrlTargets, lambda);
            return invOp.getQubitsOut();
          });
    }
    llvm::SmallVector<Value> results;
    llvm::append_range(results, ctrlOp.getTargetsOut());
    llvm::append_range(results, ctrlOp.getControlsOut());
    rewriter.replaceOp(op, results);
  } else if (op.getIsAdjoint()) {
    auto invOp = InvOp::create(rewriter, loc, targets, lambda);
    rewriter.replaceOp(op, invOp.getQubitsOut());
  }
}

/**
 * @brief Creates a (potentially modified) QCO operation from a Jeff operation.
 *
 * @details
 * This helper centralizes the "direct vs. ctrl/inv-wrapped" decision and uses
 * index sequences to forward the desired number of targets and parameters into
 * the QCO op builder.
 *
 * @tparam QCOOpType The QCO operation type to create
 * @tparam JeffOpType The Jeff operation type to convert from
 * @tparam TargetIndices Indices of target operands to forward
 * @tparam ParamIndices Indices of parameters to forward
 *
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param controls The control qubits (type-converted) of the operation
 * @param targets The target qubits (type-converted) of the operation
 * @param parameters The parameters of the operation
 */
template <typename QCOOpType, typename JeffOpType, std::size_t... TargetIndices,
          std::size_t... ParamIndices>
static LogicalResult
createGateFromJeff(JeffOpType& op, ConversionPatternRewriter& rewriter,
                   ValueRange controls, ValueRange targets,
                   ValueRange parameters,
                   std::index_sequence<TargetIndices...> /*targetIndices*/,
                   std::index_sequence<ParamIndices...> /*paramIndices*/) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, targets[TargetIndices]...,
                                           parameters[ParamIndices]...);
    return success();
  }

  auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
    auto qcoOp =
        QCOOpType::create(rewriter, op.getLoc(), innerTargets[TargetIndices]...,
                          parameters[ParamIndices]...);
    return qcoOp.getOutputQubits();
  };
  createModified(op, rewriter, controls, targets, lambda);
  return success();
}

template <typename QCOOpType, typename JeffOpType, std::size_t NumTargets,
          std::size_t NumParams>
static LogicalResult
createGateFromJeffArity(JeffOpType& op, ConversionPatternRewriter& rewriter,
                        ValueRange controls, ValueRange targets,
                        ValueRange parameters = {}) {
  if (targets.size() != NumTargets) {
    return rewriter.notifyMatchFailure(
        op, "Unexpected number of target qubits for Jeff-to-QCO conversion");
  }
  if (parameters.size() != NumParams) {
    return rewriter.notifyMatchFailure(
        op, "Unexpected number of parameters for Jeff-to-QCO conversion");
  }

  return createGateFromJeff<QCOOpType, JeffOpType>(
      op, rewriter, controls, targets, parameters,
      std::make_index_sequence<NumTargets>{},
      std::make_index_sequence<NumParams>{});
}

/**
 * @brief Creates a qco.barrier operation from a jeff.custom operation
 *
 * @param op The jeff.custom operation instance to convert
 * @param adaptor The OpAdaptor of the jeff.custom operation
 * @param rewriter The pattern rewriter
 */
static void createBarrierOp(jeff::CustomOp& op, jeff::CustomOpAdaptor& adaptor,
                            ConversionPatternRewriter& rewriter) {
  auto targets = adaptor.getInTargetQubits();
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<BarrierOp>(op, targets);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp = BarrierOp::create(rewriter, op.getLoc(), innerTargets);
      return qcoOp.getQubitsOut();
    };
    createModified(op, rewriter, adaptor.getInCtrlQubits(), targets, lambda);
  }
}

/**
 * @brief Gets the name of the entry point from the module attributes
 */
static llvm::StringRef getEntryPointName(Operation* op) {
  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module) {
    llvm::reportFatalInternalError("Expected a module operation");
  }

  auto entryPointAttr = module->getAttr("jeff.entrypoint");
  if (!entryPointAttr) {
    llvm::reportFatalInternalError(
        "Module is missing 'jeff.entrypoint' attribute");
  }
  auto entryPoint = llvm::cast<IntegerAttr>(entryPointAttr).getUInt();

  auto stringsAttr = module->getAttr("jeff.strings");
  if (!stringsAttr) {
    llvm::reportFatalInternalError(
        "Module is missing 'jeff.strings' attribute");
  }
  auto strings = llvm::cast<ArrayAttr>(stringsAttr);

  if (entryPoint >= strings.size()) {
    llvm::reportFatalInternalError("Entry point index is out of bounds");
  }

  return llvm::cast<mlir::StringAttr>(strings[entryPoint]).getValue();
}

/**
 * @brief Cleans up the module after conversion
 *
 * @param op The module operation to clean up
 * @return LogicalResult Success or failure of the cleanup
 */
static LogicalResult cleanUp(Operation* op) {
  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module) {
    return failure();
  }

  // Remove module attributes
  module->removeAttr("jeff.entrypoint");
  module->removeAttr("jeff.strings");
  module->removeAttr("jeff.tool");
  module->removeAttr("jeff.toolVersion");
  module->removeAttr("jeff.version");
  module->removeAttr("jeff.versionMinor");
  module->removeAttr("jeff.versionPatch");

  return success();
}

namespace {

/**
 * @brief Converts jeff.qureg_alloc to qtensor.alloc
 *
 * @par Example:
 * ```mlir
 * %qureg = jeff.qureg_alloc(%c3) : !jeff.qureg
 * ```
 * is converted to
 * ```mlir
 * %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertJeffQuregAllocOpToQCO final
    : OpConversionPattern<jeff::QuregAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QuregAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto sizeValue = getConstantIntValue(adaptor.getNumQubits());
    Type tensorType;
    Value size;
    if (sizeValue.has_value()) {
      tensorType = RankedTensorType::get({*sizeValue},
                                         QubitType::get(rewriter.getContext()));
      size = arith::ConstantOp::create(rewriter, op.getLoc(),
                                       rewriter.getIndexAttr(*sizeValue))
                 .getResult();
    } else {
      tensorType = RankedTensorType::get({ShapedType::kDynamic},
                                         QubitType::get(rewriter.getContext()));
      size = arith::IndexCastOp::create(rewriter, op.getLoc(),
                                        rewriter.getIndexType(),
                                        adaptor.getNumQubits())
                 .getResult();
    }
    rewriter.replaceOpWithNewOp<qtensor::AllocOp>(op, tensorType, size);
    return success();
  }
};

/**
 * @brief Converts jeff.qureg_extract_index to qtensor.extract
 *
 * @par Example:
 * ```mlir
 * %qureg_out, %q = jeff.qureg_extract_index(%c0) %qureg_in : !jeff.qureg,
 * !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %tensor_out, %q = qtensor.extract %tensor_in[%c0]: tensor<3x!qco.qubit>
 * ```
 */
struct ConvertJeffQuregExtractIndexOpToQCO final
    : OpConversionPattern<jeff::QuregExtractIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QuregExtractIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto index = arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), adaptor.getIndex());
    rewriter.replaceOpWithNewOp<qtensor::ExtractOp>(op, adaptor.getInQreg(),
                                                    index.getResult());
    return success();
  }
};

/**
 * @brief Converts jeff.qureg_insert_index to qtensor.insert
 *
 * @par Example:
 * ```mlir
 * %qureg_out = jeff.qureg_insert_index(%c0) %qureg_in %q : !jeff.qureg
 * ```
 * is converted to
 * ```mlir
 * %tensor_out = qtensor.insert %q into %tensor_in[%c0] : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertJeffQuregInsertIndexOpToQCO final
    : OpConversionPattern<jeff::QuregInsertIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QuregInsertIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto index = arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), adaptor.getIndex());
    rewriter.replaceOpWithNewOp<qtensor::InsertOp>(
        op, adaptor.getInQubit(), adaptor.getInQreg(), index.getResult());
    return success();
  }
};

/**
 * @brief Converts jeff.qureg_free_zero to qtensor.dealloc
 *
 * @par Example:
 * ```mlir
 * jeff.qureg_free_zero %qureg : !jeff.qureg
 * ```
 * is converted to
 * ```mlir
 * qtensor.dealloc %tensor : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertJeffQuregFreeZeroOpToQCO final
    : OpConversionPattern<jeff::QuregFreeZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QuregFreeZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qtensor::DeallocOp>(op, adaptor.getQreg());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_alloc to qco.alloc
 *
 * @par Example:
 * ```mlir
 * %q = jeff.qubit_alloc : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = qco.alloc : !qco.qubit
 * ```
 */
struct ConvertJeffQubitAllocOpToQCO final
    : OpConversionPattern<jeff::QubitAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitAllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<AllocOp>(op);
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_free to qco.reset + qco.sink
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit
 * qco.sink %q_out : !qco.qubit
 * ```
 */
struct ConvertJeffQubitFreeOpToQCO final
    : OpConversionPattern<jeff::QubitFreeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto resetOp = ResetOp::create(rewriter, op.getLoc(), adaptor.getInQubit());
    rewriter.replaceOpWithNewOp<SinkOp>(op, resetOp.getQubitOut());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_free_zero to qco.sink
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free_zero %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * qco.sink %q : !qco.qubit
 * ```
 */
struct ConvertJeffQubitFreeZeroOpToQCO final
    : OpConversionPattern<jeff::QubitFreeZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitFreeZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<SinkOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_measure to qco.measure + qco.sink
 *
 * @par Example:
 * ```mlir
 * %result = jeff.qubit_measure %q_in : !i1
 * ```
 * is converted to
 * ```mlir
 * %q_out, %result = qco.measure %q_in : !qco.qubit
 * qco.sink %q_out : !qco.qubit
 * ```
 */
struct ConvertJeffQubitMeasureOpToQCO final
    : OpConversionPattern<jeff::QubitMeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitMeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto measureOp = MeasureOp::create(rewriter, loc, adaptor.getInQubit());
    SinkOp::create(rewriter, loc, measureOp.getQubitOut());
    rewriter.replaceOp(op, measureOp.getResult());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_measure_nd to qco.measure
 *
 * @par Example:
 * ```mlir
 * %q_out, %result = jeff.qubit_measure_nd %q_in : !jeff.qubit, i1
 * ```
 * is converted to
 * ```mlir
 * %q_out, %result = qco.measure %q_in : !qco.qubit
 * ```
 */
struct ConvertJeffQubitMeasureNDOpToQCO final
    : OpConversionPattern<jeff::QubitMeasureNDOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitMeasureNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<MeasureOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.reset to qco.qubit_reset
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.qubit_reset %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit
 * ```
 */
struct ConvertJeffQubitResetOpToQCO final
    : OpConversionPattern<jeff::QubitResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ResetOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.gphase to qco.gphase
 *
 * @par Example:
 * ```mlir
 * jeff.gphase(%theta) {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8}
 * ```
 * is converted to
 * ```mlir
 * qco.gphase(%theta)
 * ```
 */
struct ConvertJeffGPhaseOpToQCO final : OpConversionPattern<jeff::GPhaseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::GPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
      rewriter.replaceOpWithNewOp<GPhaseOp>(op, op.getRotation());
    } else {
      auto lambda = [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {
        GPhaseOp::create(rewriter, op.getLoc(), op.getRotation());
        return {};
      };
      createModified(op, rewriter, adaptor.getInCtrlQubits(), {}, lambda);
    }

    return success();
  }
};

/**
 * @brief Converts one-target, zero-parameter Jeff gate to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.x {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8}
 * %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.x %q_in : !qco.qubit
 * ```
 */
template <typename JeffOpType, typename QCOOpType>
struct ConvertJeffOneTargetZeroParameterToQCO final
    : OpConversionPattern<JeffOpType> {
  using OpConversionPattern<JeffOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JeffOpType op, JeffOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    return createGateFromJeffArity<QCOOpType, JeffOpType, 1, 0>(
        op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInQubit());
  }
};

/**
 * @brief Converts one-target, one-parameter Jeff gate to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.rx(%theta) {is_adjoint = false, num_ctrls = 0 : i8, power = 1 :
 * i8} %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.rx(%theta) %q_in : !qco.qubit
 * ```
 */
template <typename JeffOpType, typename QCOOpType>
struct ConvertJeffOneTargetOneParameterToQCO final
    : OpConversionPattern<JeffOpType> {
  using OpConversionPattern<JeffOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JeffOpType op, JeffOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    return createGateFromJeffArity<QCOOpType, JeffOpType, 1, 1>(
        op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        op.getRotation());
  }
};

/**
 * @brief Converts jeff.u to qco.u
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.u(%theta, %phi, %lambda) {is_adjoint = false, num_ctrls = 0 :
 * i8, power = 1 : i8} %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.u(%theta, %phi, %lambda) %q_in : !qco.qubit
 * ```
 */
struct ConvertJeffUOpToQCO final : OpConversionPattern<jeff::UOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::UOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    return createGateFromJeffArity<UOp, jeff::UOp, 1, 3>(
        op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        {op.getTheta(), op.getPhi(), op.getLambda()});
  }
};

/**
 * @brief Converts jeff.swap to qco.swap
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = jeff.swap {is_adjoint = false, num_ctrls = 0 : i8, power =
 * 1 : i8} %q0_in %q1_in : !jeff.qubit !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.swap %q0_in, %q1_in : !qco.qubit, !qco.qubit
 * ```
 */
struct ConvertJeffSwapOpToQCO final : OpConversionPattern<jeff::SwapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::SwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    return createGateFromJeffArity<SWAPOp, jeff::SwapOp, 2, 0>(
        op, rewriter, adaptor.getInCtrlQubits(),
        {adaptor.getInQubitOne(), adaptor.getInQubitTwo()});
  }
};

/**
 * @brief Converts jeff.custom to the corresponding QCO operation
 *
 * @par Example:
 * ```mlir
 * %q_out:2 = jeff.custom "iswap"() {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.iswap %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 */
struct ConvertJeffCustomOpToQCO final : OpConversionPattern<jeff::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto controls = adaptor.getInCtrlQubits();
    auto targets = adaptor.getInTargetQubits();
    auto params = op.getParams();
    auto name = op.getName();

    if (name == "sx") {
      if (targets.size() != 1 || !params.empty()) {
        return rewriter.notifyMatchFailure(
            op, "Custom sx expects exactly one target and no parameters");
      }
      return createGateFromJeffArity<SXOp, jeff::CustomOp, 1, 0>(
          op, rewriter, controls, targets, params);
    }
    if (name == "barrier") {
      if (!params.empty()) {
        return rewriter.notifyMatchFailure(
            op, "Custom barrier operations must not have parameters");
      }
      createBarrierOp(op, adaptor, rewriter);
      return success();
    }
    if (name == "r") {
      if (targets.size() != 1 || params.size() != 2) {
        return rewriter.notifyMatchFailure(
            op, "Custom r expects one target and two parameters");
      }
      return createGateFromJeffArity<ROp, jeff::CustomOp, 1, 2>(
          op, rewriter, controls, targets, params);
    }
    if (name == "iswap") {
      if (targets.size() != 2 || !params.empty()) {
        return rewriter.notifyMatchFailure(
            op, "Custom iswap expects two targets and no parameters");
      }
      return createGateFromJeffArity<iSWAPOp, jeff::CustomOp, 2, 0>(
          op, rewriter, controls, targets, params);
    }
    if (name == "dcx") {
      if (targets.size() != 2 || !params.empty()) {
        return rewriter.notifyMatchFailure(
            op, "Custom dcx expects two targets and no parameters");
      }
      return createGateFromJeffArity<DCXOp, jeff::CustomOp, 2, 0>(
          op, rewriter, controls, targets, params);
    }
    if (name == "ecr") {
      if (targets.size() != 2 || !params.empty()) {
        return rewriter.notifyMatchFailure(
            op, "Custom ecr expects two targets and no parameters");
      }
      return createGateFromJeffArity<ECROp, jeff::CustomOp, 2, 0>(
          op, rewriter, controls, targets, params);
    }
    if (name == "xx_plus_yy") {
      if (targets.size() != 2 || params.size() != 2) {
        return rewriter.notifyMatchFailure(
            op, "Custom xx_plus_yy expects two targets and two parameters");
      }
      return createGateFromJeffArity<XXPlusYYOp, jeff::CustomOp, 2, 2>(
          op, rewriter, controls, targets, params);
    }
    if (name == "xx_minus_yy") {
      if (targets.size() != 2 || params.size() != 2) {
        return rewriter.notifyMatchFailure(
            op, "Custom xx_minus_yy expects two targets and two parameters");
      }
      return createGateFromJeffArity<XXMinusYYOp, jeff::CustomOp, 2, 2>(
          op, rewriter, controls, targets, params);
    }
    return rewriter.notifyMatchFailure(op,
                                       "Unsupported custom operation: " + name);
  }
};

/**
 * @brief Converts jeff.ppr to the corresponding QCO operation
 *
 * @par Example:
 * ```mlir
 * %q_out:2 = jeff.ppr(%theta, [1, 1]) {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.rxx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 */
struct ConvertJeffPPROpToQCO final : OpConversionPattern<jeff::PPROp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::PPROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto pauliGates = op.getPauliGates();
    auto targets = adaptor.getInQubits();
    auto controls = adaptor.getInCtrlQubits();
    if (pauliGates.size() != 2 || targets.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Only PPR operations with exactly 2 Pauli gates are supported");
    }

    if (pauliGates[0] == 1 && pauliGates[1] == 1) {
      return createGateFromJeffArity<RXXOp, jeff::PPROp, 2, 1>(
          op, rewriter, controls, targets, op.getRotation());
    }
    if (pauliGates[0] == 2 && pauliGates[1] == 2) {
      return createGateFromJeffArity<RYYOp, jeff::PPROp, 2, 1>(
          op, rewriter, controls, targets, op.getRotation());
    }
    if (pauliGates[0] == 3 && pauliGates[1] == 1) {
      return createGateFromJeffArity<RZXOp, jeff::PPROp, 2, 1>(
          op, rewriter, controls, targets, op.getRotation());
    }
    if (pauliGates[0] == 3 && pauliGates[1] == 3) {
      return createGateFromJeffArity<RZZOp, jeff::PPROp, 2, 1>(
          op, rewriter, controls, targets, op.getRotation());
    }

    return rewriter.notifyMatchFailure(op, "Unsupported PPR operation");
  }
};

/**
 * @brief Converts the Jeff-style main function to a QCO-style main function
 *
 * @par Example:
 * ```mlir
 * func.func @main() -> () {
 *   return
 * }
 * ```
 * is converted to
 * ```mlir
 * func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
 *   %0 = arith.constant 0 : i64
 *   return %0
 * }
 * ```
 */
struct ConvertJeffMainToQCO final : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getSymName() != getEntryPointName(op->getParentOfType<ModuleOp>())) {
      return failure();
    }

    if (op.getBlocks().size() != 1) {
      return failure();
    }
    auto* block = &op.getBlocks().front();

    auto* returnOp = block->getTerminator();
    if (!llvm::isa<func::ReturnOp>(returnOp)) {
      return failure();
    }

    // Update function signature and add passthrough attribute
    rewriter.startOpModification(op);
    auto* ctx = rewriter.getContext();
    op.setType(FunctionType::get(ctx, {}, {rewriter.getI64Type()}));
    auto entryPointAttr = StringAttr::get(ctx, "entry_point");
    op->setAttr("passthrough", ArrayAttr::get(ctx, {entryPointAttr}));
    rewriter.finalizeOpModification(op);

    // Replace return operation
    rewriter.setInsertionPointToStart(block);
    auto constOp = arith::ConstantOp::create(rewriter, op.getLoc(),
                                             rewriter.getI64IntegerAttr(0));

    rewriter.setInsertionPointToEnd(block);
    func::ReturnOp::create(rewriter, op.getLoc(), constOp.getResult());

    rewriter.eraseOp(returnOp);

    return success();
  }
};

/**
 * @brief Type converter for Jeff-to-QCO conversion
 *
 * @details
 * Converts `!jeff.qubit` to `!qco.qubit` and `!jeff.qureg` to
 * `!tensor<?x!qco.qubit>`.
 */
class JeffToQCOTypeConverter final : public TypeConverter {
public:
  explicit JeffToQCOTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](jeff::QubitType /*type*/) -> Type {
      return QubitType::get(ctx);
    });

    addConversion([ctx](jeff::QuregType type) -> Type {
      return RankedTensorType::get({type.getLength()}, QubitType::get(ctx));
    });
  }
};

/**
 * @brief Pass for converting Jeff operations to QCO operations
 */
struct JeffToQCO final : impl::JeffToQCOBase<JeffToQCO> {
  using JeffToQCOBase::JeffToQCOBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    const JeffToQCOTypeConverter typeConverter(context);

    // Configure conversion target
    target.addIllegalDialect<jeff::JeffDialect>();
    target.addLegalDialect<QCODialect, qtensor::QTensorDialect,
                           arith::ArithDialect, math::MathDialect,
                           tensor::TensorDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !(op.getSymName() == getEntryPointName(module) &&
               op.getFunctionType().getResults().empty());
    });
    target.addLegalOp<func::ReturnOp>();

    // Register operation conversion patterns
    jeff::populateJeffToNativeConversionPatterns(patterns);
    patterns.add<
        ConvertJeffQuregAllocOpToQCO, ConvertJeffQuregExtractIndexOpToQCO,
        ConvertJeffQuregInsertIndexOpToQCO, ConvertJeffQuregFreeZeroOpToQCO,
        ConvertJeffQubitAllocOpToQCO, ConvertJeffQubitFreeOpToQCO,
        ConvertJeffQubitFreeZeroOpToQCO, ConvertJeffQubitMeasureOpToQCO,
        ConvertJeffQubitMeasureNDOpToQCO, ConvertJeffQubitResetOpToQCO,
        ConvertJeffGPhaseOpToQCO,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::IOp, IdOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::XOp, XOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::YOp, YOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::ZOp, ZOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::HOp, HOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::SOp, SOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::TOp, TOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::RxOp, RXOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::RyOp, RYOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::RzOp, RZOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::R1Op, POp>,
        ConvertJeffUOpToQCO, ConvertJeffSwapOpToQCO, ConvertJeffCustomOpToQCO,
        ConvertJeffPPROpToQCO, ConvertJeffMainToQCO>(typeConverter, context);

    // Apply the conversion
    if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }

    if (cleanUp(module).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
