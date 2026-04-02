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

#include "mlir/Conversion/GateTable.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <jeff/Conversion/JeffToNative/JeffToNative.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param controls The control qubits of the operation
 * @param targets The target qubits of the operation
 * @param lambda A lambda function that creates the inner QCO operation and
 * returns its results
 */
template <typename QCOOpType, typename JeffOpType>
static void createModified(
    JeffOpType& op, ConversionPatternRewriter& rewriter,
    const llvm::SmallVector<Value>& controls,
    const llvm::SmallVector<Value>& targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> lambda) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0) {
    qco::CtrlOp ctrlOp;
    if (!op.getIsAdjoint()) {
      ctrlOp = qco::CtrlOp::create(rewriter, loc, controls, targets, lambda);
    } else {
      ctrlOp = qco::CtrlOp::create(
          rewriter, loc, controls, targets,
          [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
            auto invOp = qco::InvOp::create(rewriter, loc, ctrlTargets, lambda);
            return invOp.getQubitsOut();
          });
    }
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getIsAdjoint()) {
    auto invOp = qco::InvOp::create(rewriter, loc, targets, lambda);
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
 * @tparam ResultExtractor Callable returning QCO results as SmallVector<Value>
 *
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param controls The control qubits (type-converted) of the operation
 * @param targets The target qubits (type-converted) of the operation
 * @param parameters The parameters of the operation
 * @param extractResults Callable extracting the QCO op results for region-yield
 */
template <typename QCOOpType, typename JeffOpType, std::size_t... TargetIndices,
          std::size_t... ParamIndices, typename ResultExtractor>
static void
createGateFromJeff(JeffOpType& op, ConversionPatternRewriter& rewriter,
                   const llvm::SmallVector<Value>& controls,
                   const llvm::SmallVector<Value>& targets,
                   const llvm::SmallVector<Value>& parameters,
                   std::index_sequence<TargetIndices...> /*targetIndices*/,
                   std::index_sequence<ParamIndices...> /*paramIndices*/,
                   const ResultExtractor& extractResults) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, targets[TargetIndices]...,
                                           parameters[ParamIndices]...);
    return;
  }

  auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
    auto qcoOp =
        QCOOpType::create(rewriter, op.getLoc(), innerTargets[TargetIndices]...,
                          parameters[ParamIndices]...);
    return extractResults(qcoOp);
  };
  createModified<QCOOpType, JeffOpType>(
      op, rewriter, controls,
      {targets
           [TargetIndices]...}, // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
      lambda);
}

template <typename QCOOpType, typename JeffOpType, std::size_t NumTargets,
          std::size_t NumParams>
static void
createGateFromJeffArity(JeffOpType& op, ConversionPatternRewriter& rewriter,
                        const llvm::SmallVector<Value>& controls,
                        const llvm::SmallVector<Value>& targets,
                        const llvm::SmallVector<Value>& parameters) {
  static_assert((NumTargets == 1) || (NumTargets == 2),
                "Only 1- and 2-target gates are supported here");

  auto extractResults = [&](auto qcoOp) -> llvm::SmallVector<Value> {
    if constexpr (NumTargets == 1) {
      return {qcoOp.getQubitOut()};
    } else {
      return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
    }
  };

  createGateFromJeff<QCOOpType, JeffOpType>(
      op, rewriter, controls, targets, parameters,
      std::make_index_sequence<NumTargets>{},
      std::make_index_sequence<NumParams>{}, extractResults);
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
    rewriter.replaceOpWithNewOp<qco::BarrierOp>(op, targets);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp = qco::BarrierOp::create(rewriter, op.getLoc(), innerTargets);
      return qcoOp.getQubitsOut();
    };
    createModified<qco::BarrierOp>(op, rewriter, adaptor.getInCtrlQubits(),
                                   targets, lambda);
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
    auto size = arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), adaptor.getNumQubits());
    rewriter.replaceOpWithNewOp<qtensor::AllocOp>(op, size.getResult());
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
    rewriter.replaceOpWithNewOp<qco::AllocOp>(op);
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
    auto resetOp =
        qco::ResetOp::create(rewriter, op.getLoc(), adaptor.getInQubit());
    rewriter.replaceOpWithNewOp<qco::SinkOp>(op, resetOp.getQubitOut());
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
    rewriter.replaceOpWithNewOp<qco::SinkOp>(op, adaptor.getInQubit());
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
    auto measureOp =
        qco::MeasureOp::create(rewriter, loc, adaptor.getInQubit());
    qco::SinkOp::create(rewriter, loc, measureOp.getQubitOut());
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
    rewriter.replaceOpWithNewOp<qco::MeasureOp>(op, adaptor.getInQubit());
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
    rewriter.replaceOpWithNewOp<qco::ResetOp>(op, adaptor.getInQubit());
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
      rewriter.replaceOpWithNewOp<qco::GPhaseOp>(op, op.getRotation());
    } else {
      auto lambda = [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {
        qco::GPhaseOp::create(rewriter, op.getLoc(), op.getRotation());
        return {};
      };
      createModified<qco::GPhaseOp, jeff::GPhaseOp>(
          op, rewriter, adaptor.getInCtrlQubits(), {}, lambda);
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

    createGateFromJeffArity<QCOOpType, JeffOpType, 1, 0>(
        op, rewriter, adaptor.getInCtrlQubits(), {adaptor.getInQubit()}, {});

    return success();
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

    createGateFromJeffArity<QCOOpType, JeffOpType, 1, 1>(
        op, rewriter, adaptor.getInCtrlQubits(), {adaptor.getInQubit()},
        {op.getRotation()});

    return success();
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

    createGateFromJeffArity<qco::UOp, jeff::UOp, 1, 3>(
        op, rewriter, adaptor.getInCtrlQubits(), {adaptor.getInQubit()},
        {op.getTheta(), op.getPhi(), op.getLambda()});

    return success();
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

    createGateFromJeffArity<qco::SWAPOp, jeff::SwapOp, 2, 0>(
        op, rewriter, adaptor.getInCtrlQubits(),
        {adaptor.getInQubitOne(), adaptor.getInQubitTwo()}, {});

    return success();
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

    if (op.getName() == "sx") {
      if (op.getInTargetQubits().size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "Custom SX operations must have exactly one target qubit");
      }
      createGateFromJeffArity<qco::SXOp, jeff::CustomOp, 1, 0>(
          op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits(),
          {});
      return success();
    }

    if (op.getName() == "barrier") {
      createBarrierOp(op, adaptor, rewriter);
      return success();
    }

#define MQT_HANDLE_JEFF_CUSTOM_TO_QCO(                                         \
    KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, JEFF_OP,                   \
    JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME, JEFF_PPR, QIR_KIND, QIR_FN)           \
  do {                                                                         \
    if constexpr ((JEFF_KIND) == ::mlir::mqt::gates::JeffKind::Custom &&       \
                  !(JEFF_BASE_ADJOINT)) {                                      \
      if (op.getName() == #JEFF_CUSTOM_NAME) {                                 \
        if constexpr ((TARGETS) == 1 && (PARAMS) == 0) {                       \
          if (op.getInTargetQubits().size() != 1) {                            \
            return rewriter.notifyMatchFailure(                                \
                op, "Custom operations must have exactly one target qubit");   \
          }                                                                    \
          createGateFromJeffArity<QCO_OP, jeff::CustomOp, 1, 0>(               \
              op, rewriter, adaptor.getInCtrlQubits(),                         \
              {adaptor.getInTargetQubits()[0]}, {});                           \
          return success();                                                    \
        } else if constexpr ((TARGETS) == 1 && (PARAMS) == 2) {                \
          if (op.getInTargetQubits().size() != 1) {                            \
            return rewriter.notifyMatchFailure(                                \
                op, "Custom operations must have exactly one target qubit");   \
          }                                                                    \
          createGateFromJeffArity<QCO_OP, jeff::CustomOp, 1, 2>(               \
              op, rewriter, adaptor.getInCtrlQubits(),                         \
              {adaptor.getInTargetQubits()[0]}, op.getParams());               \
          return success();                                                    \
        } else if constexpr ((TARGETS) == 2 && (PARAMS) == 0) {                \
          createGateFromJeffArity<QCO_OP, jeff::CustomOp, 2, 0>(               \
              op, rewriter, adaptor.getInCtrlQubits(),                         \
              adaptor.getInTargetQubits(), {});                                \
          return success();                                                    \
        } else if constexpr ((TARGETS) == 2 && (PARAMS) == 2) {                \
          createGateFromJeffArity<QCO_OP, jeff::CustomOp, 2, 2>(               \
              op, rewriter, adaptor.getInCtrlQubits(),                         \
              adaptor.getInTargetQubits(), op.getParams());                    \
          return success();                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (false);

    MQT_GATE_TABLE(MQT_HANDLE_JEFF_CUSTOM_TO_QCO)

#undef MQT_HANDLE_JEFF_CUSTOM_TO_QCO

    return rewriter.notifyMatchFailure(op, "Unsupported custom operation: " +
                                               op.getName());
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

#define MQT_HANDLE_JEFF_PPR_TO_QCO(                                            \
    KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, JEFF_OP,                   \
    JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME, JEFF_PPR, QIR_KIND, QIR_FN)           \
  do {                                                                         \
    if constexpr ((JEFF_KIND) == ::mlir::mqt::gates::JeffKind::PPR) {          \
      if (pauliGates[0] == (JEFF_PPR).p0 && pauliGates[1] == (JEFF_PPR).p1) {  \
        createGateFromJeffArity<QCO_OP, jeff::PPROp, 2, 1>(                    \
            op, rewriter, controls, targets, {op.getRotation()});              \
        return success();                                                      \
      }                                                                        \
    }                                                                          \
  } while (false);

    MQT_GATE_TABLE(MQT_HANDLE_JEFF_PPR_TO_QCO)

#undef MQT_HANDLE_JEFF_PPR_TO_QCO

    return rewriter.notifyMatchFailure(op, "Unsupported PPR operation");

    return success();
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
      return qco::QubitType::get(ctx);
    });

    addConversion([ctx](jeff::QuregType /*type*/) -> Type {
      return RankedTensorType::get({ShapedType::kDynamic},
                                   qco::QubitType::get(ctx));
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
        ConvertJeffOneTargetZeroParameterToQCO<jeff::IOp, qco::IdOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::XOp, qco::XOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::YOp, qco::YOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::ZOp, qco::ZOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::HOp, qco::HOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::SOp, qco::SOp>,
        ConvertJeffOneTargetZeroParameterToQCO<jeff::TOp, qco::TOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::RxOp, qco::RXOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::RyOp, qco::RYOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::RzOp, qco::RZOp>,
        ConvertJeffOneTargetOneParameterToQCO<jeff::R1Op, qco::POp>,
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
