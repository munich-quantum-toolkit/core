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

#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
 * @brief Creates a one-target, zero-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param controls The control qubits of the operation
 * @param target The target qubit of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void createOneTargetZeroParameter(
    JeffOpType& op, ConversionPatternRewriter& rewriter,
    const llvm::SmallVector<Value>& controls, Value target) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, target);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), innerTargets[0]);
      return {qcoOp.getQubitOut()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, {target},
                                          lambda);
  }
}

/**
 * @brief Creates a one-target, one-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param target The target qubit of the operation
 * @param parameter The parameter of the operation
 * @param controls The control qubits of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void createOneTargetOneParameter(
    JeffOpType& op, ConversionPatternRewriter& rewriter, Value parameter,
    const llvm::SmallVector<Value>& controls, Value target) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, target, parameter);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp =
          QCOOpType::create(rewriter, op.getLoc(), innerTargets[0], parameter);
      return {qcoOp.getQubitOut()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, {target},
                                          lambda);
  }
}

/**
 * @brief Creates a one-target, two-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param target The target qubit of the operation
 * @param parameters The parameters of the operation
 * @param controls The control qubits of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void
createOneTargetTwoParameter(JeffOpType& op, ConversionPatternRewriter& rewriter,
                            const llvm::SmallVector<Value>& parameters,
                            const llvm::SmallVector<Value>& controls,
                            Value target) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, target, parameters[0],
                                           parameters[1]);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), innerTargets[0],
                                     parameters[0], parameters[1]);
      return {qcoOp.getQubitOut()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, {target},
                                          lambda);
  }
}

/**
 * @brief Creates a one-target, three-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param target The target qubit of the operation
 * @param parameters The parameters of the operation
 * @param controls The control qubits of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void createOneTargetThreeParameter(
    JeffOpType& op, ConversionPatternRewriter& rewriter,
    const llvm::SmallVector<Value>& parameters,
    const llvm::SmallVector<Value>& controls, Value target) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, target, parameters[0],
                                           parameters[1], parameters[2]);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp =
          QCOOpType::create(rewriter, op.getLoc(), innerTargets[0],
                            parameters[0], parameters[1], parameters[2]);
      return {qcoOp.getQubitOut()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, {target},
                                          lambda);
  }
}

/**
 * @brief Creates a two-target, zero-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param controls The control qubits of the operation
 * @param targets The target qubits of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void
createTwoTargetZeroParameter(JeffOpType& op,
                             ConversionPatternRewriter& rewriter,
                             const llvm::SmallVector<Value>& controls,
                             const llvm::SmallVector<Value>& targets) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, targets[0], targets[1]);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), innerTargets[0],
                                     innerTargets[1]);
      return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, targets,
                                          lambda);
  }
}

/**
 * @brief Creates a two-target, one-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param targets The target qubits of the operation
 * @param parameter The parameter of the operation
 * @param controls The control qubits of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void
createTwoTargetOneParameter(JeffOpType& op, ConversionPatternRewriter& rewriter,
                            Value parameter,
                            const llvm::SmallVector<Value>& controls,
                            const llvm::SmallVector<Value>& targets) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, targets[0], targets[1],
                                           parameter);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp = QCOOpType::create(rewriter, op.getLoc(), innerTargets[0],
                                     innerTargets[1], parameter);
      return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, targets,
                                          lambda);
  }
}

/**
 * @brief Creates a two-target, two-parameter QCO operation from a Jeff
 * operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param rewriter The pattern rewriter
 * @param targets The target qubits of the operation
 * @param parameters The parameters of the operation
 * @param controls The control qubits of the operation
 */
template <typename QCOOpType, typename JeffOpType>
static void
createTwoTargetTwoParameter(JeffOpType& op, ConversionPatternRewriter& rewriter,
                            const llvm::SmallVector<Value>& parameters,
                            const llvm::SmallVector<Value>& controls,
                            const llvm::SmallVector<Value>& targets) {
  if (op.getNumCtrls() == 0 && !op.getIsAdjoint()) {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, targets[0], targets[1],
                                           parameters[0], parameters[1]);
  } else {
    auto lambda = [&](ValueRange innerTargets) -> llvm::SmallVector<Value> {
      auto qcoOp =
          QCOOpType::create(rewriter, op.getLoc(), innerTargets[0],
                            innerTargets[1], parameters[0], parameters[1]);
      return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
    };
    createModified<QCOOpType, JeffOpType>(op, rewriter, controls, targets,
                                          lambda);
  }
}

/**
 * @brief Converts a one-target, zero-parameter Jeff operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam JeffOpAdaptorType The OpAdaptor type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param adaptor The OpAdaptor of the Jeff operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename JeffOpType, typename JeffOpAdaptorType>
static LogicalResult
convertOneTargetZeroParameter(JeffOpType& op, JeffOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  if (op.getPower() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Operations with power != 1 are not yet supported");
  }

  createOneTargetZeroParameter<QCOOpType>(
      op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInQubit());

  return success();
}

/**
 * @brief Converts a one-target, one-parameter Jeff operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam JeffOpAdaptorType The OpAdaptor type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param adaptor The OpAdaptor of the Jeff operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename JeffOpType, typename JeffOpAdaptorType>
static LogicalResult
convertOneTargetOneParameter(JeffOpType& op, JeffOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  if (op.getPower() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Operations with power != 1 are not yet supported");
  }

  createOneTargetOneParameter<QCOOpType>(op, rewriter, op.getRotation(),
                                         adaptor.getInCtrlQubits(),
                                         adaptor.getInQubit());

  return success();
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
 * @brief Cleans up the main function after conversion
 *
 * @details
 * The `passthrough` attribute is added, the trivial return operation is
 * replaced, and the function type is fixed.
 *
 * @param main The main function to clean up
 * @return LogicalResult Success or failure of the cleanup
 */
static LogicalResult cleanUpMain(func::FuncOp main) {
  if (main.getBlocks().size() != 1) {
    return failure();
  }
  auto* block = &main.getBlocks().front();

  auto* ctx = main.getContext();
  auto loc = main.getLoc();
  OpBuilder builder(ctx);

  // Add passthrough attribute
  auto entryPointAttr = StringAttr::get(ctx, "entry_point");
  main->setAttr("passthrough", ArrayAttr::get(ctx, {entryPointAttr}));

  // Remove trivial return operation
  auto* returnOp = block->getTerminator();
  if (!llvm::isa<func::ReturnOp>(returnOp)) {
    return failure();
  }
  returnOp->erase();

  // Add return operation
  builder.setInsertionPointToStart(block);
  auto constOp =
      arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(0));

  builder.setInsertionPointToEnd(block);
  func::ReturnOp::create(builder, loc, constOp.getResult());

  // Fix return type
  main.setType(FunctionType::get(ctx, {}, {builder.getI64Type()}));

  return success();
}

/**
 * @brief Cleans up the module after conversion
 *
 * @details
 * The main function is identified and cleaned up, and module attributes are
 * removed.
 *
 * @param op The module operation to clean up
 * @return LogicalResult Success or failure of the cleanup
 */
static LogicalResult cleanUp(Operation* op) {
  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module) {
    return failure();
  }

  auto entryPointAttr = module->getAttr("jeff.entrypoint");
  if (!entryPointAttr) {
    return failure();
  }
  auto entryPoint = llvm::cast<IntegerAttr>(entryPointAttr).getUInt();

  auto stringsAttr = module->getAttr("jeff.strings");
  if (!stringsAttr) {
    return failure();
  }
  auto strings = llvm::cast<ArrayAttr>(stringsAttr);

  if (entryPoint >= strings.size()) {
    return failure();
  }
  auto mainName = llvm::cast<mlir::StringAttr>(strings[entryPoint]).getValue();

  bool mainFound = false;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.getSymName() == mainName) {
      mainFound = true;
      if (cleanUpMain(funcOp).failed()) {
        return failure();
      }
    }
  }

  if (!mainFound) {
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
 * @brief Converts jeff.qubit_free to qco.reset + qco.dealloc
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit
 * qco.dealloc %q_out : !qco.qubit
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
    rewriter.replaceOpWithNewOp<qco::DeallocOp>(op, resetOp.getQubitOut());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_free_zero to qco.dealloc
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free_zero %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * qco.dealloc %q : !qco.qubit
 * ```
 */
struct ConvertJeffQubitFreeZeroOpToQCO final
    : OpConversionPattern<jeff::QubitFreeZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitFreeZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qco::DeallocOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_measure to qco.measure + qco.dealloc
 *
 * @par Example:
 * ```mlir
 * %result = jeff.qubit_measure %q_in : !i1
 * ```
 * is converted to
 * ```mlir
 * %q_out, %result = qco.measure %q_in : !qco.qubit
 * qco.dealloc %q_out : !qco.qubit
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
    qco::DeallocOp::create(rewriter, loc, measureOp.getQubitOut());
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

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QCO)          \
  struct ConvertJeff##OP_CLASS_JEFF##ToQCO final                               \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<qco::OP_CLASS_QCO>(op, adaptor,     \
                                                              rewriter);       \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IOp, IdOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, XOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, YOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, ZOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, HOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, SOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, TOp)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QCO)           \
  struct ConvertJeff##OP_CLASS_JEFF##ToQCO final                               \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<qco::OP_CLASS_QCO>(op, adaptor,      \
                                                             rewriter);        \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RxOp, RXOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RyOp, RYOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RzOp, RZOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(R1Op, POp)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

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

    createOneTargetThreeParameter<qco::UOp>(
        op, rewriter, {op.getTheta(), op.getPhi(), op.getLambda()},
        adaptor.getInCtrlQubits(), adaptor.getInQubit());

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

    createTwoTargetZeroParameter<qco::SWAPOp>(
        op, rewriter, adaptor.getInCtrlQubits(),
        {adaptor.getInQubitOne(), adaptor.getInQubitTwo()});

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

    auto name = op.getName();
    if (name == "sx") {
      if (op.getInTargetQubits().size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "Custom SX operations must have exactly one target qubit");
      }
      createOneTargetZeroParameter<qco::SXOp>(op, rewriter,
                                              adaptor.getInCtrlQubits(),
                                              adaptor.getInTargetQubits()[0]);
    } else if (name == "r") {
      if (op.getInTargetQubits().size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "Custom R operations must have exactly one target qubit");
      }
      createOneTargetTwoParameter<qco::ROp>(op, rewriter, op.getParams(),
                                            adaptor.getInCtrlQubits(),
                                            adaptor.getInTargetQubits()[0]);
    } else if (name == "iswap") {
      createTwoTargetZeroParameter<qco::iSWAPOp>(
          op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits());
    } else if (name == "dcx") {
      createTwoTargetZeroParameter<qco::DCXOp>(
          op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits());
    } else if (name == "ecr") {
      createTwoTargetZeroParameter<qco::ECROp>(
          op, rewriter, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits());
    } else if (name == "xx_minus_yy") {
      createTwoTargetTwoParameter<qco::XXMinusYYOp>(
          op, rewriter, op.getParams(), adaptor.getInCtrlQubits(),
          adaptor.getInTargetQubits());
    } else if (name == "xx_plus_yy") {
      createTwoTargetTwoParameter<qco::XXPlusYYOp>(op, rewriter, op.getParams(),
                                                   adaptor.getInCtrlQubits(),
                                                   adaptor.getInTargetQubits());
    } else if (name == "barrier") {
      createBarrierOp(op, adaptor, rewriter);
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported custom operation: " +
                                                 name);
    }

    return success();
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
      createTwoTargetOneParameter<qco::RXXOp>(op, rewriter, op.getRotation(),
                                              controls, targets);
    } else if (pauliGates[0] == 2 && pauliGates[1] == 2) {
      createTwoTargetOneParameter<qco::RYYOp>(op, rewriter, op.getRotation(),
                                              controls, targets);
    } else if (pauliGates[0] == 3 && pauliGates[1] == 1) {
      createTwoTargetOneParameter<qco::RZXOp>(op, rewriter, op.getRotation(),
                                              controls, targets);
    } else if (pauliGates[0] == 3 && pauliGates[1] == 3) {
      createTwoTargetOneParameter<qco::RZZOp>(op, rewriter, op.getRotation(),
                                              controls, targets);
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported PPR operation");
    }

    return success();
  }
};

/**
 * @brief Converts jeff.int_const1 to arith.constant
 *
 * @par Example:
 * ```mlir
 * %0 = jeff.int_const1(true) : i1
 * ```
 * is converted to
 * ```mlir
 * %0 = arith.constant true : i1
 * ```
 */
struct ConvertJeffIntConst1OpToArith final
    : OpConversionPattern<jeff::IntConst1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::IntConst1Op op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
    return success();
  }
};

/**
 * @brief Converts jeff.int_const64 to arith.constant
 *
 * @par Example:
 * ```mlir
 * %0 = jeff.int_const64(3) : i64
 * ```
 * is converted to
 * ```mlir
 * %0 = arith.constant 3 : i64
 * ```
 */
struct ConvertJeffIntConst64OpToArith final
    : OpConversionPattern<jeff::IntConst64Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::IntConst64Op op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
    return success();
  }
};

/**
 * @brief Converts jeff.float_const64 to arith.constant
 *
 * @par Example:
 * ```mlir
 * %0 = jeff.float_const64(0.3) : f64
 * ```
 * is converted to
 * ```mlir
 * %0 = arith.constant 0.3 : f64
 * ```
 */
struct ConvertJeffFloatConst64OpToArith final
    : OpConversionPattern<jeff::FloatConst64Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::FloatConst64Op op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
    return success();
  }
};

/**
 * @brief Type converter for Jeff-to-QCO conversion
 *
 * @details
 * Converts `!jeff.qubit` to `!qco.qubit`.
 */
class JeffToQCOTypeConverter final : public TypeConverter {
public:
  explicit JeffToQCOTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](jeff::QubitType /*type*/) -> Type {
      return qco::QubitType::get(ctx);
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
    target.addLegalDialect<QCODialect, arith::ArithDialect>();

    // Register operation conversion patterns
    patterns
        .add<ConvertJeffQubitAllocOpToQCO, ConvertJeffQubitFreeOpToQCO,
             ConvertJeffQubitFreeZeroOpToQCO, ConvertJeffQubitMeasureOpToQCO,
             ConvertJeffQubitMeasureNDOpToQCO, ConvertJeffQubitResetOpToQCO,
             ConvertJeffGPhaseOpToQCO, ConvertJeffIOpToQCO, ConvertJeffXOpToQCO,
             ConvertJeffYOpToQCO, ConvertJeffZOpToQCO, ConvertJeffHOpToQCO,
             ConvertJeffSOpToQCO, ConvertJeffTOpToQCO, ConvertJeffRxOpToQCO,
             ConvertJeffRyOpToQCO, ConvertJeffRzOpToQCO, ConvertJeffR1OpToQCO,
             ConvertJeffUOpToQCO, ConvertJeffSwapOpToQCO,
             ConvertJeffCustomOpToQCO, ConvertJeffPPROpToQCO,
             ConvertJeffIntConst1OpToArith, ConvertJeffIntConst64OpToArith,
             ConvertJeffFloatConst64OpToArith>(typeConverter, context);

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

} // namespace mlir
