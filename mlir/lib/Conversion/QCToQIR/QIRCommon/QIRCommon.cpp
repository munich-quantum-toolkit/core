/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQIR/QIRCommon/QIRCommon.h"

#include "mlir/Conversion/GateTable.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace mlir {
using namespace qc;
using namespace qir;

LogicalResult LoweringState::ensureAllocationMode(AllocationMode requested,
                                                  Operation* op) {
  if (allocationMode == AllocationMode::Unset) {
    allocationMode = requested;
    return success();
  }
  if (allocationMode == requested) {
    return success();
  }
  return op->emitOpError(
      "cannot mix static and dynamic qubit allocation modes in conversion");
}

QCToQIRTypeConverter::QCToQIRTypeConverter(MLIRContext* ctx)
    : LLVMTypeConverter(ctx) {
  addConversion([ctx](QubitType) { return LLVM::LLVMPointerType::get(ctx); });
  addConversion([ctx](MemRefType type) -> Type {
    if (isa<QubitType>(type.getElementType())) {
      return LLVM::LLVMPointerType::get(ctx);
    }
    return type;
  });
};

/**
 * @brief Helper to convert a QC operation to a LLVM CallOp
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOpAdaptorType The OpAdaptor type of the QC operation
 * @param op The QC operation instance to convert
 * @param adaptor The OpAdaptor of the QC operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @param numTargets The number of targets
 * @param numParams The number of parameters
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOpAdaptorType>
static LogicalResult
convertUnitaryToCallOp(QCOpType& op, QCOpAdaptorType& adaptor,
                       ConversionPatternRewriter& rewriter, MLIRContext* ctx,
                       LoweringState& state, StringRef fnName,
                       size_t numTargets, size_t numParams) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> controls =
      inCtrlOp != 0 ? state.controls : SmallVector<Value>{};
  const size_t numCtrls = controls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numParams + numCtrls + numTargets);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto floatType = Float64Type::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  for (size_t i = 0; i < numTargets; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add parameter types
  for (size_t i = 0; i < numParams; ++i) {
    argumentTypes.push_back(floatType);
  }

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numParams + numCtrls + numTargets);
  operands.append(controls.begin(), controls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.inCtrlOp--;
    if (state.inCtrlOp == 0) {
      state.controls.clear();
    }
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

namespace {

/**
 * @brief Generic converter for unitary QC ops to QIR calls.
 *
 * @details
 * Many QC gates lower to a QIR runtime call where the callee name depends on
 * the number of active controls. This helper factors out that boilerplate
 * without relying on preprocessor macros.
 *
 * @par Examples
 * The examples below illustrate the lowering shapes for unitary gates that
 * are registered through `MQT_GATE_TABLE` in `populateQCToQIRPatterns`.
 *
 * @par One target, zero parameters
 * ```mlir
 * qc.x %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr) -> ()
 * ```
 *
 * @par One target, one parameter
 * ```mlir
 * qc.rx(%theta) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__rx__body(%q, %theta) : (!llvm.ptr, f64) -> ()
 * ```
 *
 * @par One target, two parameters
 * ```mlir
 * qc.r(%theta, %phi) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__r__body(%q, %theta, %phi) : (!llvm.ptr, f64, f64)
 * -> ()
 * ```
 *
 * @par One target, three parameters
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__u3__body(%q, %theta, %phi, %lambda)
 *     : (!llvm.ptr, f64, f64, f64) -> ()
 * ```
 *
 * @par Two targets, zero parameters
 * ```mlir
 * qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) ->
 * ()
 * ```
 *
 * @par Two targets, one parameter
 * ```mlir
 * qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__rxx__body(%q0, %q1, %theta)
 *     : (!llvm.ptr, !llvm.ptr, f64) -> ()
 * ```
 *
 * @par Two targets, two parameters
 * ```mlir
 * qc.xx_plus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__xx_plus_yy__body(%q0, %q1, %theta, %beta)
 *     : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
 * ```
 *
 * @tparam OpType The QC operation type to convert
 * @tparam NumTargets Number of target qubits for this operation
 * @tparam NumParams Number of floating-point parameters for this operation
 * @tparam GetFnName Function that maps numCtrls -> QIR function name
 */
template <typename OpType, std::size_t NumTargets, std::size_t NumParams,
          auto GetFnName>
struct ConvertQCUnitaryOpQIR : StatefulOpConversionPattern<OpType> {
  using StatefulOpConversionPattern<OpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, OpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    const auto inCtrlOp = state.inCtrlOp;
    const size_t numCtrls = inCtrlOp != 0 ? state.controls.size() : 0;
    const auto fnName = GetFnName(numCtrls);
    return convertUnitaryToCallOp(op, adaptor, rewriter, this->getContext(),
                                  state, fnName, NumTargets, NumParams);
  }
};

/**
 * @brief Converts qc.static to llvm.inttoptr
 *
 * @details
 * Converts a static qubit reference to an LLVM pointer by creating a constant
 * with the qubit index and converting it to a pointer. The pointer is cached
 * in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q0 = qc.static 0 : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQCStaticOp final : StatefulOpConversionPattern<StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto index = static_cast<int64_t>(op.getIndex());
    auto& state = getState();
    if (failed(state.ensureAllocationMode(AllocationMode::Static,
                                          op.getOperation()))) {
      return failure();
    }

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Switch to entry block
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    // Get or create a pointer to the qubit
    Value qubit;
    if (const auto it = state.staticQubits.find(index);
        it != state.staticQubits.end()) {
      // Reuse existing pointer
      qubit = it->second;
    } else {
      // Create and cache for reuse
      qubit = createPointerFromIndex(rewriter, op.getLoc(), index);
      state.staticQubits.try_emplace(index, qubit);
    }
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

// GPhaseOp

/**
 * @brief Converts qc.gphase to QIR gphase
 *
 * @par Example:
 * ```mlir
 * qc.gphase(%theta)
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__gphase__body(%theta) : (f64) -> ()
 * ```
 */
struct ConvertQCGPhaseOp final : StatefulOpConversionPattern<GPhaseOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(GPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (state.inCtrlOp != 0) {
      return op.emitError("Controlled GPhaseOps cannot be converted to QIR");
    }
    return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(), state,
                                  QIR_GPHASE, 0, 1);
  }
};

// BarrierOp

/**
 * @brief Erases qc.barrier operation, as it is a no-op in QIR
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Inlines qc.ctrl region removes the operation
 */
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    if (state.inCtrlOp != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "Nested CtrlOps are not supported");
    }

    // Update modifier information
    state.inCtrlOp = op.getNumBodyUnitaries();
    state.controls = llvm::to_vector(adaptor.getControls());

    // Inline block and remove operation
    rewriter.inlineBlockBefore(&op.getRegion().front(), op,
                               adaptor.getTargets());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erases qc.yield operation
 */
struct ConvertQCYieldOp final : StatefulOpConversionPattern<YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                   LoweringState& state) {
  OpBuilder builder(ctx);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  builder.setInsertionPointToStart(state.entryBlock);

  auto initSig = LLVM::LLVMFunctionType::get(voidType, ptrType);
  auto initDec =
      getOrCreateFunctionDeclaration(builder, main, QIR_INITIALIZE, initSig);
  auto zero = LLVM::ZeroOp::create(builder, main->getLoc(), ptrType);
  LLVM::CallOp::create(builder, main->getLoc(), initDec, zero.getResult());
}

void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                        LoweringState& state) {
  OpBuilder builder(ctx);
  builder.setInsertionPoint(&main.getBlocks().back().back());
  emitOutputRecording(builder, main,
                      llvm::to_vector(llvm::make_second_range(state.cregs)),
                      state.staticResults);
}

void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
  // Note: `MQT_GATE_TABLE` is defined in `mlir/Conversion/GateTable.h`.
#define MQT_ADD_QC_TO_QIR_UNITARY(KEY, TARGETS, PARAMS, QCO_OP, QC_OP, QIR_FN) \
  patterns.add<ConvertQCUnitaryOpQIR<QC_OP, (TARGETS), (PARAMS), &(QIR_FN)>>(  \
      typeConverter, ctx, &state);
  MQT_GATE_TABLE(MQT_ADD_QC_TO_QIR_UNITARY)
#undef MQT_ADD_QC_TO_QIR_UNITARY

  patterns.add<ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCYieldOp,
               ConvertQCStaticOp, ConvertQCGPhaseOp>(typeConverter, ctx,
                                                     &state);
}

Value resolveMeasurementResult(LoweringState& state, Operation* op,
                               ConversionPatternRewriter& rewriter) {
  if (const auto it = state.returnedCregs.find(op);
      it != state.returnedCregs.end()) {
    const auto [allocOp, index] = it->second;
    auto& reg = state.cregs[allocOp];
    // Dynamic indices are handled by `resolveDynamicMeasurementResult`
    const auto bit = getConstantIntValue(index);
    assert(bit && "index must be a constant");
    return reg.results[*bit];
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(state.entryBlock->getTerminator());
  const auto index = static_cast<int64_t>(state.staticResults.size());
  const auto record = state.returnedStaticResults.contains(op);
  auto result = createPointerFromIndex(rewriter, op->getLoc(), index);
  state.staticResults.try_emplace(
      index, qir::StaticResult{.pointer = result, .record = record});
  return result;
}

void stripReturnedMeasurements(Operation* moduleOp, LoweringState& state) {
  moduleOp->walk([&](func::FuncOp funcOp) {
    // Check whether the given function is the main entrypoint
    auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough");
    bool isEntryPoint = false;
    if (passthrough) {
      isEntryPoint = llvm::any_of(passthrough, [](Attribute attr) {
        auto strAttr = dyn_cast<StringAttr>(attr);
        return strAttr && strAttr.getValue() == "entry_point";
      });
    }
    if (!isEntryPoint) {
      return;
    }

    funcOp.walk([&](func::ReturnOp returnOp) {
      SmallVector<Value> keptOperands;
      SmallVector<Type> keptReturnTypes;

      for (auto operand : returnOp.getOperands()) {
        if (auto measureOp = operand.getDefiningOp<MeasureOp>()) {
          state.returnedStaticResults.insert(measureOp.getOperation());
        } else if (auto allocOp = operand.getDefiningOp<memref::AllocOp>()) {
          const std::string label = "c" + std::to_string(state.cregs.size());
          const auto size = allocOp.getType().getShape()[0];
          auto& reg =
              state.cregs.try_emplace(allocOp.getOperation()).first->second;
          reg.label = label;
          reg.size = size;
          reg.results.assign(size, Value{});
          for (auto* user : allocOp.getResult().getUsers()) {
            auto storeOp = dyn_cast<memref::StoreOp>(user);
            if (!storeOp) {
              continue;
            }
            auto measureOp =
                storeOp.getValueToStore().getDefiningOp<MeasureOp>();
            if (!measureOp) {
              continue;
            }
            state.returnedCregs.try_emplace(measureOp.getOperation(),
                                            allocOp.getOperation(),
                                            storeOp.getIndices()[0]);
          }
        } else {
          keptOperands.push_back(operand);
          keptReturnTypes.push_back(operand.getType());
        }
      }

      if (keptOperands.empty() && !returnOp.getOperands().empty()) {
        OpBuilder builder(returnOp);
        auto zero =
            arith::ConstantIntOp::create(builder, returnOp.getLoc(), 0, 64);
        keptOperands.push_back(zero);
        keptReturnTypes.push_back(zero.getType());
      }

      returnOp.getOperandsMutable().assign(keptOperands);

      funcOp.setFunctionType(FunctionType::get(
          funcOp.getContext(), funcOp.getFunctionType().getInputs(),
          keptReturnTypes));
    });
  });
}

} // namespace mlir
