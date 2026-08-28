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

#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace mlir {
using namespace qc;
using namespace qir;

template <typename Callback>
static void walkOperationsIteratively(Operation* root, Callback&& callback) {
  SmallVector<Operation*> worklist{root};
  while (!worklist.empty()) {
    Operation* operation = worklist.pop_back_val();
    callback(operation);
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (Operation& nested : block) {
          worklist.push_back(&nested);
        }
      }
    }
  }
}

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
  addConversion(
      [ctx](cbit::RegisterType) { return LLVM::LLVMPointerType::get(ctx); });
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
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @param numTargets The number of targets
 * @param numParams The number of parameters
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOpAdaptorType>
static LogicalResult
convertUnitaryToCallOp(QCOpType& op, QCOpAdaptorType& adaptor,
                       ConversionPatternRewriter& rewriter,
                       LoweringState& state, StringRef fnName,
                       const size_t numTargets, const size_t numParams) {
  // Query state for modifier information
  const SmallVector<Value> controls =
      state.inCtrlOp ? state.controls : SmallVector<Value>{};
  auto convertedOperands = adaptor.getOperands();
  auto targets = convertedOperands.take_front(numTargets);
  auto parameters = convertedOperands.drop_front(numTargets);
  assert(parameters.size() == numParams && "unexpected gate parameter count");

  // Clean up modifier information
  if (state.inCtrlOp) {
    state.inCtrlOp = false;
    state.controls.clear();
  }

  qir::emitQISCall(rewriter, op, op.getLoc(), parameters, controls, targets,
                   fnName);
  rewriter.eraseOp(op);
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
 * are registered through the shared QIR gate table in
 * `populateQCToQIRPatterns`.
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
 * llvm.call @__quantum__qis__rx__body(%theta, %q) : (f64, !llvm.ptr) -> ()
 * ```
 *
 * @par One target, two parameters
 * ```mlir
 * qc.r(%theta, %phi) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__prx__body(%theta, %phi, %q)
 *     : (f64, f64, !llvm.ptr) -> ()
 * ```
 *
 * @par One target, three parameters
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__u3__body(%theta, %phi, %lambda, %q)
 *     : (f64, f64, f64, !llvm.ptr) -> ()
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
 * llvm.call @__quantum__qis__rxx__body(%theta, %q0, %q1)
 *     : (f64, !llvm.ptr, !llvm.ptr) -> ()
 * ```
 *
 * @par Two targets, two parameters
 * ```mlir
 * qc.xx_plus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__xx_plus_yy__body(%theta, %beta, %q0, %q1)
 *     : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()
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
    const size_t numCtrls = state.inCtrlOp ? state.controls.size() : 0;
    const auto fnName = GetFnName(numCtrls);
    return convertUnitaryToCallOp(op, adaptor, rewriter, state, fnName,
                                  NumTargets, NumParams);
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
    if (state.inCtrlOp) {
      return op.emitError("Controlled GPhaseOps cannot be converted to QIR");
    }
    return convertUnitaryToCallOp(op, adaptor, rewriter, state, QIR_GPHASE, 0,
                                  1);
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

    if (state.inCtrlOp) {
      return rewriter.notifyMatchFailure(op,
                                         "Nested CtrlOps are not supported");
    }

    if (op.getNumBodyUnitaries() > 1) {
      return rewriter.notifyMatchFailure(
          op, "CtrlOps with multiple body unitaries are not supported. Run the "
              "unroll-modifiers pass before the conversion");
    }

    // Empty control bodies and controls around no-op unitaries do not need
    // lowering state. In particular, barrier lowering erases the operation
    // without consuming that state, which would otherwise control the next
    // gate.
    auto bodyUnitary = op.getNumBodyUnitaries() == 1 ? op.getBodyUnitary(0)
                                                     : UnitaryOpInterface{};
    if (bodyUnitary && !isa<BarrierOp, IdOp>(bodyUnitary.getOperation())) {
      state.inCtrlOp = true;
      state.controls = llvm::to_vector(adaptor.getControls());
    }

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
  SmallVector<qir::ClassicalRegister> returnedRegisters;
  returnedRegisters.reserve(state.returnedCregs.size());
  for (const auto registerIndex : state.returnedCregs) {
    returnedRegisters.push_back(state.cregs[registerIndex]);
  }
  emitOutputRecording(builder, main, returnedRegisters, state.staticResults);
}

void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  patterns.add<ConvertQCUnitaryOpQIR<qc::KEY##Op, (TARGETS), (PARAMS),         \
                                     &getFnName##GETTER>>(typeConverter, ctx,  \
                                                          &state);
#include "mlir/Conversion/GateTable.def"

  patterns.add<ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCYieldOp,
               ConvertQCStaticOp, ConvertQCGPhaseOp>(typeConverter, ctx,
                                                     &state);
}

Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(state.entryBlock->getTerminator());
  const auto index = static_cast<int64_t>(state.staticResults.size());
  const auto record = state.returnedStaticResults.contains(op);
  auto result = createPointerFromIndex(rewriter, op->getLoc(), index);
  state.staticResults.try_emplace(
      index, qir::StaticResult{.pointer = result, .record = record});
  return result;
}

LogicalResult validateQIRConversionInput(ModuleOp moduleOp,
                                         bool requireSingleBlock,
                                         LoweringState& state) {
  if (failed(mqt::verifyProgramMetadata(moduleOp))) {
    return failure();
  }
  for (Operation& operation : moduleOp.getBody()->getOperations()) {
    const auto symbol = SymbolTable::getSymbolName(&operation);
    if (!symbol || !symbol.getValue().starts_with("__quantum__")) {
      continue;
    }
    if (auto function = dyn_cast<func::FuncOp>(operation);
        function && function.isExternal()) {
      continue;
    }
    if (auto function = dyn_cast<LLVM::LLVMFuncOp>(operation);
        function && function.isExternal()) {
      continue;
    }
    return operation.emitError()
           << "QIR conversion reserves runtime symbol " << symbol.getValue()
           << " for a function declaration";
  }
  func::FuncOp entryPoint;
  for (auto function : moduleOp.getOps<func::FuncOp>()) {
    if (!mqt::isEntryPoint(function)) {
      continue;
    }
    if (entryPoint) {
      moduleOp.emitError("QIR conversion requires exactly one program entry "
                         "function marked with mqt.entry_point");
      return failure();
    }
    entryPoint = function;
  }
  if (!entryPoint) {
    moduleOp.emitError(
        "QIR conversion requires a program entry function marked with "
        "mqt.entry_point");
    return failure();
  }
  if (entryPoint.isExternal()) {
    entryPoint.emitError("QIR conversion requires a defined entry function");
    return failure();
  }
  if (entryPoint.getNumArguments() != 0) {
    entryPoint.emitError(
        "QIR conversion does not support entry-function arguments");
    return failure();
  }
  if (requireSingleBlock && !entryPoint.getBody().hasOneBlock()) {
    entryPoint.emitError(
        "QIR Base Profile conversion requires a single-block entry function");
    return failure();
  }
  if (requireSingleBlock &&
      !isa<func::ReturnOp>(entryPoint.getBody().front().getTerminator())) {
    entryPoint.emitError(
        "QIR Base Profile conversion requires straight-line control flow "
        "ending in func.return");
    return failure();
  }

  bool invalid = false;
  walkOperationsIteratively(moduleOp, [&](Operation* operation) {
    if (invalid || operation == moduleOp || operation == entryPoint) {
      return;
    }
    const auto dialect = operation->getName().getDialectNamespace();
    if (dialect != qc::QCDialect::getDialectNamespace() &&
        dialect != cbit::CBitDialect::getDialectNamespace() &&
        dialect != memref::MemRefDialect::getDialectNamespace()) {
      return;
    }
    if (operation->getParentOfType<func::FuncOp>() != entryPoint) {
      operation->emitError(
          "QIR conversion only supports QC, CBit, and MemRef operations in "
          "the program entry function");
      invalid = true;
    }
  });
  if (invalid) {
    return failure();
  }

  if (requireSingleBlock) {
    walkOperationsIteratively(moduleOp, [&](Operation* operation) {
      if (invalid || operation == moduleOp || isa<CtrlOp>(operation) ||
          !isa<RegionBranchOpInterface>(operation)) {
        return;
      }
      operation->emitError(
          "QIR Base Profile conversion does not support region-based control "
          "flow other than qc.ctrl");
      invalid = true;
    });
    if (invalid) {
      return failure();
    }

    walkOperationsIteratively(entryPoint, [&](Operation* operation) {
      if (invalid || operation == entryPoint) {
        return;
      }
      const auto dialect = operation->getName().getDialectNamespace();
      if (dialect != qc::QCDialect::getDialectNamespace() &&
          dialect != cbit::CBitDialect::getDialectNamespace() &&
          dialect != memref::MemRefDialect::getDialectNamespace()) {
        return;
      }
      for (Operation* parent = operation->getParentOp();
           parent && parent != entryPoint; parent = parent->getParentOp()) {
        if (parent->getNumRegions() == 0 || isa<CtrlOp>(parent)) {
          continue;
        }
        operation->emitError(
            "QIR Base Profile conversion does not support QC, CBit, or "
            "MemRef operations nested in preserved region operations");
        invalid = true;
        return;
      }
    });
    if (invalid) {
      return failure();
    }
  }

  walkOperationsIteratively(entryPoint, [&](Operation* operation) {
    if (auto op = dyn_cast<memref::AllocOp>(operation)) {
      const auto type = op.getType();
      if (type.getRank() != 1 || !isa<QubitType>(type.getElementType())) {
        op.emitError("QIR conversion only supports generic memrefs for "
                     "one-dimensional qc.qubit registers; use CBit for "
                     "classical registers");
        invalid = true;
      }
    }
    if (auto op = dyn_cast<memref::LoadOp>(operation)) {
      const auto type = cast<MemRefType>(op.getMemref().getType());
      if (type.getRank() != 1 || op.getIndices().size() != 1 ||
          !isa<QubitType>(type.getElementType())) {
        op.emitError("QIR conversion only supports one-dimensional qubit "
                     "register loads with exactly one index");
        invalid = true;
      }
    }
  });
  if (invalid) {
    return failure();
  }

  if (!requireSingleBlock) {
    Operation* staticAllocation = nullptr;
    Operation* dynamicAllocation = nullptr;
    walkOperationsIteratively(entryPoint, [&](Operation* operation) {
      if (isa<StaticOp>(operation)) {
        staticAllocation = operation;
      } else if (isa<AllocOp, memref::AllocOp>(operation)) {
        dynamicAllocation = operation;
      }
    });
    if (staticAllocation && dynamicAllocation) {
      dynamicAllocation->emitError(
          "QIR Adaptive Profile conversion cannot mix static and dynamic "
          "qubit allocations");
      return failure();
    }
    return success();
  }

  SmallVector<std::pair<Value, int64_t>> loadedRegisterElements;
  uint64_t freshStaticQubitIds = 0;
  walkOperationsIteratively(entryPoint, [&](Operation* operation) {
    if (auto op = dyn_cast<StaticOp>(operation)) {
      const auto rawIndex = op.getIndex();
      if (rawIndex >=
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        op.emitError("static qubit index exceeds the supported QIR range");
        invalid = true;
        return;
      }
      const auto index = static_cast<int64_t>(rawIndex);
      state.nextStaticQubitIndex =
          std::max(state.nextStaticQubitIndex, index + 1);
      return;
    }
    if (auto op = dyn_cast<AllocOp>(operation)) {
      ++freshStaticQubitIds;
      return;
    }
    if (auto op = dyn_cast<memref::AllocOp>(operation)) {
      const auto type = op.getType();
      if (type.isDynamicDim(0)) {
        op.emitError("QIR Base Profile conversion requires statically sized "
                     "one-dimensional qc.qubit memrefs");
        invalid = true;
      }
      return;
    }
    auto op = dyn_cast<memref::LoadOp>(operation);
    if (!op) {
      return;
    }
    auto allocation = op.getMemref().getDefiningOp<memref::AllocOp>();
    const auto index = getConstantIntValue(op.getIndices().front());
    if (!allocation || !index || *index < 0 ||
        *index >= allocation.getType().getDimSize(0)) {
      op.emitError("QIR Base Profile conversion requires a constant, "
                   "in-bounds index into a direct qubit-register allocation");
      invalid = true;
      return;
    }
    const std::pair<Value, int64_t> element{op.getMemref(), *index};
    if (!llvm::is_contained(loadedRegisterElements, element)) {
      loadedRegisterElements.push_back(element);
      ++freshStaticQubitIds;
    }
  });
  if (invalid) {
    return failure();
  }
  const auto availableStaticQubitIds = static_cast<uint64_t>(
      std::numeric_limits<int64_t>::max() - state.nextStaticQubitIndex);
  if (freshStaticQubitIds > availableStaticQubitIds) {
    entryPoint.emitError(
        "QIR Base Profile conversion exhausts the supported static qubit "
        "index range");
    return failure();
  }
  return success();
}

LogicalResult prepareClassicalResults(Operation* moduleOp,
                                      LoweringState& state) {
  constexpr uint64_t maxClassicalResultSlots = 1U << 20;
  uint64_t numClassicalResultSlots = 0;
  bool exceedsResultLimit = false;
  walkOperationsIteratively(moduleOp, [&](Operation* operation) {
    auto allocOp = dyn_cast<cbit::AllocOp>(operation);
    if (!allocOp || exceedsResultLimit) {
      return;
    }
    const auto width = allocOp.getResult().getType().getWidth();
    if (width <= 0 || static_cast<uint64_t>(width) >
                          maxClassicalResultSlots - numClassicalResultSlots) {
      allocOp.emitError() << "QIR conversion supports at most "
                          << maxClassicalResultSlots
                          << " classical result slots per module";
      exceedsResultLimit = true;
      return;
    }
    numClassicalResultSlots += static_cast<uint64_t>(width);
  });
  if (exceedsResultLimit) {
    return failure();
  }

  struct ReturnRewrite {
    func::FuncOp function;
    func::ReturnOp returnOp;
    SmallVector<Value> keptOperands;
    SmallVector<Type> keptTypes;
    bool needsStatusResult = false;
    bool recordsClassicalOutput = false;
  };

  bool hasInvalidMemory = false;
  SmallVector<cbit::StoreOp> consumedStores;
  SmallVector<ReturnRewrite> returnRewrites;
  SmallVector<func::FuncOp> entryPoints;
  walkOperationsIteratively(moduleOp, [&](Operation* operation) {
    if (auto function = dyn_cast<func::FuncOp>(operation);
        function && mqt::isEntryPoint(function)) {
      entryPoints.push_back(function);
    }
  });
  for (auto funcOp : entryPoints) {

    walkOperationsIteratively(funcOp, [&](Operation* operation) {
      if (operation->getParentOfType<func::FuncOp>() != funcOp) {
        return;
      }
      auto allocOp = dyn_cast<cbit::AllocOp>(operation);
      if (!allocOp) {
        return;
      }
      const auto [it, inserted] = state.cregIndices.try_emplace(
          allocOp.getOperation(), state.cregs.size());
      if (inserted) {
        state.cregs.emplace_back();
      }
      auto& reg = state.cregs[it->second];
      reg.record = false;
      if (const auto name = allocOp->getAttrOfType<StringAttr>(
              mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())) {
        reg.label = name.str();
      }
      const auto size = allocOp.getResult().getType().getWidth();
      reg.size = size;
      reg.results.assign(static_cast<size_t>(size), Value{});
    });

    const auto markRegisterForRecording = [&](size_t registerIndex) {
      auto& reg = state.cregs[registerIndex];
      if (reg.record) {
        return;
      }
      if (reg.label.empty()) {
        reg.label = "c" + std::to_string(state.returnedCregs.size());
      }
      reg.record = true;
      state.returnedCregs.push_back(registerIndex);
    };

    walkOperationsIteratively(funcOp, [&](Operation* operation) {
      if (operation->getParentOfType<func::FuncOp>() != funcOp) {
        return;
      }
      auto returnOp = dyn_cast<func::ReturnOp>(operation);
      if (!returnOp) {
        return;
      }
      ReturnRewrite rewrite{.function = funcOp, .returnOp = returnOp};

      for (auto operand : returnOp.getOperands()) {
        if (auto measureOp = operand.getDefiningOp<MeasureOp>()) {
          state.returnedStaticResults.insert(measureOp.getOperation());
          rewrite.recordsClassicalOutput = true;
        } else if (auto allocOp = operand.getDefiningOp<cbit::AllocOp>();
                   allocOp &&
                   state.cregIndices.contains(allocOp.getOperation())) {
          markRegisterForRecording(
              state.cregIndices.at(allocOp.getOperation()));
          rewrite.recordsClassicalOutput = true;
        } else {
          rewrite.keptOperands.push_back(operand);
          rewrite.keptTypes.push_back(operand.getType());
        }
      }

      rewrite.needsStatusResult =
          rewrite.keptOperands.empty() && !returnOp.getOperands().empty();
      returnRewrites.push_back(std::move(rewrite));
    });

    if (returnRewrites.size() > 1) {
      auto recordedReturn =
          llvm::find_if(returnRewrites, [](const ReturnRewrite& rewrite) {
            return rewrite.recordsClassicalOutput;
          });
      if (recordedReturn != returnRewrites.end()) {
        recordedReturn->returnOp.emitError(
            "QIR conversion requires a single entry-function return when "
            "recording classical outputs");
        return failure();
      }
    }

    walkOperationsIteratively(funcOp, [&](Operation* operation) {
      if (operation->getParentOfType<func::FuncOp>() != funcOp) {
        return;
      }
      auto storeOp = dyn_cast<cbit::StoreOp>(operation);
      if (!storeOp) {
        return;
      }
      auto allocOp = storeOp.getReg().getDefiningOp<cbit::AllocOp>();
      if (!allocOp || !state.cregIndices.contains(allocOp.getOperation())) {
        storeOp.emitError(
            "QIR conversion requires direct CBit register allocations");
        hasInvalidMemory = true;
        return;
      }
      const auto registerIndex = state.cregIndices.at(allocOp.getOperation());
      if (!state.cregs[registerIndex].record) {
        return;
      }
      auto measureOp = storeOp.getValue().getDefiningOp<MeasureOp>();
      if (!measureOp) {
        storeOp.emitError(
            "QIR conversion does not support non-measurement stores to "
            "returned CBit registers");
        hasInvalidMemory = true;
        return;
      }
      const auto destination =
          std::pair<size_t, Value>{registerIndex, storeOp.getIndex()};
      const auto [it, inserted] = state.cregMeasurements.try_emplace(
          measureOp.getOperation(), destination);
      if (!inserted && it->second != destination) {
        storeOp.emitError(
            "a measurement result cannot be stored in multiple classical "
            "register locations during QIR conversion");
        hasInvalidMemory = true;
      }
      consumedStores.push_back(storeOp);
    });
  }
  if (hasInvalidMemory) {
    return failure();
  }

  DenseMap<Operation*, SmallVector<Type>> loweredReturnTypes;
  for (auto& rewrite : returnRewrites) {
    if (rewrite.needsStatusResult) {
      rewrite.keptTypes.push_back(
          IntegerType::get(rewrite.function.getContext(), 64));
    }
    const auto [it, inserted] = loweredReturnTypes.try_emplace(
        rewrite.function.getOperation(), rewrite.keptTypes);
    if (!inserted && it->second != rewrite.keptTypes) {
      rewrite.returnOp.emitError(
          "QIR conversion requires every entry-function return to have the "
          "same lowered result types");
      return failure();
    }
  }

  for (auto& rewrite : returnRewrites) {
    if (rewrite.needsStatusResult) {
      OpBuilder builder(rewrite.returnOp);
      auto zero = arith::ConstantIntOp::create(
          builder, rewrite.returnOp.getLoc(), 0, 64);
      rewrite.keptOperands.push_back(zero);
    }
    rewrite.returnOp.getOperandsMutable().assign(rewrite.keptOperands);
    rewrite.function.setFunctionType(FunctionType::get(
        rewrite.function.getContext(),
        rewrite.function.getFunctionType().getInputs(), rewrite.keptTypes));
  }
  for (auto storeOp : consumedStores) {
    storeOp.erase();
  }
  return success();
}

} // namespace mlir
