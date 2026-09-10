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
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
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

LogicalResult finalizeQIRConversion(ModuleOp moduleOp, ConversionTarget& target,
                                    LLVMTypeConverter& typeConverter) {
  auto* ctx = moduleOp.getContext();
  RewritePatternSet patterns(ctx);
  target.addIllegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           math::MathDialect>();
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    return failure();
  }
  PassManager manager(ctx);
  manager.addPass(createReconcileUnrealizedCastsPass());
  return manager.run(moduleOp);
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

/// Helper to convert a QC operation to a LLVM CallOp
///
/// @tparam QCOpType The operation type of the QC operation
/// @tparam QCOpAdaptorType The OpAdaptor type of the QC operation
/// @param op The QC operation instance to convert
/// @param adaptor The OpAdaptor of the QC operation
/// @param rewriter The pattern rewriter
/// @param controls Converted controls for this gate
/// @param fnName The name of the QIR function to call
/// @param numTargets The number of targets
/// @param numParams The number of parameters
/// @return LogicalResult Success or failure of the conversion
template <typename QCOpType, typename QCOpAdaptorType>
static LogicalResult
convertUnitaryToCallOp(QCOpType& op, QCOpAdaptorType& adaptor,
                       ConversionPatternRewriter& rewriter, ValueRange controls,
                       StringRef fnName, const size_t numTargets,
                       const size_t numParams) {
  auto convertedOperands = adaptor.getOperands();
  auto targets = convertedOperands.take_front(numTargets);
  auto parameters = convertedOperands.drop_front(numTargets);
  assert(parameters.size() == numParams && "unexpected gate parameter count");

  qir::emitQISCall(rewriter, op, op.getLoc(), parameters, controls, targets,
                   fnName);
  rewriter.eraseOp(op);
  return success();
}

namespace {

/// Generic converter for unitary QC ops to QIR calls.
///
/// Many QC gates lower to a QIR runtime call where the callee name depends on
/// the number of active controls. This helper factors out that boilerplate
/// without relying on preprocessor macros.
///
/// @par Examples
/// The examples below illustrate the lowering shapes for unitary gates that
/// are registered through the shared QIR gate table in
/// `populateQCToQIRPatterns`.
///
/// @par One target, zero parameters
/// ```mlir
/// qc.x %q : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr) -> ()
/// ```
///
/// @par One target, one parameter
/// ```mlir
/// qc.rx(%theta) %q : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__rx__body(%theta, %q) : (f64, !llvm.ptr) -> ()
/// ```
///
/// @par One target, two parameters
/// ```mlir
/// qc.r(%theta, %phi) %q : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__prx__body(%theta, %phi, %q)
///     : (f64, f64, !llvm.ptr) -> ()
/// ```
///
/// @par One target, three parameters
/// ```mlir
/// qc.u(%theta, %phi, %lambda) %q : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__u3__body(%theta, %phi, %lambda, %q)
///     : (f64, f64, f64, !llvm.ptr) -> ()
/// ```
///
/// @par Two targets, zero parameters
/// ```mlir
/// qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) ->
/// ()
/// ```
///
/// @par Two targets, one parameter
/// ```mlir
/// qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__rxx__body(%theta, %q0, %q1)
///     : (f64, !llvm.ptr, !llvm.ptr) -> ()
/// ```
///
/// @par Two targets, two parameters
/// ```mlir
/// qc.xx_plus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__xx_plus_yy__body(%theta, %beta, %q0, %q1)
///     : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()
/// ```
///
/// @tparam OpType The QC operation type to convert
/// @tparam NumTargets Number of target qubits for this operation
/// @tparam NumParams Number of floating-point parameters for this operation
/// @tparam GetFnName Function that maps numCtrls -> QIR function name
template <typename OpType, std::size_t NumTargets, std::size_t NumParams,
          auto GetFnName>
struct ConvertQCUnitaryOpQIR : StatefulOpConversionPattern<OpType> {
  using StatefulOpConversionPattern<OpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, OpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    const auto it = state.controlledGates.find(op);
    ValueRange controls = it != state.controlledGates.end()
                              ? ValueRange(it->second)
                              : ValueRange{};
    const auto fnName = GetFnName(controls.size());
    auto result = convertUnitaryToCallOp(op, adaptor, rewriter, controls,
                                         fnName, NumTargets, NumParams);
    if (it != state.controlledGates.end()) {
      state.controlledGates.erase(it);
    }
    return result;
  }
};

/// Converts qc.static to llvm.inttoptr
///
/// Converts a static qubit reference to an LLVM pointer by creating a constant
/// with the qubit index and converting it to a pointer. The pointer is cached
/// in the lowering state for reuse.
///
/// @par Example:
/// ```mlir
/// %q0 = qc.static 0 : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// %c0 = llvm.mlir.constant(0 : i64) : i64
/// %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
/// ```
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

/// Converts qc.gphase to QIR gphase
///
/// @par Example:
/// ```mlir
/// qc.gphase(%theta)
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__gphase__body(%theta) : (f64) -> ()
/// ```
struct ConvertQCGPhaseOp final : StatefulOpConversionPattern<GPhaseOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(GPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (state.controlledGates.contains(op)) {
      return op.emitError("Controlled GPhaseOps cannot be converted to QIR");
    }
    return convertUnitaryToCallOp(op, adaptor, rewriter, ValueRange{},
                                  QIR_GPHASE, 0, 1);
  }
};

// BarrierOp

/// Erases qc.barrier operation, as it is a no-op in QIR
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Inlines qc.ctrl region removes the operation
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    if (state.controlledGates.contains(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "Nested CtrlOps are not supported");
    }

    if (op.getNumBodyUnitaries() > 1) {
      return rewriter.notifyMatchFailure(
          op, "CtrlOps with multiple body unitaries are not supported. Run the "
              "unroll-modifiers pass before the conversion");
    }

    auto bodyUnitary = op.getNumBodyUnitaries() == 1 ? op.getBodyUnitary(0)
                                                     : UnitaryOpInterface{};
    if (bodyUnitary && !isa<BarrierOp, IdOp>(bodyUnitary.getOperation())) {
      state.controlledGates.try_emplace(bodyUnitary.getOperation(),
                                        llvm::to_vector(adaptor.getControls()));
    }

    // Inline block and remove operation
    rewriter.inlineBlockBefore(&op.getRegion().front(), op,
                               adaptor.getTargets());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Erases qc.yield operation
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
    returnedRegisters.push_back(std::move(state.cregs[registerIndex]));
  }
  emitOutputRecording(builder, main, returnedRegisters, state.scalarResults);
}

void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  patterns.add<ConvertQCUnitaryOpQIR<qc::KEY##Op, (TARGETS), (PARAMS),         \
                                     &getFnName##GETTER>>(typeConverter, ctx,  \
                                                          &state);
#include "mlir/Conversion/GateTable.def"

  patterns.add<ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCYieldOp,
               ConvertQCStaticOp, ConvertQCGPhaseOp>(typeConverter, ctx,
                                                     &state);
}

Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter, bool dynamic) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(state.entryBlock->getTerminator());
  const auto index = static_cast<int64_t>(state.scalarResults.size());
  const auto record = state.returnedScalarResults.contains(op);
  Value result;
  if (dynamic) {
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto signature = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto declaration = getOrCreateFunctionDeclaration(
        rewriter, op, QIR_RESULT_ALLOC, signature);
    auto zero = LLVM::ZeroOp::create(rewriter, op->getLoc(), ptrType);
    result = LLVM::CallOp::create(rewriter, op->getLoc(), declaration,
                                  zero.getResult())
                 .getResult();
  } else {
    result = createPointerFromIndex(rewriter, op->getLoc(), index);
  }
  state.scalarResults.try_emplace(
      index, qir::StaticResult{.pointer = result, .record = record});
  return result;
}

LogicalResult prepareClassicalResults(Operation* moduleOp,
                                      LoweringState& state) {
  bool hasInvalidMemory = false;
  moduleOp->walk([&](Operation* operation) {
    if (!isa<func::CallOp, func::CallIndirectOp>(operation)) {
      return;
    }
    const auto isRegister = [](Type type) {
      return isa<cbit::RegisterType>(type);
    };
    if (llvm::any_of(operation->getOperandTypes(), isRegister) ||
        llvm::any_of(operation->getResultTypes(), isRegister)) {
      operation->emitError(
          "QIR conversion does not support CBit registers in calls; "
          "read or write scalar values before the call");
      hasInvalidMemory = true;
    }
  });
  if (hasInvalidMemory) {
    return failure();
  }
  auto funcOp = mqt::getEntryPoint(cast<ModuleOp>(moduleOp));
  SmallVector<func::ReturnOp> returns;
  funcOp.walk([&](func::ReturnOp op) { returns.push_back(op); });
  if (returns.size() != 1) {
    return funcOp.emitError(
        "QIR output requires a single return in the entry function");
  }
  auto returnOp = returns.front();
  SmallVector<Value> keptOperands;
  SmallVector<Type> keptReturnTypes;
  SmallVector<cbit::StoreOp> consumedStores;
  DominanceInfo dominance(funcOp);

  funcOp.walk([&](memref::AllocOp allocOp) {
    const auto type = allocOp.getType();
    if (type.getRank() != 1 || !isa<QubitType>(type.getElementType())) {
      allocOp.emitError(
          "QIR conversion only supports generic memrefs for "
          "one-dimensional qc.qubit registers; use CBit for classical "
          "registers");
      hasInvalidMemory = true;
    }
  });

  funcOp.walk([&](cbit::AllocOp allocOp) {
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
  });

  const auto markRegisterForRecording = [&](const size_t registerIndex) {
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

  for (auto operand : returnOp.getOperands()) {
    if (auto measureOp = operand.getDefiningOp<MeasureOp>()) {
      state.returnedScalarResults.insert(measureOp.getOperation());
    } else if (auto allocOp = operand.getDefiningOp<cbit::AllocOp>();
               allocOp && state.cregIndices.contains(allocOp.getOperation())) {
      markRegisterForRecording(state.cregIndices.at(allocOp.getOperation()));
    } else {
      keptOperands.push_back(operand);
      keptReturnTypes.push_back(operand.getType());
    }
  }

  funcOp.walk([&](cbit::StoreOp storeOp) {
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
    auto* indexProducer = storeOp.getIndex().getDefiningOp();
    bool canFuse =
        measureOp->getBlock() == storeOp->getBlock() &&
        (dominance.dominates(storeOp.getIndex(), measureOp) ||
         (indexProducer && indexProducer->hasTrait<OpTrait::ConstantLike>()));
    for (auto* next = measureOp->getNextNode();
         canFuse && next != storeOp.getOperation();
         next = next->getNextNode()) {
      /// These unscoped quantum effects cannot access CBit storage.
      if (isa<qc::AllocOp, qc::DeallocOp, qc::GPhaseOp>(next)) {
        continue;
      }
      if (auto otherStore = dyn_cast<cbit::StoreOp>(next);
          otherStore && otherStore.getReg() == storeOp.getReg()) {
        const auto index = getConstantIntValue(storeOp.getIndex());
        const auto otherIndex = getConstantIntValue(otherStore.getIndex());
        if (index && otherIndex && *index != *otherIndex) {
          continue;
        }
      }
      const auto effects = getEffectsRecursively(next);
      canFuse = effects && llvm::all_of(*effects, [](const auto& effect) {
                  auto value = effect.getValue();
                  if (!value) {
                    return false;
                  }
                  if (isa<QubitType>(value.getType())) {
                    return true;
                  }
                  auto memref = dyn_cast<MemRefType>(value.getType());
                  return memref && isa<QubitType>(memref.getElementType());
                });
    }
    if (!canFuse) {
      storeOp.emitError("QIR output cannot fuse this measurement/store pair: "
                        "require the same "
                        "block, an index available at measurement, and no "
                        "intervening classical memory effects");
      hasInvalidMemory = true;
      return;
    }
    const auto destination =
        std::pair<size_t, Value>{registerIndex, storeOp.getIndex()};
    const auto [it, inserted] = state.cregMeasurements.try_emplace(
        measureOp.getOperation(), destination);
    if (!inserted && it->second != destination) {
      storeOp.emitError("a measurement result cannot be stored in multiple "
                        "classical register locations during QIR conversion");
      hasInvalidMemory = true;
    }
    consumedStores.push_back(storeOp);
  });
  if (hasInvalidMemory) {
    return failure();
  }

  if (keptOperands.empty() && !returnOp.getOperands().empty()) {
    OpBuilder builder(returnOp);
    auto zero = arith::ConstantIntOp::create(builder, returnOp.getLoc(), 0, 64);
    keptOperands.push_back(zero);
    keptReturnTypes.push_back(zero.getType());
  }
  returnOp.getOperandsMutable().assign(keptOperands);
  funcOp.setFunctionType(FunctionType::get(funcOp.getContext(),
                                           funcOp.getFunctionType().getInputs(),
                                           keptReturnTypes));
  for (auto storeOp : consumedStores) {
    storeOp.erase();
  }
  return success();
}

} // namespace mlir
