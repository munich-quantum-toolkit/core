/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCToQIR/QIRCommon/QIRCommon.h"

#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace mlir {
using namespace qc;
using namespace qir;

namespace {

class ContiguousCopyOpLowering final
    : public ConvertOpToLLVMPattern<memref::CopyOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp copy, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!sourceType || !targetType ||
        !memref::isStaticShapeAndContiguousRowMajor(sourceType) ||
        !memref::isStaticShapeAndContiguousRowMajor(targetType)) {
      return failure();
    }
    auto elementType =
        getTypeConverter()->convertType(sourceType.getElementType());
    if (!elementType) {
      return failure();
    }
    auto loc = copy.getLoc();
    auto source =
        MemRefDescriptor(adaptor.getSource())
            .bufferPtr(rewriter, loc, *getTypeConverter(), sourceType);
    auto target =
        MemRefDescriptor(adaptor.getTarget())
            .bufferPtr(rewriter, loc, *getTypeConverter(), targetType);
    auto arrayType =
        LLVM::LLVMArrayType::get(elementType, sourceType.getNumElements());
    auto value = LLVM::LoadOp::create(rewriter, loc, arrayType, source,
                                      /*alignment=*/1);
    LLVM::StoreOp::create(rewriter, loc, value, target, /*alignment=*/1);
    rewriter.eraseOp(copy);
    return success();
  }
};

} // namespace

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

void registerQIRClassicalTensorDialects(DialectRegistry& registry) {
  registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
                  linalg::LinalgDialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
}

LogicalResult finalizeQIRConversion(ModuleOp moduleOp, ConversionTarget& target,
                                    LLVMTypeConverter& typeConverter) {
  auto* ctx = moduleOp.getContext();
  // Read-only tensors and explicit destinations need no alias analysis.
  bufferization::BufferizationOptions options;
  options.allowUnknownOps = true;
  options.opFilter.allowOperation<arith::ConstantOp, tensor::ExtractOp,
                                  bufferization::MaterializeInDestinationOp>();
  bufferization::BufferizationState state;
  if (failed(bufferization::bufferizeOp(moduleOp, options, state))) {
    return failure();
  }

  if (moduleOp
          .walk([](Operation* op) {
            if (isa<memref::AllocOp, memref::DeallocOp, memref::ReallocOp>(
                    op)) {
              op->emitError("QIR classical storage requires stack allocation");
              return WalkResult::interrupt();
            }
            if (auto alloca = dyn_cast<memref::AllocaOp>(op);
                alloca && !alloca.getType().hasStaticShape()) {
              op->emitError(
                  "QIR classical stack storage requires a static shape");
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  // QIR arrays provide loads/stores, not memcpy or the MLIR memrefCopy runtime.
  if (moduleOp
          .walk([](memref::CopyOp copy) {
            auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
            auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
            if (!sourceType || !targetType) {
              copy.emitError("QIR classical copies require ranked memrefs");
              return WalkResult::interrupt();
            }
            if (sourceType.hasStaticShape() &&
                sourceType.getNumElements() == 0) {
              copy.erase();
              return WalkResult::advance();
            }
            if (memref::isStaticShapeAndContiguousRowMajor(sourceType) &&
                memref::isStaticShapeAndContiguousRowMajor(targetType)) {
              return WalkResult::advance();
            }
            IRRewriter rewriter(copy.getContext());
            rewriter.setInsertionPoint(copy);
            auto structuredCopy = linalg::CopyOp::create(
                rewriter, copy.getLoc(), copy.getSource(), copy.getTarget());
            if (failed(linalg::linalgOpToLoops(rewriter, structuredCopy))) {
              copy.emitError("could not lower strided copy to loops");
              return WalkResult::interrupt();
            }
            rewriter.eraseOp(structuredCopy);
            rewriter.eraseOp(copy);
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  RewritePatternSet patterns(ctx);
  target.addIllegalDialect<affine::AffineDialect, arith::ArithDialect,
                           cf::ControlFlowDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           linalg::LinalgDialect, tensor::TensorDialect,
                           bufferization::BufferizationDialect>();
  LLVMTypeConverter memoryTypeConverter(ctx);
  populateSCFToControlFlowConversionPatterns(patterns);
  memref::populateExpandStridedMetadataPatterns(patterns);
  populateAffineToStdConversionPatterns(patterns);
  patterns.add<ContiguousCopyOpLowering>(memoryTypeConverter,
                                         PatternBenefit(2));
  populateFinalizeMemRefToLLVMConversionPatterns(memoryTypeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
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

CompilerTarget::OperationMatch
LoweringState::nativeOperation(StringRef name, ValueRange qubits,
                               ValueRange parameters,
                               size_t numControls) const {
  SmallVector<CompilerTarget::SiteId> sites;
  for (auto qubit : qubits) {
    auto pointer = qubit.getDefiningOp<LLVM::IntToPtrOp>();
    auto index = pointer ? getConstantIntValue(pointer.getArg()) : std::nullopt;
    if (!index) {
      sites.clear();
      break;
    }
    sites.push_back(*index);
  }
  SmallVector<std::optional<double>> values;
  for (auto parameter : parameters) {
    values.push_back(mqt::valueToDouble(parameter));
  }
  return nativeTarget->findOperation(
      name, qubits.size(), values,
      sites.size() == qubits.size()
          ? std::optional<ArrayRef<CompilerTarget::SiteId>>(sites)
          : std::nullopt,
      numControls);
}

LogicalResult LoweringState::nameNativeCall(LLVM::CallOp call, StringRef name,
                                            ValueRange qubits) const {
  if (!nativeTarget) {
    return success();
  }
  const auto* capability = nativeOperation(name, qubits, {}).capability;
  if (capability == nullptr) {
    return call.emitError("target does not support emitted QIR operation");
  }
  call->setAttr(
      qir::QIR_NATIVE_NAME_ATTR,
      StringAttr::get(
          call.getContext(),
          (Twine("__quantum__qis__") + capability->name() + "__body").str()));
  return success();
}

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
static LogicalResult convertUnitaryToCallOp(
    QCOpType& op, QCOpAdaptorType& adaptor, ConversionPatternRewriter& rewriter,
    ValueRange controls, StringRef fnName, const size_t numTargets,
    const size_t numParams, LoweringState& state, StringRef genericName = {}) {
  auto convertedOperands = adaptor.getOperands();
  auto targets = convertedOperands.take_front(numTargets);
  auto parameters = convertedOperands.drop_front(numTargets);
  assert(parameters.size() == numParams && "unexpected gate parameter count");

  std::string nativeName;
  if (state.nativeTarget) {
    auto unitary = cast<UnitaryOpInterface>(op.getOperation());
    SmallVector<Value> qubits(controls);
    llvm::append_range(qubits, targets);
    const auto symbol = unitary.getBaseSymbol();
    const auto [capability, genericControls] =
        state.nativeOperation(symbol, qubits, parameters, controls.size());
    if (capability == nullptr) {
      return op.emitError("target does not support emitted QIR operation");
    }
    if (genericControls) {
      fnName = genericName;
    }
    nativeName = (Twine("__quantum__qis__") + capability->name() +
                  (genericControls ? "__ctl" : "__body"))
                     .str();
  }
  auto call = qir::emitQISCall(rewriter, op, op.getLoc(), parameters, controls,
                               targets, fnName);
  if (!nativeName.empty()) {
    call->setAttr(qir::QIR_NATIVE_NAME_ATTR,
                  rewriter.getStringAttr(nativeName));
  }
  rewriter.eraseOp(op);
  return success();
}

namespace {

/// Lower a unitary gate to a QIR call selected by its active control count.
///
/// For example, qc.rx(θ) becomes
/// `__quantum__qis__rx__body(θ, qubit)` without controls. The shared
/// qir::emitQISCall helper owns argument packing for controlled calls.
///
/// @tparam GetFnName Map the number of controls to the QIR callee name.
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
    auto result =
        convertUnitaryToCallOp(op, adaptor, rewriter, controls, fnName,
                               NumTargets, NumParams, state, GetFnName(3));
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

    const OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    Value qubit;
    if (const auto it = state.staticQubits.find(index);
        it != state.staticQubits.end()) {
      qubit = it->second;
    } else {
      qubit = createPointerFromIndex(rewriter, op.getLoc(), index);
      state.staticQubits.try_emplace(index, qubit);
    }
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

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
                                  QIR_GPHASE, 0, 1, state);
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

    auto unitaries = op.getBody()->getOps<UnitaryOpInterface>();
    auto it = unitaries.begin();
    UnitaryOpInterface bodyUnitary;
    if (it != unitaries.end()) {
      bodyUnitary = *it;
      if (++it != unitaries.end()) {
        return rewriter.notifyMatchFailure(
            op,
            "CtrlOps with multiple body unitaries are not supported. Run the "
            "unroll-modifiers pass before the conversion");
      }
    }

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
  auto returnOp = cast<LLVM::ReturnOp>(&main.getBlocks().back().back());
  builder.setInsertionPoint(returnOp);
  auto returned = returnOp.getOperand(0);
  SmallVector<qir::OutputRecord> outputs;
  outputs.reserve(state.returnedOutputs.size());
  for (const auto& output : state.returnedOutputs) {
    if (const auto* index = std::get_if<int64_t>(&output)) {
      auto value = builder.createOrFold<LLVM::ExtractValueOp>(
          returnOp.getLoc(), returned, ArrayRef<int64_t>{*index});
      outputs.push_back({
          .label = "__unnamed__" + std::to_string(outputs.size()),
          .value = value,
      });
      continue;
    }
    auto* operation = std::get<Operation*>(output);
    if (const auto it = state.cregIndices.find(operation);
        it != state.cregIndices.end()) {
      const auto& reg = state.cregs[it->second];
      outputs.push_back({.label = reg.label, .value = &reg});
    } else {
      outputs.push_back(state.measuredResults.at(operation));
    }
  }
  if (isa<LLVM::LLVMStructType>(returned.getType())) {
    auto status = builder.createOrFold<LLVM::ExtractValueOp>(
        returnOp.getLoc(), returned, ArrayRef<int64_t>{0});
    returnOp->setOperands(status);
    main.setFunctionType(
        LLVM::LLVMFunctionType::get(status.getType(), main.getArgumentTypes()));
  }
  emitOutputRecording(builder, main, outputs);
}

void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  patterns.add<ConvertQCUnitaryOpQIR<qc::KEY##Op, (TARGETS), (PARAMS),         \
                                     &getFnName##GETTER>>(typeConverter, ctx,  \
                                                          &state);
#include "mqt/Conversion/GateTable.def"

  patterns.add<ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCYieldOp,
               ConvertQCStaticOp, ConvertQCGPhaseOp>(typeConverter, ctx,
                                                     &state);
}

Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter, bool dynamic) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(state.entryBlock->getTerminator());
  const auto index = static_cast<int64_t>(state.scalarResults.size());
  if (auto it = state.measuredResults.find(op);
      it != state.measuredResults.end()) {
    it->second.label = "__unnamed__" + std::to_string(index);
  }
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
  state.scalarResults.push_back(result);
  return result;
}

/// Summarize the interference before each store in one block traversal.
///
/// Only stores to distinct constant-index locations in direct allocations can
/// be crossed.
static DenseSet<Operation*> findStoreFusionCandidates(Block* block) {
  DenseSet<Operation*> candidates;
  DenseMap<Operation*, size_t> measurements;
  DenseMap<std::pair<Value, int64_t>, size_t> lastStoreAtIndex;
  size_t lastStore = 0;
  size_t lastBarrier = 0;
  size_t position = 0;
  for (auto& operation : *block) {
    ++position;
    if (isa<MeasureOp>(operation)) {
      measurements[&operation] = position;
    }
    if (auto store = dyn_cast<cbit::StoreOp>(operation)) {
      const auto index = getConstantIntValue(store.getIndex());
      const bool knownLocation =
          index && store.getReg().getDefiningOp<cbit::AllocOp>();
      auto conflict = std::max(lastBarrier, lastStore);
      if (knownLocation) {
        conflict = std::max(lastBarrier,
                            lastStoreAtIndex.lookup({store.getReg(), *index}));
      }
      auto measure = store.getValue().getDefiningOp<MeasureOp>();
      if (measure && measurements.lookup(measure.getOperation()) > conflict) {
        candidates.insert(store.getOperation());
      }
      if (knownLocation) {
        lastStore = position;
        lastStoreAtIndex[{store.getReg(), *index}] = position;
      } else {
        lastBarrier = position;
      }
      continue;
    }
    // Fresh allocations and quantum effects cannot access existing CBit
    // storage.
    if (isa<cbit::AllocOp, qc::AllocOp, qc::DeallocOp, qc::GPhaseOp>(
            operation)) {
      continue;
    }
    const auto effects = getEffectsRecursively(&operation);
    if (!effects || !llvm::all_of(*effects, [](const auto& effect) {
          auto value = effect.getValue();
          if (!value) {
            return false;
          }
          if (isa<QubitType>(value.getType())) {
            return true;
          }
          auto memref = dyn_cast<MemRefType>(value.getType());
          return memref && isa<QubitType>(memref.getElementType());
        })) {
      lastBarrier = position;
    }
  }
  return candidates;
}

LogicalResult prepareClassicalResults(Operation* moduleOp, LoweringState& state,
                                      bool allowComputedOutputs,
                                      bool deferMeasurementStores) {
  auto funcOp = mqt::getEntryPoint(cast<ModuleOp>(moduleOp));
  const TargetEnvironmentAnalysis environment(moduleOp);
  if (environment &&
      environment.environment().target().nativeOperationsKind() ==
          CompilerTarget::NativeOperations::Kind::Explicit) {
    state.nativeTarget = environment.environment().target();
  }
  bool hasInvalidMemory = false;
  moduleOp->walk([&](Operation* operation) {
    operation->removeAttr(qir::QIR_NATIVE_NAME_ATTR);
    if (isa<cbit::AllocOp, MeasureOp>(operation) &&
        operation->getParentOfType<func::FuncOp>() != funcOp) {
      operation->emitError("QIR conversion requires measurements and CBit "
                           "allocations in the entry function");
      hasInvalidMemory = true;
    }
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
  SmallVector<func::ReturnOp> returns;
  funcOp.walk([&](func::ReturnOp op) { returns.push_back(op); });
  if (returns.size() != 1) {
    return funcOp.emitError(
        "QIR output requires a single return in the entry function");
  }
  auto returnOp = returns.front();
  SmallVector<cbit::StoreOp> consumedStores;
  DominanceInfo dominance(funcOp);
  IRRewriter rewriter(moduleOp->getContext());

  funcOp.walk([&](memref::AllocOp allocOp) {
    const auto type = allocOp.getType();
    if (isa<QubitType>(type.getElementType()) && type.getRank() != 1) {
      allocOp.emitError(
          "QIR conversion only supports one-dimensional qc.qubit registers");
      hasInvalidMemory = true;
    }
  });

  funcOp.walk([&](cbit::AllocOp allocOp) {
    state.cregIndices[allocOp.getOperation()] = state.cregs.size();
    auto& reg = state.cregs.emplace_back();
    reg.record = false;
    if (const auto name = allocOp->getAttrOfType<StringAttr>(
            mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())) {
      reg.label = name.str();
    }
    const auto size = allocOp.getResult().getType().getWidth();
    reg.size = size;
  });

  if (hasInvalidMemory) {
    return failure();
  }
  size_t registerCount = 0;
  SmallVector<Value> scalarOutputs;
  for (auto operand : returnOp.getOperands()) {
    auto* definition = operand.getDefiningOp();
    if (isa_and_nonnull<MeasureOp>(definition)) {
      state.measuredResults.try_emplace(
          definition, qir::OutputRecord{
                          .label = "__unnamed__" +
                                   std::to_string(state.returnedOutputs.size()),
                          .value = Value{},
                      });
      state.returnedOutputs.push_back(definition);
    } else if (const auto it = state.cregIndices.find(definition);
               it != state.cregIndices.end()) {
      auto& reg = state.cregs[it->second];
      if (!reg.record) {
        if (reg.label.empty()) {
          reg.label = "c" + std::to_string(registerCount);
        }
        ++registerCount;
        reg.record = true;
      }
      state.returnedOutputs.push_back(definition);
    } else if (allowComputedOutputs && operand.getType().isInteger(1)) {
      scalarOutputs.push_back(operand);
      state.returnedOutputs.emplace_back(
          static_cast<int64_t>(scalarOutputs.size()));
    } else {
      return returnOp.emitError(
          "QIR entry return supports only direct measurements and CBit "
          "allocations, plus Boolean scalars in Adaptive Profile; other "
          "scalar program outputs are not supported");
    }
  }

  if (allowComputedOutputs) {
    funcOp.walk([&](cbit::StoreOp storeOp) {
      auto allocOp = storeOp.getReg().getDefiningOp<cbit::AllocOp>();
      if (allocOp && !storeOp.getValue().getDefiningOp<MeasureOp>()) {
        auto& reg = state.cregs[state.cregIndices.at(allocOp.getOperation())];
        reg.booleanStorage = reg.record;
      }
    });
  }

  const auto useBooleanStorage = [&](size_t registerIndex) {
    if (!allowComputedOutputs) {
      return false;
    }
    state.cregs[registerIndex].booleanStorage = true;
    return true;
  };

  DenseMap<Block*, DenseSet<Operation*>> fusionCandidates;
  funcOp.walk([&](cbit::StoreOp storeOp) {
    auto allocOp = storeOp.getReg().getDefiningOp<cbit::AllocOp>();
    if (!allocOp && allowComputedOutputs) {
      // Adaptive conversion checks forwarded registers after lowering SCF.
      return;
    }
    if (!allocOp) {
      storeOp.emitError(
          "QIR conversion requires direct CBit register allocations");
      hasInvalidMemory = true;
      return;
    }
    const auto registerIndex = state.cregIndices.at(allocOp.getOperation());
    if (!state.cregs[registerIndex].record ||
        state.cregs[registerIndex].booleanStorage) {
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
    bool canFuse = measureOp->getBlock() == storeOp->getBlock();
    if (canFuse && measureOp->getNextNode() != storeOp.getOperation()) {
      const auto [candidates, newBlock] =
          fusionCandidates.try_emplace(storeOp->getBlock());
      if (newBlock) {
        candidates->second = findStoreFusionCandidates(storeOp->getBlock());
      }
      canFuse = candidates->second.contains(storeOp.getOperation());
    }
    if (canFuse && !dominance.dominates(storeOp.getIndex(), measureOp)) {
      canFuse = succeeded(moveValueDefinitions(rewriter, storeOp.getIndex(),
                                               measureOp, dominance));
    }
    if (!canFuse) {
      if (useBooleanStorage(registerIndex)) {
        return;
      }
      storeOp.emitError("QIR output cannot fuse this measurement/store pair: "
                        "require the same block, an index available or safely "
                        "computable before measurement, and no intervening "
                        "classical memory effects");
      hasInvalidMemory = true;
      return;
    }
    const auto destination =
        std::pair<size_t, Value>{registerIndex, storeOp.getIndex()};
    const auto [it, inserted] = state.cregMeasurements.try_emplace(
        measureOp.getOperation(), destination);
    if (!inserted && it->second != destination) {
      if (useBooleanStorage(registerIndex)) {
        return;
      }
      storeOp.emitError("a measurement result cannot be stored in multiple "
                        "classical register locations during QIR conversion");
      hasInvalidMemory = true;
    }
    consumedStores.push_back(storeOp);
  });
  if (hasInvalidMemory) {
    return failure();
  }
  DenseSet<size_t> sharedRegisters;
  for (auto* measurement : llvm::make_first_range(state.measuredResults)) {
    if (auto it = state.cregMeasurements.find(measurement);
        it != state.cregMeasurements.end()) {
      sharedRegisters.insert(it->second.first);
    }
  }
  if (!sharedRegisters.empty()) {
    DenseMap<std::pair<size_t, int64_t>, Operation*> storedMeasurements;
    DenseSet<size_t> dynamicStores;
    for (auto store : consumedStores) {
      const auto reg = state.cregIndices.at(store.getReg().getDefiningOp());
      if (!sharedRegisters.contains(reg)) {
        continue;
      }
      if (const auto index = getConstantIntValue(store.getIndex())) {
        auto* value = store.getValue().getDefiningOp();
        const auto [it, inserted] =
            storedMeasurements.try_emplace({reg, *index}, value);
        if (!inserted && it->second != value) {
          it->second = nullptr;
        }
      } else {
        dynamicStores.insert(reg);
      }
    }
    for (auto* measurement : llvm::make_first_range(state.measuredResults)) {
      auto it = state.cregMeasurements.find(measurement);
      if (it == state.cregMeasurements.end()) {
        continue;
      }
      const auto& destination = it->second;
      const auto index = getConstantIntValue(destination.second);
      if (!index || dynamicStores.contains(destination.first) ||
          storedMeasurements.lookup({destination.first, *index}) !=
              measurement) {
        if (useBooleanStorage(destination.first)) {
          continue;
        }
        measurement->emitError(
            "returned scalar measurement cannot share overwritten register "
            "storage; require distinct constant store indices");
        hasInvalidMemory = true;
      }
    }
    if (hasInvalidMemory) {
      return failure();
    }
  }

  llvm::erase_if(consumedStores, [&](cbit::StoreOp store) {
    return state.cregs[state.cregIndices.at(store.getReg().getDefiningOp())]
        .booleanStorage;
  });
  state.cregMeasurements.remove_if([&](const auto& measurement) {
    return state.cregs[measurement.second.first].booleanStorage;
  });

  OpBuilder builder(returnOp);
  auto zero = arith::ConstantIntOp::create(builder, returnOp.getLoc(), 0, 64);
  scalarOutputs.insert(scalarOutputs.begin(), zero);
  returnOp.getOperandsMutable().assign(scalarOutputs);
  funcOp.setFunctionType(FunctionType::get(funcOp.getContext(),
                                           funcOp.getFunctionType().getInputs(),
                                           returnOp.getOperandTypes()));
  for (auto storeOp : consumedStores) {
    auto measureOp = storeOp.getValue().getDefiningOp<MeasureOp>();
    auto allocOp = storeOp.getReg().getDefiningOp<cbit::AllocOp>();
    if (allocOp->getBlock() == measureOp->getBlock() &&
        measureOp->isBeforeInBlock(allocOp)) {
      rewriter.moveOpBefore(allocOp, measureOp);
    }
    if (deferMeasurementStores) {
      state.deferredMeasurementStores[storeOp] = measureOp;
    } else {
      storeOp.erase();
    }
  }
  return success();
}

void finalizeClassicalResults(LoweringState& state) {
  for (auto [store, measurement] : state.deferredMeasurementStores) {
    auto storeOp = cast<cbit::StoreOp>(store);
    state.cregMeasurements.at(measurement).second = storeOp.getIndex();
    storeOp.erase();
  }
  state.deferredMeasurementStores.clear();
}

} // namespace mlir
