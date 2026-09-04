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

#include "mlir/Conversion/ConversionUtils.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/FunctionUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir {

using namespace qco;
using namespace qc;

namespace {

/// Qubit allocation mode
enum class AllocationMode : std::uint8_t {
  Unset,  //!< No allocation mode has been established yet.
  Static, //!< The module uses static qubit allocation.
  Dynamic //!< The module uses dynamic qubit allocation.
};

/// State object for tracking qubit allocation mode.
///
/// Used to track whether a function uses static or dynamic qubit allocation.
/// This is used to determine whether to convert `qco.sink` to `qc.dealloc` (for
/// dynamic qubits) or simply erase it (for static qubits). This is also used to
/// catch cases of mixed allocation modes being used, which is not supported.
struct LoweringState {
  /// Per-region map from a register's indices to its loaded qubit values.
  DenseMap<Region*, DenseMap<Value, DenseMap<Value, Value>>> qubitValues;
  /// Original qubit argument positions, retained while signatures are
  /// rewritten.
  DenseMap<Operation*, SmallVector<unsigned>> qubitArguments;
  /// The qubit allocation mode used in the module
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Sets or validates the allocation mode, or emits an error if it conflicts.
  [[nodiscard]] LogicalResult ensureAllocationMode(AllocationMode requestedMode,
                                                   Operation* op) {
    if (allocationMode == AllocationMode::Unset) {
      allocationMode = requestedMode;
      return success();
    }
    if (allocationMode == requestedMode) {
      return success();
    }
    return op->emitOpError(
        "cannot mix static and dynamic qubit allocation modes in QCO program");
  }
};

/// Base class for conversion patterns that need access to lowering state
///
/// Extends OpConversionPattern to provide access to a shared LoweringState
/// object, which is used to track the allocation mode of the module.
/// @tparam OpType The QCO operation type to be converted.
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

/// Moves the operations from one region into another.
///
/// Moves the operations from the source region into the target region.
/// The target region replaces the uses of the old block arguments with the
/// @p replacementValues and erases the unused block arguments.
///
/// @param sourceRegion Source region where the operations are moved from
/// @param targetRegion Target region where the operations are moved to
/// @param replacementValues Values to replace the uses of the arguments
/// @param rewriter PatternRewriter of the current conversion pass
static void inlineRegion(Region& sourceRegion, Region& targetRegion,
                         ValueRange replacementValues,
                         ConversionPatternRewriter& rewriter) {
  rewriter.inlineRegionBefore(sourceRegion, targetRegion, targetRegion.end());
  auto& block = targetRegion.front();
  assert(block.getNumArguments() == replacementValues.size() &&
         "Number of replacement values must match number of block arguments");
  TypeConverter::SignatureConversion signature(block.getNumArguments());
  for (auto [arg, replacementVal] :
       llvm::zip_equal(block.getArguments(), replacementValues)) {
    signature.remapInput(arg.getArgNumber(), replacementVal);
  }
  rewriter.applySignatureConversion(&block, signature);
}

[[nodiscard]] static bool isQuantumStateType(const Type type) {
  if (isa<qco::QubitType, qc::QubitType>(type)) {
    return true;
  }
  if (const auto tensor = dyn_cast<RankedTensorType>(type)) {
    return isa<qco::QubitType>(tensor.getElementType());
  }
  const auto memref = dyn_cast<MemRefType>(type);
  return memref && isa<qc::QubitType>(memref.getElementType());
}

[[nodiscard]] static SmallVector<Value>
selectConvertedState(ValueRange originalValues, ValueRange convertedValues,
                     const bool selectQuantum) {
  assert(originalValues.size() == convertedValues.size());
  SmallVector<Value> selected;
  for (auto [original, converted] :
       llvm::zip_equal(originalValues, convertedValues)) {
    if (isQuantumStateType(original.getType()) == selectQuantum) {
      selected.push_back(converted);
    }
  }
  return selected;
}

static void inlineSCFRegion(Region& sourceRegion, Region& targetRegion,
                            const unsigned int offset, ValueRange originalState,
                            ValueRange quantumReplacements,
                            ConversionPatternRewriter& rewriter) {
  rewriter.inlineRegionBefore(sourceRegion, targetRegion, targetRegion.end());
  auto& block = targetRegion.front();
  assert(block.getNumArguments() == offset + originalState.size() &&
         "region arguments must match the original loop state");

  TypeConverter::SignatureConversion signature(block.getNumArguments());
  for (auto arg : block.getArguments().take_front(offset)) {
    signature.addInputs(arg.getArgNumber(), arg.getType());
  }
  size_t quantumIndex = 0;
  for (const auto [index, original] : llvm::enumerate(originalState)) {
    if (!isQuantumStateType(original.getType())) {
      signature.addInputs(offset + index, original.getType());
      continue;
    }
    assert(quantumIndex < quantumReplacements.size() &&
           "missing replacement for quantum loop state");
    signature.remapInput(offset + index, quantumReplacements[quantumIndex++]);
  }
  assert(quantumIndex == quantumReplacements.size() &&
         "unused replacement for quantum loop state");
  rewriter.applySignatureConversion(&block, signature);
}

[[nodiscard]] static SmallVector<Value>
combineConvertedResults(TypeRange originalTypes, ValueRange classicalResults,
                        ValueRange quantumReplacements) {
  SmallVector<Value> replacements;
  replacements.reserve(originalTypes.size());
  size_t classicalIndex = 0;
  size_t quantumIndex = 0;
  for (const auto type : originalTypes) {
    if (isQuantumStateType(type)) {
      assert(quantumIndex < quantumReplacements.size());
      replacements.push_back(quantumReplacements[quantumIndex++]);
    } else {
      assert(classicalIndex < classicalResults.size());
      replacements.push_back(classicalResults[classicalIndex++]);
    }
  }
  assert(classicalIndex == classicalResults.size());
  assert(quantumIndex == quantumReplacements.size());
  return replacements;
}

#define GEN_PASS_DEF_QCOTOQC
#include "mlir/Conversion/QCOToQC/QCOToQC.h.inc"

namespace {

/// Type converter for QCO-to-QC conversion
///
/// Handles type conversion between the QCO and QC dialects.
/// The primary conversion is from !qco.qubit to !qc.qubit, which
/// represents the semantic shift from value types to reference types.
///
/// Qubit tensor types preserve their shape during conversion: a statically
/// shaped `tensor<Nx!qco.qubit>` becomes `memref<Nx!qc.qubit>`, while a
/// dynamically shaped `tensor<?x!qco.qubit>` becomes `memref<?x!qc.qubit>`.
///
/// Other types (integers, booleans, etc.) pass through unchanged via
/// the identity conversion.
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
      if (isa<qco::QubitType>(type.getElementType())) {
        return MemRefType::get(type.getShape(), qc::QubitType::get(ctx));
      }
      return type;
    });
  }
};

} // namespace

[[nodiscard]] static LogicalResult
collectFunctionQubitArguments(ModuleOp moduleOp, LoweringState& state) {
  for (auto function : moduleOp.getOps<func::FuncOp>()) {
    auto& qubitArguments = state.qubitArguments[function];
    for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
      if (isa<qco::QubitType>(type)) {
        qubitArguments.emplace_back(index);
      }
    }
    if (qubitArguments.empty()) {
      continue;
    }
    if (function.getNumResults() < qubitArguments.size() ||
        llvm::any_of(function.getResultTypes().take_back(qubitArguments.size()),
                     [](Type type) { return !isa<qco::QubitType>(type); })) {
      return function.emitOpError()
             << "must return one trailing qubit for each qubit argument";
    }
    const auto firstQubitResult =
        function.getNumResults() - qubitArguments.size();
    for (unsigned index = firstQubitResult; index < function.getNumResults();
         ++index) {
      if (auto attrs = function.getResultAttrDict(index);
          attrs && !attrs.empty()) {
        return function.emitOpError(
            "cannot preserve attributes on pass-through qubit results in QC");
      }
    }
    if (function.isDeclaration()) {
      continue;
    }
    if (!function.getBody().hasOneBlock()) {
      return function.emitOpError()
             << "with qubit arguments must have one outer block";
    }
    auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
    if (!returnOp) {
      return function.emitOpError("must terminate with func.return");
    }
    auto returnedQubits =
        returnOp.getOperands().take_back(qubitArguments.size());
    for (auto [argument, value] :
         llvm::zip_equal(qubitArguments, returnedQubits)) {
      auto origin = qco::traceQubitArgument(function, value);
      if (failed(origin) || *origin != argument) {
        return function.emitOpError()
               << "must return its qubit arguments positionally";
      }
    }
  }
  return success();
}

namespace {

struct ConvertFuncOp final : StatefulOpConversionPattern<func::FuncOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor,
                  ConversionPatternRewriter& rewriter) const override {
    TypeConverter::SignatureConversion signature(op.getNumArguments());
    if (failed(getTypeConverter()->convertSignatureArgs(op.getArgumentTypes(),
                                                        signature))) {
      return failure();
    }
    SmallVector<Type> inputs;
    if (failed(
            getTypeConverter()->convertTypes(op.getArgumentTypes(), inputs))) {
      return failure();
    }

    const auto& qubitArguments = getState().qubitArguments[op];
    const auto firstQubitResult = op.getNumResults() - qubitArguments.size();
    SmallVector<Type> results;
    if (failed(getTypeConverter()->convertTypes(
            op.getResultTypes().take_front(firstQubitResult), results))) {
      return failure();
    }
    SmallVector<DictionaryAttr> resultAttrs;
    for (unsigned index = 0; index < firstQubitResult; ++index) {
      resultAttrs.emplace_back(op.getResultAttrDict(index));
    }

    rewriter.modifyOpInPlace(op, [&] {
      op.setType(rewriter.getFunctionType(inputs, results));
      function_interface_impl::setAllResultAttrDicts(op, resultAttrs);
    });
    if (!op.isExternal() &&
        failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(),
                                           &signature))) {
      return failure();
    }
    return success();
  }
};

struct ConvertFuncReturnOp final : StatefulOpConversionPattern<func::ReturnOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto function = op->getParentOfType<func::FuncOp>();
    const auto numQubitArguments = getState().qubitArguments[function].size();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(
        op, adaptor.getOperands().drop_back(numQubitArguments));
    return success();
  }
};

struct ConvertFuncCallOp final : StatefulOpConversionPattern<func::CallOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getCalleeAttr());
    if (!callee) {
      return rewriter.notifyMatchFailure(op, "callee is not defined");
    }
    const auto& qubitArguments = getState().qubitArguments[callee];
    const auto firstQubitResult = op.getNumResults() - qubitArguments.size();
    auto resultAttrs = op.getResAttrsAttr();
    if (resultAttrs &&
        llvm::any_of(resultAttrs.getValue().take_back(qubitArguments.size()),
                     [](Attribute attr) {
                       return !cast<DictionaryAttr>(attr).empty();
                     })) {
      return op.emitOpError(
          "cannot preserve attributes on pass-through qubit results in QC");
    }

    SmallVector<Type> keptResultTypes(op.getResultTypes());
    keptResultTypes.resize(firstQubitResult);
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(keptResultTypes, resultTypes))) {
      return failure();
    }
    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     resultTypes, adaptor.getOperands());
    call->setAttrs(op->getAttrs());
    if (resultAttrs) {
      call.setResAttrsAttr(rewriter.getArrayAttr(
          resultAttrs.getValue().take_front(firstQubitResult)));
    }

    SmallVector<Value> replacements;
    llvm::append_range(replacements, call.getResults());
    for (const auto argument : qubitArguments) {
      replacements.emplace_back(adaptor.getOperands()[argument]);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct ConvertQCOCallOp final : OpConversionPattern<qco::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (auto attrs = op.getResAttrsAttr();
        attrs && llvm::any_of(attrs, [](Attribute attr) {
          return !cast<DictionaryAttr>(attr).empty();
        })) {
      return op.emitOpError(
          "cannot preserve unitary call result attributes in QC");
    }
    auto call = qc::CallOp::create(rewriter, op.getLoc(), op.getCalleeAttr(),
                                   adaptor.getOperands());
    call->setAttrs(op->getAttrs());
    call.removeResAttrsAttr();
    rewriter.replaceOp(op, adaptor.getOperands().take_back(op.getNumResults()));
    return success();
  }
};

/// Converts qtensor.alloc to memref.alloc
///
/// @par Example:
/// ```mlir
/// %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
/// ```
/// is converted to
/// ```mlir
/// %memref = memref.alloc(%c3) : memref<3x!qc.qubit>
/// ```
struct ConvertQTensorAllocOp final
    : StatefulOpConversionPattern<qtensor::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic,
                                               op.getOperation()))) {
      return failure();
    }
    auto qubitType = qc::QubitType::get(op.getContext());
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto memrefType = MemRefType::get(tensorType.getShape(), qubitType);

    memref::AllocOp alloc;
    if (tensorType.hasStaticShape()) {
      // Static size: no dynamic size operand needed
      alloc = memref::AllocOp::create(rewriter, op.getLoc(), memrefType);
    } else {
      // Dynamic size: forward the runtime size operand
      alloc = memref::AllocOp::create(rewriter, op.getLoc(), memrefType,
                                      op.getSize());
    }
    alloc->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

/// Converts qtensor.extract to memref.load
///
/// @par Example:
/// ```mlir
/// %tensor_out, %q = qtensor.extract %tensor_in[%c0]: tensor<3x!qco.qubit>
/// ```
/// is converted to
/// ```mlir
/// %q = memref.load %memref[%c0] : memref<3x!qc.qubit>
/// ```
struct ConvertQTensorExtractOp final
    : StatefulOpConversionPattern<qtensor::ExtractOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitValues =
        getState().qubitValues[op->getParentRegion()][adaptor.getTensor()];
    if (auto qubit = qubitValues.lookup(adaptor.getIndex())) {
      rewriter.replaceOp(op, {adaptor.getTensor(), qubit});
      return success();
    }

    auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                       adaptor.getTensor(), adaptor.getIndex())
                    .getResult();
    qubitValues[adaptor.getIndex()] = load;
    rewriter.replaceOp(op, {adaptor.getTensor(), load});
    return success();
  }
};

/// Converts qtensor.insert to an in-place memref.store.
struct ConvertQTensorInsertOp final
    : StatefulOpConversionPattern<qtensor::InsertOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& qubitValues =
        state.qubitValues[op->getParentRegion()][adaptor.getDest()];
    if (qubitValues.lookup(adaptor.getIndex()) == adaptor.getScalar()) {
      rewriter.replaceOp(op, adaptor.getDest());
      return success();
    }

    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getScalar(),
                            adaptor.getDest(), ValueRange{adaptor.getIndex()});
    for (auto& caches : llvm::make_second_range(state.qubitValues)) {
      caches.erase(adaptor.getDest());
    }
    state.qubitValues[op->getParentRegion()][adaptor.getDest()]
                     [adaptor.getIndex()] = adaptor.getScalar();
    rewriter.replaceOp(op, adaptor.getDest());
    return success();
  }
};

/// Converts qtensor.dealloc to memref.dealloc
///
/// @par Example:
/// ```mlir
/// qtensor.dealloc %tensor : tensor<3x!qco.qubit>
/// ```
/// is converted to
/// ```mlir
/// memref.dealloc %memref : memref<3x!qc.qubit>
/// ```
struct ConvertQTensorDeallocOp final : OpConversionPattern<qtensor::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, adaptor.getTensor());
    return success();
  }
};

template <typename QCOOpType, typename QCOpType, std::size_t NumTargets,
          std::size_t NumParams>
struct ConvertQCOGateToQC final : OpConversionPattern<QCOOpType> {
  using OpConversionPattern<QCOOpType>::OpConversionPattern;

  /// Generic QCO gate conversion helper (value semantics -> reference).
  ///
  /// This helper relies on a strict operand ordering contract provided by the
  /// dialect conversion framework:
  /// - `adaptor.getOperands()` is expected to be ordered as
  ///   `targets...` followed by `parameters...`.
  /// - The first @p NumTargets operands are the (type-converted) QC target
  /// qubits.
  /// - The remaining @p NumParams operands are the gate parameters.
  ///
  /// `matchAndRewrite` passes the full adapted operand list to `createGate`,
  /// which forwards the first @p NumTargets values (converted targets) and the
  /// following @p NumParams values (parameters, unchanged type through the
  /// converter) to `QCOpType::create(...)`. It then replaces the original QCO
  /// op with the created QC targets via `rewriter.replaceOp(op, qcTargets)`.
  ///
  /// The values of @p NumTargets and @p NumParams are compile-time constants
  /// and define this contract for each instantiation.
  ///
  /// @see ConvertQCOGateToQC
  /// @see createGate
  /// @see matchAndRewrite
  /// @see addGatePattern
  template <std::size_t... TargetIndices, std::size_t... ParamIndices>
  static void createGate(ConversionPatternRewriter& rewriter, Location loc,
                         ValueRange qcOperands,
                         std::index_sequence<TargetIndices...> /*tgt*/,
                         std::index_sequence<ParamIndices...> /*par*/) {
    QCOpType::create(rewriter, loc, qcOperands[TargetIndices]...,
                     qcOperands[NumTargets + ParamIndices]...);
  }

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcOperands = adaptor.getOperands();
    assert(qcOperands.size() == (NumTargets + NumParams) &&
           "Unexpected number of operands for QCO->QC gate conversion");
    auto qcTargets = qcOperands.take_front(NumTargets);

    createGate(rewriter, op.getLoc(), qcOperands,
               std::make_index_sequence<NumTargets>{},
               std::make_index_sequence<NumParams>{});
    rewriter.replaceOp(op, qcTargets);
    return success();
  }
};

/// Converts a variadic dense qco.unitary to its reference-semantics form.
struct ConvertQCOUnitaryOp final : OpConversionPattern<qco::UnitaryOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::UnitaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcQubits = adaptor.getQubitsIn();
    qc::UnitaryOp::create(rewriter, op.getLoc(), op.getMatrix(), qcQubits);
    rewriter.replaceOp(op, qcQubits);
    return success();
  }
};

} // namespace

template <typename QCOOp, typename QCOp, std::size_t Targets,
          std::size_t Params>
static void addGatePattern(RewritePatternSet& patterns,
                           TypeConverter& typeConverter, MLIRContext* context) {
  patterns.add<ConvertQCOGateToQC<QCOOp, QCOp, Targets, Params>>(typeConverter,
                                                                 context);
}

namespace {

/// Converts qco.alloc to qc.alloc
///
/// @par Example:
/// ```mlir
/// %q = qco.alloc : !qco.qubit
/// ```
/// is converted to
/// ```mlir
/// %q = qc.alloc : !qc.qubit
/// ```
struct ConvertQCOAllocOp final : StatefulOpConversionPattern<qco::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic,
                                               op.getOperation()))) {
      return failure();
    }

    // Create qc.alloc
    rewriter.replaceOpWithNewOp<qc::AllocOp>(op);

    return success();
  }
};

/// Converts qco.sink to qc.dealloc.
///
/// In QCO, qubits have value/linear semantics and must be consumed explicitly
/// (via `qco.sink`). In QC, qubits have reference semantics; for dynamic qubits
/// we materialize this end-of-lifetime as `qc.dealloc`. Static qubits do not
/// need explicit deallocation, so we simply erase the `qco.sink` operation.
///
/// The OpAdaptor automatically provides the type-converted qubit operand
/// (`!qc.qubit` instead of `!qco.qubit`), so we simply pass it through to the
/// new operation when needed.
///
/// Example transformation:
/// ```mlir
/// qco.sink %q_qco : !qco.qubit
/// // becomes:
/// qc.dealloc %q_qc : !qc.qubit
/// ```
struct ConvertQCOSinkOp final : StatefulOpConversionPattern<SinkOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto allocationMode = getState().allocationMode;
    if (allocationMode == AllocationMode::Unset) {
      return op.emitOpError(
          "cannot exist without an established qubit allocation mode");
    }

    if (allocationMode == AllocationMode::Static) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<DeallocOp>(op, adaptor.getQubit());
    return success();
  }
};

/// Converts qco.static to qc.static
///
/// Static qubits represent references to hardware-mapped or fixed-position
/// qubits identified by an index. The conversion preserves the index attribute
/// and creates the corresponding qc.static operation.
///
/// Example transformation:
/// ```mlir
/// %q0 = qco.static 0 : !qco.qubit
/// // becomes:
/// %q = qc.static 0 : !qc.qubit
/// ```
struct ConvertQCOStaticOp final : StatefulOpConversionPattern<qco::StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Static,
                                               op.getOperation()))) {
      return failure();
    }

    // Create qc.static with the same index
    rewriter.replaceOpWithNewOp<qc::StaticOp>(op, op.getIndex());
    return success();
  }
};

/// Converts qco.measure to qc.measure
///
/// Measurement demonstrates the key semantic difference between the dialects:
/// - QCO (value semantics): Consumes input qubit, returns both output qubit
///   and classical bit result
/// - QC (reference semantics): Measures qubit in-place, returns only the
///   classical bit result
///
/// The OpAdaptor provides the input qubit already converted to !qc.qubit.
/// Since QC operations are in-place, we return the same qubit reference
/// alongside the measurement bit. MLIR's conversion infrastructure
/// automatically routes subsequent uses of the QCO output qubit to this QC
/// reference.
///
/// @par Example:
/// ```mlir
/// %q_out, %c = qco.measure %q_in : !qco.qubit
/// ```
/// is converted to
/// ```mlir
/// %c = qc.measure %q : !qc.qubit -> i1
/// ```
struct ConvertQCOMeasureOp final : OpConversionPattern<qco::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // OpAdaptor provides the already type-converted input qubit
    auto qcQubit = adaptor.getQubitIn();

    // Create qc.measure (in-place operation, returns only bit)
    auto qcOp = qc::MeasureOp::create(rewriter, op.getLoc(), qcQubit);

    auto measureBit = qcOp.getResult();

    // Replace both results: qubit output → same qc reference, bit → new bit
    rewriter.replaceOp(op, {qcQubit, measureBit});

    return success();
  }
};

/// Converts qco.reset to qc.reset
///
/// Reset operations force a qubit to the |0⟩ state:
/// - QCO (value semantics): Consumes input qubit, returns reset output qubit
/// - QC (reference semantics): Resets qubit in-place, no result value
///
/// The OpAdaptor provides the input qubit already converted to !qc.qubit.
/// Since QC's reset is in-place, we return the same qubit reference.
/// MLIR's conversion infrastructure automatically routes subsequent uses of
/// the QCO output qubit to this QC reference.
///
/// Example transformation:
/// ```mlir
/// %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
/// // becomes:
/// qc.reset %q : !qc.qubit
/// // %q_out uses are replaced with %q (the adaptor-converted input)
/// ```
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

/// Converts a zero-target, one-parameter QCO gate to QC
///
/// @tparam QCOOpType The operation type of the QCO gate
/// @tparam QCOpType The operation type of the QC gate
///
/// @par Example:
/// ```mlir
/// qco.gphase(%theta)
/// ```
/// is converted to
/// ```mlir
/// qc.gphase(%theta)
/// ```
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

/// Converts qco.barrier to qc.barrier
///
/// @par Example:
/// ```mlir
/// %q_out:2 = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
/// !qco.qubit, !qco.qubit
/// ```
/// is converted to
/// ```mlir
/// qc.barrier %q0, %q1 : !qc.qubit, !qc.qubit
/// ```
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

/// Converts qco.ctrl to qc.ctrl
///
/// @par Example:
/// ```mlir
/// %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%a_in = %q1_in) {
///   %a_res = qco.x %a_in : !qco.qubit -> !qco.qubit
///   qco.yield %a_res : !qco.qubit
/// } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
/// ```
/// is converted to
/// ```mlir
/// qc.ctrl(%q0) targets(%a0 = %q1) {
///   qc.x %a0 : !qc.qubit
/// } : !qc.qubit
/// ```
struct ConvertQCOCtrlOp final : OpConversionPattern<qco::CtrlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create qc.ctrl operation
    auto qcOp = qc::CtrlOp::create(
        rewriter, op.getLoc(), adaptor.getControlsIn(), adaptor.getTargetsIn());

    if (failed(moveRegion(op.getRegion(), qcOp.getRegion(), rewriter,
                          getTypeConverter()))) {
      return failure();
    }

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, adaptor.getOperands());

    return success();
  }
};

/// Converts qco.inv to qc.inv
///
/// @par Example:
/// ```mlir
/// %q0_out = qco.inv (%a_in = %q0_in) {
///   %a_res = qco.s %a_in : !qco.qubit -> !qco.qubit
///   qco.yield %a_res : !qco.qubit
/// } : {!qco.qubit} -> {!qco.qubit}
/// ```
/// is converted to
/// ```mlir
/// qc.inv {
///   qc.s %q0 : !qc.qubit
/// }
/// ```
struct ConvertQCOInvOp final : OpConversionPattern<qco::InvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::InvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create qc.inv operation
    auto qcOp = qc::InvOp::create(rewriter, op.getLoc(), adaptor.getQubitsIn());

    if (failed(moveRegion(op.getRegion(), qcOp.getRegion(), rewriter,
                          getTypeConverter()))) {
      return failure();
    }

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, adaptor.getOperands());

    return success();
  }
};

/// Converts qco.pow to qc.pow
///
/// @par Example:
/// ```mlir
/// %q0_out = qco.pow(%exponent) (%a_in = %q0_in) {
///   %a_res = qco.s %a_in : !qco.qubit -> !qco.qubit
///   qco.yield %a_res
/// } : {!qco.qubit} -> {!qco.qubit}
/// ```
/// is converted to
/// ```mlir
/// qc.pow(%exponent) (%a0 = %q0) {
///   qc.s %a0 : !qc.qubit
/// } : !qc.qubit
/// ```
struct ConvertQCOPowOp final : OpConversionPattern<qco::PowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create qc.pow operation with exponent and qubit operands
    auto qcOp = qc::PowOp::create(rewriter, op.getLoc(), adaptor.getExponent(),
                                  adaptor.getQubitsIn());

    if (failed(moveRegion(op.getRegion(), qcOp.getRegion(), rewriter,
                          getTypeConverter()))) {
      return failure();
    }

    // Replace the output qubits with the same QC references
    rewriter.replaceOp(op, adaptor.getQubitsIn());

    return success();
  }
};

/// Converts qco.yield to qc.yield or to scf.yield if the parent is a
/// scf::IfOp or scf::IndexSwitchOp.
///
/// @par Example:
/// ```mlir
/// qco.yield %targets : !qco.qubit
/// ```
/// is converted to
/// ```mlir
/// qc.yield
/// ```
struct ConvertQCOYieldOp final : OpConversionPattern<qco::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(
          op, adaptor.getTargets().take_front(ifOp.getNumResults()));
    } else if (auto switchOp =
                   dyn_cast<scf::IndexSwitchOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(
          op, adaptor.getTargets().take_front(switchOp.getNumResults()));
    } else {
      rewriter.replaceOpWithNewOp<qc::YieldOp>(op);
    }

    return success();
  }
};

/// Converts scf.for with value semantics to scf.for with memory
/// semantics for qubit values while preserving classical loop-carried state.
///
/// @par Example:
/// ```mlir
/// %targets_out = scf.for %iv = %lb to %ub step %step iter_args(%arg0 =
/// %qtensor) -> (tensor<3x!qco.qubit) {
///   %t0, %q0 = qtensor.extract %arg0[%iv] : tensor<3x!qco.qubit>
///   %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
///   %insert = qtensor.insert %q1 into %t1[%iv] : tensor<3x!qco.qubit>
///   scf.yield %t1 : tensor<3x!qco.qubit>
/// }
/// ```
/// is converted to
/// ```mlir
/// scf.for %iv = %lb to %ub step %step {
///   %q0 = qc.load %memref[%iv] : !memref<3x!qc.qubit>
///   qc.h %q0 : !qc.qubit
/// }
/// ```
struct ConvertQCOSCFForOp final : OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto classicalInits =
        selectConvertedState(op.getInitArgs(), adaptor.getInitArgs(), false);
    const auto quantumInits =
        selectConvertedState(op.getInitArgs(), adaptor.getInitArgs(), true);
    auto newFor = scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), classicalInits);
    newFor->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    // Erase default block
    rewriter.eraseBlock(&newFor.getRegion().front());

    // Inline the region, retaining classical state as block arguments and
    // replacing quantum state with the reference-semantic QC values.
    inlineSCFRegion(op.getRegion(), newFor.getRegion(), 1, op.getInitArgs(),
                    quantumInits, rewriter);

    rewriter.replaceOp(op, combineConvertedResults(op.getResultTypes(),
                                                   newFor.getResults(),
                                                   quantumInits));

    return success();
  }
};

/// Converts scf.while with value semantics to scf.while with memory
/// semantics for qubit values while preserving classical loop-carried state.
///
/// @par Example:
/// ```mlir
/// %targets_out = scf.while (%arg0 = %q0) : (!qco.qubit) -> !qco.qubit {
///   %q1, %cond = qco.measure %arg0 : !qco.qubit
///   scf.condition(%cond) %q1 : !qco.qubit
/// } do {
/// ^bb0(%arg0: !qco.qubit):
///   %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
///   scf.yield %q2 : !qco.qubit
/// }
/// ```
/// is converted to
/// ```mlir
/// scf.while : () -> () {
///   %cond = qc.measure %q0 : !qc.qubit -> i1
///   scf.condition(%cond)
/// } do {
///   qc.h %q0 : !qc.qubit
///   scf.yield
/// }
/// ```
struct ConvertQCOSCFWhileOp final : OpConversionPattern<scf::WhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto classicalInits =
        selectConvertedState(op.getInits(), adaptor.getInits(), false);
    const auto quantumInits =
        selectConvertedState(op.getInits(), adaptor.getInits(), true);
    SmallVector<Type> classicalResultTypes;
    for (const auto type : op.getResultTypes()) {
      if (!isQuantumStateType(type)) {
        classicalResultTypes.push_back(type);
      }
    }
    auto newWhileOp = scf::WhileOp::create(
        rewriter, op->getLoc(), classicalResultTypes, classicalInits);

    // The before region receives initial-state types, while the after region
    // receives result-state types. Quantum state in both regions maps back to
    // the same reference-semantic QC values.
    inlineSCFRegion(op.getBefore(), newWhileOp.getBefore(), 0, op.getInits(),
                    quantumInits, rewriter);
    inlineSCFRegion(op.getAfter(), newWhileOp.getAfter(), 0, op.getResults(),
                    quantumInits, rewriter);

    rewriter.replaceOp(op, combineConvertedResults(op.getResultTypes(),
                                                   newWhileOp.getResults(),
                                                   quantumInits));

    return success();
  }
};

/// Converts qco.if to scf.if
///
/// @par Example:
/// ```mlir
/// %targets_out = qco.if %cond args(%arg0 = %q0) -> (!qco.qubit) {
///   %q1 = qco.h %arg0 : !qco.qubit -> !qco.qubit
///   qco.yield %q1 : !qco.qubit
/// } else args(%arg0 = %q0) {
///   qco.yield %arg0 : !qco.qubit
/// }
/// ```
/// is converted to
/// ```mlir
/// scf.if %cond {
///   qc.h %q0 : !qc.qubit
/// }
/// ```
struct ConvertQCOIfOp final : OpConversionPattern<IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> classicalResultTypes;
    if (failed(getTypeConverter()->convertTypes(
            op.getClassicalResults().getTypes(), classicalResultTypes))) {
      return failure();
    }
    const bool keepElseRegion =
        !classicalResultTypes.empty() ||
        op.getElseRegion().front().getOperations().size() > 1;

    // Create the new if operation
    auto newIf = scf::IfOp::create(rewriter, op.getLoc(), classicalResultTypes,
                                   adaptor.getCondition(), keepElseRegion);
    auto& newThenRegion = newIf.getThenRegion();
    auto& oldElseRegion = op.getElseRegion();
    // Erase the default empty then block
    rewriter.eraseBlock(&newThenRegion.front());

    // Inline the region and replace the block arguments
    inlineRegion(op.getThenRegion(), newThenRegion, adaptor.getQubits(),
                 rewriter);

    // Inline the else block when it has observable operations or must produce
    // classical results.
    if (keepElseRegion) {
      rewriter.eraseBlock(&newIf.getElseRegion().front());
      inlineRegion(oldElseRegion, newIf.getElseRegion(), adaptor.getQubits(),
                   rewriter);
    }

    SmallVector<Value> replacements(newIf.getResults());
    llvm::append_range(replacements, adaptor.getQubits());
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

/// Converts qco.index_switch to scf.index_switch
///
/// @par Example:
/// ```mlir
/// %result = qco.index_switch %condition -> !qco.qubit
/// case 0 args(%arg0 = %q0) {
///   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
///   qco.yield %q1 : !qco.qubit
/// }
/// default args(%arg0 = %q0) {
///   %q2 = qco.z %arg0 : !qco.qubit -> !qco.qubit
///   qco.yield %q2 : !qco.qubit
/// }
/// ```
/// is converted to
/// ```mlir
/// scf.index_switch %condition
/// case 0 {
///   qc.x %q0 : !qc.qubit
/// }
/// default {
///   qc.z %q0 : !qc.qubit
/// }
/// ```
struct ConvertQCOIndexSwitchOp final : OpConversionPattern<IndexSwitchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexSwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> classicalResultTypes;
    if (failed(getTypeConverter()->convertTypes(
            op.getClassicalResults().getTypes(), classicalResultTypes))) {
      return failure();
    }
    auto newOp = scf::IndexSwitchOp::create(
        rewriter, op.getLoc(), classicalResultTypes, adaptor.getArg(),
        adaptor.getCases(), op.getNumCases());

    const auto oldRegions = op.getCaseRegions();
    const auto newCaseRegions = newOp.getCaseRegions();
    for (size_t i = 0; i < op.getNumCases(); ++i) {
      inlineRegion(oldRegions[i], newCaseRegions[i], adaptor.getTargets(),
                   rewriter);
    }

    inlineRegion(op.getDefaultRegion(), newOp.getDefaultRegion(),
                 adaptor.getTargets(), rewriter);

    SmallVector<Value> replacements(newOp.getResults());
    llvm::append_range(replacements, adaptor.getTargets());
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

/// Converts scf.yield with value semantics to scf.yield with memory
/// semantics for qubit values while retaining classical yielded values.
///
/// @par Example:
/// ```mlir
/// scf.yield %targets
/// ```
/// is converted to
/// ```mlir
/// scf.yield
/// ```
struct ConvertQCOSCFYieldOp final : OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        op, selectConvertedState(op.getResults(), adaptor.getResults(), false));
    return success();
  }
};

/// Converts scf.condition with value semantics to scf.condition with
/// memory semantics for qubit values while retaining classical state
///
/// @par Example:
/// ```mlir
/// scf.condition(%cond) %targets
/// ```
/// is converted to
/// ```mlir
/// scf.condition(%cond)
/// ```
struct ConvertQCOSCFConditionOp final : OpConversionPattern<scf::ConditionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        op, adaptor.getCondition(),
        selectConvertedState(op.getArgs(), adaptor.getArgs(), false));

    return success();
  }
};

/// Pass implementation for QCO-to-QC conversion
///
/// This pass converts QCO dialect operations (value semantics) to
/// QC dialect operations (reference semantics). The conversion is useful
/// for lowering optimized SSA-form code back to a hardware-oriented
/// representation suitable for backend code generation.
///
/// The conversion leverages MLIR's built-in type conversion infrastructure:
/// The TypeConverter handles !qco.qubit → !qc.qubit transformations,
/// and the OpAdaptor automatically provides type-converted operands to each
/// conversion pattern. This eliminates the need for manual state tracking.
///
/// Key semantic transformation:
/// - QCO operations form explicit SSA chains where each operation consumes
///   inputs and produces new outputs
/// - QC operations modify qubits in-place using references
/// - The conversion maps each QCO SSA chain to a single QC reference,
///   with MLIR's conversion framework automatically handling the plumbing
///
/// The pass operates through:
/// 1. Type conversion: !qco.qubit → !qc.qubit
/// 2. Operation conversion: Each QCO op converted to its QC equivalent
/// 3. Automatic operand mapping: OpAdaptors provide converted operands
/// 4. Function/control-flow adaptation: Signatures updated to use QC types
struct QCOToQC final : impl::QCOToQCBase<QCOToQC> {
  using QCOToQCBase::QCOToQCBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto moduleOp = getOperation();

    // Create state object to track the qubit addressing mode
    LoweringState state;
    if (failed(collectFunctionQubitArguments(moduleOp, state))) {
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> unitaryFunctions;
    for (auto function : moduleOp.getOps<func::FuncOp>()) {
      if (mqt::isUnitaryFunction(function)) {
        unitaryFunctions.emplace_back(function);
        function->removeAttr(mqt::MQTDialect::UnitaryAttrHelper::getNameStr());
      }
    }
    auto unitaryGuard = llvm::make_scope_exit([&] {
      for (auto function : unitaryFunctions) {
        mqt::setUnitaryFunction(function);
      }
    });

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCOToQCTypeConverter typeConverter(context);

    // Configure conversion target
    target.addIllegalDialect<QCODialect, qtensor::QTensorDialect>();
    target
        .addLegalDialect<cbit::CBitDialect, QCDialect, memref::MemRefDialect>();

    target.addDynamicallyLegalDialect<scf::SCFDialect>([](Operation* op) {
      // Some types are not converted yet so QC and QCO types have to be
      // checked.
      auto isQubitType = [](Type t) {
        return TypeSwitch<Type, bool>(t)
            .Case<qc::QubitType, qco::QubitType>([](auto) { return true; })
            .Case<MemRefType>([](MemRefType t) {
              return isa<qc::QubitType>(t.getElementType());
            })
            .Case<RankedTensorType>([](RankedTensorType t) {
              return isa<qco::QubitType>(t.getElementType());
            })
            .Default([](auto) { return false; });
      };

      return !llvm::any_of(op->getOperandTypes(), isQubitType);
    });

    // Register operation conversion patterns that do not need state tracking
    patterns
        .add<ConvertQTensorDeallocOp, ConvertQCOMeasureOp, ConvertQCOResetOp,
             ConvertQCOUnitaryOp,
             ConvertQCOZeroTargetOneParameterToQC<qco::GPhaseOp, qc::GPhaseOp>>(
            typeConverter, context);

#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  addGatePattern<qco::KEY##Op, qc::KEY##Op, (TARGETS), (PARAMS)>(              \
      patterns, typeConverter, context);
#include "mlir/Conversion/GateTable.def"

    patterns.add<ConvertQCOBarrierOp, ConvertQCOCtrlOp, ConvertQCOInvOp,
                 ConvertQCOPowOp, ConvertQCOYieldOp, ConvertQCOIfOp,
                 ConvertQCOIndexSwitchOp, ConvertQCOSCFWhileOp,
                 ConvertQCOSCFConditionOp, ConvertQCOSCFYieldOp,
                 ConvertQCOSCFForOp>(typeConverter, context);

    // Register operation conversion patterns that need state tracking
    patterns.add<ConvertQTensorExtractOp, ConvertQTensorInsertOp,
                 ConvertQTensorAllocOp, ConvertQCOAllocOp, ConvertQCOStaticOp,
                 ConvertQCOSinkOp>(typeConverter, context, &state);

    // QCO qubit arguments are returned positionally and become in-place QC
    // references again.
    patterns.add<ConvertFuncOp>(typeConverter, context, &state);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    patterns.add<ConvertFuncReturnOp>(typeConverter, context, &state);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    patterns.add<ConvertFuncCallOp>(typeConverter, context, &state);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    patterns.add<ConvertQCOCallOp>(typeConverter, context);

    // Conversion of qco types in control-flow ops (e.g., cf.br, cf.cond_br)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir
