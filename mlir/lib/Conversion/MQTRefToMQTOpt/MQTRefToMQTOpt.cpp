/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any ref gate operation to the same
// gate operation in the opt dialect
#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTRefGateOp<::mqt::ir::ref::gate, ::mqt::ir::opt::gate>>(   \
          typeConverter, context, &state);

#include "mlir/Conversion/MQTRefToMQTOpt/MQTRefToMQTOpt.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>
#include <vector>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTREFTOMQTOPT
#include "mlir/Conversion/MQTRefToMQTOpt/MQTRefToMQTOpt.h.inc"

namespace {

struct LoweringState {
  /// @brief Map each initial ref qubit to its latest opt qubit.
  llvm::DenseMap<Value, Value> qubitMap;
  /// @brief Map each initial ref qubit to its index.
  llvm::DenseMap<Value, Value> qubitIndexMap;
  /// @brief Map each initial ref register to its refQubits.
  llvm::DenseMap<Value, std::vector<Value>> qregQubitsMap;
};

template <typename OpType>
class StatefulOpConversionPattern : public mlir::OpConversionPattern<OpType> {
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(mlir::TypeConverter& typeConverter,
                              mlir::MLIRContext* context, LoweringState* state)
      : mlir::OpConversionPattern<OpType>(typeConverter, context),
        state_(state) {}

  /// @brief Return the state object as reference.
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

bool isQubitType(const MemRefType type) {
  return llvm::isa<ref::QubitType>(type.getElementType());
}

bool isQubitType(memref::AllocOp op) { return isQubitType(op.getType()); }

bool isQubitType(memref::DeallocOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

bool isQubitType(memref::LoadOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

} // namespace

class MQTRefToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit MQTRefToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](ref::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // MemRefType conversion
    addConversion([ctx](MemRefType type) -> Type {
      if (isQubitType(type)) {
        return MemRefType::get(type.getShape(), opt::QubitType::get(ctx));
      }
      return type;
    });
  }
};

struct ConvertMQTRefMemRefAlloc final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern<
      memref::AllocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    const auto& qubitType = opt::QubitType::get(rewriter.getContext());
    auto memRefType = MemRefType::get(op.getType().getShape(), qubitType);

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memRefType,
                                                 op.getDynamicSizes());

    return success();
  }
};

struct ConvertMQTRefMemRefDealloc final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern<
      memref::DeallocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    auto refMemRef = op.getMemref();
    const auto& optMemRef = adaptor.getMemref();

    // iterate over all the qubits that were extracted from the register
    for (const auto& refQubit : getState().qregQubitsMap[refMemRef]) {
      const auto& optQubit = getState().qubitMap[refQubit];
      auto index = getState().qubitIndexMap[refQubit];

      auto storeOp = rewriter.create<memref::StoreOp>(
          op.getLoc(), optQubit, optMemRef, ValueRange{index});

      // move it before the current dealloc operation
      storeOp->moveBefore(op);

      // erase the refQubit entry from the maps
      getState().qubitMap.erase(refQubit);
      getState().qubitIndexMap.erase(refQubit);
    }

    // erase the register from the map
    getState().qregQubitsMap.erase(refMemRef);

    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, optMemRef);

    return success();
  }
};

struct ConvertMQTRefAllocQubit final
    : StatefulOpConversionPattern<ref::AllocQubitOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::AllocQubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = op.getQubit();
    auto optOp = rewriter.replaceOpWithNewOp<opt::AllocQubitOp>(op);
    const auto& optQubit = optOp.getQubit();
    getState().qubitMap.try_emplace(refQubit, optQubit);
    return success();
  }
};

struct ConvertMQTRefDeallocQubit final
    : StatefulOpConversionPattern<ref::DeallocQubitOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::DeallocQubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = op.getQubit();
    const auto& optQubit = getState().qubitMap[refQubit];
    rewriter.replaceOpWithNewOp<opt::DeallocQubitOp>(op, optQubit);
    getState().qubitMap.erase(refQubit);
    return success();
  }
};

struct ConvertMQTRefMemRefLoad final
    : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern<
      memref::LoadOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    // create new operation
    auto optLoadOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, adaptor.getMemref(), adaptor.getIndices());

    const auto& refMemRef = op.getMemref();
    const auto& refQubit = op.getResult();
    const auto& optQubit = optLoadOp.getResult();

    // put the pair of the ref qubit and the latest opt qubit in the map
    getState().qubitMap.try_emplace(refQubit, optQubit);

    // add entry to qubitIndexMap
    getState().qubitIndexMap.try_emplace(refQubit,
                                         adaptor.getIndices().front());

    // append the entry to the qregQubitsMap to store which qubits that were
    // extracted from the register
    getState().qregQubitsMap[refMemRef].emplace_back(refQubit);

    return success();
  }
};

struct ConvertMQTRefMeasure final
    : StatefulOpConversionPattern<ref::MeasureOp> {
  using StatefulOpConversionPattern<
      ref::MeasureOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& refQubit = op.getInQubit();

    // get the latest opt qubit from the map and add them to the vector
    const Value optQubit = getState().qubitMap[refQubit];

    // create new operation
    auto optOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), qubitType, op.getOutBit().getType(), optQubit);

    auto outOptQubit = optOp.getOutQubit();
    auto newBit = optOp.getOutBit();

    getState().qubitMap[refQubit] = outOptQubit;

    // replace the old operation results with the new bits and delete
    // old operation
    rewriter.replaceOp(op, newBit);

    return success();
  }
};

struct ConvertMQTRefReset final : StatefulOpConversionPattern<ref::ResetOp> {
  using StatefulOpConversionPattern<ref::ResetOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::ResetOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& refQubit = op.getInQubit();

    // get the latest opt qubit from the map and add them to the vector
    const Value optQubit = getState().qubitMap[refQubit];

    // create new operation
    auto optOp =
        rewriter.create<opt::ResetOp>(op.getLoc(), qubitType, optQubit);

    auto outOptQubit = optOp.getOutQubit();

    getState().qubitMap[refQubit] = outOptQubit;

    // delete the old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTRefQubit final : StatefulOpConversionPattern<ref::QubitOp> {
  using StatefulOpConversionPattern<ref::QubitOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::QubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    // create new operation
    auto optOp =
        rewriter.create<opt::QubitOp>(op.getLoc(), qubitType, op.getIndex());

    // collect ref and opt SSA value
    const auto& refQubit = op.getQubit();
    const auto& optQubit = optOp.getQubit();

    // map ref to opt
    getState().qubitMap[refQubit] = optQubit;

    // replace the old operation result with the new result and delete
    // old operation
    rewriter.replaceOp(op, optQubit);

    return success();
  }
};

template <typename MQTGateRefOp, typename MQTGateOptOp>
struct ConvertMQTRefGateOp final : StatefulOpConversionPattern<MQTGateRefOp> {
  using StatefulOpConversionPattern<MQTGateRefOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateRefOp op, typename MQTGateRefOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Map ref qubits to opt qubits
    auto mapQubits = [&](const auto& refQubits) {
      std::vector<Value> optQubits;
      for (const auto& refQubit : refQubits) {
        optQubits.emplace_back(this->getState().qubitMap[refQubit]);
      }
      return optQubits;
    };

    auto optInQubits = mapQubits(op.getInQubits());
    auto optPosCtrlQubitsValues = mapQubits(op.getPosCtrlInQubits());
    auto optNegCtrlQubitsValues = mapQubits(op.getNegCtrlInQubits());

    // Get optional attributes
    auto staticParams = op.getStaticParams()
                            ? DenseF64ArrayAttr::get(rewriter.getContext(),
                                                     *op.getStaticParams())
                            : DenseF64ArrayAttr{};
    auto paramMask = op.getParamsMask()
                         ? DenseBoolArrayAttr::get(rewriter.getContext(),
                                                   *op.getParamsMask())
                         : DenseBoolArrayAttr{};

    // Create new operation
    auto optOp = rewriter.create<MQTGateOptOp>(
        op.getLoc(), ValueRange(optInQubits).getTypes(),
        ValueRange(optPosCtrlQubitsValues).getTypes(),
        ValueRange(optNegCtrlQubitsValues).getTypes(), staticParams, paramMask,
        op.getParams(), optInQubits, optPosCtrlQubitsValues,
        optNegCtrlQubitsValues);

    // Update qubit map
    const auto& optResults = optOp.getAllOutQubits();
    for (size_t i = 0; i < op.getAllInQubits().size(); i++) {
      this->getState().qubitMap[op.getAllInQubits()[i]] = optResults[i];
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MQTRefToMQTOpt final : impl::MQTRefToMQTOptBase<MQTRefToMQTOpt> {
  using MQTRefToMQTOptBase::MQTRefToMQTOptBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTRefToMQTOptTypeConverter typeConverter(context);

    target.addIllegalDialect<ref::MQTRefDialect>();
    target.addLegalDialect<opt::MQTOptDialect>();

    target.addDynamicallyLegalOp<memref::AllocOp>(
        [&](memref::AllocOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::DeallocOp>(
        [&](memref::DeallocOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::LoadOp>(
        [&](memref::LoadOp op) { return !isQubitType(op); });
    target.addLegalOp<memref::StoreOp>();

    patterns.add<ConvertMQTRefMemRefAlloc>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMemRefDealloc>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMemRefLoad>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefAllocQubit>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefDeallocQubit>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefQubit>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMeasure>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefReset>(typeConverter, context, &state);

    ADD_CONVERT_PATTERN(GPhaseOp)
    ADD_CONVERT_PATTERN(IOp)
    ADD_CONVERT_PATTERN(BarrierOp)
    ADD_CONVERT_PATTERN(HOp)
    ADD_CONVERT_PATTERN(XOp)
    ADD_CONVERT_PATTERN(YOp)
    ADD_CONVERT_PATTERN(ZOp)
    ADD_CONVERT_PATTERN(SOp)
    ADD_CONVERT_PATTERN(SdgOp)
    ADD_CONVERT_PATTERN(TOp)
    ADD_CONVERT_PATTERN(TdgOp)
    ADD_CONVERT_PATTERN(VOp)
    ADD_CONVERT_PATTERN(VdgOp)
    ADD_CONVERT_PATTERN(UOp)
    ADD_CONVERT_PATTERN(U2Op)
    ADD_CONVERT_PATTERN(POp)
    ADD_CONVERT_PATTERN(SXOp)
    ADD_CONVERT_PATTERN(SXdgOp)
    ADD_CONVERT_PATTERN(RXOp)
    ADD_CONVERT_PATTERN(RYOp)
    ADD_CONVERT_PATTERN(RZOp)
    ADD_CONVERT_PATTERN(SWAPOp)
    ADD_CONVERT_PATTERN(iSWAPOp)
    ADD_CONVERT_PATTERN(iSWAPdgOp)
    ADD_CONVERT_PATTERN(PeresOp)
    ADD_CONVERT_PATTERN(PeresdgOp)
    ADD_CONVERT_PATTERN(DCXOp)
    ADD_CONVERT_PATTERN(ECROp)
    ADD_CONVERT_PATTERN(RXXOp)
    ADD_CONVERT_PATTERN(RYYOp)
    ADD_CONVERT_PATTERN(RZZOp)
    ADD_CONVERT_PATTERN(RZXOp)
    ADD_CONVERT_PATTERN(XXminusYYOp)
    ADD_CONVERT_PATTERN(XXplusYYOp)

    // conversion of mqtopt types in func.func signatures
    // does not work for now as signature needs to be changed
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    // conversion of mqtref types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtref types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtref types in control-flow ops; e.g. cf.br
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
