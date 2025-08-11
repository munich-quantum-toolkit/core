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
#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinAttributes.h>
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
  /// @brief Holds metadata for qubits.
  struct QubitData {
    /// @brief The ref register from which the qubit was extracted.
    Value qReg;
    /// @brief Index given as a value.
    Value index;
    /// @brief Index given as an attribute.
    IntegerAttr indexAttr;
  };

  /// @brief Map each initial ref qubit to its latest opt qubit.
  llvm::DenseMap<Value, Value> qubitMap;
  /// @brief Map each initial ref qubit to its metadata.
  llvm::DenseMap<Value, QubitData> qubitDataMap;
  /// @brief Map each initial ref register to its latest opt register.
  llvm::DenseMap<Value, Value> qregMap;
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
} // namespace

class MQTRefToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit MQTRefToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](ref::DynamicQubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // QregType conversion
    addConversion([ctx](ref::QubitRegisterType /*type*/) -> Type {
      return opt::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertMQTRefAlloc final : StatefulOpConversionPattern<ref::AllocOp> {
  using StatefulOpConversionPattern<ref::AllocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // save the ref register
    const auto& refQreg = op.getQreg();

    // prepare return type
    const auto& qregType = opt::QubitRegisterType::get(rewriter.getContext());

    // prepare size attribute
    auto sizeAttr = op.getSizeAttr()
                        ? rewriter.getI64IntegerAttr(
                              static_cast<int64_t>(*op.getSizeAttr()))
                        : IntegerAttr{};

    // replace the ref alloc operation with an opt alloc operation
    auto mqtoptOp = rewriter.replaceOpWithNewOp<opt::AllocOp>(
        op, qregType, op.getSize(), sizeAttr);

    // put the pair of the ref register and the latest opt register in the map
    getState().qregMap.try_emplace(refQreg, mqtoptOp.getQreg());

    return success();
  }
};

struct ConvertMQTRefDealloc final
    : StatefulOpConversionPattern<ref::DeallocOp> {
  using StatefulOpConversionPattern<
      ref::DeallocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare return type
    const auto& qregType = opt::QubitRegisterType::get(rewriter.getContext());

    const auto& refQreg = op.getQreg();
    auto& optQreg = getState().qregMap[refQreg];

    // iterate over all the qubits that were extracted from the register
    for (const auto& refQubit : getState().qregQubitsMap[refQreg]) {
      const auto& qubitData = getState().qubitDataMap[refQubit];
      const auto& optQubit = getState().qubitMap[refQubit];

      auto optInsertOp = rewriter.create<opt::InsertOp>(
          op.getLoc(), qregType, optQreg, optQubit, qubitData.index,
          qubitData.indexAttr);

      // move it before the current dealloc operation
      optInsertOp->moveBefore(op);

      // update the optQreg
      optQreg = optInsertOp.getOutQreg();

      // erase the refQubit entry from the maps
      getState().qubitMap.erase(refQubit);
      getState().qubitDataMap.erase(refQubit);
    }

    // erase the register from the maps
    getState().qregMap.erase(refQreg);
    getState().qregQubitsMap.erase(refQreg);

    // replace the ref dealloc operation with an opt dealloc operation
    rewriter.replaceOpWithNewOp<opt::DeallocOp>(op, optQreg);

    return success();
  }
};

struct ConvertMQTRefExtract final
    : StatefulOpConversionPattern<ref::ExtractOp> {
  using StatefulOpConversionPattern<
      ref::ExtractOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::ExtractOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // create result types
    const auto& qregType = opt::QubitRegisterType::get(rewriter.getContext());
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& refQreg = op.getInQreg();
    const auto& optQreg = getState().qregMap[refQreg];

    // create new operation
    auto mqtoptOp = rewriter.create<opt::ExtractOp>(
        op.getLoc(), qregType, qubitType, optQreg, op.getIndex(),
        op.getIndexAttrAttr());

    const auto& refQubit = op.getOutQubit();
    const auto& optQubit = mqtoptOp.getOutQubit();
    const auto& newOptQreg = mqtoptOp.getOutQreg();

    // put the pair of the ref qubit and the latest opt qubit in the map
    getState().qubitMap.try_emplace(refQubit, optQubit);

    // update the latest opt register of the initial ref register
    getState().qregMap[refQreg] = newOptQreg;

    // add an entry to the qubitDataMap to store the indices and the register
    // for the insertOperation
    getState().qubitDataMap.try_emplace(refQubit, refQreg, op.getIndex(),
                                        op.getIndexAttrAttr());

    // append the entry to the qregQubitsMap to store which qubits that were
    // extracted from the register
    getState().qregQubitsMap[refQreg].emplace_back(refQubit);

    // replace the old operation result with the new result and delete
    // old operation
    rewriter.replaceOp(op, optQubit);

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

    const auto& refQubits = op.getInQubits();

    // get the latest opt qubit from the map and add them to the vector
    std::vector<Value> optQubits;
    for (const auto& refQubit : refQubits) {
      optQubits.emplace_back(getState().qubitMap[refQubit]);
    }
    // create the result types
    const std::vector<Type> qubitTypes(optQubits.size(), qubitType);

    // create new operation
    auto mqtoptOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), qubitTypes, op.getOutBits().getTypes(), optQubits);

    const auto& outOptQubits = mqtoptOp.getOutQubits();
    const auto& newBits = mqtoptOp.getOutBits();

    // iterate over all qubits
    for (size_t i = 0; i < refQubits.size(); i++) {
      // update the latest opt qubit of the initial ref qubit
      getState().qubitMap[refQubits[i]] = outOptQubits[i];
    }

    // replace the old operation results with the new bits and delete
    // old operation
    rewriter.replaceOp(op, newBits);

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
    auto mqtoptOp = rewriter.create<MQTGateOptOp>(
        op.getLoc(), ValueRange(optInQubits).getTypes(),
        ValueRange(optPosCtrlQubitsValues).getTypes(),
        ValueRange(optNegCtrlQubitsValues).getTypes(), staticParams, paramMask,
        op.getParams(), optInQubits, optPosCtrlQubitsValues,
        optNegCtrlQubitsValues);

    // Update qubit map
    const auto& optResults = mqtoptOp.getAllOutQubits();
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
    patterns.add<ConvertMQTRefAlloc>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefDealloc>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefExtract>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMeasure>(typeConverter, context, &state);

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
    ADD_CONVERT_PATTERN(XXminusYY)
    ADD_CONVERT_PATTERN(XXplusYY)

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
