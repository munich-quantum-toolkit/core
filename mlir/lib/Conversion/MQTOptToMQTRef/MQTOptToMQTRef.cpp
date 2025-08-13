/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any opt gate operation to the same
// gate operation in the ref dialect
#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTOptGateOp<::mqt::ir::opt::gate, ::mqt::ir::ref::gate>>(   \
          typeConverter, context);

#include "mlir/Conversion/MQTOptToMQTRef/MQTOptToMQTRef.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <cstdint>
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

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTOPTTOMQTREF
#include "mlir/Conversion/MQTOptToMQTRef/MQTOptToMQTRef.h.inc"

class MQTOptToMQTRefTypeConverter final : public TypeConverter {
public:
  explicit MQTOptToMQTRefTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](opt::QubitType /*type*/) -> Type {
      return ref::QubitType::get(ctx);
    });

    // QregType conversion
    addConversion([ctx](opt::QubitRegisterType /*type*/) -> Type {
      return ref::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertMQTOptAlloc final : OpConversionPattern<opt::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare return type
    const auto& qregType = ref::QubitRegisterType::get(rewriter.getContext());

    // prepare size attribute
    auto sizeAttr = op.getSizeAttr()
                        ? rewriter.getI64IntegerAttr(
                              static_cast<int64_t>(*op.getSizeAttr()))
                        : IntegerAttr{};

    // replace the opt alloc operation with a ref alloc operation
    rewriter.replaceOpWithNewOp<ref::AllocOp>(op, qregType, op.getSize(),
                                              sizeAttr);

    return success();
  }
};

struct ConvertMQTOptDealloc final : OpConversionPattern<opt::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(const opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ref::DeallocOp>(op, adaptor.getQreg());
    return success();
  }
};

struct ConvertMQTOptExtract final : OpConversionPattern<opt::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    const auto& qubitType = ref::QubitType::get(rewriter.getContext());

    const auto& refQreg = adaptor.getInQreg();

    // create new operation
    auto mqtrefOp = rewriter.create<ref::ExtractOp>(op.getLoc(), qubitType,
                                                    refQreg, adaptor.getIndex(),
                                                    adaptor.getIndexAttrAttr());

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, ValueRange({refQreg, mqtrefOp.getOutQubit()}));
    return success();
  }
};

struct ConvertMQTOptInsert final : OpConversionPattern<opt::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // replace the result of the old operation with the new result and delete
    // old operation
    rewriter.replaceOp(op, adaptor.getInQreg());
    return success();
  }
};

struct ConvertMQTOptMeasure final : OpConversionPattern<opt::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    const auto& oldBit = op.getOutBit();
    const auto& refQubit = adaptor.getInQubit();

    // create new operation
    auto mqtrefOp = rewriter.create<ref::MeasureOp>(op.getLoc(),
                                                    oldBit.getType(), refQubit);

    const auto& newBit = mqtrefOp.getOutBit();

    // concatenate the ref qubit and the bit
    const SmallVector<Value> newValues{refQubit, newBit};

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, newValues);
    return success();
  }
};

struct ConvertMQTOptReset final : OpConversionPattern<opt::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = adaptor.getInQubit();

    // create new operation
    rewriter.create<ref::ResetOp>(op.getLoc(), refQubit);

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, refQubit);
    return success();
  }
};

template <typename MQTGateOptOp, typename MQTGateRefOp>
struct ConvertMQTOptGateOp final : OpConversionPattern<MQTGateOptOp> {
  using OpConversionPattern<MQTGateOptOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOptOp op, typename MQTGateOptOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get all the input qubits including the ctrl qubits
    const auto& refInQubitsValues = adaptor.getInQubits();
    const auto& refPosCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto& refNegCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // append them to a single vector
    SmallVector<Value> refQubitsValues;
    refQubitsValues.reserve(refInQubitsValues.size() +
                            refPosCtrlQubitsValues.size() +
                            refNegCtrlQubitsValues.size());
    refQubitsValues.append(refInQubitsValues.begin(), refInQubitsValues.end());
    refQubitsValues.append(refPosCtrlQubitsValues.begin(),
                           refPosCtrlQubitsValues.end());
    refQubitsValues.append(refNegCtrlQubitsValues.begin(),
                           refNegCtrlQubitsValues.end());

    // get the static params and paramMask if they exist
    auto staticParams = op.getStaticParams()
                            ? DenseF64ArrayAttr::get(rewriter.getContext(),
                                                     *op.getStaticParams())
                            : DenseF64ArrayAttr{};
    auto paramMask = op.getParamsMask()
                         ? DenseBoolArrayAttr::get(rewriter.getContext(),
                                                   *op.getParamsMask())
                         : DenseBoolArrayAttr{};

    // create new operation
    rewriter.create<MQTGateRefOp>(
        op.getLoc(), staticParams, paramMask, op.getParams(), refInQubitsValues,
        refPosCtrlQubitsValues, refNegCtrlQubitsValues);

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, refQubitsValues);

    return success();
  }
};

struct MQTOptToMQTRef final : impl::MQTOptToMQTRefBase<MQTOptToMQTRef> {
  using MQTOptToMQTRefBase::MQTOptToMQTRefBase;
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTOptToMQTRefTypeConverter typeConverter(context);

    target.addIllegalDialect<opt::MQTOptDialect>();
    target.addLegalDialect<ref::MQTRefDialect>();

    patterns
        .add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptInsert,
             ConvertMQTOptExtract, ConvertMQTOptMeasure, ConvertMQTOptReset>(
            typeConverter, context);

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
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // conversion of mqtopt types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtopt types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtopt types in control-flow ops; e.g. cf.br
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
