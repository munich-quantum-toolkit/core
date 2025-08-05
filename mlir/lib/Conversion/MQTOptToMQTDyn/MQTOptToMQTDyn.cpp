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
// gate operation in the dyn dialect
#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTOptGateOp<::mqt::ir::opt::gate, ::mqt::ir::dyn::gate>>(   \
          typeConverter, context);

#include "mlir/Conversion/MQTOptToMQTDyn/MQTOptToMQTDyn.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

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

#define GEN_PASS_DEF_MQTOPTTOMQTDYN
#include "mlir/Conversion/MQTOptToMQTDyn/MQTOptToMQTDyn.h.inc"

class MQTOptToMQTDynTypeConverter final : public TypeConverter {
public:
  explicit MQTOptToMQTDynTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](opt::QubitType /*type*/) -> Type {
      return dyn::QubitType::get(ctx);
    });

    // QregType conversion
    addConversion([ctx](opt::QubitRegisterType /*type*/) -> Type {
      return dyn::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertMQTOptAlloc final : OpConversionPattern<opt::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare return type
    const auto& qregType = dyn::QubitRegisterType::get(rewriter.getContext());

    // prepare size attribute
    auto sizeAttr = op.getSizeAttr()
                        ? rewriter.getI64IntegerAttr(
                              static_cast<int64_t>(*op.getSizeAttr()))
                        : IntegerAttr{};

    // replace the opt alloc operation with a dyn alloc operation
    rewriter.replaceOpWithNewOp<dyn::AllocOp>(op, qregType, op.getSize(),
                                              sizeAttr);

    return success();
  }
};

struct ConvertMQTOptDealloc final : OpConversionPattern<opt::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(const opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<dyn::DeallocOp>(op, adaptor.getQreg());
    return success();
  }
};

struct ConvertMQTOptExtract final : OpConversionPattern<opt::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    const auto& qubitType = dyn::QubitType::get(rewriter.getContext());

    const auto& dynQreg = adaptor.getInQreg();

    // create new operation
    auto mqtdynOp = rewriter.create<dyn::ExtractOp>(op.getLoc(), qubitType,
                                                    dynQreg, adaptor.getIndex(),
                                                    adaptor.getIndexAttrAttr());

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, ValueRange({dynQreg, mqtdynOp.getOutQubit()}));
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

    const auto& oldBits = op.getOutBits();
    const auto& dynQubits = adaptor.getInQubits();

    // create new operation
    auto mqtdynOp = rewriter.create<dyn::MeasureOp>(
        op.getLoc(), oldBits.getTypes(), dynQubits);

    const auto& newBits = mqtdynOp.getOutBits();

    // concatenate the dyn qubits and the bits
    SmallVector<Value> newValues;
    newValues.reserve(dynQubits.size() + newBits.size());
    newValues.append(dynQubits.begin(), dynQubits.end());
    newValues.append(newBits.begin(), newBits.end());

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, newValues);
    return success();
  }
};

template <typename MQTGateOptOp, typename MQTGateDynOp>
struct ConvertMQTOptGateOp final : OpConversionPattern<MQTGateOptOp> {
  using OpConversionPattern<MQTGateOptOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOptOp op, typename MQTGateOptOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get all the input qubits including the ctrl qubits
    const auto& dynInQubitsValues = adaptor.getInQubits();
    const auto& dynPosCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto& dynNegCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // append them to a single vector
    SmallVector<Value> dynQubitsValues;
    dynQubitsValues.reserve(dynInQubitsValues.size() +
                            dynPosCtrlQubitsValues.size() +
                            dynNegCtrlQubitsValues.size());
    dynQubitsValues.append(dynInQubitsValues.begin(), dynInQubitsValues.end());
    dynQubitsValues.append(dynPosCtrlQubitsValues.begin(),
                           dynPosCtrlQubitsValues.end());
    dynQubitsValues.append(dynNegCtrlQubitsValues.begin(),
                           dynNegCtrlQubitsValues.end());

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
    rewriter.create<MQTGateDynOp>(
        op.getLoc(), staticParams, paramMask, op.getParams(), dynInQubitsValues,
        dynPosCtrlQubitsValues, dynNegCtrlQubitsValues);

    // replace the results of the old operation with the new results and delete
    // old operation
    rewriter.replaceOp(op, dynQubitsValues);

    return success();
  }
};

struct MQTOptToMQTDyn final : impl::MQTOptToMQTDynBase<MQTOptToMQTDyn> {
  using MQTOptToMQTDynBase::MQTOptToMQTDynBase;
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTOptToMQTDynTypeConverter typeConverter(context);

    target.addIllegalDialect<opt::MQTOptDialect>();
    target.addLegalDialect<dyn::MQTDynDialect>();

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptInsert,
                 ConvertMQTOptExtract, ConvertMQTOptMeasure>(typeConverter,
                                                             context);

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
