/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTOptGateOp<::mqt::ir::opt::gate, ::mqt::ir::dyn::gate>>(   \
          typeConverter, context);

#include "mlir/Conversion/MQTOptToMQTDyn/MQTOptToMQTDyn.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <cstddef>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>
#include <vector>

using namespace mlir;

namespace mqt::ir {

#define GEN_PASS_DEF_MQTOPTTOMQTDYN
#include <mlir/Conversion/MQTOptToMQTDyn/MQTOptToMQTDyn.h.inc>

class MQTOptToMQTDynTypeConverter : public TypeConverter {
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

struct ConvertMQTOptAlloc : public OpConversionPattern<opt::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    auto qregType = dyn::QubitRegisterType::get(rewriter.getContext());

    // create new operation
    auto mqtdynOp = rewriter.create<dyn::AllocOp>(
        op.getLoc(), qregType, adaptor.getSize(), adaptor.getSizeAttrAttr());

    auto optQreg = op.getQreg();
    auto dynQreg = mqtdynOp.getQreg();

    // get a view of the opt register users that can be modified
    const std::vector<Operation*> optQregUsers(optQreg.getUsers().begin(),
                                               optQreg.getUsers().end());

    // update the operand of the opt register user
    for (auto* user : optQregUsers) {
      user->replaceUsesOfWith(optQreg, dynQreg);
    }

    // erase the old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTOptDealloc : public OpConversionPattern<opt::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create new operation
    auto mqtdynOp =
        rewriter.create<dyn::DeallocOp>(op.getLoc(), adaptor.getQreg());

    // replace old operation with new operation
    rewriter.replaceOp(op, mqtdynOp);

    return success();
  }
};

struct ConvertMQTOptExtract : public OpConversionPattern<opt::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    auto qubitType = dyn::QubitType::get(rewriter.getContext());

    auto dynQreg = adaptor.getInQreg();

    // create new operation
    auto mqtdynOp = rewriter.create<dyn::ExtractOp>(op.getLoc(), qubitType,
                                                    dynQreg, adaptor.getIndex(),
                                                    adaptor.getIndexAttrAttr());

    auto optQubit = op.getOutQubit();
    auto dynQubit = mqtdynOp.getOutQubit();
    auto optQreg = op.getOutQreg();

    // get a view of the opt qubit users that can be modified
    const std::vector<Operation*> optQubitUsers(optQubit.getUsers().begin(),
                                                optQubit.getUsers().end());

    // update the operand of the opt qubit user
    for (auto* user : optQubitUsers) {
      user->replaceUsesOfWith(optQubit, dynQubit);
    }

    // get a view of the opt register users that can be modified
    const std::vector<Operation*> optQregUsers(optQreg.getUsers().begin(),
                                               optQreg.getUsers().end());

    // update the operand of the opt register user
    for (auto* user : optQregUsers) {
      user->replaceUsesOfWith(optQreg, dynQreg);
    }

    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTOptInsert : public OpConversionPattern<opt::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    auto optQreg = op.getOutQreg();
    auto dynQreg = adaptor.getInQreg();

    // get a view of the opt register users that can be modified
    const std::vector<Operation*> optQregUsers(optQreg.getUsers().begin(),
                                               optQreg.getUsers().end());

    // update the operand of the opt register user
    for (auto* user : optQregUsers) {
      user->replaceUsesOfWith(optQreg, dynQreg);
    }

    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTOptMeasure : public OpConversionPattern<opt::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    auto oldBits = op.getOutBits();
    auto dynQubits = adaptor.getInQubits();
    auto optQubits = op.getOutQubits();

    // create new operation
    auto mqtdynOp = rewriter.create<dyn::MeasureOp>(
        op.getLoc(), oldBits.getTypes(), dynQubits);

    auto newBits = mqtdynOp.getOutBits();

    Value dynQubit = nullptr;
    Value optQubit = nullptr;
    Value oldBit = nullptr;
    Value newBit = nullptr;

    // iterate over the qubits and bits
    for (size_t i = 0; i < optQubits.size(); i++) {
      dynQubit = dynQubits[i];
      optQubit = optQubits[i];
      oldBit = oldBits[i];
      newBit = newBits[i];

      // get a view of the opt qubit users that can be modified
      const std::vector<mlir::Operation*> qubitUsers(
          optQubit.getUsers().begin(), optQubit.getUsers().end());

      // update the operand of the opt qubit user
      for (auto* user : qubitUsers) {
        user->replaceUsesOfWith(optQubit, dynQubit);
      }

      const std::vector<mlir::Operation*> bitUsers(oldBit.getUsers().begin(),
                                                   oldBit.getUsers().end());
      // iterate over the users of the old bit and replace the old bit with the
      // new bit
      for (auto* user : bitUsers) {
        user->replaceUsesOfWith(oldBit, newBit);
      }
    }

    // erase the previous operation
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename MQTGateOptOp, typename MQTGateDynOp>
struct ConvertMQTOptGateOp : public OpConversionPattern<MQTGateOptOp> {
  using OpConversionPattern<MQTGateOptOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOptOp op, typename MQTGateOptOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get all the input qubits including the ctrl qubits
    auto dynInQubitsValues = adaptor.getInQubits();
    auto dynPosCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto dynNegCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // append them to a single vector
    std::vector<Value> allDynInputQubits(dynInQubitsValues.begin(),
                                         dynInQubitsValues.end());
    allDynInputQubits.insert(allDynInputQubits.end(),
                             dynPosCtrlQubitsValues.begin(),
                             dynPosCtrlQubitsValues.end());
    allDynInputQubits.insert(allDynInputQubits.end(),
                             dynNegCtrlQubitsValues.begin(),
                             dynNegCtrlQubitsValues.end());

    // get the static params and paramMask if they exist
    DenseF64ArrayAttr staticParams = nullptr;
    if (auto optionalParams = op.getStaticParams()) {
      staticParams = mlir::DenseF64ArrayAttr::get(rewriter.getContext(),
                                                  optionalParams.value());
    } else {
      staticParams = DenseF64ArrayAttr{};
    }
    DenseBoolArrayAttr paramMask = nullptr;
    if (auto optionalMask = op.getParamsMask()) {
      paramMask = mlir::DenseBoolArrayAttr::get(rewriter.getContext(),
                                                optionalMask.value());
    } else {
      paramMask = DenseBoolArrayAttr{};
    }
    // create new operation
    rewriter.create<MQTGateDynOp>(
        op.getLoc(), staticParams, paramMask, op.getParams(), dynInQubitsValues,
        dynPosCtrlQubitsValues, dynNegCtrlQubitsValues);

    Value optQubit = nullptr;
    Value dynQubit = nullptr;
    auto optResults = op->getResults();

    // iterate over all opt qubits
    for (size_t i = 0; i < optResults.size(); i++) {
      optQubit = optResults[i];
      dynQubit = allDynInputQubits[i];

      // get a view of the opt qubit users that can be modified
      const std::vector<mlir::Operation*> qubitUsers(
          optQubit.getUsers().begin(), optQubit.getUsers().end());

      // update the operand of the opt qubit user
      for (auto* user : qubitUsers) {
        user->replaceUsesOfWith(optQubit, dynQubit);
      }
    }

    // erase the previous operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct MQTOptToMQTDyn : impl::MQTOptToMQTDynBase<MQTOptToMQTDyn> {
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
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtopt types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtopt types in control-flow ops; e.g. cf.br
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
