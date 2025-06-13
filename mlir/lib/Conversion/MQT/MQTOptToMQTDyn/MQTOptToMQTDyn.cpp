/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */
#include "mlir/Conversion/MQT/MQTOptToMQTDyn/MQTOptToMQTDyn.h"

#include "mlir/Dialect/Common/Compat.h"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <cassert>
#include <cstddef>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>
namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DEF_MQTOPTTOMQTDYN
#include <mlir/Conversion/MQT/MQTOptToMQTDyn/MQTOptToMQTDyn.h.inc>

class MQTOptToMQTDynTypeConverter : public TypeConverter {
public:
  explicit MQTOptToMQTDynTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](::mqt::ir::opt::QubitType /*type*/) -> Type {
      return ::mqt::ir::dyn::QubitType::get(ctx);
    });

    // QregType conversion
    addConversion([ctx](::mqt::ir::opt::QubitRegisterType /*type*/) -> Type {
      return ::mqt::ir::dyn::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertMQTOptAlloc
    : public OpConversionPattern<::mqt::ir::opt::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    auto qregType =
        ::mqt::ir::dyn::QubitRegisterType::get(rewriter.getContext());

    // create new operation
    auto mqtdynOp = rewriter.create<::mqt::ir::dyn::AllocOp>(
        op.getLoc(), qregType, adaptor.getSize(), adaptor.getSizeAttrAttr());

    // // replace the operand of the result qreg users with the input qreg of
    // the operation

    auto qreg = op->getResult(0);
    std::vector<mlir::Operation*> qregUsers(qreg.getUsers().begin(),
                                            qreg.getUsers().end());
    for (auto* user : llvm::reverse(qregUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtdynOp) && user != mqtdynOp && user != op) {
        // replace uses of the old qubit with the new qubit result
        user->replaceUsesOfWith(qreg, mqtdynOp->getResult(0));
      }
    }
    // erase the old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTOptDealloc
    : public OpConversionPattern<::mqt::ir::opt::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create new operation
    auto mqtdynOp = rewriter.create<::mqt::ir::dyn::DeallocOp>(
        op.getLoc(), ::mlir::TypeRange({}), adaptor.getQreg());

    // replace old operation with new operation
    rewriter.replaceOp(op, mqtdynOp);
    return success();
  }
};

struct ConvertMQTOptExtract
    : public OpConversionPattern<::mqt::ir::opt::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result types
    auto qubitType = ::mqt::ir::dyn::QubitType::get(rewriter.getContext());

    // create new operation
    auto mqtdynOp = rewriter.create<::mqt::ir::dyn::ExtractOp>(
        op.getLoc(), qubitType, adaptor.getInQreg(), adaptor.getIndex(),
        adaptor.getIndexAttrAttr());

    auto oldQubit = op->getResult(1);
    auto newQubit = mqtdynOp->getResult(0);
    std::vector<mlir::Operation*> qubitUsers(oldQubit.getUsers().begin(),
                                             oldQubit.getUsers().end());

    // replace the operand of the result qubit users with the input qubit of the
    // operation
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtdynOp) && user != mqtdynOp && user != op) {
        // replace uses of the old qubit with the new qubit result
        user->replaceUsesOfWith(oldQubit, newQubit);
      }
    }

    // replace the operand of the result qreg users with the input qreg of the
    // operation
    auto oldQreg = op->getResult(0);
    std::vector<mlir::Operation*> qregUsers(oldQreg.getUsers().begin(),
                                            oldQreg.getUsers().end());
    for (auto* user : llvm::reverse(qregUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtdynOp) && user != mqtdynOp && user != op) {
        // replace uses of the old qubit with the new qubit result
        user->replaceUsesOfWith(oldQreg, adaptor.getInQreg());
      }
    }

    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTOptInsert
    : public OpConversionPattern<::mqt::ir::opt::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // replace the operand of the result qreg users with the input qreg of the
    // operation
    auto oldQreg = op->getResult(0);
    std::vector<mlir::Operation*> qregUsers(oldQreg.getUsers().begin(),
                                            oldQreg.getUsers().end());
    for (auto* user : llvm::reverse(qregUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(op) && user != op) {
        // replace uses of the old qubit with the new qubit result
        user->replaceUsesOfWith(oldQreg, adaptor.getInQreg());
      }
    }
    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTOptMeasure
    : public OpConversionPattern<::mqt::ir::opt::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare result types
    auto bitType = rewriter.getI1Type();

    // create new operation
    auto mqtdynOp = rewriter.create<::mqt::ir::dyn::MeasureOp>(
        op.getLoc(), bitType, adaptor.getInQubits()[0]);

    auto optBit = op->getResult(1);

    auto optQubit = op->getResult(0);

    std::vector<mlir::Operation*> resultUsers(optBit.getUsers().begin(),
                                              optBit.getUsers().end());

    std::vector<mlir::Operation*> qubitUsers(optQubit.getUsers().begin(),
                                             optQubit.getUsers().end());

    // replace the operand of the result qubit users with the input qubit of the
    // current operation
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtdynOp) && user != mqtdynOp && user != op) {

        // replace the previous use of the input qubit with the result qubit
        user->replaceUsesOfWith(optQubit, adaptor.getInQubits()[0]);
      }
    }
    // replace the old bit with new bit
    for (auto* user : llvm::reverse(resultUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtdynOp) && user != mqtdynOp && user != op) {
        // replace uses of the old bit with the new bit result
        user->replaceUsesOfWith(optBit, mqtdynOp->getResult(0));
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename MQTGateOp>
struct ConvertMQTDynGateOp : public OpConversionPattern<MQTGateOp> {
  using OpConversionPattern<MQTGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOp op, typename MQTGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get all the input qubits including the ctrl qubits
    auto inQubitsValues = adaptor.getInQubits();
    auto posCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // append them to a single vector
    SmallVector<Value> values(inQubitsValues.begin(), inQubitsValues.end());
    values.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    values.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // get the static params and paramMask if they exist
    auto staticParams =
        op.getStaticParams().has_value()
            ? DenseF64ArrayAttr::get(rewriter.getContext(),
                                     op.getStaticParams().value())
            : mlir::DenseF64ArrayAttr{};
    auto paramMask = op.getParamsMask().has_value()
                         ? DenseBoolArrayAttr::get(rewriter.getContext(),
                                                   op.getParamsMask().value())
                         : mlir::DenseBoolArrayAttr{};

    // create new operation
    Operation* mqtdynOp;

    if (llvm::isa<::mqt::ir::opt::XOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::XOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::GPhaseOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::GPhaseOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::IOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::IOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::BarrierOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::BarrierOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::HOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::HOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::YOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::YOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::ZOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::ZOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::SOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::SOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::SdgOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::SdgOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::TOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::TOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::TdgOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::TdgOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::VOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::VOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::VdgOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::VdgOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::UOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::UOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::U2Op>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::U2Op>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::POp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::POp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::SXOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::SXOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::SXdgOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::SXdgOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RXOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RXOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RYOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RYOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RZOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RZOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::SWAPOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::SWAPOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::iSWAPOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::iSWAPOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::iSWAPdgOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::iSWAPdgOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::PeresOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::PeresOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::PeresdgOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::PeresdgOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::DCXOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::DCXOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::ECROp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::ECROp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RXXOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RXXOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RYYOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RYYOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RZZOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RZZOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::RZXOp>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::RZXOp>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::XXminusYY>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::XXminusYY>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::opt::XXplusYY>(op)) {
      mqtdynOp = rewriter.create<::mqt::ir::dyn::XXplusYY>(
          op.getLoc(), staticParams, paramMask, adaptor.getParams(),
          inQubitsValues, posCtrlQubitsValues, negCtrlQubitsValues);
    }

    // update the users of the gate operation results with the input qubits
    Value value;
    auto oldResults = op->getResults();
    for (size_t i = 0; i < oldResults.size(); i++) {
      value = oldResults[i];
      std::vector<mlir::Operation*> users(value.getUsers().begin(),
                                          value.getUsers().end());

      // Iterate over users in reverse order to update their operands properly
      for (auto* user : llvm::reverse(users)) {

        // Only consider operations after the current operation
        if (!user->isBeforeInBlock(mqtdynOp) && user != op) {

          user->replaceUsesOfWith(value, values[i]);
        }
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
    target.addIllegalDialect<::mqt::ir::opt::MQTOptDialect>();
    target.addLegalDialect<::mqt::ir::dyn::MQTDynDialect>();

    RewritePatternSet patterns(context);
    MQTOptToMQTDynTypeConverter typeConverter(context);

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptInsert,
                 ConvertMQTOptExtract, ConvertMQTOptMeasure>(typeConverter,
                                                             context);

    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::XOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::GPhaseOp>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::IOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::BarrierOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::HOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::YOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::ZOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::SOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::SdgOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::TOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::TdgOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::VOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::VdgOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::U2Op>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::POp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::SXOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::SXdgOp>>(typeConverter,
                                                              context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RXOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RYOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RZOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::SWAPOp>>(typeConverter,
                                                              context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::iSWAPOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::iSWAPdgOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::PeresOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::PeresdgOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::DCXOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::ECROp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RXXOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RYYOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RZZOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::RZXOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::XXminusYY>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::XXplusYY>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::opt::UOp>>(typeConverter,
                                                           context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mlir::mqt::ir::conversions