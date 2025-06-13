/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */
#include "mlir/Conversion/MQT/MQTDynToMQTOpt/MQTDynToMQTOpt.h"

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

#define GEN_PASS_DEF_MQTDYNTOMQTOPT
#include <mlir/Conversion/MQT/MQTDynToMQTOpt/MQTDynToMQTOpt.h.inc>

class MQTDynToMQTOptTypeConverter : public TypeConverter {
public:
  explicit MQTDynToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](::mqt::ir::dyn::QubitType /*type*/) -> Type {
      return ::mqt::ir::opt::QubitType::get(ctx);
    });

    // QuregType conversion
    addConversion([ctx](::mqt::ir::dyn::QubitRegisterType /*type*/) -> Type {
      return ::mqt::ir::opt::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertMQTDynAlloc
    : public OpConversionPattern<::mqt::ir::dyn::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::dyn::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    auto quregType =
        ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());

    // create new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::AllocOp>(
        op.getLoc(), quregType, adaptor.getSize(), adaptor.getSizeAttrAttr());

    // replace old operation with new operation

    auto qureg = op->getResult(0);
    std::vector<mlir::Operation*> quregUsers(qureg.getUsers().begin(),
                                             qureg.getUsers().end());
    for (auto* user : llvm::reverse(quregUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        // replace uses of the old qubit with the new qubit result
        user->replaceUsesOfWith(qureg, mqtoptOp->getResult(0));
      }
    }
    // rewriter.replaceOp(op, mqtoptOp);
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTDynDealloc
    : public OpConversionPattern<::mqt::ir::dyn::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::dyn::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::DeallocOp>(
        op.getLoc(), ::mlir::TypeRange({}), adaptor.getQreg());

    // replace old operation with new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct ConvertMQTDynExtract
    : public OpConversionPattern<::mqt::ir::dyn::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::dyn::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result types
    auto quregType =
        ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());
    auto qubitType = ::mqt::ir::opt::QubitType::get(rewriter.getContext());

    // create new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::ExtractOp>(
        op.getLoc(), quregType, qubitType, adaptor.getInQreg(),
        adaptor.getIndex(), adaptor.getIndexAttrAttr());

    // get the alloc operation to find the dealloc operation
    auto qureg = mqtoptOp.getInQreg();
    std::vector<mlir::Operation*> quregUsers(qureg.getUsers().begin(),
                                             qureg.getUsers().end());

    for (auto* user : quregUsers) {

      // Only consider operations after the current operation
      if (auto deallocOp = dyn_cast<::mqt::ir::dyn::DeallocOp>(user)) {
        // create insert operation

        auto mqtoptInsertOp = rewriter.create<::mqt::ir::opt::InsertOp>(
            mqtoptOp.getLoc(), quregType, mqtoptOp->getResult(0),
            mqtoptOp->getResult(1), mqtoptOp.getIndex(),
            mqtoptOp.getIndexAttrAttr());

        // move insert operation to the end before the dealloc operation
        mqtoptInsertOp->moveBefore(deallocOp);
        deallocOp->replaceUsesOfWith(mqtoptOp.getInQreg(),
                                     mqtoptInsertOp->getResult(0));
      } else if (auto insertOp = dyn_cast<::mqt::ir::opt::InsertOp>(user)) {

        auto mqtoptInsertOp = rewriter.create<::mqt::ir::opt::InsertOp>(
            mqtoptOp.getLoc(), quregType, insertOp->getResult(0),
            mqtoptOp->getResult(1), mqtoptOp.getIndex(),
            mqtoptOp.getIndexAttrAttr());
        mqtoptInsertOp->moveAfter(insertOp);
        std::vector<mlir::Operation*> insertUsers(
            insertOp->getResult(0).getUsers().begin(),
            insertOp->getResult(0).getUsers().end());
        for (auto* insertUser : insertUsers) {

          // Only consider operations after the current operation
          if (!insertUser->isBeforeInBlock(mqtoptInsertOp) &&
              insertUser != mqtoptInsertOp) {

            // replace uses of the old qubit with the new qubit result
            insertUser->replaceUsesOfWith(insertOp->getResult(0),
                                          mqtoptInsertOp->getResult(0));
          }
        }
        //    insertUsers[0]->replaceUsesOfWith(insertOp->getResult(0),
        //                                        mqtoptInsertOp->getResult(0));
      }
    }

    auto oldQubit = op->getResult(0);
    auto newQubit = mqtoptOp->getResult(1);
    std::vector<mlir::Operation*> qubitUsers(oldQubit.getUsers().begin(),
                                             oldQubit.getUsers().end());

    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        // replace uses of the old qubit with the new qubit result
        user->replaceUsesOfWith(oldQubit, newQubit);
      }
    }

    for (auto* user : quregUsers) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        // doesnt work
        user->replaceUsesOfWith(qureg, mqtoptOp->getResult(0));
      }
    }

    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTDynMeasure
    : public OpConversionPattern<::mqt::ir::dyn::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mqt::ir::dyn::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare result types
    auto qubitType = ::mqt::ir::opt::QubitType::get(rewriter.getContext());
    auto bitType = rewriter.getI1Type();

    // create new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::MeasureOp>(
        op.getLoc(), qubitType, bitType, adaptor.getInQubits()[0]);

    auto dynResult = op->getResult(0);

    std::vector<mlir::Operation*> resultUsers(dynResult.getUsers().begin(),
                                              dynResult.getUsers().end());

    std::vector<mlir::Operation*> qubitUsers(
        op.getInQubits()[0].getUsers().begin(),
        op.getInQubits()[0].getUsers().end());

    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {

        // replace the previous use of the input qubit with the result qubit
        user->replaceUsesOfWith(op.getInQubits()[0], mqtoptOp->getResult(0));
      }
    }
    for (auto* user : llvm::reverse(resultUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        // replace uses of the old bit with the new bit result
        user->replaceUsesOfWith(dynResult, mqtoptOp->getResult(1));
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
    Operation* mqtoptOp;

    if (llvm::isa<::mqt::ir::dyn::XOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::XOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::GPhaseOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::GPhaseOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::IOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::IOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::BarrierOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::BarrierOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::HOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::HOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::YOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::YOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::ZOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::ZOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::SOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::SOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::SdgOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::SdgOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);

    } else if (llvm::isa<::mqt::ir::dyn::TOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::TOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::TdgOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::TdgOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::VOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::VOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::VdgOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::VdgOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::UOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::UOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::U2Op>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::U2Op>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::POp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::POp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::SXOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::SXOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::SXdgOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::SXdgOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RXOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RXOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RYOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RYOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RZOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RZOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::SWAPOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::SWAPOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::iSWAPOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::iSWAPOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::iSWAPdgOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::iSWAPdgOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::PeresOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::PeresOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::PeresdgOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::PeresdgOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::DCXOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::DCXOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::ECROp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::ECROp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RXXOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RXXOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RYYOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RYYOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RZZOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RZZOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::RZXOp>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RZXOp>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::XXminusYY>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::XXminusYY>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else if (llvm::isa<::mqt::ir::dyn::XXplusYY>(op)) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::XXplusYY>(
          op.getLoc(), inQubitsValues.getType(), posCtrlQubitsValues.getType(),
          negCtrlQubitsValues.getType(), staticParams, paramMask,
          adaptor.getParams(), inQubitsValues, posCtrlQubitsValues,
          negCtrlQubitsValues);
    } else {
      return failure();
    }
    // replace the uses of the input qubits with the result of the new
    // operation
    Value value;
    for (size_t i = 0; i < values.size(); i++) {
      value = values[i];
      std::vector<mlir::Operation*> users(value.getUsers().begin(),
                                          value.getUsers().end());

      // Iterate over users in reverse order to update their operands properly
      for (auto* user : llvm::reverse(users)) {

        // Only consider operations after the current operation
        if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp &&
            user != op) {

          user->replaceUsesOfWith(value, mqtoptOp->getResult(i));
        }
      }
    }

    // erase the previous operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct MQTDynToMQTOpt : impl::MQTDynToMQTOptBase<MQTDynToMQTOpt> {
  using MQTDynToMQTOptBase::MQTDynToMQTOptBase;
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ConversionTarget target(*context);
    target.addIllegalDialect<::mqt::ir::dyn::MQTDynDialect>();
    target.addLegalDialect<::mqt::ir::opt::MQTOptDialect>();

    RewritePatternSet patterns(context);
    MQTDynToMQTOptTypeConverter typeConverter(context);

    patterns.add<ConvertMQTDynAlloc, ConvertMQTDynExtract, ConvertMQTDynMeasure,
                 ConvertMQTDynDealloc>(typeConverter, context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::XOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::GPhaseOp>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::IOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::BarrierOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::HOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::YOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::ZOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::SOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::SdgOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::TOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::TdgOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::VOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::VdgOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::U2Op>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::POp>>(typeConverter,
                                                           context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::SXOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::SXdgOp>>(typeConverter,
                                                              context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RXOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RYOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RZOp>>(typeConverter,
                                                            context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::SWAPOp>>(typeConverter,
                                                              context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::iSWAPOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::iSWAPdgOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::PeresOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::PeresdgOp>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::DCXOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::ECROp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RXXOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RYYOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RZZOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::RZXOp>>(typeConverter,
                                                             context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::XXminusYY>>(typeConverter,
                                                                 context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::XXplusYY>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTDynGateOp<::mqt::ir::dyn::UOp>>(typeConverter,
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