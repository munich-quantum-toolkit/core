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
      .add<ConvertMQTDynGateOp<::mqt::ir::dyn::gate, ::mqt::ir::opt::gate>>(   \
          typeConverter, context);

#include "mlir/Conversion/MQTDynToMQTOpt/MQTDynToMQTOpt.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/IR/ValueRange.h"

#include <cstddef>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>
#include <vector>
namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTDYNTOMQTOPT
#include <mlir/Conversion/MQTDynToMQTOpt/MQTDynToMQTOpt.h.inc>

namespace {
// struct to store the metadata for qubits
struct QubitData {
  Value qReg;
  Value index;
  IntegerAttr indexAttr;
  QubitData(Value qReg, Value index, IntegerAttr indexAttr)
      : qReg(qReg), index(index), indexAttr(indexAttr) {}
  QubitData() : qReg(nullptr), index(nullptr), indexAttr(nullptr) {}
};
llvm::DenseMap<mlir::Value, mlir::Value>& getQubitMap() {
  static llvm::DenseMap<mlir::Value, mlir::Value> qubitMap;
  return qubitMap;
}
llvm::DenseMap<mlir::Value, mlir::Value>& getQregMap() {
  static llvm::DenseMap<mlir::Value, mlir::Value> qregMap;
  return qregMap;
}
llvm::DenseMap<mlir::Value, QubitData>& getQubitDataMap() {
  static llvm::DenseMap<mlir::Value, QubitData> qubitDataMap;
  return qubitDataMap;
}

} // namespace
class MQTDynToMQTOptTypeConverter : public TypeConverter {
public:
  explicit MQTDynToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](dyn::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // QregType conversion
    addConversion([ctx](dyn::QubitRegisterType /*type*/) -> Type {
      return opt::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertMQTDynAlloc : public OpConversionPattern<dyn::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result type
    auto qregType = opt::QubitRegisterType::get(rewriter.getContext());

    // create new operation
    auto mqtoptOp = rewriter.create<opt::AllocOp>(
        op.getLoc(), qregType, adaptor.getSize(), adaptor.getSizeAttrAttr());

    auto dynQreg = op.getQreg();
    auto optQreg = mqtoptOp.getQreg();

    // put the pair of the dyn register and the latest opt register in the map
    getQregMap().insert({dynQreg, optQreg});

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTDynDealloc : public OpConversionPattern<dyn::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare return type

    auto qregType =
        ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());

    auto dynQreg = op.getQreg();

    // get the maps for the qubits and the qubitdata
    auto& qubitMap = getQubitMap();
    auto& qubitDataMap = getQubitDataMap();
    auto& qregMap = getQregMap();
    // iterate over all qubits to check if the qubit needs to be inserted in the
    // register before the dealloc operation is called
    for (auto qubitPair = qubitMap.begin(); qubitPair != qubitMap.end();) {
      auto dynQubit = qubitPair->getFirst();
      auto optQreg = qregMap[dynQreg];

      // work around to delete the inserted qubit from the maps
      auto toErase = qubitPair;
      ++qubitPair;
      // check if the qubit is extracted from the same register
      if (dynQreg == qubitDataMap[dynQubit].qReg) {
        auto optQubit = toErase->second;

        // create insert operation
        auto optInsertOp = rewriter.create<opt::InsertOp>(
            op.getLoc(), qregType, optQreg, optQubit,
            qubitDataMap[dynQubit].index, qubitDataMap[dynQubit].indexAttr);

        // move it before the current dealloc operation
        optInsertOp->moveBefore(op);
        // update the latest opt register of the initial dyn register
        qregMap[dynQreg] = optInsertOp.getOutQreg();

        // remove the qubits from the maps
        qubitMap.erase(toErase);
        qubitDataMap.erase(dynQubit);
      }
    }
    auto optQreg = qregMap[dynQreg];

    // create the new dealloc operation
    auto mqtoptOp = rewriter.create<opt::DeallocOp>(op.getLoc(), optQreg);

    // erase the register from the map
    qregMap.erase(dynQreg);

    // replace old operation with new operation
    rewriter.replaceOp(op, mqtoptOp);

    return success();
  }
};

struct ConvertMQTDynExtract : public OpConversionPattern<dyn::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // create result types
    auto qregType = opt::QubitRegisterType::get(rewriter.getContext());
    auto qubitType = opt::QubitType::get(rewriter.getContext());

    // get the latest opt register from the map
    auto dynQreg = op.getInQreg();
    auto optQreg = getQregMap()[dynQreg];
    // create new operation
    auto mqtoptOp = rewriter.create<opt::ExtractOp>(
        op.getLoc(), qregType, qubitType, optQreg, adaptor.getIndex(),
        adaptor.getIndexAttrAttr());

    auto dynQubit = op.getOutQubit();
    auto optQubit = mqtoptOp.getOutQubit();
    auto newQreg = mqtoptOp.getOutQreg();

    ////put the pair of the dyn qubit and the latest opt qubit in the map
    getQubitMap().insert({dynQubit, optQubit});

    // update the latest opt register of the initial dyn register
    getQregMap()[dynQreg] = newQreg;

    // add an entry to the qubitDataMap to store the indices and the register
    // for the insertOperation
    getQubitDataMap().insert({dynQubit, QubitData(dynQreg, adaptor.getIndex(),
                                                  adaptor.getIndexAttrAttr())});
    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTDynMeasure : public OpConversionPattern<dyn::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare result types
    auto qubitType = opt::QubitType::get(rewriter.getContext());
    auto bitType = rewriter.getI1Type();

    auto& qubitMap = getQubitMap();
    // get the latest opt qubit from the map

    auto dynQubit = op.getInQubits()[0];
    auto oldQubit = qubitMap[dynQubit];
    // create new operation
    auto mqtoptOp = rewriter.create<opt::MeasureOp>(op.getLoc(), qubitType,
                                                    bitType, oldQubit);

    auto oldBit = op->getResult(0);
    auto newBit = mqtoptOp->getResult(1);
    auto newQubit = mqtoptOp->getResult(0);

    // update the latest opt qubit of the initial dyn qubit
    qubitMap[dynQubit] = newQubit;

    // iterate over the users of the old bit and replace the operands with
    // the new one
    for (auto* user : oldBit.getUsers()) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        user->replaceUsesOfWith(oldBit, newBit);
      }
    }

    // erase old operation
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename MQTGateDynOp, typename MQTGateOptOp>
struct ConvertMQTDynGateOp : public OpConversionPattern<MQTGateDynOp> {
  using OpConversionPattern<MQTGateDynOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateDynOp op, typename MQTGateDynOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get all the input qubits including the ctrl qubits
    auto dynInQubitsValues = op.getInQubits();
    auto dynPosCtrlQubitsValues = op.getPosCtrlInQubits();
    auto dynNegCtrlQubitsValues = op.getNegCtrlInQubits();
    auto dynAllQubits = op.getAllInQubits();

    // get the latest opt qubit of all dyn input qubits
    std::vector<Value> optInQubits;
    std::vector<Value> optPosCtrlQubitsValues;
    std::vector<Value> optNegCtrlQubitsValues;

    for (auto dynQubit : dynInQubitsValues) {
      optInQubits.push_back(getQubitMap()[dynQubit]);
    }
    for (auto dynQubit : dynPosCtrlQubitsValues) {
      optPosCtrlQubitsValues.push_back(getQubitMap()[dynQubit]);
    }
    for (auto dynQubit : dynNegCtrlQubitsValues) {
      optNegCtrlQubitsValues.push_back(getQubitMap()[dynQubit]);
    }

    // append them to a single vector
    std::vector<Value> allOptQubits(optInQubits.begin(), optInQubits.end());
    allOptQubits.insert(allOptQubits.end(), optPosCtrlQubitsValues.begin(),
                        optPosCtrlQubitsValues.end());
    allOptQubits.insert(allOptQubits.end(), optNegCtrlQubitsValues.begin(),
                        optNegCtrlQubitsValues.end());

    // get the static params and paramMask if they exist
    DenseF64ArrayAttr staticParams = nullptr;
    if (auto params = op.getStaticParams()) {
      staticParams =
          mlir::DenseF64ArrayAttr::get(rewriter.getContext(), params.value());
    } else {
      staticParams = DenseF64ArrayAttr{};
    }
    DenseBoolArrayAttr paramMask = nullptr;
    if (auto mask = op.getParamsMask()) {
      paramMask =
          mlir::DenseBoolArrayAttr::get(rewriter.getContext(), mask.value());
    } else {
      paramMask = DenseBoolArrayAttr{};
    }

    // create new operation
    Operation* mqtoptOp = rewriter.create<MQTGateOptOp>(
        op.getLoc(), ValueRange(optInQubits).getTypes(),
        ValueRange(optPosCtrlQubitsValues).getTypes(),
        ValueRange(optNegCtrlQubitsValues).getTypes(), staticParams, paramMask,
        adaptor.getParams(), optInQubits, optPosCtrlQubitsValues,
        optNegCtrlQubitsValues);

    // iterate over all the dyn input qubits and update their latest opt qubit
    Value dynQubit = nullptr;
    for (size_t i = 0; i < dynAllQubits.size(); i++) {
      dynQubit = dynAllQubits[i];
      getQubitMap()[dynQubit] = mqtoptOp->getResult(i);
    }

    // erase the old operation
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
    RewritePatternSet patterns(context);
    MQTDynToMQTOptTypeConverter typeConverter(context);

    target.addIllegalDialect<dyn::MQTDynDialect>();
    target.addLegalDialect<opt::MQTOptDialect>();

    patterns.add<ConvertMQTDynAlloc, ConvertMQTDynExtract, ConvertMQTDynMeasure,
                 ConvertMQTDynDealloc>(typeConverter, context);

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

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
