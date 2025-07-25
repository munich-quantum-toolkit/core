/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any dyn gate operation to the same
// gate operation in the opt dialect
#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTDynGateOp<::mqt::ir::dyn::gate, ::mqt::ir::opt::gate>>(   \
          typeConverter, context, qubitMap);

#include "mlir/Conversion/MQTDynToMQTOpt/MQTDynToMQTOpt.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

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

#define GEN_PASS_DEF_MQTDYNTOMQTOPT
#include "mlir/Conversion/MQTDynToMQTOpt/MQTDynToMQTOpt.h.inc"

namespace {
// struct to store the metadata for qubits
struct QubitData {
  // dyn register from where the qubit was extracted
  Value qReg;
  // index given as a value
  Value index;
  // index given as an attribute
  IntegerAttr indexAttr;
};
} // namespace

class MQTDynToMQTOptTypeConverter final : public TypeConverter {
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

struct ConvertMQTDynAlloc final : OpConversionPattern<dyn::AllocOp> {
  llvm::DenseMap<Value, Value>* qregMap;

  explicit ConvertMQTDynAlloc(const TypeConverter& typeConverter,
                              MLIRContext* context,
                              llvm::DenseMap<Value, Value>& qregMap)
      : OpConversionPattern(typeConverter, context), qregMap(&qregMap) {}

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // save the dyn register
    const auto& dynQreg = op.getQreg();

    // prepare return type
    const auto& qregType = opt::QubitRegisterType::get(rewriter.getContext());

    // prepare size attribute
    auto sizeAttr = op.getSizeAttr()
                        ? rewriter.getI64IntegerAttr(
                              static_cast<int64_t>(*op.getSizeAttr()))
                        : IntegerAttr{};

    // replace the dyn alloc operation with an opt alloc operation
    auto mqtoptOp = rewriter.replaceOpWithNewOp<opt::AllocOp>(
        op, qregType, op.getSize(), sizeAttr);

    // put the pair of the dyn register and the latest opt register in the map
    qregMap->try_emplace(dynQreg, mqtoptOp.getQreg());

    return success();
  }
};

struct ConvertMQTDynDealloc final : OpConversionPattern<dyn::DeallocOp> {

  llvm::DenseMap<Value, Value>* qubitMap;
  llvm::DenseMap<Value, QubitData>* qubitDataMap;
  llvm::DenseMap<Value, Value>* qregMap;
  llvm::DenseMap<Value, std::vector<Value>>* qregQubitsMap;

  explicit ConvertMQTDynDealloc(
      const TypeConverter& typeConverter, MLIRContext* context,
      llvm::DenseMap<Value, Value>& qubitMap,
      llvm::DenseMap<Value, QubitData>& qubitDataMap,
      llvm::DenseMap<Value, Value>& qregMap,
      llvm::DenseMap<Value, std::vector<Value>>& qregQubitsMap)
      : OpConversionPattern(typeConverter, context), qubitMap(&qubitMap),
        qubitDataMap(&qubitDataMap), qregMap(&qregMap),
        qregQubitsMap(&qregQubitsMap) {}

  LogicalResult
  matchAndRewrite(dyn::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare return type
    const auto& qregType = opt::QubitRegisterType::get(rewriter.getContext());

    const auto& dynQreg = op.getQreg();
    auto& optQreg = (*qregMap)[dynQreg];

    // get a reference to avoid dereferencing multiple times in loops
    auto& qubitDataMapRef = *qubitDataMap;
    auto& qubitMapRef = *qubitMap;
    auto& dynQubits = (*qregQubitsMap)[dynQreg];

    // iterate over all the qubits that were extracted from the register
    for (const auto& dynQubit : dynQubits) {
      const auto& qubitData = qubitDataMapRef[dynQubit];
      const auto& optQubit = qubitMapRef[dynQubit];

      auto optInsertOp = rewriter.create<opt::InsertOp>(
          op.getLoc(), qregType, optQreg, optQubit, qubitData.index,
          qubitData.indexAttr);

      // move it before the current dealloc operation
      optInsertOp->moveBefore(op);

      // update the optQreg
      optQreg = optInsertOp.getOutQreg();

      // erase the dynQubit entry from the maps
      qubitMapRef.erase(dynQubit);
      qubitDataMapRef.erase(dynQubit);
    }

    // erase the register from the maps
    qregMap->erase(dynQreg);
    qregQubitsMap->erase(dynQreg);

    // replace the dyn dealloc operation with an opt dealloc operation
    rewriter.replaceOpWithNewOp<opt::DeallocOp>(op, optQreg);

    return success();
  }
};

struct ConvertMQTDynExtract final : OpConversionPattern<dyn::ExtractOp> {
  llvm::DenseMap<Value, Value>* qubitMap;
  llvm::DenseMap<Value, QubitData>* qubitDataMap;
  llvm::DenseMap<Value, Value>* qregMap;
  llvm::DenseMap<Value, std::vector<Value>>* qregQubitsMap;

  explicit ConvertMQTDynExtract(
      const TypeConverter& typeConverter, MLIRContext* context,
      llvm::DenseMap<Value, Value>& qubitMap,
      llvm::DenseMap<Value, QubitData>& qubitDataMap,
      llvm::DenseMap<Value, Value>& qregMap,
      llvm::DenseMap<Value, std::vector<Value>>& qregQubitsMap)
      : OpConversionPattern(typeConverter, context), qubitMap(&qubitMap),
        qubitDataMap(&qubitDataMap), qregMap(&qregMap),
        qregQubitsMap(&qregQubitsMap) {}

  LogicalResult
  matchAndRewrite(dyn::ExtractOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // create result types
    const auto& qregType = opt::QubitRegisterType::get(rewriter.getContext());
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& dynQreg = op.getInQreg();
    const auto& optQreg = (*qregMap)[dynQreg];

    // create new operation
    auto mqtoptOp = rewriter.create<opt::ExtractOp>(
        op.getLoc(), qregType, qubitType, optQreg, op.getIndex(),
        op.getIndexAttrAttr());

    const auto& dynQubit = op.getOutQubit();
    const auto& optQubit = mqtoptOp.getOutQubit();
    const auto& newOptQreg = mqtoptOp.getOutQreg();

    // put the pair of the dyn qubit and the latest opt qubit in the map
    qubitMap->try_emplace(dynQubit, optQubit);

    // update the latest opt register of the initial dyn register
    (*qregMap)[dynQreg] = newOptQreg;

    // add an entry to the qubitDataMap to store the indices and the register
    // for the insertOperation
    qubitDataMap->try_emplace(dynQubit, dynQreg, op.getIndex(),
                              op.getIndexAttrAttr());

    // append the entry to the qregQubitsMap to store which qubits that were
    // extracted from the register
    (*qregQubitsMap)[dynQreg].emplace_back(dynQubit);

    // replace the old operation result with the new result and delete
    // old operation
    rewriter.replaceOp(op, optQubit);

    return success();
  }
};

struct ConvertMQTDynMeasure final : OpConversionPattern<dyn::MeasureOp> {
  llvm::DenseMap<Value, Value>* qubitMap;

  explicit ConvertMQTDynMeasure(const TypeConverter& typeConverter,
                                MLIRContext* context,
                                llvm::DenseMap<Value, Value>& qubitMap)
      : OpConversionPattern(typeConverter, context), qubitMap(&qubitMap) {}

  LogicalResult
  matchAndRewrite(dyn::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& dynQubits = op.getInQubits();
    // get a reference to avoid dereferencing multiple times in loops
    auto& qubitMapRef = *qubitMap;

    std::vector<Value> optQubits;

    // get the latest opt qubit from the map and add them to the vector
    for (const auto& dynQubit : dynQubits) {
      optQubits.emplace_back(qubitMapRef[dynQubit]);
    }
    // create the result types
    const std::vector<Type> qubitTypes(optQubits.size(), qubitType);

    // create new operation
    auto mqtoptOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), qubitTypes, op.getOutBits().getTypes(), optQubits);

    const auto& outOptQubits = mqtoptOp.getOutQubits();
    const auto& newBits = mqtoptOp.getOutBits();

    // iterate over all qubits
    for (size_t i = 0; i < dynQubits.size(); i++) {
      // update the latest opt qubit of the initial dyn qubit
      qubitMapRef[dynQubits[i]] = outOptQubits[i];
    }

    // replace the old operation results with the new bits and delete
    // old operation
    rewriter.replaceOp(op, newBits);

    return success();
  }
};

template <typename MQTGateDynOp, typename MQTGateOptOp>
struct ConvertMQTDynGateOp final : OpConversionPattern<MQTGateDynOp> {
  llvm::DenseMap<Value, Value>* qubitMap;

  explicit ConvertMQTDynGateOp(TypeConverter& typeConverter,
                               MLIRContext* context,
                               llvm::DenseMap<Value, Value>& qubitMap)
      : OpConversionPattern<MQTGateDynOp>(typeConverter, context),
        qubitMap(&qubitMap) {}

  LogicalResult
  matchAndRewrite(MQTGateDynOp op, typename MQTGateDynOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // get a reference to avoid dereferencing multiple times in loops
    auto& qubitMapRef = *qubitMap;

    // Map dyn qubits to opt qubits
    auto mapQubits = [&](const auto& dynQubits) {
      std::vector<Value> optQubits;
      for (const auto& dynQubit : dynQubits) {
        optQubits.emplace_back(qubitMapRef[dynQubit]);
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
      qubitMapRef[op.getAllInQubits()[i]] = optResults[i];
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MQTDynToMQTOpt final : impl::MQTDynToMQTOptBase<MQTDynToMQTOpt> {
  using MQTDynToMQTOptBase::MQTDynToMQTOptBase;

  // map each initial dyn qubit to its latest opt qubit
  llvm::DenseMap<Value, Value> qubitMap;
  // map each initial dyn qubit to its metadata
  llvm::DenseMap<Value, QubitData> qubitDataMap;
  // map each initial dyn register to its latest opt register
  llvm::DenseMap<Value, Value> qregMap;
  // map each initial dyn register to its dynQubits
  llvm::DenseMap<Value, std::vector<Value>> qregQubitsMap;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTDynToMQTOptTypeConverter typeConverter(context);

    target.addIllegalDialect<dyn::MQTDynDialect>();
    target.addLegalDialect<opt::MQTOptDialect>();
    patterns.add<ConvertMQTDynAlloc>(typeConverter, context, qregMap);
    patterns.add<ConvertMQTDynDealloc>(typeConverter, context, qubitMap,
                                       qubitDataMap, qregMap, qregQubitsMap);
    patterns.add<ConvertMQTDynExtract>(typeConverter, context, qubitMap,
                                       qubitDataMap, qregMap, qregQubitsMap);
    patterns.add<ConvertMQTDynMeasure>(typeConverter, context, qubitMap);

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
    // conversion of mqtdyn types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtdyn types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // conversion of mqtdyn types in control-flow ops; e.g. cf.br
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
