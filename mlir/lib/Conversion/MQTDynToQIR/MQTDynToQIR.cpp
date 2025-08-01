/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any dyn gate operation to a llvm
// call operation that adheres to the qir specification

#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns.add<ConvertMQTDynGateOpQIR<dyn::gate>>(typeConverter, context);

#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTDYNTOQIR
#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h.inc"

namespace {
// add function declaration at the beginning if it does not exist already and
// return the function declaration
LLVM::LLVMFuncOp getFunctionDeclaration(PatternRewriter& rewriter,
                                        Operation* op, StringRef fnSymbol,
                                        Type fnType) {
  auto* fnDecl = SymbolTable::lookupNearestSymbolFrom(
      op, rewriter.getStringAttr(fnSymbol));

  if (fnDecl == nullptr) {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto mod = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(mod.getBody());

    fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
  }

  return cast<LLVM::LLVMFuncOp>(fnDecl);
}

class MQTDynToQIRTypeConverter final : public TypeConverter {
public:
  explicit MQTDynToQIRTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });
    // Qubit conversion
    addConversion([ctx](dyn::QubitType /*type*/) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });
    // QregType conversion
    addConversion([ctx](dyn::QubitRegisterType /*type*/) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });
  }
};

} // namespace

struct ConvertMQTDynAllocQIR final : OpConversionPattern<dyn::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_allocate_array";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 64));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // create a constantOp if the size is an attribute
    auto size = adaptor.getSize();
    if (!size) {
      size =
          rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getSizeAttrAttr());
    }

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, size);
    return success();
  }
};
struct ConvertMQTDynDeallocQIR final : OpConversionPattern<dyn::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_release_array";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

struct ConvertMQTDynExtractQIR final : OpConversionPattern<dyn::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__array_get_element_ptr_1d";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 64)});

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // create a constantOp if the index is an attribute
    auto index = adaptor.getIndex();
    if (!index) {
      index =
          rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getIndexAttrAttr());
    }

    // create the new callOp
    const auto elemPtr =
        rewriter
            .create<LLVM::CallOp>(op.getLoc(), fnDecl,
                                  ValueRange{adaptor.getInQreg(), index})
            .getResult();

    // replace the old operation with a loadOp
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, LLVM::LLVMPointerType::get(ctx), elemPtr);

    return success();
  }
};

template <typename MQTGateDynOp>
struct ConvertMQTDynGateOpQIR final : OpConversionPattern<MQTGateDynOp> {
  using OpConversionPattern<MQTGateDynOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateDynOp op, typename MQTGateDynOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // get all the values
    const auto& params = adaptor.getParams();
    const auto& inQubits = adaptor.getInQubits();
    const auto& posCtrlQubits = adaptor.getPosCtrlInQubits();
    const auto& negCtrlQubits = adaptor.getNegCtrlInQubits();

    // concatenate all the types
    SmallVector<Type> types;
    types.reserve(params.size() + inQubits.size() + posCtrlQubits.size() +
                  negCtrlQubits.size());
    types.append(params.getTypes().begin(), params.getTypes().end());
    types.append(inQubits.getTypes().begin(), inQubits.getTypes().end());
    types.append(posCtrlQubits.getTypes().begin(),
                 posCtrlQubits.getTypes().end());
    types.append(negCtrlQubits.getTypes().begin(),
                 negCtrlQubits.getTypes().end());

    // concatenate all the values
    SmallVector<Value> operands;
    operands.reserve(params.size() + inQubits.size() + posCtrlQubits.size() +
                     negCtrlQubits.size());
    operands.append(params.begin(), params.end());
    operands.append(inQubits.begin(), inQubits.end());
    operands.append(posCtrlQubits.begin(), posCtrlQubits.end());
    operands.append(negCtrlQubits.begin(), negCtrlQubits.end());

    // get the name of the gate
    const StringRef name = op->getName().getStringRef().split('.').second;
    std::string fnName;

    // add leading c's depending on the number of control qubits
    const auto ctrQubitsCount = posCtrlQubits.size() + negCtrlQubits.size();
    fnName.insert(0, ctrQubitsCount, 'c');

    // check if it is a cnot gate
    if (name == "x" && ctrQubitsCount == 1) {
      fnName = ("__quantum__qis__" + fnName + "not__body");
    } else {
      fnName = ("__quantum__qis__" + fnName + name + "__body").str();
    }

    // create the signature of the function
    const auto qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types);

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with a callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);

    return success();
  }
};

struct ConvertMQTDynMeasureQIR final : OpConversionPattern<dyn::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  SmallVector<Operation*>* measureConstants;
  Operation* constantOp;
  explicit ConvertMQTDynMeasureQIR(TypeConverter& typeConverter,
                                   MLIRContext* context,
                                   SmallVector<Operation*>& measureConstants,
                                   Operation* constantOp)
      : OpConversionPattern(typeConverter, context),
        measureConstants(&measureConstants), constantOp(constantOp) {}
  LogicalResult
  matchAndRewrite(dyn::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // create name and signature of the new function
    StringRef fnName = "__quantum__qis__m__body";
    auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // get the function declaration
    auto fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
    auto newOp = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fnDecl, adaptor.getInQubits());
    fnName = "__quantum__rt__result_record_output";
    qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{newOp->getResult(0),
                   measureConstants->front()->getResult(0)});
    measureConstants->erase(&measureConstants->front());

    fnName = "__quantum__rt__result_update_reference_count";
    qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 32)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{newOp->getResult(0), constantOp->getResult(0)});

    return success();
  }
};

struct MQTDynToQIR final : impl::MQTDynToQIRBase<MQTDynToQIR> {
  using MQTDynToQIRBase::MQTDynToQIRBase;

  // collect all the necessary operations for the measure operation conversion
  static Operation*
  collectMeasureConstants(Operation* op, SmallVector<Operation*>& operations) {

    auto module = llvm::dyn_cast<ModuleOp>(op);
    LLVM::LLVMFuncOp main;

    // find the main function
    for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
      if (auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough")) {
        for (auto attr : passthrough) {
          if (auto strAttr = dyn_cast<StringAttr>(attr)) {

            if (strAttr.getValue() == "entry_point") {

              main = funcOp;
            }
          }
        }
      }
    }

    // get the first block in the main function
    auto& firstBlock = *(main.getBlocks().begin());
    Operation* result = nullptr;
    // walk through the block and collect all addressOfOp and get the -1
    // constant value
    firstBlock.walk([&](Operation* op) {
      if (auto addressOfOp = llvm::dyn_cast<LLVM::AddressOfOp>(op)) {
        operations.emplace_back(addressOfOp);
      }

      if (auto constantOp = llvm::dyn_cast<LLVM::ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
          // check if the value is -1
          if (intAttr.getInt() == -1) {
            result = op;
          }
        }
      }
    });

    return result;
  }

  static void addInitialize(Operation* op) {
    // get the first block in the main function
    auto module = llvm::dyn_cast<ModuleOp>(op);
    LLVM::LLVMFuncOp main;
    // find the main function
    for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
      if (auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough")) {
        for (auto attr : passthrough) {
          if (auto strAttr = dyn_cast<StringAttr>(attr)) {

            if (strAttr.getValue() == "entry_point") {

              main = funcOp;
            }
          }
        }
      }
    }
    auto& firstBlock = *(main.getBlocks().begin());
    Operation* zeroOperation = nullptr;
    // find the zeroOp
    firstBlock.walk([&](Operation* op) {
      if (auto zeroOp = llvm::dyn_cast<LLVM::ZeroOp>(op)) {
        zeroOperation = zeroOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    // set the builder to the 2nd last operation in the first block
    OpBuilder builder(main.getBody());
    auto& ops = firstBlock.getOperations();
    const auto insertPoint = std::prev(ops.end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // get the function declaration
    const StringRef fnName = "__quantum__rt__initialize";
    auto* fnDecl =
        SymbolTable::lookupNearestSymbolFrom(op, builder.getStringAttr(fnName));

    // create and insert the initialize operation
    builder.create<LLVM::CallOp>(op->getLoc(),
                                 static_cast<LLVM::LLVMFuncOp>(fnDecl),
                                 ValueRange{zeroOperation->getResult(0)});
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTDynToQIRTypeConverter typeConverter(context);
    target.addIllegalDialect<dyn::MQTDynDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    SmallVector<Operation*> addressOfOps;
    auto* constantOp = collectMeasureConstants(module, addressOfOps);
    addInitialize(module);

    patterns.add<ConvertMQTDynAllocQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynDeallocQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynExtractQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynMeasureQIR>(typeConverter, context, addressOfOps,
                                          constantOp);
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
