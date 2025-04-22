/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/Catalyst/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DEF_CATALYSTQUANTUMTOMQTOPT
#include "mlir/Conversion/Catalyst/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

using namespace mlir;

class CatalystQuantumToMQTOptTypeConverter : public TypeConverter {
public:
  explicit CatalystQuantumToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](Type type) { return type; });

    // Convert source QuregType to target QubitRegisterType
    addConversion([ctx](catalyst::quantum::QuregType /*type*/) -> Type {
      return ::mqt::ir::opt::QubitRegisterType::get(ctx);
    });

    // Convert source QubitType to target QubitType
    addConversion([ctx](catalyst::quantum::QubitType /*type*/) -> Type {
      return ::mqt::ir::opt::QubitType::get(ctx);
    });
  }
};

struct ConvertQuantumAlloc
    : public OpConversionPattern<catalyst::quantum::AllocOp> {

  ConvertQuantumAlloc(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::AllocOp>(typeConverter,
                                                        context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto const nQubitsValue = adaptor.getNqubits();
    auto const nQubitsIntegerAttr = adaptor.getNqubitsAttrAttr();

    // Prepare the result type(s)
    auto resultType =
        ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());

    // Create the new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::AllocOp>(
        op.getLoc(), resultType, nQubitsValue, nQubitsIntegerAttr);

    // Get the result of the new operation, which represents the qubit register
    auto trgtQreg = mqtoptOp->getResult(0);

    // Collect the users of the original operation to update their operands
    std::vector<mlir::Operation*> users(op->getUsers().begin(),
                                        op->getUsers().end());

    // Iterate over the users in reverse order
    for (auto* user : llvm::reverse(users)) {
      // Update the operand of the user operation to the new qubit register
      user->replaceUsesOfWith(op.getResult(), trgtQreg);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuantumDealloc
    : public OpConversionPattern<catalyst::quantum::DeallocOp> {

  ConvertQuantumDealloc(const TypeConverter& typeConverter,
                        MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::DeallocOp>(typeConverter,
                                                          context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto qregValue = adaptor.getQreg();

    // Prepare the result type(s)
    auto resultTypes = ::mlir::TypeRange({});

    // Create the new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::DeallocOp>(
        op.getLoc(), resultTypes, qregValue);

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct ConvertQuantumMeasure
    : public OpConversionPattern<catalyst::quantum::MeasureOp> {

  ConvertQuantumMeasure(const TypeConverter& typeConverter,
                        MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::MeasureOp>(typeConverter,
                                                          context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto inQubit = adaptor.getInQubit();

    // Prepare the result type(s)
    auto qubitType = ::mqt::ir::opt::QubitType::get(rewriter.getContext());
    auto bitType = rewriter.getI1Type();

    // Create the new operation
    auto mqtOp = rewriter.create<::mqt::ir::opt::MeasureOp>(
        op.getLoc(), mlir::TypeRange{qubitType}, mlir::TypeRange{bitType},
        mlir::ValueRange{inQubit});

    // Because the results (bit and qubit) have changed order, we need to
    // manually update their uses
    auto catalystMeasure = op->getResult(0); // bit
    auto catalystQubit = op->getResult(1);   // qubit

    auto mqtQubit = mqtOp->getResult(0);
    auto mqtMeasure = mqtOp->getResult(1);

    // Collect the users of the original qubit
    std::vector<mlir::Operation*> qubitUsers(catalystQubit.getUsers().begin(),
                                             catalystQubit.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtOp) && user != mqtOp && user != op) {
        // Update operands in the user operation
        user->replaceUsesOfWith(catalystQubit, mqtQubit);
      }
    }

    // Collect the users of the original measurement bit
    std::vector<mlir::Operation*> measureUsers(
        catalystMeasure.getUsers().begin(), catalystMeasure.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(measureUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtOp) && user != mqtOp && user != op) {
        // Update operands in the user operation
        user->replaceUsesOfWith(catalystMeasure, mqtMeasure);
      }
    }

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuantumExtract
    : public OpConversionPattern<catalyst::quantum::ExtractOp> {

  ConvertQuantumExtract(const TypeConverter& typeConverter,
                        MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::ExtractOp>(typeConverter,
                                                          context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s), and attribute(s)
    auto qregValue = adaptor.getQreg();
    auto idxValue = adaptor.getIdx();
    auto idxIntegerAttr = adaptor.getIdxAttrAttr();

    // Prepare the result type(s)
    auto resultType0 =
        ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());
    auto resultType1 = ::mqt::ir::opt::QubitType::get(rewriter.getContext());

    // Create the new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::ExtractOp>(
        op.getLoc(), resultType0, resultType1, qregValue, idxValue,
        idxIntegerAttr);

    auto inQreg = op->getOperand(0);
    auto outQreg = mqtoptOp->getResult(0);

    // Collect the users of the original input qubit register to update their
    // operands
    std::vector<mlir::Operation*> users(inQreg.getUsers().begin(),
                                        inQreg.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(users)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        user->replaceUsesOfWith(inQreg, outQreg);
      }
    }

    // Collect the users of the original output qubit
    auto oldQubit = op->getResult(0);
    auto newQubit = mqtoptOp->getResult(1);

    std::vector<mlir::Operation*> qubitUsers(oldQubit.getUsers().begin(),
                                             oldQubit.getUsers().end());

    // Iterate over qubit users in reverse order
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        user->replaceUsesOfWith(oldQubit, newQubit);
      }
    }

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuantumInsert
    : public OpConversionPattern<catalyst::quantum::InsertOp> {

  ConvertQuantumInsert(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::InsertOp>(typeConverter,
                                                         context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto inQregValue = adaptor.getInQreg();
    auto qubitValue = adaptor.getQubit();
    auto idxValue = adaptor.getIdx();
    auto idxIntegerAttr = adaptor.getIdxAttrAttr();

    // Prepare the result type(s)
    auto resultType =
        ::mqt::ir::opt::QubitRegisterType::get(rewriter.getContext());

    // Create the new operation
    auto mqtoptOp = rewriter.create<::mqt::ir::opt::InsertOp>(
        op.getLoc(), resultType, inQregValue, qubitValue, idxValue,
        idxIntegerAttr);

    auto targetQreg = mqtoptOp->getResult(0);
    auto sourceQreg = op->getResult(0);

    // Collect the users of the original out qubit register to update their
    // operands
    std::vector<mlir::Operation*> users(sourceQreg.getUsers().begin(),
                                        sourceQreg.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(users)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        user->replaceUsesOfWith(sourceQreg, targetQreg);
      }
    }
    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuantumCustomOp
    : public OpConversionPattern<catalyst::quantum::CustomOp> {

  ConvertQuantumCustomOp(const TypeConverter& typeConverter,
                         MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::CustomOp>(typeConverter,
                                                         context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto gateName = op.getGateName();
    auto paramsValues = adaptor.getParams();
    auto allQubitsValues = adaptor.getInQubits();
    auto inNegCtrlQubitsValues = mlir::ValueRange(); // TODO: not available yet

    // Can be manipulated later
    llvm::SmallVector<mlir::Value> inQubitsVec(allQubitsValues.begin(),
                                               allQubitsValues.end());
    llvm::SmallVector<mlir::Value> inCtrlQubitsVec;

    llvm::SmallVector<bool> paramsMaskVec;
    llvm::SmallVector<double> staticParamsVec;
    llvm::SmallVector<Value> finalParamValues;

    // Read attributes
    auto maskAttr = op->getAttrOfType<DenseBoolArrayAttr>("params_mask");
    auto staticParamsAttr =
        op->getAttrOfType<DenseF64ArrayAttr>("static_params");

    // Total length of combined parameter list
    size_t totalParams = 0;
    if (maskAttr) {
      totalParams = maskAttr.size();
    } else {
      totalParams = staticParamsAttr
                        ? staticParamsAttr.size() + paramsValues.size()
                        : paramsValues.size();
    }

    // Pointers to step through static/dynamic values
    size_t staticIdx = 0;
    size_t dynamicIdx = 0;

    // Build final mask + values in order
    for (size_t i = 0; i < totalParams; ++i) {
      bool const isStatic = (maskAttr ? maskAttr[i] : false);

      paramsMaskVec.emplace_back(isStatic);

      if (isStatic) {
        assert(staticParamsAttr && "Missing static_params for static mask");
        staticParamsVec.emplace_back(staticParamsAttr[staticIdx++]);
      } else {
        assert(dynamicIdx < paramsValues.size() &&
               "Too few dynamic parameters");
        finalParamValues.emplace_back(paramsValues[dynamicIdx++]);
      }
    }

    auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), staticParamsVec);
    auto paramsMask =
        DenseBoolArrayAttr::get(rewriter.getContext(), paramsMaskVec);

    if (gateName == "CNOT" || gateName == "CY" || gateName == "CZ" ||
        gateName == "CRX" || gateName == "CRY" || gateName == "CRZ" ||
        gateName == "ControlledPhaseShift") {

      assert(inQubitsVec.size() == 2 && "Expected 1 control + 1 target qubit");
      inCtrlQubitsVec.push_back(inQubitsVec[0]);
      inQubitsVec = {inQubitsVec[1]};

    } else if (gateName == "Toffoli") {

      assert(inQubitsVec.size() == 3 && "Expected 2 controls + 1 target qubit");
      inCtrlQubitsVec.push_back(inQubitsVec[0]);
      inCtrlQubitsVec.push_back(inQubitsVec[1]);
      inQubitsVec = {inQubitsVec[2]};
    }

    // Final ValueRanges to pass into create<> ops
    mlir::ValueRange inQubitsValues(inQubitsVec);
    mlir::ValueRange inCtrlQubitsValues(inCtrlQubitsVec);

    // Prepare the result type(s)
    mlir::Type qubitType =
        ::mqt::ir::opt::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubitsValues.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Create the new operation
    Operation* mqtoptOp = nullptr;

    if (gateName.compare("Hadamard") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::HOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliX") == 0 ||
               gateName.compare("CNOT") == 0 ||
               gateName.compare("Toffoli") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::XOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliY") == 0 || gateName.compare("CY") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::YOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliZ") == 0 || gateName.compare("CZ") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::ZOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("SWAP") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::SWAPOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("RX") == 0 || gateName.compare("CRX") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RXOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("RY") == 0 || gateName.compare("CRY") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RYOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("RZ") == 0 || gateName.compare("CRZ") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::RZOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PhaseShift") == 0 ||
               gateName.compare("ControlledPhaseShift") == 0) {
      mqtoptOp = rewriter.create<::mqt::ir::opt::POp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else {
      llvm::errs() << "Unsupported gate: " << gateName << "\n";
      return failure();
    }

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct CatalystQuantumToMQTOpt
    : impl::CatalystQuantumToMQTOptBase<CatalystQuantumToMQTOpt> {
  using CatalystQuantumToMQTOptBase::CatalystQuantumToMQTOptBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    std::cout << "Cmap: " << cMap << "\n";

    ConversionTarget target(*context);
    target.addLegalDialect<::mqt::ir::opt::MQTOptDialect>();
    target.addIllegalDialect<catalyst::quantum::QuantumDialect>();

    // Mark operations legal, that have no equivalent in the target dialect
    target.addLegalOp<catalyst::quantum::DeviceInitOp>();
    target.addLegalOp<catalyst::quantum::DeviceReleaseOp>();
    target.addLegalOp<catalyst::quantum::NamedObsOp>();
    target.addLegalOp<catalyst::quantum::ExpvalOp>();
    target.addLegalOp<catalyst::quantum::FinalizeOp>();
    target.addLegalOp<catalyst::quantum::ComputationalBasisOp>();
    target.addLegalOp<catalyst::quantum::StateOp>();
    target.addLegalOp<catalyst::quantum::InitializeOp>();

    RewritePatternSet patterns(context);
    CatalystQuantumToMQTOptTypeConverter typeConverter(context);

    patterns.add<ConvertQuantumAlloc, ConvertQuantumDealloc,
                 ConvertQuantumExtract, ConvertQuantumMeasure,
                 ConvertQuantumInsert, ConvertQuantumCustomOp>(typeConverter,
                                                               context);

    // Boilerplate code to prevent: unresolved materialization
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
  }
};

} // namespace mlir::mqt::ir::conversions
