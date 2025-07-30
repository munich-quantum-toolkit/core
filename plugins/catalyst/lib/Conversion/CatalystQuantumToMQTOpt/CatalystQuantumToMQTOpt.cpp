/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <cassert>
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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_CATALYSTQUANTUMTOMQTOPT
#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

using namespace mlir;

class CatalystQuantumToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit CatalystQuantumToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](const Type type) { return type; });

    // Convert source QuregType to target QubitRegisterType
    addConversion([ctx](catalyst::quantum::QuregType /*type*/) -> Type {
      return opt::QubitRegisterType::get(ctx);
    });

    // Convert source QubitType to target QubitType
    addConversion([ctx](catalyst::quantum::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });
  }
};

struct ConvertQuantumAlloc final
    : OpConversionPattern<catalyst::quantum::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    auto resultType = opt::QubitRegisterType::get(rewriter.getContext());

    // Create the new operation
    const auto mqtoptOp = rewriter.create<opt::AllocOp>(
        op.getLoc(), resultType, adaptor.getNqubits(),
        adaptor.getNqubitsAttrAttr());

    // Get the result of the new operation, which represents the qubit register
    const auto targetQreg = mqtoptOp->getResult(0);

    // Collect the users of the original operation to update their operands

    // Iterate over the users in reverse order
    for (std::vector users(op->getUsers().begin(), op->getUsers().end());
         auto* user : llvm::reverse(users)) {
      // Update the operand of the user operation to the new qubit register
      user->replaceUsesOfWith(op.getResult(), targetQreg);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuantumDealloc final
    : OpConversionPattern<catalyst::quantum::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create the new operation
    const auto mqtoptOp = rewriter.create<opt::DeallocOp>(
        op.getLoc(), TypeRange({}), adaptor.getQreg());

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct ConvertQuantumMeasure final
    : OpConversionPattern<catalyst::quantum::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    const auto qubitType = opt::QubitType::get(rewriter.getContext());
    const auto bitType = rewriter.getI1Type();

    // Create the new operation
    const auto mqtOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), TypeRange{qubitType}, TypeRange{bitType},
        adaptor.getInQubit());

    // Because the results (bit and qubit) have changed order, we need to
    // manually update their uses
    const auto catalystMeasure = op->getResult(0); // bit
    const auto catalystQubit = op->getResult(1);   // qubit

    const auto mqtQubit = mqtOp->getResult(0);
    const auto mqtMeasure = mqtOp->getResult(1);

    // Collect the users of the original qubit
    std::vector qubitUsers(catalystQubit.getUsers().begin(),
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
    std::vector measureUsers(catalystMeasure.getUsers().begin(),
                             catalystMeasure.getUsers().end());

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

struct ConvertQuantumExtract final
    : OpConversionPattern<catalyst::quantum::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    auto resultType0 = opt::QubitRegisterType::get(rewriter.getContext());
    auto resultType1 = opt::QubitType::get(rewriter.getContext());

    // Create the new operation
    const auto mqtoptOp = rewriter.create<opt::ExtractOp>(
        op.getLoc(), resultType0, resultType1, adaptor.getQreg(),
        adaptor.getIdx(), adaptor.getIdxAttrAttr());

    const auto inQreg = op->getOperand(0);
    const auto outQreg = mqtoptOp->getResult(0);

    // Collect the users of the original input qubit register to update their
    // operands
    std::vector users(inQreg.getUsers().begin(), inQreg.getUsers().end());

    // Iterate over users in reverse order to update their operands properly
    for (auto* user : llvm::reverse(users)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(mqtoptOp) && user != mqtoptOp && user != op) {
        user->replaceUsesOfWith(inQreg, outQreg);
      }
    }

    // Collect the users of the original output qubit
    const auto oldQubit = op->getResult(0);
    const auto newQubit = mqtoptOp->getResult(1);

    std::vector qubitUsers(oldQubit.getUsers().begin(),
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

struct ConvertQuantumInsert final
    : OpConversionPattern<catalyst::quantum::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    auto resultType = opt::QubitRegisterType::get(rewriter.getContext());

    // Create the new operation
    const auto mqtoptOp = rewriter.create<opt::InsertOp>(
        op.getLoc(), resultType, adaptor.getInQreg(), adaptor.getQubit(),
        adaptor.getIdx(), adaptor.getIdxAttrAttr());

    const auto targetQreg = mqtoptOp->getResult(0);
    const auto sourceQreg = op->getResult(0);

    // Collect the users of the original out qubit register to update their
    // operands
    std::vector users(sourceQreg.getUsers().begin(),
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

struct ConvertQuantumCustomOp final
    : OpConversionPattern<catalyst::quantum::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto gateName = op.getGateName();
    auto paramsValues = adaptor.getParams();
    auto allQubitsValues = adaptor.getInQubits();
    auto inNegCtrlQubitsValues = ValueRange(); // TODO: not available yet

    // Can be manipulated later
    SmallVector<Value> inQubitsVec(allQubitsValues.begin(),
                                   allQubitsValues.end());
    SmallVector<Value> inCtrlQubitsVec;

    SmallVector<bool> paramsMaskVec;
    SmallVector<double> staticParamsVec;
    SmallVector<Value> finalParamValues;

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
      inCtrlQubitsVec.emplace_back(inQubitsVec[0]);
      inQubitsVec = {inQubitsVec[1]};

    } else if (gateName == "Toffoli") {

      assert(inQubitsVec.size() == 3 && "Expected 2 controls + 1 target qubit");
      inCtrlQubitsVec.emplace_back(inQubitsVec[0]);
      inCtrlQubitsVec.emplace_back(inQubitsVec[1]);
      inQubitsVec = {inQubitsVec[2]};
    }

    // Final ValueRanges to pass into create<> ops
    const ValueRange inQubitsValues(inQubitsVec);
    const ValueRange inCtrlQubitsValues(inCtrlQubitsVec);

    // Create the new operation
    Operation* mqtoptOp = nullptr;

    if (gateName.compare("Hadamard") == 0) {
      mqtoptOp = rewriter.create<opt::HOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliX") == 0 ||
               gateName.compare("CNOT") == 0 ||
               gateName.compare("Toffoli") == 0) {
      mqtoptOp = rewriter.create<opt::XOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliY") == 0 || gateName.compare("CY") == 0) {
      mqtoptOp = rewriter.create<opt::YOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliZ") == 0 || gateName.compare("CZ") == 0) {
      mqtoptOp = rewriter.create<opt::ZOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("SWAP") == 0) {
      mqtoptOp = rewriter.create<opt::SWAPOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("RX") == 0 || gateName.compare("CRX") == 0) {
      mqtoptOp = rewriter.create<opt::RXOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("RY") == 0 || gateName.compare("CRY") == 0) {
      mqtoptOp = rewriter.create<opt::RYOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("RZ") == 0 || gateName.compare("CRZ") == 0) {
      mqtoptOp = rewriter.create<opt::RZOp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else if (gateName.compare("PhaseShift") == 0 ||
               gateName.compare("ControlledPhaseShift") == 0) {
      mqtoptOp = rewriter.create<opt::POp>(
          op.getLoc(), inQubitsValues.getType(), inCtrlQubitsValues.getType(),
          inNegCtrlQubitsValues.getType(), staticParams, paramsMask,
          paramsValues, inQubitsValues, inCtrlQubitsValues,
          inNegCtrlQubitsValues);
    } else {
      llvm::errs() << "Unsupported gate: " << gateName << "\n";
      return failure();
    }

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct CatalystQuantumToMQTOpt final
    : impl::CatalystQuantumToMQTOptBase<CatalystQuantumToMQTOpt> {
  using CatalystQuantumToMQTOptBase::CatalystQuantumToMQTOptBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<opt::MQTOptDialect>();
    target.addIllegalDialect<catalyst::quantum::QuantumDialect>();

    // Mark operations legal, that have no equivalent in the target dialect
    // TODO: how to handle them properly?
    target.addLegalOp<
        catalyst::quantum::DeviceInitOp, catalyst::quantum::DeviceReleaseOp,
        catalyst::quantum::NamedObsOp, catalyst::quantum::ExpvalOp,
        catalyst::quantum::FinalizeOp, catalyst::quantum::ComputationalBasisOp,
        catalyst::quantum::StateOp, catalyst::quantum::InitializeOp,
        catalyst::quantum::ComputationalBasisOp>();

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

} // namespace mqt::ir::conversions
