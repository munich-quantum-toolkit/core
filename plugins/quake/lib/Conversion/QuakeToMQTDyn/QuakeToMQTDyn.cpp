/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QuakeToMQTDyn/QuakeToMQTDyn.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <cudaq/Optimizer/Dialect/Quake/QuakeDialect.h>
#include <cudaq/Optimizer/Dialect/Quake/QuakeOps.h>
#include <cudaq/Optimizer/Dialect/Quake/QuakeTypes.h>
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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_QUAKETOMQTDYN
#include "mlir/Conversion/QuakeToMQTDyn/QuakeToMQTDyn.h.inc"

using namespace mlir;

class QuakeToMQTDynTypeConverter : public TypeConverter {
public:
  explicit QuakeToMQTDynTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    // Allow all types to pass through unmodified if needed
    addConversion([](Type type) { return type; });
  }
};

struct ConvertQuakeAlloca
    : public OpConversionPattern<cudaq::quake::quake_AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::quake_AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Create the new operation
    auto mqtDynOp = rewriter.create<::mqt::ir::dyn::AllocOp>(
        op.getLoc(), adaptor.getSize());

    // Get the result of the new operation
    auto mqtDynReg = mqtDynOp->getResult(0);

    // Collect the users of the original operation to update their operands
    std::vector<mlir::Operation*> users(op->getUsers().begin(),
                                        op->getUsers().end());

    // Iterate over the users in reverse order
    for (auto* user : llvm::reverse(users)) {
      // Update the operand of the user operation to the new qubit register
      user->replaceUsesOfWith(op.getResult(), mqtDynReg);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct QuakeToMQTDyn
    : impl::QuakeToMQTDynBase<QuakeToMQTDyn> {
  using QuakeToMQTDynBase::QuakeToMQTDynBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<::mqt::ir::dyn::MQTDynDialect>();
    target.addIllegalDialect<cudaq::quake::QuakeDialect>();

    RewritePatternSet patterns(context);
    QuakeToMQTDynTypeConverter typeConverter(context);

    patterns.add<ConvertQuakeAlloca>(typeConverter,
                                                               context);

    // Boilerplate code to prevent unresolved materialization
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