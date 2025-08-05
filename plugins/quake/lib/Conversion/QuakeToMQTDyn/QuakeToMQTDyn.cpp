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

#include <cassert>
#include <cstddef>
#include <cudaq/Optimizer/Dialect/Quake/QuakeDialect.h>
#include <cudaq/Optimizer/Dialect/Quake/QuakeOps.h>
#include <cudaq/Optimizer/Dialect/Quake/QuakeTypes.h>
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

using namespace mlir;

#define GEN_PASS_DEF_QUAKETOMQTDYN
#include "mlir/Conversion/QuakeToMQTDyn/QuakeToMQTDyn.h.inc"

struct ConvertQuakeAllocaOp final : OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern<quake::AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::AllocaOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // TODO: Implement the conversion

    return success();
  }
};

struct QuakeToMQTDynTypeConverter : public TypeConverter {
  QuakeToMQTDynTypeConverter() {
    // Identity conversion
    addConversion([](Type type) { return type; });
  }
};

void populateQuakeToMQTDynPatterns(TypeConverter& converter,
                                   RewritePatternSet& patterns) {
  auto* context = patterns.getContext();
  // patterns.insert<ConvertQuakeAllocaOp>(converter, context);
}

struct QuakeToMQTDyn : impl::QuakeToMQTDynBase<QuakeToMQTDyn> {
  using QuakeToMQTDynBase::QuakeToMQTDynBase;

  void runOnOperation() override {
    auto* context = &getContext();

    QuakeToMQTDynTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    populateQuakeToMQTDynPatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<::mqt::ir::dyn::MQTDynDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();
  }
};

} // namespace mqt::ir::conversions
