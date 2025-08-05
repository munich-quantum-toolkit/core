/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTDynToQuake/MQTDynToQuake.h"

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

#define GEN_PASS_DEF_MQTDYNTOQUAKE
#include "mlir/Conversion/MQTDynToQuake/MQTDynToQuake.h.inc"

struct ConvertMQTDynAllocOp final : OpConversionPattern<dyn::AllocOp> {
  using OpConversionPattern<dyn::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // TODO: Implement the conversion

    return success();
  }
};

struct MQTDynToQuakeTypeConverter : public TypeConverter {
  MQTDynToQuakeTypeConverter() {
    // Identity conversion
    addConversion([](Type type) { return type; });
  }
};

void populateMQTDynToQuakePatterns(TypeConverter& converter,
                                   RewritePatternSet& patterns) {
  auto* context = patterns.getContext();
  patterns.insert<ConvertMQTDynAllocOp>(converter, context);
}

struct MQTDynToQuake : impl::MQTDynToQuakeBase<MQTDynToQuake> {
  using MQTDynToQuakeBase::MQTDynToQuakeBase;

  void runOnOperation() override {
    auto* context = &getContext();

    MQTDynToQuakeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    populateMQTDynToQuakePatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalDialect<mqt::ir::dyn::MQTDynDialect>();
  }
};

} // namespace mqt::ir::conversions
