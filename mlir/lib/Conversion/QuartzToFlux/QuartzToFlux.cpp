/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h"

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
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

namespace mlir {
using namespace mlir::flux;
using namespace mlir::quartz;

#define GEN_PASS_DEF_QUARTZTOFLUX
#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h.inc"

class QuartzToFluxTypeConverter final : public TypeConverter {
public:
  explicit QuartzToFluxTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](quartz::QubitType /*type*/) -> Type {
      return flux::QubitType::get(ctx);
    });
  }
};

struct QuartzToFlux final : impl::QuartzToFluxBase<QuartzToFlux> {
  using QuartzToFluxBase::QuartzToFluxBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QuartzToFluxTypeConverter typeConverter(context);

    target.addIllegalDialect<QuartzDialect>();
    target.addLegalDialect<FluxDialect>();

    // conversion of quartz types in func.func signatures
    // does not work for now as signature needs to be changed
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    // conversion of quartz types in func.return
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](const func::ReturnOp op) { return typeConverter.isLegal(op); });

    // conversion of quartz types in func.call
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](const func::CallOp op) { return typeConverter.isLegal(op); });

    // conversion of quartz types in control-flow ops; e.g. cf.br
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mlir
