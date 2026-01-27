/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToJeff/QCToJeff.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mlir {
using namespace qc;

#define GEN_PASS_DEF_QCTOJEFF
#include "mlir/Conversion/QCToJeff/QCToJeff.h.inc"

namespace {

struct LoweringState {
  /// Map from QC qubit references to their Jeff qubit SSA values
  llvm::DenseMap<Value, Value> qubitMap;
};

template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter,
                              MLIRContext* context, LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, context), state_(state) {}

  /// Returns the shared lowering state object
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

} // namespace

struct ConvertQCAllocOpToJeff final : StatefulOpConversionPattern<qc::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto jeffOp = rewriter.replaceOpWithNewOp<jeff::QubitAllocOp>(op);

    // Set up qubit map
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getResult();
    qubitMap[qcQubit] = jeffOp.getResult();

    return success();
  }
};

struct ConvertQCDeallocOpToJeff final
    : StatefulOpConversionPattern<qc::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getQubit();

    auto jeffQubit = qubitMap[qcQubit];
    auto jeffOp = rewriter.replaceOpWithNewOp<jeff::QubitFreeOp>(op, jeffQubit);

    // Erase qubit from map
    qubitMap.erase(qcQubit);

    return success();
  }
};

struct ConvertQCXOpToJeff final : StatefulOpConversionPattern<qc::XOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::XOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap;
    auto qcQubit = op.getQubitIn();

    auto jeffQubit = qubitMap[qcQubit];
    auto jeffOp = rewriter.replaceOpWithNewOp<jeff::XOp>(
        op, jeffQubit, /*in_ctrl_qubits=*/ValueRange{}, /*num_ctrls=*/0,
        /*is_adjoint=*/false,
        /*power=*/1);

    // Update qubit map
    qubitMap[qcQubit] = jeffOp.getOutQubit();

    return success();
  }
};

class QCToJeffTypeConverter final : public TypeConverter {
public:
  explicit QCToJeffTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](qc::QubitType /*type*/) -> Type {
      return jeff::QubitType::get(ctx);
    });
  }
};

struct QCToJeff final : impl::QCToJeffBase<QCToJeff> {
  using QCToJeffBase::QCToJeffBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCToJeffTypeConverter typeConverter(context);

    LoweringState state;

    // Configure conversion target: QC illegal, Jeff legal
    target.addIllegalDialect<QCDialect>();
    target.addLegalDialect<jeff::JeffDialect>();

    // Register operation conversion patterns
    patterns.add<ConvertQCAllocOpToJeff, ConvertQCDeallocOpToJeff,
                 ConvertQCXOpToJeff>(typeConverter, context, &state);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
