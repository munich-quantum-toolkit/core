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

#include <cassert>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
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

  // Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, SmallVector<Value>> controls;
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

template <typename JeffOpType, typename QCOpType>
static LogicalResult
convertOneTargetZeroParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                              LoweringState& state, bool isAdjoint) {
  auto& qubitMap = state.qubitMap;
  auto inCtrlOp = state.inCtrlOp;

  auto qcTarget = op.getQubitIn();
  auto jeffTarget = qubitMap[qcTarget];

  SmallVector<Value> jeffControls{};
  if (inCtrlOp != 0) {
    for (auto qcControl : state.controls[inCtrlOp]) {
      assert(qubitMap.contains(qcControl) && "QC qubit not found");
      jeffControls.push_back(qubitMap[qcControl]);
    }
  }

  auto jeffOp = rewriter.create<JeffOpType>(op.getLoc(), jeffTarget,
                                            /*in_ctrl_qubits=*/jeffControls,
                                            /*num_ctrls=*/jeffControls.size(),
                                            /*is_adjoint=*/isAdjoint,
                                            /*power=*/1);

  // Update qubit map and modifier information
  qubitMap[qcTarget] = jeffOp.getOutQubit();
  if (inCtrlOp != 0) {
    for (size_t i = 0; i < jeffControls.size(); ++i) {
      auto qcControl = state.controls[inCtrlOp][i];
      qubitMap[qcControl] = jeffOp.getOutCtrlQubits()[i];
    }
    state.controls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  rewriter.eraseOp(op);

  return success();
}

struct ConvertQCAllocOpToJeff final : StatefulOpConversionPattern<qc::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::AllocOp op, OpAdaptor /*adaptor*/,
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
    rewriter.replaceOpWithNewOp<jeff::QubitFreeOp>(op, jeffQubit);

    // Erase qubit from map
    qubitMap.erase(qcQubit);

    return success();
  }
};

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_QC, OP_CLASS_JEFF,           \
                                         IS_ADJOINT)                           \
  struct ConvertQC##OP_CLASS_QC##ToJeff final                                  \
      : StatefulOpConversionPattern<qc::OP_CLASS_QC> {                         \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS_QC op, OpAdaptor /*adaptor*/,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<jeff::OP_CLASS_JEFF>(               \
          op, rewriter, getState(), IS_ADJOINT);                               \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, IOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, XOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, YOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, ZOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, HOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, SOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, SOp, true)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, TOp, false)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, TOp, true)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

struct ConvertQCCtrlOpToJeff final : StatefulOpConversionPattern<qc::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::CtrlOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Update modifier information
    auto& state = getState();
    state.inCtrlOp++;
    const SmallVector<Value> controls(op.getControls().begin(),
                                      op.getControls().end());
    state.controls[state.inCtrlOp] = controls;

    // Inline region and remove operation
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQCYieldOpToJeff final : StatefulOpConversionPattern<qc::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
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
                 ConvertQCIdOpToJeff, ConvertQCXOpToJeff, ConvertQCYOpToJeff,
                 ConvertQCZOpToJeff, ConvertQCHOpToJeff, ConvertQCSOpToJeff,
                 ConvertQCSdgOpToJeff, ConvertQCTOpToJeff, ConvertQCTdgOpToJeff,
                 ConvertQCCtrlOpToJeff, ConvertQCYieldOpToJeff>(
        typeConverter, context, &state);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
