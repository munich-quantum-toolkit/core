/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <cassert>
#include <cstdint>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numbers>
#include <utility>

namespace mlir {
using namespace qco;

#define GEN_PASS_DEF_QCOTOJEFF
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h.inc"

namespace {

/**
 * @brief State object for tracking Jeff qubit values during conversion
 */
struct LoweringState {
  // Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, qco::CtrlOp> ctrlOps;
  DenseMap<int64_t, SmallVector<Value>> controlsIn;
  DenseMap<int64_t, SmallVector<Value>> controlsOut;
  DenseMap<int64_t, SmallVector<Value>> targetsIn;
  DenseMap<int64_t, SmallVector<Value>> targetsOut;
};

/**
 * @brief Base class for conversion patterns that need access to the
 * LoweringState
 *
 * @tparam OpType The QCO operation type to convert
 */
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

/**
 * @brief Converts a one-target, zero-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetZeroParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter,
                              LoweringState& state, bool isAdjoint) {
  const auto inCtrlOp = state.inCtrlOp;

  Value target;
  if (inCtrlOp != 0) {
    target = state.targetsIn[inCtrlOp][0];
  } else {
    target = adaptor.getQubitIn();
  }

  auto jeffOp = rewriter.create<JeffOpType>(
      op.getLoc(), target,
      /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp],
      /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
      /*is_adjoint=*/isAdjoint,
      /*power=*/1);

  if (inCtrlOp != 0) {
    rewriter.eraseOp(op);
    state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
    state.targetsOut[inCtrlOp] = {jeffOp.getOutQubit()};
  } else {
    rewriter.replaceOp(op, jeffOp.getOutQubit());
  }

  return success();
}

/**
 * @brief Converts a one-target, one-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetOneParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter,
                             LoweringState& state) {
  const auto inCtrlOp = state.inCtrlOp;

  Value target;
  if (inCtrlOp != 0) {
    target = state.targetsIn[inCtrlOp][0];
  } else {
    target = adaptor.getQubitIn();
  }

  auto jeffOp = rewriter.create<JeffOpType>(
      op.getLoc(), target, op.getParameter(0),
      /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp],
      /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
      /*is_adjoint=*/false,
      /*power=*/1);

  if (inCtrlOp != 0) {
    rewriter.eraseOp(op);
    state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
    state.targetsOut[inCtrlOp] = {jeffOp.getOutQubit()};
  } else {
    rewriter.replaceOp(op, jeffOp.getOutQubit());
  }

  return success();
}

/**
 * @brief Converts a one-target, three-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetThreeParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                               ConversionPatternRewriter& rewriter,
                               LoweringState& state) {
  const auto inCtrlOp = state.inCtrlOp;

  Value target;
  if (inCtrlOp != 0) {
    target = state.targetsIn[inCtrlOp][0];
  } else {
    target = adaptor.getQubitIn();
  }

  auto jeffOp = rewriter.create<JeffOpType>(
      op.getLoc(), target, op.getParameter(0), op.getParameter(1),
      op.getParameter(2),
      /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp],
      /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
      /*is_adjoint=*/false,
      /*power=*/1);

  if (inCtrlOp != 0) {
    rewriter.eraseOp(op);
    state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
    state.targetsOut[inCtrlOp] = {jeffOp.getOutQubit()};
  } else {
    rewriter.replaceOp(op, jeffOp.getOutQubit());
  }

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertTwoTargetZeroParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter,
                              LoweringState& state) {
  const auto inCtrlOp = state.inCtrlOp;

  Value target0;
  Value target1;
  if (inCtrlOp != 0) {
    target0 = state.targetsIn[inCtrlOp][0];
    target1 = state.targetsIn[inCtrlOp][1];
  } else {
    target0 = adaptor.getQubit0In();
    target1 = adaptor.getQubit1In();
  }

  auto jeffOp = rewriter.create<JeffOpType>(
      op.getLoc(), target0, target1,
      /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp],
      /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
      /*is_adjoint=*/false,
      /*power=*/1);

  if (inCtrlOp != 0) {
    rewriter.eraseOp(op);
    state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
    state.targetsOut[inCtrlOp] = {jeffOp.getOutQubitOne(),
                                  jeffOp.getOutQubitTwo()};
  } else {
    rewriter.replaceOp(op, {jeffOp.getOutQubitOne(), jeffOp.getOutQubitTwo()});
  }

  return success();
}

template <typename QCOOpType>
static void createCustomOp(QCOOpType& op, ConversionPatternRewriter& rewriter,
                           LoweringState& state,
                           const llvm::SmallVector<Value>& targets,
                           const llvm::SmallVector<Value>& params,
                           bool isAdjoint, StringRef name) {
  const auto inCtrlOp = state.inCtrlOp;

  auto jeffOp = rewriter.create<jeff::CustomOp>(
      op.getLoc(), targets,
      /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp], /*params=*/params,
      /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
      /*is_adjoint=*/isAdjoint,
      /*power=*/1, /*name=*/name, /*num_targets=*/targets.size(),
      /*num_params=*/params.size());

  if (state.inCtrlOp != 0) {
    rewriter.eraseOp(op);
    state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
    state.targetsOut[inCtrlOp] = jeffOp.getOutQubits();
  } else {
    rewriter.replaceOp(op, jeffOp.getOutQubits());
  }
}

/**
 * @brief Converts qco.alloc to jeff.qubit_alloc
 *
 * @par Example:
 * ```mlir
 * %q = qco.alloc : !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = jeff.qubit_alloc : !jeff.qubit
 * ```
 */
struct ConvertQCOAllocOpToJeff final
    : StatefulOpConversionPattern<qco::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<jeff::QubitAllocOp>(op);
    return success();
  }
};

/**
 * @brief Converts qco.dealloc to jeff.qubit_free_zero
 *
 * @par Example:
 * ```mlir
 * qco.dealloc %q : !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * jeff.qubit_free_zero %q : !jeff.qubit
 * ```
 */
struct ConvertQCODeallocOpToJeff final
    : StatefulOpConversionPattern<qco::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<jeff::QubitFreeZeroOp>(op, adaptor.getQubit());
    return success();
  }
};

/**
 * @brief Converts qco.measure to jeff.qubit_measure_nd
 *
 * @par Example:
 * ```mlir
 * %q_out, %result = qco.measure %q_in : !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %result, %q_out = jeff.qubit_measure_nd %q_in : i1, !jeff.qubit
 * ```
 */
struct ConvertQCOMeasureOpToJeff final
    : StatefulOpConversionPattern<qco::MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto jeffOp = rewriter.create<jeff::QubitMeasureNDOp>(op.getLoc(),
                                                          adaptor.getQubitIn());
    rewriter.replaceOp(op, {jeffOp.getOutQubit(), jeffOp.getResult()});
    return success();
  }
};

/**
 * @brief Converts qco.reset to jeff.qubit_reset
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.qubit_reset %q_in : !jeff.qubit
 * ```
 */
struct ConvertQCOResetOpToJeff final
    : StatefulOpConversionPattern<qco::ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<jeff::QubitResetOp>(op, adaptor.getQubitIn());
    return success();
  }
};

/**
 * @brief Converts qco.gphase to jeff.gphase
 *
 * @par Example:
 * ```mlir
 * qco.gphase(%theta)
 * ```
 * is converted to
 * ```mlir
 * jeff.gphase(%theta) {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8}
 * ```
 */
struct ConvertQCOGPhaseOpToJeff final
    : StatefulOpConversionPattern<qco::GPhaseOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::GPhaseOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    auto jeffOp = rewriter.create<jeff::GPhaseOp>(
        op.getLoc(), op.getParameter(0),
        /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp],
        /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
        /*is_adjoint=*/false,
        /*power=*/1);

    if (inCtrlOp != 0) {
      rewriter.eraseOp(op);
      state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_QCO, OP_CLASS_JEFF,          \
                                         IS_ADJOINT)                           \
  struct ConvertQCO##OP_CLASS_QCO##ToJeff final                                \
      : StatefulOpConversionPattern<qco::OP_CLASS_QCO> {                       \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS_QCO op, OpAdaptor adaptor,                   \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<jeff::OP_CLASS_JEFF>(               \
          op, adaptor, rewriter, getState(), IS_ADJOINT);                      \
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

struct ConvertQCOSXOpToJeff final : StatefulOpConversionPattern<qco::SXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::SXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    Value target;
    if (inCtrlOp != 0) {
      target = state.targetsIn[inCtrlOp][0];
    } else {
      target = adaptor.getQubitIn();
    }

    createCustomOp(op, rewriter, state, {target}, {}, false, "sx");

    return success();
  }
};

struct ConvertQCOSXdgOpToJeff final : StatefulOpConversionPattern<qco::SXdgOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::SXdgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    Value target;
    if (inCtrlOp != 0) {
      target = state.targetsIn[inCtrlOp][0];
    } else {
      target = adaptor.getQubitIn();
    }

    createCustomOp(op, rewriter, state, {target}, {}, true, "sx");

    return success();
  }
};

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS_QCO, OP_CLASS_JEFF)           \
  struct ConvertQCO##OP_CLASS_QCO##ToJeff final                                \
      : StatefulOpConversionPattern<qco::OP_CLASS_QCO> {                       \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS_QCO op, OpAdaptor adaptor,                   \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<jeff::OP_CLASS_JEFF>(                \
          op, adaptor, rewriter, getState());                                  \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, RxOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, RyOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, RzOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, R1Op)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

struct ConvertQCOU2OpToJeff final : StatefulOpConversionPattern<qco::U2Op> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::U2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    Value target;
    if (inCtrlOp != 0) {
      target = state.targetsIn[inCtrlOp][0];
    } else {
      target = adaptor.getQubitIn();
    }

    auto loc = op.getLoc();
    auto theta = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64FloatAttr(std::numbers::pi / 2));
    auto jeffOp = rewriter.create<jeff::UOp>(
        loc, target, theta.getResult(), op.getParameter(0), op.getParameter(1),
        /*in_ctrl_qubits=*/state.controlsIn[inCtrlOp],
        /*num_ctrls=*/state.controlsIn[inCtrlOp].size(),
        /*is_adjoint=*/false, /*power=*/1);

    if (inCtrlOp != 0) {
      rewriter.eraseOp(op);
      state.controlsOut[inCtrlOp] = jeffOp.getOutCtrlQubits();
      state.targetsOut[inCtrlOp] = {jeffOp.getOutQubit()};
    } else {
      rewriter.replaceOp(op, jeffOp.getOutQubit());
    }

    return success();
  }
};

struct ConvertQCOROpToJeff final : StatefulOpConversionPattern<qco::ROp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::ROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    Value target;
    if (inCtrlOp != 0) {
      target = state.targetsIn[inCtrlOp][0];
    } else {
      target = adaptor.getQubitIn();
    }

    createCustomOp(op, rewriter, state, {target},
                   {op.getParameter(0), op.getParameter(1)}, false, "r");

    return success();
  }
};

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS_QCO, OP_CLASS_JEFF)         \
  struct ConvertQCO##OP_CLASS_QCO##ToJeff final                                \
      : StatefulOpConversionPattern<qco::OP_CLASS_QCO> {                       \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS_QCO op, OpAdaptor adaptor,                   \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetThreeParameter<jeff::OP_CLASS_JEFF>(              \
          op, adaptor, rewriter, getState());                                  \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, UOp)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS_QCO, OP_CLASS_JEFF)          \
  struct ConvertQCO##OP_CLASS_QCO##ToJeff final                                \
      : StatefulOpConversionPattern<qco::OP_CLASS_QCO> {                       \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qco::OP_CLASS_QCO op, OpAdaptor adaptor,                   \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetZeroParameter<jeff::OP_CLASS_JEFF>(               \
          op, adaptor, rewriter, getState());                                  \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, SwapOp)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

struct ConvertQCOiSWAPOpToJeff final
    : StatefulOpConversionPattern<qco::iSWAPOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::iSWAPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    llvm::SmallVector<Value> targets;
    if (inCtrlOp != 0) {
      targets.push_back(state.targetsIn[inCtrlOp][0]);
      targets.push_back(state.targetsIn[inCtrlOp][1]);
    } else {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "iswap");

    return success();
  }
};

struct ConvertQCOECROpToJeff final : StatefulOpConversionPattern<qco::ECROp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::ECROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    llvm::SmallVector<Value> targets;
    if (inCtrlOp != 0) {
      targets.push_back(state.targetsIn[inCtrlOp][0]);
      targets.push_back(state.targetsIn[inCtrlOp][1]);
    } else {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "ecr");

    return success();
  }
};

struct ConvertQCODCXOpToJeff final : StatefulOpConversionPattern<qco::DCXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::DCXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    llvm::SmallVector<Value> targets;
    if (inCtrlOp != 0) {
      targets.push_back(state.targetsIn[inCtrlOp][0]);
      targets.push_back(state.targetsIn[inCtrlOp][1]);
    } else {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "dcx");

    return success();
  }
};

// TwoTargetTwoParameter

struct ConvertQCOXXMinusYYOpToJeff final
    : StatefulOpConversionPattern<qco::XXMinusYYOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::XXMinusYYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    llvm::SmallVector<Value> targets;
    if (inCtrlOp != 0) {
      targets.push_back(state.targetsIn[inCtrlOp][0]);
      targets.push_back(state.targetsIn[inCtrlOp][1]);
    } else {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    }

    createCustomOp(op, rewriter, state, targets,
                   {op.getParameter(0), op.getParameter(1)}, false,
                   "xx_minus_yy");

    return success();
  }
};

struct ConvertQCOXXPlusYYOpToJeff final
    : StatefulOpConversionPattern<qco::XXPlusYYOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::XXPlusYYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto inCtrlOp = state.inCtrlOp;

    llvm::SmallVector<Value> targets;
    if (inCtrlOp != 0) {
      targets.push_back(state.targetsIn[inCtrlOp][0]);
      targets.push_back(state.targetsIn[inCtrlOp][1]);
    } else {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    }

    createCustomOp(op, rewriter, state, targets,
                   {op.getParameter(0), op.getParameter(1)}, false,
                   "xx_plus_yy");

    return success();
  }
};

struct ConvertQCOBarrierOpToJeff final
    : StatefulOpConversionPattern<qco::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    createCustomOp(op, rewriter, state, adaptor.getQubitsIn(), {}, false,
                   "barrier");

    return success();
  }
};

/**
 * @brief Converts qco.ctrl to Jeff by inlining the region
 *
 * @par Example:
 * ```mlir
 * %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%a0_in = %q1_in) {
 *   %a1_res = qco.x %a0_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a1_res
 * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
 * ```
 * is converted to
 * ```mlir
 * %target_out, %control_out = jeff.x {is_adjoint = false, num_ctrls = 1 : i8,
 * power = 1 : i8} %target_in ctrls(%control_in) : !jeff.qubit ctrls !jeff.qubit
 * ```
 */
struct ConvertQCOCtrlOpToJeff final : StatefulOpConversionPattern<qco::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    // Set modifier information
    state.inCtrlOp++;
    state.ctrlOps[state.inCtrlOp] = op;
    const SmallVector<Value> controls(adaptor.getControlsIn().begin(),
                                      adaptor.getControlsIn().end());
    state.controlsIn[state.inCtrlOp] = controls;
    const SmallVector<Value> targets(adaptor.getTargetsIn().begin(),
                                     adaptor.getTargetsIn().end());
    state.targetsIn[state.inCtrlOp] = targets;

    // Inline region
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator(), targets);

    return success();
  }
};

/**
 * @brief Erases qco.yield operation
 */
struct ConvertQCOYieldOpToJeff final
    : StatefulOpConversionPattern<qco::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    // Replace CtrlOp
    auto ctrlOp = state.ctrlOps[state.inCtrlOp];
    SmallVector<Value> results;
    const auto& controlsOut = state.controlsOut[state.inCtrlOp];
    const auto& targetsOut = state.targetsOut[state.inCtrlOp];
    results.append(controlsOut.begin(), controlsOut.end());
    results.append(targetsOut.begin(), targetsOut.end());
    rewriter.replaceOp(ctrlOp, results);

    // Clear modifier information
    state.ctrlOps.erase(state.inCtrlOp);
    state.controlsIn.erase(state.inCtrlOp);
    state.controlsOut.erase(state.inCtrlOp);
    state.targetsIn.erase(state.inCtrlOp);
    state.targetsOut.erase(state.inCtrlOp);
    state.inCtrlOp--;

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Type converter for QCO-to-Jeff conversion
 *
 * @details
 * Converts `!qco.qubit` to `!jeff.qubit`.
 */
class QCOToJeffTypeConverter final : public TypeConverter {
public:
  explicit QCOToJeffTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](qco::QubitType /*type*/) -> Type {
      return jeff::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Pass for converting QCO operations to Jeff operations
 *
 * @details
 * TODO
 */
struct QCOToJeff final : impl::QCOToJeffBase<QCOToJeff> {
  using QCOToJeffBase::QCOToJeffBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCOToJeffTypeConverter typeConverter(context);

    LoweringState state;

    // Configure conversion target: QCO illegal, Jeff legal
    target.addIllegalDialect<QCODialect>();
    target.addLegalDialect<jeff::JeffDialect>();

    // Register operation conversion patterns
    patterns
        .add<ConvertQCOAllocOpToJeff, ConvertQCODeallocOpToJeff,
             ConvertQCOMeasureOpToJeff, ConvertQCOResetOpToJeff,
             ConvertQCOGPhaseOpToJeff, ConvertQCOIdOpToJeff,
             ConvertQCOXOpToJeff, ConvertQCOYOpToJeff, ConvertQCOZOpToJeff,
             ConvertQCOHOpToJeff, ConvertQCOSOpToJeff, ConvertQCOSdgOpToJeff,
             ConvertQCOTOpToJeff, ConvertQCOTdgOpToJeff, ConvertQCOSXOpToJeff,
             ConvertQCOSXdgOpToJeff, ConvertQCORXOpToJeff, ConvertQCORYOpToJeff,
             ConvertQCORZOpToJeff, ConvertQCOPOpToJeff, ConvertQCOROpToJeff,
             ConvertQCOU2OpToJeff, ConvertQCOUOpToJeff, ConvertQCOSWAPOpToJeff,
             ConvertQCOiSWAPOpToJeff, ConvertQCOECROpToJeff,
             ConvertQCODCXOpToJeff, ConvertQCOXXMinusYYOpToJeff,
             ConvertQCOXXPlusYYOpToJeff, ConvertQCOBarrierOpToJeff,
             ConvertQCOCtrlOpToJeff, ConvertQCOYieldOpToJeff>(typeConverter,
                                                              context, &state);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
