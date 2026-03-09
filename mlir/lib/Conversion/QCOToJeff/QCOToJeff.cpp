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

#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <utility>

namespace mlir {
using namespace qco;

#define GEN_PASS_DEF_QCOTOJEFF
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h.inc"

namespace {

/**
 * @brief State object for tracking modifier information
 */
struct LoweringState {
  // Modifier information
  bool inCtrlOp = false;
  bool inInvOp = false;
  qco::CtrlOp ctrlOp;
  qco::InvOp invOp;
  llvm::SmallVector<Value> controlsIn;
  llvm::SmallVector<Value> controlsOut;
  llvm::SmallVector<Value> targetsIn;
  llvm::SmallVector<Value> targetsOut;

  [[nodiscard]] bool inModifier() const { return inCtrlOp || inInvOp; }

  // Module information
  llvm::SmallVector<std::string> strings;
  std::string entryPointName;
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
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor instance for the QCO operation
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @param isAdjoint Whether the operation is an adjoint operation
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetZeroParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter,
                              LoweringState& state, const bool isAdjoint) {
  Value target;
  if (!state.inModifier()) {
    target = adaptor.getQubitIn();
  } else {
    target = state.targetsIn[0];
  }

  auto jeffOp = JeffOpType::create(rewriter, op.getLoc(), target,
                                   /*in_ctrl_qubits=*/state.controlsIn,
                                   /*num_ctrls=*/state.controlsIn.size(),
                                   /*is_adjoint=*/state.inInvOp ^ isAdjoint,
                                   /*power=*/1);

  if (!state.inModifier()) {
    rewriter.replaceOp(op, jeffOp.getOutQubit());
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = {jeffOp.getOutQubit()};
  }
  if (state.inCtrlOp) {
    state.controlsOut = jeffOp.getOutCtrlQubits();
  }

  return success();
}

/**
 * @brief Converts a one-target, one-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor instance for the QCO operation
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetOneParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter,
                             LoweringState& state) {
  Value target;
  if (!state.inModifier()) {
    target = adaptor.getQubitIn();
  } else {
    target = state.targetsIn[0];
  }

  auto jeffOp =
      JeffOpType::create(rewriter, op.getLoc(), target, op.getParameter(0),
                         /*in_ctrl_qubits=*/state.controlsIn,
                         /*num_ctrls=*/state.controlsIn.size(),
                         /*is_adjoint=*/state.inInvOp,
                         /*power=*/1);

  if (!state.inModifier()) {
    rewriter.replaceOp(op, jeffOp.getOutQubit());
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = {jeffOp.getOutQubit()};
  }
  if (state.inCtrlOp) {
    state.controlsOut = jeffOp.getOutCtrlQubits();
  }

  return success();
}

/**
 * @brief Converts a one-target, three-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor instance for the QCO operation
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertOneTargetThreeParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                               ConversionPatternRewriter& rewriter,
                               LoweringState& state) {
  Value target;
  if (!state.inModifier()) {
    target = adaptor.getQubitIn();
  } else {
    target = state.targetsIn[0];
  }

  auto jeffOp =
      JeffOpType::create(rewriter, op.getLoc(), target, op.getParameter(0),
                         op.getParameter(1), op.getParameter(2),
                         /*in_ctrl_qubits=*/state.controlsIn,
                         /*num_ctrls=*/state.controlsIn.size(),
                         /*is_adjoint=*/state.inInvOp,
                         /*power=*/1);

  if (!state.inModifier()) {
    rewriter.replaceOp(op, jeffOp.getOutQubit());
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = {jeffOp.getOutQubit()};
  }
  if (state.inCtrlOp) {
    state.controlsOut = jeffOp.getOutCtrlQubits();
  }

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter QCO operation to Jeff
 *
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOOpAdaptorType The OpAdaptor type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param adaptor The OpAdaptor instance for the QCO operation
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename JeffOpType, typename QCOOpType, typename QCOOpAdaptorType>
static LogicalResult
convertTwoTargetZeroParameter(QCOOpType& op, QCOOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter,
                              LoweringState& state) {
  Value target0;
  Value target1;
  if (!state.inModifier()) {
    target0 = adaptor.getQubit0In();
    target1 = adaptor.getQubit1In();
  } else {
    target0 = state.targetsIn[0];
    target1 = state.targetsIn[1];
  }

  auto jeffOp = JeffOpType::create(rewriter, op.getLoc(), target0, target1,
                                   /*in_ctrl_qubits=*/state.controlsIn,
                                   /*num_ctrls=*/state.controlsIn.size(),
                                   /*is_adjoint=*/state.inInvOp,
                                   /*power=*/1);

  if (!state.inModifier()) {
    rewriter.replaceOp(op, {jeffOp.getOutQubitOne(), jeffOp.getOutQubitTwo()});
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = {jeffOp.getOutQubitOne(), jeffOp.getOutQubitTwo()};
  }
  if (state.inCtrlOp) {
    state.controlsOut = jeffOp.getOutCtrlQubits();
  }

  return success();
}

/**
 * @brief Converts an arbitrary QCO operation to a jeff.custom operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @param targets The target qubits of the operation
 * @param params The parameters of the operation
 * @param isAdjoint Whether the operation is an adjoint operation
 * @param name The name of the custom operation
 */
template <typename QCOOpType>
static void createCustomOp(QCOOpType& op, ConversionPatternRewriter& rewriter,
                           LoweringState& state,
                           const llvm::SmallVector<Value>& targets,
                           const llvm::SmallVector<Value>& params,
                           const bool isAdjoint, StringRef name) {
  auto it = llvm::find(state.strings, name);
  if (it == state.strings.end()) {
    state.strings.emplace_back(name);
  }

  auto jeffOp = jeff::CustomOp::create(
      rewriter, op.getLoc(), targets,
      /*in_ctrl_qubits=*/state.controlsIn, /*params=*/params,
      /*num_ctrls=*/state.controlsIn.size(),
      /*is_adjoint=*/state.inInvOp ^ isAdjoint,
      /*power=*/1, /*name=*/name, /*num_targets=*/targets.size(),
      /*num_params=*/params.size());

  if (!state.inModifier()) {
    rewriter.replaceOp(op, jeffOp.getOutTargetQubits());
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = jeffOp.getOutTargetQubits();
  }
  if (state.inCtrlOp) {
    state.controlsOut = jeffOp.getOutCtrlQubits();
  }
}

/**
 * @brief Converts a compatible QCO operation to a jeff.ppr operation
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @param op The QCO operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @param targets The target qubits of the operation
 * @param pauliGates The Pauli gates defining the operation
 */
template <typename QCOOpType>
static void createPPROp(QCOOpType& op, ConversionPatternRewriter& rewriter,
                        LoweringState& state,
                        const llvm::SmallVector<Value>& targets,
                        const llvm::SmallVector<int32_t>& pauliGates) {
  auto pauliGatesAttr =
      DenseI32ArrayAttr::get(rewriter.getContext(), pauliGates);

  auto jeffOp =
      jeff::PPROp::create(rewriter, op.getLoc(), targets,
                          /*in_ctrl_qubits=*/state.controlsIn,
                          /*rotation=*/op.getParameter(0),
                          /*num_ctrls=*/state.controlsIn.size(),
                          /*is_adjoint=*/state.inInvOp,
                          /*power=*/1, /*pauli_gates=*/pauliGatesAttr);

  if (!state.inModifier()) {
    rewriter.replaceOp(op, jeffOp.getOutQubits());
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = jeffOp.getOutQubits();
  }
  if (state.inCtrlOp) {
    state.controlsOut = jeffOp.getOutCtrlQubits();
  }
}

/**
 * @brief Cleans up the module after conversion
 *
 * @param op The module operation to clean up
 * @param state The lowering state
 * @return LogicalResult Success or failure of the cleanup
 */
static LogicalResult cleanUp(Operation* op, LoweringState& state) {
  if (state.entryPointName.empty()) {
    return failure();
  }

  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module) {
    return failure();
  }

  for (auto funcOp : module.getOps<func::FuncOp>()) {
    state.strings.emplace_back(funcOp.getSymName());
  }

  auto* const it = llvm::find(state.strings, state.entryPointName);
  if (it == state.strings.end()) {
    return failure();
  }
  const auto distance = std::distance(state.strings.begin(), it);
  if (distance > std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  const auto entryPoint = static_cast<uint16_t>(distance);

  // Set module attributes
  OpBuilder builder(module.getContext());
  auto uint16Type = builder.getIntegerType(16, false);

  module->setAttr("jeff.entrypoint",
                  builder.getIntegerAttr(uint16Type, entryPoint));

  llvm::SmallVector<llvm::StringRef> stringRefs;
  stringRefs.reserve(state.strings.size());
  for (const auto& str : state.strings) {
    stringRefs.emplace_back(str);
  }
  module->setAttr("jeff.strings", builder.getStrArrayAttr(stringRefs));

  module->setAttr("jeff.tool", builder.getStringAttr("mqt-cc"));
  module->setAttr("jeff.toolVersion", builder.getStringAttr(MQT_CORE_VERSION));

  module->setAttr("jeff.version", builder.getIntegerAttr(uint16Type, 0));
  module->setAttr("jeff.versionMinor", builder.getIntegerAttr(uint16Type, 1));
  module->setAttr("jeff.versionPatch", builder.getIntegerAttr(uint16Type, 0));

  return success();
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
 * %q_out, %result = jeff.qubit_measure_nd %q_in : !jeff.qubit, i1
 * ```
 */
struct ConvertQCOMeasureOpToJeff final
    : StatefulOpConversionPattern<qco::MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<jeff::QubitMeasureNDOp>(op,
                                                        adaptor.getQubitIn());
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

    auto jeffOp =
        jeff::GPhaseOp::create(rewriter, op.getLoc(), op.getParameter(0),
                               /*in_ctrl_qubits=*/state.controlsIn,
                               /*num_ctrls=*/state.controlsIn.size(),
                               /*is_adjoint=*/state.inInvOp,
                               /*power=*/1);

    rewriter.eraseOp(op);
    if (state.inCtrlOp) {
      state.controlsOut = jeffOp.getOutCtrlQubits();
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

/**
 * @brief Converts qco.sx to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.sx %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.custom "sx"() {is_adjoint = false, num_ctrls = 0 : i8, power =
 * 1 : i8} %q_in : !jeff.qubit
 * ```
 */
struct ConvertQCOSXOpToJeff final : StatefulOpConversionPattern<qco::SXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::SXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    Value target;
    if (!state.inModifier()) {
      target = adaptor.getQubitIn();
    } else {
      target = state.targetsIn[0];
    }

    createCustomOp(op, rewriter, state, {target}, {}, false, "sx");

    return success();
  }
};

/**
 * @brief Converts qco.sxdg to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.sxdg %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.custom "sx"() {is_adjoint = true, num_ctrls = 0 : i8, power = 1
 * : i8} %q_in : !jeff.qubit
 * ```
 */
struct ConvertQCOSXdgOpToJeff final : StatefulOpConversionPattern<qco::SXdgOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::SXdgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    Value target;
    if (!state.inModifier()) {
      target = adaptor.getQubitIn();
    } else {
      target = state.targetsIn[0];
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

/**
 * @brief Converts qco.u2 to jeff.u
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.u2(%phi, %lambda) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %theta = jeff.float_const64(1.57079632679) : f64
 * %q_out = jeff.u(%theta, %phi, %lambda) {is_adjoint = false, num_ctrls = 0 :
 * i8, power = 1 : i8} %q_in : !jeff.qubit
 * ```
 */
struct ConvertQCOU2OpToJeff final : StatefulOpConversionPattern<qco::U2Op> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::U2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    Value target;
    if (!state.inModifier()) {
      target = adaptor.getQubitIn();
    } else {
      target = state.targetsIn[0];
    }

    auto loc = op.getLoc();
    auto theta = jeff::FloatConst64Op::create(
        rewriter, loc, rewriter.getF64FloatAttr(std::numbers::pi / 2));
    auto jeffOp = jeff::UOp::create(rewriter, loc, target, theta.getResult(),
                                    op.getParameter(0), op.getParameter(1),
                                    /*in_ctrl_qubits=*/state.controlsIn,
                                    /*num_ctrls=*/state.controlsIn.size(),
                                    /*is_adjoint=*/state.inInvOp, /*power=*/1);

    if (!state.inModifier()) {
      rewriter.replaceOp(op, jeffOp.getOutQubit());
    } else {
      rewriter.eraseOp(op);
      state.targetsOut = {jeffOp.getOutQubit()};
    }
    if (state.inCtrlOp) {
      state.controlsOut = jeffOp.getOutCtrlQubits();
    }

    return success();
  }
};

/**
 * @brief Converts qco.r to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.r(%theta, %phi) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.custom "r"(%theta, %phi) {is_adjoint = true, num_ctrls = 0 :
 * i8, power = 1 : i8} %q_in : !jeff.qubit
 * ```
 */
struct ConvertQCOROpToJeff final : StatefulOpConversionPattern<qco::ROp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::ROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    Value target;
    if (!state.inModifier()) {
      target = adaptor.getQubitIn();
    } else {
      target = state.targetsIn[0];
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

/**
 * @brief Converts qco.iswap to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.iswap %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.custom "iswap"() {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCOiSWAPOpToJeff final
    : StatefulOpConversionPattern<qco::iSWAPOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::iSWAPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "iswap");

    return success();
  }
};

/**
 * @brief Converts qco.ecr to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.ecr %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.custom "ecr"() {is_adjoint = false, num_ctrls = 0 : i8, power
 * = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCOECROpToJeff final : StatefulOpConversionPattern<qco::ECROp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::ECROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "ecr");

    return success();
  }
};

/**
 * @brief Converts qco.dcx to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.dcx %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.custom "dcx"() {is_adjoint = false, num_ctrls = 0 : i8, power
 * = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCODCXOpToJeff final : StatefulOpConversionPattern<qco::DCXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::DCXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "dcx");

    return success();
  }
};

// TwoTargetOneParameter

/**
 * @brief Converts qco.rxx to jeff.ppr
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.rxx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.ppr(%theta, [1, 1]) {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCORXXOpToJeff final : StatefulOpConversionPattern<qco::RXXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::RXXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createPPROp(op, rewriter, state, targets, {1, 1});

    return success();
  }
};

/**
 * @brief Converts qco.ryy to jeff.ppr
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.ryy(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.ppr(%theta, [2, 2]) {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCORYYOpToJeff final : StatefulOpConversionPattern<qco::RYYOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::RYYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createPPROp(op, rewriter, state, targets, {2, 2});

    return success();
  }
};

/**
 * @brief Converts qco.rzx to jeff.ppr
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.rzx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.ppr(%theta, [3, 1]) {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCORZXOpToJeff final : StatefulOpConversionPattern<qco::RZXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::RZXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createPPROp(op, rewriter, state, targets, {3, 1});

    return success();
  }
};

/**
 * @brief Converts qco.rzz to jeff.ppr
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.rzz(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.ppr(%theta, [3, 3]) {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCORZZOpToJeff final : StatefulOpConversionPattern<qco::RZZOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::RZZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createPPROp(op, rewriter, state, targets, {3, 3});

    return success();
  }
};

// TwoTargetTwoParameter

/**
 * @brief Converts qco.xx_minus_yy to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.xx_minus_yy(%theta, %beta) %q0_in, %q1_in :
 * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.custom "xx_minus_yy"(%theta, %beta) {is_adjoint = false,
 * num_ctrls = 0 : i8, power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCOXXMinusYYOpToJeff final
    : StatefulOpConversionPattern<qco::XXMinusYYOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::XXMinusYYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createCustomOp(op, rewriter, state, targets,
                   {op.getParameter(0), op.getParameter(1)}, false,
                   "xx_minus_yy");

    return success();
  }
};

/**
 * @brief Converts qco.xx_plus_yy to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = qco.xx_plus_yy(%theta, %beta) %q0_in, %q1_in : !qco.qubit,
 * !qco.qubit -> !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.custom "xx_plus_yy"(%theta, %beta) {is_adjoint = false,
 * num_ctrls = 0 : i8, power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCOXXPlusYYOpToJeff final
    : StatefulOpConversionPattern<qco::XXPlusYYOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::XXPlusYYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets.push_back(adaptor.getQubit0In());
      targets.push_back(adaptor.getQubit1In());
    } else {
      targets.push_back(state.targetsIn[0]);
      targets.push_back(state.targetsIn[1]);
    }

    createCustomOp(op, rewriter, state, targets,
                   {op.getParameter(0), op.getParameter(1)}, false,
                   "xx_plus_yy");

    return success();
  }
};

/**
 * @brief Converts qco.barrier to jeff.custom
 *
 * @par Example:
 * ```mlir
 * %q_out:2 = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit,
 * !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out:2 = jeff.custom "barrier"() {is_adjoint = false, num_ctrls = 0 : i8,
 * power = 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
struct ConvertQCOBarrierOpToJeff final
    : StatefulOpConversionPattern<qco::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    llvm::SmallVector<Value> targets;
    if (!state.inModifier()) {
      targets = adaptor.getQubitsIn();
    } else {
      targets = state.targetsIn;
    }

    createCustomOp(op, rewriter, state, targets, {}, false, "barrier");

    return success();
  }
};

/**
 * @brief Converts qco.ctrl to Jeff by inlining the region
 *
 * @par Example:
 * ```mlir
 * %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%a_in = %q1_in) {
 *   %a_res = qco.x %a_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a_res
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

    if (state.inCtrlOp) {
      return rewriter.notifyMatchFailure(
          op, "Nested control operations are not supported. Run the "
              "canonicalization pass before the conversion");
    }

    if (state.inInvOp) {
      return rewriter.notifyMatchFailure(
          op, "Control operations inside inversion operations are not "
              "supported. Run the canonicalization pass before the conversion");
    }

    // Set modifier information
    state.inCtrlOp = true;
    state.ctrlOp = op;
    const SmallVector<Value> controls(adaptor.getControlsIn().begin(),
                                      adaptor.getControlsIn().end());
    state.controlsIn = controls;
    const SmallVector<Value> targets(adaptor.getTargetsIn().begin(),
                                     adaptor.getTargetsIn().end());
    state.targetsIn = targets;

    // Inline region
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator(), targets);

    return success();
  }
};

/**
 * @brief Converts qco.inv to Jeff by inlining the region
 *
 * @par Example:
 * ```mlir
 * %q_out = qco.inv (%a_in = %q_in) {
 *   %a_res = qco.s %a_in : !qco.qubit -> !qco.qubit
 *   qco.yield %a_res
 * } : {!qco.qubit} -> {!qco.qubit}
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.s {is_adjoint = true, num_ctrls = 0 : i8, power = 1 : i8} %q_in
 * : !jeff.qubit
 * ```
 */
struct ConvertQCOInvOpToJeff final : StatefulOpConversionPattern<qco::InvOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::InvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    if (state.inInvOp) {
      return rewriter.notifyMatchFailure(
          op, "Nested inversion operations are not supported. Run the "
              "canonicalization pass before the conversion");
    }

    // Set modifier information
    state.inInvOp = true;
    state.invOp = op;
    if (state.targetsIn.empty()) {
      const SmallVector<Value> targets(adaptor.getQubitsIn().begin(),
                                       adaptor.getQubitsIn().end());
      state.targetsIn = targets;
    }

    // Inline region
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator(), state.targetsIn);

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

    if (state.inInvOp) {
      rewriter.replaceOp(state.invOp, state.targetsOut);

      state.inInvOp = false;
      state.invOp = nullptr;

      if (!state.inCtrlOp) {
        state.targetsIn.clear();
        state.targetsOut.clear();
      }
    } else if (state.inCtrlOp) {
      SmallVector<Value> results;
      const auto& controlsOut = state.controlsOut;
      const auto& targetsOut = state.targetsOut;
      results.append(controlsOut.begin(), controlsOut.end());
      results.append(targetsOut.begin(), targetsOut.end());
      rewriter.replaceOp(state.ctrlOp, results);

      state.inCtrlOp = false;
      state.ctrlOp = nullptr;

      state.controlsIn.clear();
      state.controlsOut.clear();
      state.targetsIn.clear();
      state.targetsOut.clear();
    }

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts the QCO-style main function to a Jeff-style main function
 *
 * @par Example:
 * ```mlir
 * func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
 *   %0 = arith.constant 0 : i64
 *   return %0
 * }
 * ```
 * is converted to
 * ```mlir
 * func.func @main() -> () {
 *   return
 * }
 * ```
 */
struct ConvertQCOMainToJeff final : StatefulOpConversionPattern<func::FuncOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto passthrough = op->getAttrOfType<ArrayAttr>("passthrough");
    if (!passthrough) {
      return failure();
    }

    if (!llvm::any_of(passthrough, [](Attribute attr) {
          const auto strAttr = llvm::dyn_cast<StringAttr>(attr);
          return strAttr && strAttr.getValue() == "entry_point";
        })) {
      return failure();
    }

    if (op.getBlocks().size() != 1) {
      return failure();
    }
    auto* block = &op.getBlocks().front();

    auto* returnOp = block->getTerminator();
    if (!llvm::isa<func::ReturnOp>(returnOp)) {
      return failure();
    }

    getState().entryPointName = op.getSymName();

    // Update function signature and remove passthrough attribute
    rewriter.startOpModification(op);
    op.setType(FunctionType::get(rewriter.getContext(), {}, {}));
    op->removeAttr("passthrough");
    rewriter.finalizeOpModification(op);

    // Replace return operation
    rewriter.setInsertionPointToEnd(block);
    func::ReturnOp::create(rewriter, returnOp->getLoc());
    rewriter.eraseOp(returnOp);

    return success();
  }
};

/**
 * @brief Converts arith.constant to Jeff
 *
 * @par Example:
 * ```mlir
 * %0 = arith.constant 0 : i64
 * ```
 * is converted to
 * ```mlir
 * %0 = jeff.int_const64(0) : i64
 * ```
 */
struct ConvertArithConstOpToJeff final
    : StatefulOpConversionPattern<arith::ConstantOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto value = op.getValue();
    return llvm::TypeSwitch<Type, LogicalResult>(op.getType())
        .Case<FloatType>([&](auto type) -> LogicalResult {
          auto floatAttr = llvm::dyn_cast<FloatAttr>(value);
          if (!floatAttr) {
            return rewriter.notifyMatchFailure(op, "Expected float attribute");
          }
          switch (type.getWidth()) {
          case 64:
            rewriter.replaceOpWithNewOp<jeff::FloatConst64Op>(op, floatAttr);
            return success();
          default:
            return rewriter.notifyMatchFailure(op, "Unsupported type");
          }
        })
        .Case<IntegerType>([&](auto type) -> LogicalResult {
          auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
          if (!intAttr) {
            return rewriter.notifyMatchFailure(op,
                                               "Expected integer attribute");
          }
          switch (type.getWidth()) {
          case 1:
            rewriter.replaceOpWithNewOp<jeff::IntConst1Op>(op, intAttr);
            return success();
          case 64:
            rewriter.replaceOpWithNewOp<jeff::IntConst64Op>(op, intAttr);
            return success();
          default:
            return rewriter.notifyMatchFailure(op, "Unsupported type");
          }
        })
        .Default([&](auto) -> LogicalResult {
          return rewriter.notifyMatchFailure(op, "Unsupported type");
        });
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
 */
struct QCOToJeff final : impl::QCOToJeffBase<QCOToJeff> {
  using QCOToJeffBase::QCOToJeffBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCOToJeffTypeConverter typeConverter(context);

    LoweringState state;

    // Configure conversion target
    target.addIllegalDialect<QCODialect>();
    target.addLegalDialect<jeff::JeffDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return !op->hasAttr("passthrough"); });
    target.addLegalOp<func::ReturnOp>();

    // Register operation conversion patterns
    patterns.add<
        ConvertQCOAllocOpToJeff, ConvertQCODeallocOpToJeff,
        ConvertQCOMeasureOpToJeff, ConvertQCOResetOpToJeff,
        ConvertQCOGPhaseOpToJeff, ConvertQCOIdOpToJeff, ConvertQCOXOpToJeff,
        ConvertQCOYOpToJeff, ConvertQCOZOpToJeff, ConvertQCOHOpToJeff,
        ConvertQCOSOpToJeff, ConvertQCOSdgOpToJeff, ConvertQCOTOpToJeff,
        ConvertQCOTdgOpToJeff, ConvertQCOSXOpToJeff, ConvertQCOSXdgOpToJeff,
        ConvertQCORXOpToJeff, ConvertQCORYOpToJeff, ConvertQCORZOpToJeff,
        ConvertQCOPOpToJeff, ConvertQCOROpToJeff, ConvertQCOU2OpToJeff,
        ConvertQCOUOpToJeff, ConvertQCOSWAPOpToJeff, ConvertQCOiSWAPOpToJeff,
        ConvertQCOECROpToJeff, ConvertQCODCXOpToJeff, ConvertQCORXXOpToJeff,
        ConvertQCORYYOpToJeff, ConvertQCORZXOpToJeff, ConvertQCORZZOpToJeff,
        ConvertQCOXXMinusYYOpToJeff, ConvertQCOXXPlusYYOpToJeff,
        ConvertQCOBarrierOpToJeff, ConvertQCOCtrlOpToJeff,
        ConvertQCOInvOpToJeff, ConvertQCOYieldOpToJeff, ConvertQCOMainToJeff,
        ConvertArithConstOpToJeff>(typeConverter, context, &state);

    // Apply the conversion
    if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }

    if (cleanUp(module, state).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
