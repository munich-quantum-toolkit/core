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
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <jeff/Conversion/NativeToJeff/NativeToJeff.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <type_traits>
#include <utility>

namespace mlir {

using namespace qco;

#define GEN_PASS_DEF_QCOTOJEFF
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h.inc"

namespace {

/** @brief Qubit allocation mode */
enum class AllocationMode : std::uint8_t {
  Unset,  //!< No allocation mode has been established yet.
  Static, //!< The module uses static qubit allocation.
  Dynamic //!< The module uses dynamic qubit allocation.
};

/**
 * @brief State object for tracking modifier information
 */
struct LoweringState {
  // Modifier information
  bool inCtrlOp = false;
  bool inInvOp = false;
  CtrlOp ctrlOp;
  InvOp invOp;
  llvm::SmallVector<Value> controlsIn;
  llvm::SmallVector<Value> controlsOut;
  llvm::SmallVector<Value> targetsIn;
  llvm::SmallVector<Value> targetsOut;

  [[nodiscard]] bool inModifier() const { return inCtrlOp || inInvOp; }

  // Module information
  llvm::SmallVector<std::string> strings;
  std::string entryPointName;

  /// The qubit allocation mode used in the module
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Sets or validates the allocation mode, or emits an error if it conflicts.
  [[nodiscard]] LogicalResult ensureAllocationMode(AllocationMode requestedMode,
                                                   Operation* op) {
    if (allocationMode == AllocationMode::Unset) {
      allocationMode = requestedMode;
      return success();
    }
    if (allocationMode == requestedMode) {
      return success();
    }
    return op->emitOpError(
        "cannot mix static and dynamic qubit allocation modes in QCO program");
  }
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
 * @brief Handles the results of a gate conversion
 *
 * @details
 * The original QCO operation is replaced or erased, and the state is updated.
 *
 * @param op The original QCO operation
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @param targetsOut The target qubits produced by the new operation
 * @param controlsOut The control qubits produced by the new operation
 */
static void handleResult(Operation* op, ConversionPatternRewriter& rewriter,
                         LoweringState& state, ValueRange targetsOut,
                         ValueRange controlsOut) {
  if (!state.inModifier()) {
    rewriter.replaceOp(op, targetsOut);
  } else {
    rewriter.eraseOp(op);
    state.targetsOut = llvm::to_vector(targetsOut);
  }
  if (state.inCtrlOp) {
    state.controlsOut = llvm::to_vector(controlsOut);
  }
}

/**
 * @brief Target operands: `adaptor.getOperands()` at the matched op, or
 * `state.targetsIn` while lowering inside `qco.ctrl` / `qco.inv`.
 *
 * @param state Lowering state.
 * @param adaptor Operand adaptor for the matched op.
 * @tparam NumParams Number of parameters to drop from the end of the operand
 * list.
 * @tparam OpAdaptor Adaptor with `getOperands()`.
 * @return ValueRange The target operands.
 */
template <size_t NumParams, typename OpAdaptor>
[[nodiscard]] static ValueRange getEffectiveTargetOperands(LoweringState& state,
                                                           OpAdaptor adaptor) {
  return state.inModifier()
             ? ValueRange(state.targetsIn)
             : ValueRange(adaptor.getOperands().drop_back(NumParams));
}

/**
 * @brief Lowers QCO gates to matching Jeff ops.
 *
 * @details Uses `getEffectiveTargetOperands` and forwards target and parameter
 * indices into `JeffOpType::create`.
 *
 * @tparam QCOOpType The QCO gate op type
 * @tparam JeffOpType The Jeff op type
 * @tparam ExtraAdjoint Whether to XOR the adjoint flag
 * @tparam TargetIndices QCO target indices to forward
 * @tparam ParamIndices QCO parameter indices to forward
 */
template <typename QCOOpType, typename JeffOpType, bool ExtraAdjoint = false,
          std::size_t... TargetIndices, std::size_t... ParamIndices>
static LogicalResult
convertJeffGate(QCOOpType op, typename QCOOpType::Adaptor adaptor,
                ConversionPatternRewriter& rewriter, LoweringState& state,
                std::index_sequence<TargetIndices...> /*targetIndices*/,
                std::index_sequence<ParamIndices...> /*paramIndices*/) {
  constexpr std::size_t numParams = sizeof...(ParamIndices);
  ValueRange targets = getEffectiveTargetOperands<numParams>(state, adaptor);
  assert(targets.size() >= sizeof...(TargetIndices) &&
         "Not enough operands available for conversion");
  ValueRange params = op.getParameters();

  auto jeffOp = JeffOpType::create(
      rewriter, op.getLoc(), targets[TargetIndices]..., params[ParamIndices]...,
      /*in_ctrl_qubits=*/state.controlsIn,
      /*num_ctrls=*/state.controlsIn.size(),
      /*is_adjoint=*/state.inInvOp ^ ExtraAdjoint,
      /*power=*/1);

  // Jeff well-known gates: leading results are transformed targets, then ctrl
  // outs (same ordering as `getOutQubit` / `getOutCtrlQubits` accessors).
  constexpr std::size_t numTargets = sizeof...(TargetIndices);
  auto results = jeffOp->getResults();
  handleResult(op, rewriter, state, results.take_front(numTargets),
               results.drop_front(numTargets));
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
                           LoweringState& state, ValueRange targets,
                           ValueRange params, const bool isAdjoint,
                           StringRef name) {
  auto* const it = llvm::find(state.strings, name);
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

  handleResult(op, rewriter, state, jeffOp.getOutTargetQubits(),
               jeffOp.getOutCtrlQubits());
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
                        LoweringState& state, ValueRange targets,
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

  handleResult(op, rewriter, state, jeffOp.getOutQubits(),
               jeffOp.getOutCtrlQubits());
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
  if (std::cmp_greater(distance, std::numeric_limits<uint16_t>::max())) {
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

namespace {

/**
 * @brief Converts qtensor.alloc to jeff.qureg_alloc
 *
 * @par Example:
 * ```mlir
 * %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * %qureg = jeff.qureg_alloc(%c3) : !jeff.qureg
 * ```
 */
struct ConvertQTensorAllocOp final
    : StatefulOpConversionPattern<qtensor::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic,
                                               op.getOperation()))) {
      return failure();
    }
    // TODO: Why is this not happening in native conversion?
    auto sizeValue = getConstantIntValue(adaptor.getSize());
    Value size;
    if (sizeValue.has_value()) {
      size = jeff::IntConst32Op::create(rewriter, op.getLoc(), *sizeValue);
    } else {
      size = adaptor.getSize();
    }
    auto qregType =
        jeff::QuregType::get(rewriter.getContext(), op.getType().getShape()[0]);
    rewriter.replaceOpWithNewOp<jeff::QuregAllocOp>(op, qregType, size);
    return success();
  }
};

/**
 * @brief Converts qtensor.extract to jeff.qureg_extract_index
 *
 * @par Example:
 * ```mlir
 * %tensor_out, %q = qtensor.extract %tensor_in[%c0]: tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * %qureg_out, %q = jeff.qureg_extract_index(%c0) %qureg_in : !jeff.qureg,
 * !jeff.qubit
 * ```
 */
struct ConvertQTensorExtractOp final
    : StatefulOpConversionPattern<qtensor::ExtractOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // TODO: Why is this not happening in native conversion?
    auto indexValue = getConstantIntValue(adaptor.getIndex());
    Value index;
    if (indexValue.has_value()) {
      index = jeff::IntConst32Op::create(rewriter, op.getLoc(), *indexValue);
    } else {
      index = adaptor.getIndex();
    }
    auto qregType = jeff::QuregType::get(
        rewriter.getContext(), op.getTensor().getType().getShape()[0]);
    rewriter.replaceOpWithNewOp<jeff::QuregExtractIndexOp>(
        op, qregType, adaptor.getTensor(), index);
    return success();
  }
};

/**
 * @brief Converts qtensor.insert to jeff.qureg_insert_index
 *
 * @par Example:
 * ```mlir
 * %tensor_out = qtensor.insert %q into %tensor_in[%c0] : tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * %qureg_out = jeff.qureg_insert_index(%c0) %qureg_in %q : !jeff.qureg
 * ```
 */
struct ConvertQTensorInsertOp final
    : StatefulOpConversionPattern<qtensor::InsertOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // TODO: Why is this not happening in native conversion?
    auto indexValue = getConstantIntValue(adaptor.getIndex());
    Value index;
    if (indexValue.has_value()) {
      index = jeff::IntConst32Op::create(rewriter, op.getLoc(), *indexValue);
    } else {
      index = adaptor.getIndex();
    }
    auto qregType = jeff::QuregType::get(rewriter.getContext(),
                                         op.getDest().getType().getShape()[0]);
    rewriter.replaceOpWithNewOp<jeff::QuregInsertIndexOp>(
        op, qregType, adaptor.getDest(), index, adaptor.getScalar());
    return success();
  }
};

/**
 * @brief Converts qtensor.dealloc to jeff.qureg_free_zero
 *
 * @par Example:
 * ```mlir
 * qtensor.dealloc %tensor : tensor<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * jeff.qureg_free_zero %qureg : !jeff.qureg
 * ```
 */
struct ConvertQTensorDeallocOp final
    : StatefulOpConversionPattern<qtensor::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qtensor::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<jeff::QuregFreeZeroOp>(op, adaptor.getTensor());
    return success();
  }
};

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
struct ConvertQCOAllocOpToJeff final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic,
                                               op.getOperation()))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<jeff::QubitAllocOp>(op);
    return success();
  }
};

/**
 * @brief Converts qco.static to jeff.qubit_alloc
 *
 * @details
 * The Jeff dialect does not model hardware-mapped or fixed-index static
 * qubits yet. As a temporary workaround (see discussion on #1626), this
 * lowers `qco.static` to the same `jeff.qubit_alloc` operation used for
 * `qco.alloc`. The static index is not represented in Jeff IR; if Jeff gains
 * static qubit support, this conversion should be revisited.
 *
 * @par Example:
 * ```mlir
 * %q = qco.static 0 : !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = jeff.qubit_alloc : !jeff.qubit
 * ```
 */
struct ConvertQCOStaticOpToJeff final : StatefulOpConversionPattern<StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Static,
                                               op.getOperation()))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<jeff::QubitAllocOp>(op);
    return success();
  }
};

/**
 * @brief Converts qco.sink to jeff.qubit_free_zero
 *
 * @par Example:
 * ```mlir
 * qco.sink %q : !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * jeff.qubit_free_zero %q : !jeff.qubit
 * ```
 */
struct ConvertQCOSinkOpToJeff final : StatefulOpConversionPattern<SinkOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor adaptor,
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
    : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
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
struct ConvertQCOResetOpToJeff final : StatefulOpConversionPattern<ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
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
struct ConvertQCOGPhaseOpToJeff final : StatefulOpConversionPattern<GPhaseOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(GPhaseOp op, OpAdaptor /*adaptor*/,
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

/**
 * @brief Converts a QCO gate that lowers to a well-known Jeff op.
 *
 * @tparam QCOOpType QCO operation type.
 * @tparam JeffOpType Jeff op type passed to `convertJeffGate` /
 * `JeffOpType::create`.
 * @tparam NumTargets Number of target operands (1 or 2 for supported gates).
 * @tparam NumParams Number of real parameters on the QCO op.
 * @tparam JeffBaseAdjoint When true, XOR with inv-modifier (e.g. S† as `jeff.s`
 * with adjoint set).
 *
 * @par Example: one target, zero parameters
 * ```mlir
 * %q_out = qco.x %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.x {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8}
 * %q_in : !jeff.qubit
 * ```
 *
 * @par Example: one target, one parameter
 * ```mlir
 * %q_out = qco.rx(%theta) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.rx(%theta) {is_adjoint = false, num_ctrls = 0 : i8, power = 1 :
 * i8} %q_in : !jeff.qubit
 * ```
 *
 * @par Example: one target, three parameters
 * ```mlir
 * %q_out = qco.u(%theta, %phi, %lambda) %q_in : !qco.qubit -> !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = jeff.u(%theta, %phi, %lambda) {is_adjoint = false, num_ctrls = 0 :
 * i8, power = 1 : i8} %q_in : !jeff.qubit
 * ```
 *
 * @par Example: two targets, zero parameters
 * ```mlir
 * %q0_out, %q1_out = qco.swap %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = jeff.swap {is_adjoint = false, num_ctrls = 0 : i8, power =
 * 1 : i8} %q0_in, %q1_in : !jeff.qubit, !jeff.qubit
 * ```
 */
template <typename QCOOpType, typename JeffOpType, std::size_t NumTargets,
          std::size_t NumParams, bool JeffBaseAdjoint>
struct ConvertQCOWellKnownGateToJeff final
    : StatefulOpConversionPattern<QCOOpType> {
  using StatefulOpConversionPattern<QCOOpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();

    return convertJeffGate<QCOOpType, JeffOpType, JeffBaseAdjoint>(
        op, adaptor, rewriter, state, std::make_index_sequence<NumTargets>{},
        std::make_index_sequence<NumParams>{});
  }
};

/**
 * @brief Conversion pattern that lowers a QCO gate to `jeff.custom`.
 *
 * @tparam QCOOpType QCO operation type to match.
 * @tparam NumTargets Number of target qubit operands (compile-time).
 * @tparam NumParams Number of real parameters taken from the QCO op
 * (compile-time).
 *
 * @details Validates operand count when not inside a modifier, collects targets
 * and parameters, then dispatches to `createCustomOp` with the configured
 * custom gate name and base adjoint flag.
 */
template <typename QCOOpType, std::size_t NumTargets, std::size_t NumParams>
struct ConvertQCOCustomGateToJeff final
    : StatefulOpConversionPattern<QCOOpType> {
  ConvertQCOCustomGateToJeff(TypeConverter& typeConverter, MLIRContext* context,
                             LoweringState* state, StringRef name,
                             const bool baseIsAdjoint)
      : StatefulOpConversionPattern<QCOOpType>(typeConverter, context, state),
        name_(name), baseIsAdjoint_(baseIsAdjoint) {}

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();

    if (!state.inModifier()) {
      const auto expected = NumTargets + NumParams;
      if (adaptor.getOperands().size() != expected) {
        return op.emitOpError()
               << "expected " << expected
               << " operands (targets + parameters) for QCO→Jeff custom gate "
                  "conversion, got "
               << adaptor.getOperands().size();
      }
    }

    ValueRange targets = getEffectiveTargetOperands<NumParams>(state, adaptor);
    assert(targets.size() >= NumTargets &&
           "Not enough operands available for conversion");

    createCustomOp(op, rewriter, state, targets, op.getParameters(),
                   baseIsAdjoint_, name_);
    return success();
  }

private:
  StringRef name_;
  bool baseIsAdjoint_;
};

/**
 * @brief Conversion pattern that lowers a QCO gate to `jeff.ppr`.
 *
 * @tparam QCOOpType QCO operation type (expected: two targets, one rotation
 * param).
 *
 * @details Selects two target operands (respecting modifier state) and builds
 * the Pauli tuple from the constructor-supplied encodings `p0_` and `p1_`.
 */
template <typename QCOOpType>
struct ConvertQCOPPRGateToJeff final : StatefulOpConversionPattern<QCOOpType> {
  ConvertQCOPPRGateToJeff(TypeConverter& typeConverter, MLIRContext* context,
                          LoweringState* state, const int32_t p0,
                          const int32_t p1)
      : StatefulOpConversionPattern<QCOOpType>(typeConverter, context, state),
        p0_(p0), p1_(p1) {}

  LogicalResult
  matchAndRewrite(QCOOpType op, QCOOpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();

    ValueRange targets = getEffectiveTargetOperands<1>(state, adaptor);
    assert(targets.size() >= 2 &&
           "Not enough operands available for conversion");
    createPPROp(op, rewriter, state, targets, {p0_, p1_});
    return success();
  }

private:
  int32_t p0_;
  int32_t p1_;
};

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
struct ConvertQCOU2OpToJeff final : StatefulOpConversionPattern<U2Op> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(U2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    ValueRange targets = getEffectiveTargetOperands<2>(state, adaptor);
    assert(!targets.empty() && "Not enough operands available for conversion");
    auto target = targets.front();

    auto loc = op.getLoc();
    auto theta = jeff::FloatConst64Op::create(
        rewriter, loc, rewriter.getF64FloatAttr(std::numbers::pi / 2));
    auto jeffOp = jeff::UOp::create(rewriter, loc, target, theta.getResult(),
                                    op.getParameter(0), op.getParameter(1),
                                    /*in_ctrl_qubits=*/state.controlsIn,
                                    /*num_ctrls=*/state.controlsIn.size(),
                                    /*is_adjoint=*/state.inInvOp, /*power=*/1);

    handleResult(op, rewriter, state, jeffOp.getOutQubit(),
                 jeffOp.getOutCtrlQubits());

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
    : StatefulOpConversionPattern<BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    ValueRange targets = getEffectiveTargetOperands<0>(state, adaptor);

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
struct ConvertQCOCtrlOpToJeff final : StatefulOpConversionPattern<CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(CtrlOp op, OpAdaptor adaptor,
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
    state.controlsIn = llvm::to_vector(adaptor.getControlsIn());
    state.targetsIn = llvm::to_vector(adaptor.getTargetsIn());

    // Inline region
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator(), state.targetsIn);

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
struct ConvertQCOInvOpToJeff final : StatefulOpConversionPattern<InvOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(InvOp op, OpAdaptor adaptor,
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
      state.targetsIn = llvm::to_vector(adaptor.getQubitsIn());
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
struct ConvertQCOYieldOpToJeff final : StatefulOpConversionPattern<YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor /*adaptor*/,
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
      state.controlsOut.append(state.targetsOut);
      rewriter.replaceOp(state.ctrlOp, state.controlsOut);

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
 * @brief Type converter for QCO-to-Jeff conversion
 *
 * @details
 * Converts `!qco.qubit` to `!jeff.qubit` and `tensor<?x!qco.qubit>` to
 * `!jeff.qureg`.
 */
class QCOToJeffTypeConverter final : public TypeConverter {
public:
  explicit QCOToJeffTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](QubitType /*type*/) -> Type {
      return jeff::QubitType::get(ctx);
    });

    addConversion([ctx](RankedTensorType type) -> Type {
      if (llvm::isa<QubitType>(type.getElementType())) {
        return jeff::QuregType::get(ctx, type.getShape()[0]);
      }
      return type;
    });
  }
};

/**
 * @brief Helper for `static_assert` fallbacks in constexpr dispatch (always
 * false).
 *
 * @details The non-type template parameter pack exists only so failed branches
 * can use a dependent `false` value inside `static_assert`.
 */
template <auto...> struct AlwaysFalse : std::false_type {};

/** @brief QCO→Jeff gate lowering category. */
enum class JeffKind : std::uint8_t {
  /// Lower to a Jeff gate from the standard `WellKnownGate` set (Jeff spec:
  /// `QubitGate.gate.wellKnown`).
  WellKnown,
  Custom,       //!< Lower to jeff.custom with a name string.
  PPR,          //!< Lower to jeff.ppr with Pauli-gate encoding.
  SpecialU2ToU, //!< Lower qco.u2 via jeff.u with injected theta=pi/2.
};

/** @brief Pauli encoding for PPR lowering (1=X, 2=Y, 3=Z). */
struct PPRPaulis {
  std::int32_t p0;
  std::int32_t p1;
};

} // namespace

/**
 * @brief Registers one QCO→Jeff rewrite pattern for a gate described at compile
 * time.
 *
 * @tparam Kind How to lower: well-known Jeff op, `jeff.custom`, `jeff.ppr`, or
 *        special-case `qco.u2` → `jeff.u`.
 * @tparam Targets Number of target qubits for the QCO op.
 * @tparam Params Number of real parameters on the QCO op.
 * @tparam QCOOpType MLIR QCO operation type.
 * @tparam JeffOpType Jeff operation type for `JeffKind::WellKnown` (or `void`
 * for custom/PPR paths that do not use it).
 * @tparam JeffBaseAdjoint For well-known ops: whether the Jeff op represents
 * the adjoint of the QCO base gate (e.g. S† as `jeff.s` with adjoint set).
 * @param patterns Pattern set to add to.
 * @param typeConverter QCO→Jeff type converter passed to patterns.
 * @param context MLIR context.
 * @param state Shared lowering state pointer target (patterns store `&state`).
 * @param customName Custom gate name when `Kind` is `JeffKind::Custom` (ignored
 *        otherwise).
 * @param ppr Pauli indices when `Kind` is `JeffKind::PPR` (ignored otherwise).
 *
 * @details Dispatches at compile time to the appropriate conversion pattern.
 * Ill-formed combinations trigger `static_assert` with a message referencing
 * this function.
 */
template <JeffKind Kind, std::size_t Targets, std::size_t Params,
          typename QCOOpType, typename JeffOpType, bool JeffBaseAdjoint>
static void addQCOToJeffGatePattern(RewritePatternSet& patterns,
                                    TypeConverter& typeConverter,
                                    MLIRContext* context, LoweringState& state,
                                    StringRef customName = {},
                                    const PPRPaulis& ppr = {}) {
  if constexpr (Kind == JeffKind::WellKnown) {
    if constexpr ((Targets == 1 && Params == 0) ||
                  (Targets == 1 && Params == 1) ||
                  (Targets == 1 && Params == 3) ||
                  (Targets == 2 && Params == 0)) {
      patterns.add<ConvertQCOWellKnownGateToJeff<QCOOpType, JeffOpType, Targets,
                                                 Params, JeffBaseAdjoint>>(
          typeConverter, context, &state);
    } else {
      static_assert(AlwaysFalse<Kind, Targets, Params>::value,
                    "addQCOToJeffGatePattern: unhandled JeffKind::WellKnown "
                    "arity/params");
    }
  } else if constexpr (Kind == JeffKind::Custom) {
    patterns.add<ConvertQCOCustomGateToJeff<QCOOpType, Targets, Params>>(
        typeConverter, context, &state, customName, JeffBaseAdjoint);
  } else if constexpr (Kind == JeffKind::PPR) {
    static_assert(Targets == 2 && Params == 1,
                  "QCOToJeff PPR lowering expects exactly 2 targets and 1 "
                  "parameter");
    patterns.add<ConvertQCOPPRGateToJeff<QCOOpType>>(typeConverter, context,
                                                     &state, ppr.p0, ppr.p1);
  } else if constexpr (Kind == JeffKind::SpecialU2ToU) {
    static_assert(std::is_same_v<QCOOpType, U2Op> && Targets == 1 &&
                      Params == 2,
                  "QCOToJeff SpecialU2ToU is only implemented for qco.u2");
    patterns.add<ConvertQCOU2OpToJeff>(typeConverter, context, &state);
  } else {
    static_assert(AlwaysFalse<Kind, Targets, Params>::value,
                  "addQCOToJeffGatePattern: unhandled JeffKind");
  }
}

namespace {

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
    target.addIllegalDialect<QCODialect, qtensor::QTensorDialect,
                             arith::ArithDialect, math::MathDialect,
                             tensor::TensorDialect>();
    target.addLegalDialect<jeff::JeffDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return !op->hasAttr("passthrough"); });
    target.addLegalOp<func::ReturnOp>();

    // Register operation conversion patterns
    jeff::populateNativeToJeffConversionPatterns(patterns);
    patterns.add<ConvertQTensorAllocOp, ConvertQTensorExtractOp,
                 ConvertQTensorInsertOp, ConvertQTensorDeallocOp,
                 ConvertQCOAllocOpToJeff, ConvertQCOStaticOpToJeff,
                 ConvertQCOSinkOpToJeff, ConvertQCOMeasureOpToJeff,
                 ConvertQCOResetOpToJeff, ConvertQCOGPhaseOpToJeff>(
        typeConverter, context, &state);

    using JK = JeffKind;
    using PP = PPRPaulis;

    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, IdOp, jeff::IOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, XOp, jeff::XOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, YOp, jeff::YOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, ZOp, jeff::ZOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, HOp, jeff::HOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, SOp, jeff::SOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, SdgOp, jeff::SOp, true>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, TOp, jeff::TOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 0, TdgOp, jeff::TOp, true>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::Custom, 1, 0, SXOp, void, false>(
        patterns, typeConverter, context, state, "sx");
    addQCOToJeffGatePattern<JK::Custom, 1, 0, SXdgOp, void, true>(
        patterns, typeConverter, context, state, "sx");
    addQCOToJeffGatePattern<JK::WellKnown, 1, 1, RXOp, jeff::RxOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 1, RYOp, jeff::RyOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 1, RZOp, jeff::RzOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 1, POp, jeff::R1Op, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::Custom, 1, 2, ROp, void, false>(
        patterns, typeConverter, context, state, "r");
    addQCOToJeffGatePattern<JK::SpecialU2ToU, 1, 2, U2Op, jeff::UOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 1, 3, UOp, jeff::UOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::WellKnown, 2, 0, SWAPOp, jeff::SwapOp, false>(
        patterns, typeConverter, context, state);
    addQCOToJeffGatePattern<JK::Custom, 2, 0, iSWAPOp, void, false>(
        patterns, typeConverter, context, state, "iswap");
    addQCOToJeffGatePattern<JK::Custom, 2, 0, DCXOp, void, false>(
        patterns, typeConverter, context, state, "dcx");
    addQCOToJeffGatePattern<JK::Custom, 2, 0, ECROp, void, false>(
        patterns, typeConverter, context, state, "ecr");
    addQCOToJeffGatePattern<JK::PPR, 2, 1, RXXOp, void, false>(
        patterns, typeConverter, context, state, "_", PP{.p0 = 1, .p1 = 1});
    addQCOToJeffGatePattern<JK::PPR, 2, 1, RYYOp, void, false>(
        patterns, typeConverter, context, state, "_", PP{.p0 = 2, .p1 = 2});
    addQCOToJeffGatePattern<JK::PPR, 2, 1, RZXOp, void, false>(
        patterns, typeConverter, context, state, "_", PP{.p0 = 3, .p1 = 1});
    addQCOToJeffGatePattern<JK::PPR, 2, 1, RZZOp, void, false>(
        patterns, typeConverter, context, state, "_", PP{.p0 = 3, .p1 = 3});
    addQCOToJeffGatePattern<JK::Custom, 2, 2, XXPlusYYOp, void, false>(
        patterns, typeConverter, context, state, "xx_plus_yy");
    addQCOToJeffGatePattern<JK::Custom, 2, 2, XXMinusYYOp, void, false>(
        patterns, typeConverter, context, state, "xx_minus_yy");

    patterns.add<ConvertQCOBarrierOpToJeff, ConvertQCOCtrlOpToJeff,
                 ConvertQCOInvOpToJeff, ConvertQCOYieldOpToJeff,
                 ConvertQCOMainToJeff>(typeConverter, context, &state);

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

} // namespace

} // namespace mlir
