/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQIR/QIRCommon/QIRCommon.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/OpenQASMAttributes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/Utils/AngleConversion.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace mlir {
using namespace qc;
using namespace qir;

LogicalResult LoweringState::ensureAllocationMode(AllocationMode requested,
                                                  Operation* op) {
  if (allocationMode == AllocationMode::Unset) {
    allocationMode = requested;
    return success();
  }
  if (allocationMode == requested) {
    return success();
  }
  return op->emitOpError(
      "cannot mix static and dynamic qubit allocation modes in conversion");
}

QCToQIRTypeConverter::QCToQIRTypeConverter(MLIRContext* ctx)
    : LLVMTypeConverter(ctx) {
  addConversion([ctx](QubitType) { return LLVM::LLVMPointerType::get(ctx); });
  addConversion([ctx](MemRefType type) -> Type {
    if (isa<QubitType>(type.getElementType()) || isClassicalBitRegister(type)) {
      return LLVM::LLVMPointerType::get(ctx);
    }
    return type;
  });
};

/**
 * @brief Helper to convert a QC operation to a LLVM CallOp
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOpAdaptorType The OpAdaptor type of the QC operation
 * @param op The QC operation instance to convert
 * @param adaptor The OpAdaptor of the QC operation
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @param numTargets The number of targets
 * @param numParams The number of parameters
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOpAdaptorType>
static LogicalResult
convertUnitaryToCallOp(QCOpType& op, QCOpAdaptorType& adaptor,
                       ConversionPatternRewriter& rewriter,
                       LoweringState& state, StringRef fnName,
                       const size_t numTargets, const size_t numParams) {
  // Query state for modifier information
  const SmallVector<Value> controls =
      state.inCtrlOp ? state.controls : SmallVector<Value>{};
  const auto convertedOperands = adaptor.getOperands();
  const auto targets = convertedOperands.take_front(numTargets);
  const auto parameters = convertedOperands.drop_front(numTargets);
  assert(parameters.size() == numParams && "unexpected gate parameter count");

  // Clean up modifier information
  if (state.inCtrlOp) {
    state.inCtrlOp = false;
    state.controls.clear();
  }

  qir::emitQISCall(rewriter, op, op.getLoc(), parameters, controls, targets,
                   fnName);
  rewriter.eraseOp(op);
  return success();
}

namespace {

/**
 * @brief Generic converter for unitary QC ops to QIR calls.
 *
 * @details
 * Many QC gates lower to a QIR runtime call where the callee name depends on
 * the number of active controls. This helper factors out that boilerplate
 * without relying on preprocessor macros.
 *
 * @par Examples
 * The examples below illustrate the lowering shapes for unitary gates that
 * are registered through the shared QIR gate table in
 * `populateQCToQIRPatterns`.
 *
 * @par One target, zero parameters
 * ```mlir
 * qc.x %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr) -> ()
 * ```
 *
 * @par One target, one parameter
 * ```mlir
 * qc.rx(%theta) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__rx__body(%theta, %q) : (f64, !llvm.ptr) -> ()
 * ```
 *
 * @par One target, two parameters
 * ```mlir
 * qc.r(%theta, %phi) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__prx__body(%theta, %phi, %q)
 *     : (f64, f64, !llvm.ptr) -> ()
 * ```
 *
 * @par One target, three parameters
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__u3__body(%theta, %phi, %lambda, %q)
 *     : (f64, f64, f64, !llvm.ptr) -> ()
 * ```
 *
 * @par Two targets, zero parameters
 * ```mlir
 * qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) ->
 * ()
 * ```
 *
 * @par Two targets, one parameter
 * ```mlir
 * qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__rxx__body(%theta, %q0, %q1)
 *     : (f64, !llvm.ptr, !llvm.ptr) -> ()
 * ```
 *
 * @par Two targets, two parameters
 * ```mlir
 * qc.xx_plus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__xx_plus_yy__body(%theta, %beta, %q0, %q1)
 *     : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()
 * ```
 *
 * @tparam OpType The QC operation type to convert
 * @tparam NumTargets Number of target qubits for this operation
 * @tparam NumParams Number of floating-point parameters for this operation
 * @tparam GetFnName Function that maps numCtrls -> QIR function name
 */
template <typename OpType, std::size_t NumTargets, std::size_t NumParams,
          auto GetFnName>
struct ConvertQCUnitaryOpQIR : StatefulOpConversionPattern<OpType> {
  using StatefulOpConversionPattern<OpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, OpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    const size_t numCtrls = state.inCtrlOp ? state.controls.size() : 0;
    const auto fnName = GetFnName(numCtrls);
    return convertUnitaryToCallOp(op, adaptor, rewriter, state, fnName,
                                  NumTargets, NumParams);
  }
};

/**
 * @brief Converts qc.static to llvm.inttoptr
 *
 * @details
 * Converts a static qubit reference to an LLVM pointer by creating a constant
 * with the qubit index and converting it to a pointer. The pointer is cached
 * in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q0 = qc.static 0 : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQCStaticOp final : StatefulOpConversionPattern<StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto index = static_cast<int64_t>(op.getIndex());
    auto& state = getState();
    if (failed(state.ensureAllocationMode(AllocationMode::Static,
                                          op.getOperation()))) {
      return failure();
    }

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Switch to entry block
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    // Get or create a pointer to the qubit
    Value qubit;
    if (const auto it = state.staticQubits.find(index);
        it != state.staticQubits.end()) {
      // Reuse existing pointer
      qubit = it->second;
    } else {
      // Create and cache for reuse
      qubit = createPointerFromIndex(rewriter, op.getLoc(), index);
      state.staticQubits.try_emplace(index, qubit);
    }
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

// GPhaseOp

/**
 * @brief Converts qc.gphase to QIR gphase
 *
 * @par Example:
 * ```mlir
 * qc.gphase(%theta)
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__gphase__body(%theta) : (f64) -> ()
 * ```
 */
struct ConvertQCGPhaseOp final : StatefulOpConversionPattern<GPhaseOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(GPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (state.inCtrlOp) {
      return op.emitError("Controlled GPhaseOps cannot be converted to QIR");
    }
    return convertUnitaryToCallOp(op, adaptor, rewriter, state, QIR_GPHASE, 0,
                                  1);
  }
};

// BarrierOp

/**
 * @brief Erases qc.barrier operation, as it is a no-op in QIR
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Inlines qc.ctrl region removes the operation
 */
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    if (state.inCtrlOp) {
      return rewriter.notifyMatchFailure(op,
                                         "Nested CtrlOps are not supported");
    }

    if (op.getNumBodyUnitaries() > 1) {
      return rewriter.notifyMatchFailure(
          op, "CtrlOps with multiple body unitaries are not supported. Run the "
              "unroll-modifiers pass before the conversion");
    }

    // Empty control bodies and controls around no-op unitaries do not need
    // lowering state. In particular, barrier lowering erases the operation
    // without consuming that state, which would otherwise control the next
    // gate.
    auto bodyUnitary = op.getNumBodyUnitaries() == 1 ? op.getBodyUnitary(0)
                                                     : UnitaryOpInterface{};
    if (bodyUnitary && !isa<BarrierOp, IdOp>(bodyUnitary.getOperation())) {
      state.inCtrlOp = true;
      state.controls = llvm::to_vector(adaptor.getControls());
    }

    // Inline block and remove operation
    rewriter.inlineBlockBefore(&op.getRegion().front(), op,
                               adaptor.getTargets());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erases qc.yield operation
 */
struct ConvertQCYieldOp final : StatefulOpConversionPattern<YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                   LoweringState& state) {
  OpBuilder builder(ctx);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  builder.setInsertionPointToStart(state.entryBlock);

  auto initSig = LLVM::LLVMFunctionType::get(voidType, ptrType);
  auto initDec =
      getOrCreateFunctionDeclaration(builder, main, QIR_INITIALIZE, initSig);
  auto zero = LLVM::ZeroOp::create(builder, main->getLoc(), ptrType);
  LLVM::CallOp::create(builder, main->getLoc(), initDec, zero.getResult());
}

void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                        LoweringState& state) {
  OpBuilder builder(ctx);
  builder.setInsertionPoint(&main.getBlocks().back().back());
  SmallVector<qir::ClassicalRegister> returnedRegisters;
  returnedRegisters.reserve(state.returnedCregs.size());
  for (const auto registerIndex : state.returnedCregs) {
    returnedRegisters.push_back(state.cregs[registerIndex]);
  }
  emitOutputRecording(builder, main, returnedRegisters, state.staticResults);
}

void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  patterns.add<ConvertQCUnitaryOpQIR<qc::KEY##Op, (TARGETS), (PARAMS),         \
                                     &getFnName##GETTER>>(typeConverter, ctx,  \
                                                          &state);
#include "mlir/Conversion/GateTable.def"

  patterns.add<ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCYieldOp,
               ConvertQCStaticOp, ConvertQCGPhaseOp>(typeConverter, ctx,
                                                     &state);
}

Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(state.entryBlock->getTerminator());
  const auto index = static_cast<int64_t>(state.staticResults.size());
  const auto record = state.returnedStaticResults.contains(op);
  auto result = createPointerFromIndex(rewriter, op->getLoc(), index);
  state.staticResults.try_emplace(
      index, qir::StaticResult{.pointer = result, .record = record});
  return result;
}

static bool isLeadingZeroInitialization(memref::StoreOp storeOp,
                                        memref::AllocOp allocOp) {
  if (storeOp->getBlock() != allocOp->getBlock() ||
      !allocOp->isBeforeInBlock(storeOp) ||
      !matchPattern(storeOp.getValueToStore(), m_Zero())) {
    return false;
  }

  const auto type = allocOp.getType();
  const auto index = getConstantIntValue(storeOp.getIndices().front());
  if (!index || type.isDynamicDim(0) || *index < 0 ||
      *index >= type.getDimSize(0)) {
    return false;
  }

  for (const auto& use : allocOp.getMemref().getUses()) {
    auto* user = use.getOwner();
    auto* ancestor = storeOp->getBlock()->findAncestorOpInBlock(*user);
    if (ancestor == nullptr || ancestor == storeOp ||
        !ancestor->isBeforeInBlock(storeOp)) {
      continue;
    }

    auto previousStore = dyn_cast<memref::StoreOp>(user);
    if (ancestor != user || !previousStore ||
        !matchPattern(previousStore.getValueToStore(), m_Zero())) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] static bool isEntryPoint(const func::FuncOp funcOp) {
  const auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough");
  return passthrough && llvm::any_of(passthrough, [](const Attribute attr) {
           const auto strAttr = dyn_cast<StringAttr>(attr);
           return strAttr && strAttr.getValue() == "entry_point";
         });
}

[[nodiscard]] static bool
isUnsignedDivisionSafetyAssert(cf::AssertOp assertion) {
  const auto message = assertion.getMsg();
  if (message != "division by zero" && message != "modulo by zero") {
    return false;
  }
  Value divisor;
  if (auto comparison = assertion.getArg().getDefiningOp<arith::CmpIOp>()) {
    if (comparison.getPredicate() != arith::CmpIPredicate::ne) {
      return false;
    }
    if (matchPattern(comparison.getRhs(), m_Zero())) {
      divisor = comparison.getLhs();
    } else if (matchPattern(comparison.getLhs(), m_Zero())) {
      divisor = comparison.getRhs();
    } else {
      return false;
    }
  } else if (assertion.getArg().getType().isInteger(1)) {
    divisor = assertion.getArg();
  } else {
    return false;
  }
  return llvm::any_of(divisor.getUsers(), [&](Operation* user) {
    if (message == "division by zero") {
      auto division = dyn_cast<arith::DivUIOp>(user);
      return division && division.getRhs() == divisor;
    }
    auto remainder = dyn_cast<arith::RemUIOp>(user);
    return remainder && remainder.getRhs() == divisor;
  });
}

[[nodiscard]] static LogicalResult
validateQIRSourceInterface(Operation* moduleOp,
                           const QIRTargetProfile profile) {
  bool invalid = false;
  SmallVector<cf::AssertOp> redundantDivisionAssertions;
  moduleOp->walk([&](func::FuncOp funcOp) {
    if (!isEntryPoint(funcOp)) {
      return;
    }
    if (!funcOp.getArguments().empty()) {
      SmallVector<std::string> names;
      names.reserve(funcOp.getNumArguments());
      for (const auto [index, argument] :
           llvm::enumerate(funcOp.getArguments())) {
        const auto metadata = funcOp.getArgAttrOfType<DictionaryAttr>(
            index, "mqt.openqasm.scalar");
        const auto name =
            metadata ? metadata.getAs<StringAttr>("name") : StringAttr{};
        names.push_back(name ? name.str()
                             : "argument #" + std::to_string(index));
      }
      funcOp.emitError()
          << "QIR 2.1 entry points may not take parameters; specialize source "
             "inputs before QIR conversion (inputs: "
          << llvm::join(names, ", ") << ")";
      invalid = true;
    }
    const auto dynamicAngle = funcOp.walk([&](Operation* operation) {
      for (const auto result : operation->getResults()) {
        if (const auto quantized = mqt::angle::matchQuantizedRadians(result);
            quantized && !matchPattern(quantized->bits, m_Constant())) {
          return WalkResult::interrupt();
        }
        if (const auto source = mqt::angle::matchFloatToBits(result);
            source && !matchPattern(*source, m_Constant())) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (dynamicAngle.wasInterrupted()) {
      funcOp.emitError()
          << "QIR 2.1 profiles do not permit the runtime instructions "
             "required by dynamic OpenQASM angle conversion; specialize "
             "angle values before QIR conversion";
      invalid = true;
    }
    const auto dynamicAngleArithmetic = funcOp.walk([&](Operation* operation) {
      if (!operation->hasAttrOfType<UnitAttr>(openqasm::ANGLE_VALUE_ATTR) ||
          operation->getNumResults() != 1 ||
          matchPattern(operation->getResult(0), m_Constant())) {
        return WalkResult::advance();
      }
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation)) {
        return WalkResult::interrupt();
      }
      if (const auto resize = mqt::angle::matchResize(operation->getResult(0));
          resize && resize->targetWidth < resize->sourceWidth &&
          !matchPattern(resize->source, m_Constant())) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (dynamicAngleArithmetic.wasInterrupted()) {
      funcOp.emitError()
          << "QIR 2.1 profiles do not define the wraparound or lossy "
             "truncation behavior required by dynamic OpenQASM angle "
             "arithmetic; specialize angle values before QIR conversion";
      invalid = true;
    }
    if (profile == QIRTargetProfile::Base) {
      const auto dynamicAngleComputation =
          funcOp.walk([&](Operation* operation) {
            if (!operation->hasAttr(openqasm::ANGLE_VALUE_ATTR) &&
                !operation->hasAttr(openqasm::ANGLE_OPERANDS_ATTR)) {
              return WalkResult::advance();
            }
            const auto isDynamic = [](const Value value) {
              return !matchPattern(value, m_Constant());
            };
            if (llvm::any_of(operation->getOperands(), isDynamic) ||
                llvm::any_of(operation->getResults(), isDynamic)) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
      if (dynamicAngleComputation.wasInterrupted()) {
        funcOp.emitError()
            << "the QIR 2.1 Base Profile does not permit dynamic classical "
               "angle computation; use the Adaptive Profile or specialize "
               "angle values before QIR conversion";
        invalid = true;
      }
    }
    funcOp.walk([&](Operation* operation) {
      if (!isa<math::CtPopOp, LLVM::FshlOp, LLVM::FshrOp>(operation)) {
        return;
      }
      operation->emitError()
          << "QIR 2.1 profiles do not permit population-count or "
             "funnel-shift intrinsics; specialize popcount and rotation "
             "operands before QIR conversion";
      invalid = true;
    });
    funcOp.walk([&](cf::AssertOp assertion) {
      if (isUnsignedDivisionSafetyAssert(assertion)) {
        redundantDivisionAssertions.push_back(assertion);
        return;
      }
      assertion.emitError()
          << "QIR 2.1 profiles do not permit external runtime assertion "
             "machinery; specialize the checked value before QIR conversion";
      invalid = true;
    });
    funcOp.walk([&](func::ReturnOp returnOp) {
      SmallVector<std::string> unsupportedOutputs;
      for (const auto [index, operand] :
           llvm::enumerate(returnOp.getOperands())) {
        if (operand.getDefiningOp<MeasureOp>()) {
          continue;
        }
        if (auto allocOp = operand.getDefiningOp<memref::AllocOp>();
            allocOp && isClassicalBitRegister(allocOp.getType())) {
          continue;
        }
        const auto scalarMetadata =
            index < funcOp.getNumResults()
                ? funcOp.getResultAttr(index, "mqt.openqasm.scalar")
                : Attribute{};
        const auto isStatus = returnOp.getNumOperands() == 1 &&
                              operand.getType().isInteger(64) &&
                              !scalarMetadata;
        if (isStatus) {
          continue;
        }
        const auto metadata = dyn_cast_or_null<DictionaryAttr>(scalarMetadata);
        const auto name =
            metadata ? metadata.getAs<StringAttr>("name") : StringAttr{};
        unsupportedOutputs.push_back(name ? name.str()
                                          : "result #" + std::to_string(index));
      }
      if (!unsupportedOutputs.empty()) {
        returnOp.emitError()
            << "QIR profile entry points return only an i64 status; source "
               "scalar outputs require output-record lowering and are not "
               "supported (outputs: "
            << llvm::join(unsupportedOutputs, ", ") << ")";
        invalid = true;
      }
    });
  });
  for (auto assertion : redundantDivisionAssertions) {
    assertion.erase();
  }
  return failure(invalid);
}

LogicalResult prepareClassicalResults(Operation* moduleOp, LoweringState& state,
                                      const QIRTargetProfile profile) {
  if (failed(validateQIRSourceInterface(moduleOp, profile))) {
    return failure();
  }
  bool hasInvalidMemory = false;
  SmallVector<memref::StoreOp> consumedStores;
  moduleOp->walk([&](func::FuncOp funcOp) {
    // Check whether the given function is the main entrypoint
    if (!isEntryPoint(funcOp)) {
      return;
    }

    funcOp.walk([&](memref::AllocOp allocOp) {
      auto type = allocOp.getType();
      if (!isClassicalBitRegister(type)) {
        if (type.getRank() != 1 || !isa<QubitType>(type.getElementType())) {
          allocOp.emitError(
              "QIR conversion only supports one-dimensional memrefs of i1 "
              "classical results or qc.qubit registers");
          hasInvalidMemory = true;
        }
        return;
      }

      const auto [it, inserted] = state.cregIndices.try_emplace(
          allocOp.getOperation(), state.cregs.size());
      if (inserted) {
        state.cregs.emplace_back();
      }
      auto& reg = state.cregs[it->second];
      reg.record = false;
      if (const auto name = allocOp->getAttrOfType<StringAttr>(
              utils::CLASSICAL_REGISTER_NAME_ATTR)) {
        reg.label = name.str();
      }
      const auto size = type.getShape()[0];
      if (size != ShapedType::kDynamic) {
        reg.size = size;
        reg.results.assign(size, Value{});
      }
    });

    funcOp.walk([&](memref::StoreOp storeOp) {
      const auto type = dyn_cast<MemRefType>(storeOp.getMemref().getType());
      if (!type || !isClassicalBitRegister(type)) {
        return;
      }
      auto allocOp = storeOp.getMemref().getDefiningOp<memref::AllocOp>();
      auto measureOp = storeOp.getValueToStore().getDefiningOp<MeasureOp>();
      if (!allocOp || !state.cregIndices.contains(allocOp.getOperation())) {
        storeOp.emitError(
            "QIR conversion only supports storing direct measurement results "
            "in classical result registers");
        hasInvalidMemory = true;
        return;
      }
      if (!measureOp) {
        if (isLeadingZeroInitialization(storeOp, allocOp)) {
          consumedStores.push_back(storeOp);
          return;
        }
        storeOp.emitError(
            "QIR conversion only supports storing direct measurement results "
            "or leading zero initialization in classical result registers");
        hasInvalidMemory = true;
        return;
      }
      const auto destination =
          std::pair<size_t, Value>{state.cregIndices.at(allocOp.getOperation()),
                                   storeOp.getIndices()[0]};
      const auto [it, inserted] = state.cregMeasurements.try_emplace(
          measureOp.getOperation(), destination);
      if (!inserted && it->second != destination) {
        storeOp.emitError(
            "a measurement result cannot be stored in multiple classical "
            "register locations during QIR conversion");
        hasInvalidMemory = true;
      }
      consumedStores.push_back(storeOp);
    });

    const auto markRegisterForRecording = [&](const size_t registerIndex) {
      auto& reg = state.cregs[registerIndex];
      if (reg.record) {
        return;
      }
      if (reg.label.empty()) {
        reg.label = "c" + std::to_string(state.returnedCregs.size());
      }
      reg.record = true;
      state.returnedCregs.push_back(registerIndex);
    };

    funcOp.walk([&](func::ReturnOp returnOp) {
      SmallVector<Value> keptOperands;
      SmallVector<Type> keptReturnTypes;

      for (const auto operand : returnOp.getOperands()) {
        if (auto measureOp = operand.getDefiningOp<MeasureOp>()) {
          if (const auto it =
                  state.cregMeasurements.find(measureOp.getOperation());
              it != state.cregMeasurements.end()) {
            markRegisterForRecording(it->second.first);
          } else {
            state.returnedStaticResults.insert(measureOp.getOperation());
          }
        } else if (auto allocOp = operand.getDefiningOp<memref::AllocOp>();
                   allocOp &&
                   state.cregIndices.contains(allocOp.getOperation())) {
          markRegisterForRecording(
              state.cregIndices.at(allocOp.getOperation()));
        } else {
          keptOperands.push_back(operand);
          keptReturnTypes.push_back(operand.getType());
        }
      }

      if (keptOperands.empty()) {
        OpBuilder builder(returnOp);
        auto zero =
            arith::ConstantIntOp::create(builder, returnOp.getLoc(), 0, 64);
        keptOperands.push_back(zero);
        keptReturnTypes.push_back(zero.getType());
      }

      returnOp.getOperandsMutable().assign(keptOperands);

      funcOp.setFunctionType(FunctionType::get(
          funcOp.getContext(), funcOp.getFunctionType().getInputs(),
          keptReturnTypes));
    });
  });
  if (hasInvalidMemory) {
    return failure();
  }
  for (auto storeOp : consumedStores) {
    storeOp.erase();
  }
  return success();
}

} // namespace mlir
