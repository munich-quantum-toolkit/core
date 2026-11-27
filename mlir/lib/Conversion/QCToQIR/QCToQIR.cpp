/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCToQIR/QCToQIR.h"

#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Conversion/QCToQIR/QIRCommon.h"
#include "mqt/Dialect/CBit/IR/CBitAttributes.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QIR/QIRDefinitions.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace mlir {

#define GEN_PASS_DEF_QCTOQIR
#include "mqt/Conversion/QCToQIR/QCToQIR.h.inc"

//===----------------------------------------------------------------------===//
// Base Profile
//===----------------------------------------------------------------------===//

namespace base {

using namespace qc;
using namespace qir;

/// Return the result pointer that @p op writes to through its fused stores and
/// erase those stores, or return a null value without fused stores.
static FailureOr<Value>
resolveRegisterMeasurement(LoweringState& state, MeasureOp op,
                           ConversionPatternRewriter& rewriter) {
  const auto stores = getFusedStores(op);
  if (stores.empty()) {
    return Value{};
  }
  auto store = stores.front();
  const auto indexValue = getConstantIntValue(store.getIndex());
  if (!indexValue) {
    op->emitError("QIR Base Profile requires constant classical-register "
                  "measurement indices");
    return failure();
  }
  const auto& results =
      state.cregs[state.cregIndices.at(store.getReg().getDefiningOp())].results;
  if (*indexValue < 0 || static_cast<size_t>(*indexValue) >= results.size()) {
    op->emitError("classical-register measurement index is out of bounds");
    return failure();
  }
  for (auto fused : stores) {
    rewriter.eraseOp(fused);
  }
  return results[static_cast<size_t>(*indexValue)];
}

namespace {

/// Converts `cbit.alloc` to static result
/// pointers represented by `llvm.inttoptr` operations
///
/// Allocate static result pointers for each bit in a classical register.
struct ConvertCBitAllocOp final : StatefulOpConversionPattern<cbit::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto it = state.cregIndices.find(op.getOperation());
    if (it == state.cregIndices.end()) {
      return op.emitError(
          "QIR conversion requires CBit allocations in the entry function");
    }
    auto& reg = state.cregs[it->second];
    const auto* size = std::get_if<int64_t>(&reg.size);
    if (size == nullptr) {
      op.emitError(
          "QIR Base Profile requires statically sized classical registers");
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());
    reg.results.reserve(static_cast<size_t>(*size));
    for (int64_t i = 0; i < *size; ++i) {
      const auto index = static_cast<int64_t>(state.numScalarResults++);
      auto result = createPointerFromIndex(rewriter, op.getLoc(), index);
      reg.results.push_back(result);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct RejectCBitLoadOp final : OpConversionPattern<cbit::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::LoadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& /*rewriter*/) const override {
    return op.emitError(
        "QIR Base Profile does not support classical-register loads");
  }
};

struct RejectCBitReadOp final : OpConversionPattern<cbit::ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::ReadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& /*rewriter*/) const override {
    return op.emitError(
        "QIR Base Profile does not support classical-register reads");
  }
};

struct RejectCBitWriteOp final : OpConversionPattern<cbit::WriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::WriteOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& /*rewriter*/) const override {
    return op.emitError(
        "QIR Base Profile does not support classical-register writes");
  }
};

struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts a qubit-register `memref.load` to `llvm.inttoptr`
///
/// Converts a load operation to an LLVM pointer by creating a constant with the
/// next available static qubit index and converting it to a pointer. The
/// pointer is cached in the lowering state for reuse.
///
/// @par Example:
/// ```mlir
/// %q0 = memref.load %memref[%c0] : memref<3x!qc.qubit>
/// ```
/// is converted to
/// ```mlir
/// %c0 = llvm.mlir.constant(0 : i64) : i64
/// %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
/// ```
struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto shape = op.getMemref().getType().getShape();
    if (shape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }
    const auto index = getConstantIntValue(op.getIndices().front());
    if (!index || ShapedType::isDynamic(shape.front()) ||
        !op.getMemref().getDefiningOp<memref::AllocOp>()) {
      return op.emitError("QIR Base Profile requires constant indices into "
                          "statically allocated qubit registers");
    }
    if (*index < 0 || *index >= shape.front()) {
      return op.emitError("qubit-register index is out of bounds");
    }
    auto& qubit = state.staticRegisterQubits[{op.getMemref(), *index}];
    if (!qubit) {
      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(state.entryBlock->getTerminator());
      const auto id = static_cast<int64_t>(state.staticQubits.size());
      qubit = createPointerFromIndex(rewriter, op.getLoc(), id);
      state.staticQubits.try_emplace(id, qubit);
    }
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

/// Erases memref.dealloc during the QIR Base Profile conversion
struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts qc.alloc to llvm.inttoptr
///
/// Converts a qubit allocation to an LLVM pointer by creating a constant
/// with the next available static qubit index and converting it to a pointer.
/// The pointer is cached in the lowering state for reuse.
///
/// @par Example:
/// ```mlir
/// %q = qc.alloc : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// %c0 = llvm.mlir.constant(0 : i64) : i64
/// %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
/// ```
struct ConvertQCAllocOp final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    const auto nqubits = state.staticQubits.size();
    auto qubit = createPointerFromIndex(rewriter, op.getLoc(),
                                        static_cast<int64_t>(nqubits));
    state.staticQubits.try_emplace(static_cast<int64_t>(nqubits), qubit);
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

/// Erases qc.dealloc during the QIR Base Profile conversion
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts qc.measure to QIR measurement
///
/// For measurements with register information, a static result is used at
/// the given index + register offset. Otherwise a static result at
/// the next index is used.
///
/// @par Example (without register):
/// ```mlir
/// %result = qc.measure %q : !qc.qubit -> i1
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__mz__body(%q, %b) : (!llvm.ptr, !llvm.ptr) -> ()
/// ```
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    OpBuilder::InsertionGuard guard(rewriter);

    auto registerResult = resolveRegisterMeasurement(state, op, rewriter);
    if (failed(registerResult)) {
      return failure();
    }
    auto result = *registerResult;
    if (!result) {
      result = getResultPtr(state, op.getOperation(), rewriter);
    }

    if (auto it = state.measuredResults.find(op);
        it != state.measuredResults.end()) {
      it->second.value = result;
    }

    // Preserve instruction order until terminal measurements are verified.
    rewriter.setInsertionPoint(op);
    auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSig);
    auto call = LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                                     ValueRange{adaptor.getQubit(), result});
    if (failed(state.nameNativeCall(call, "measure", adaptor.getQubit()))) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

/// Populates the Base Profile conversion patterns.
static void populatePatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
  patterns.add<ConvertCBitAllocOp, ConvertMemRefAllocOp, ConvertMemRefLoadOp,
               ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCMeasureOp,
               ConvertQCDeallocOp>(typeConverter, ctx, &state);
  patterns.add<RejectCBitLoadOp, RejectCBitReadOp, RejectCBitWriteOp>(
      typeConverter, ctx);
}

/// Arrange the entry point into initialization, gates, measurements, and
/// output blocks, connected in that order by unconditional branches.
static void ensureBlocks(LLVM::LLVMFuncOp main, LoweringState& state) {
  // Get the existing block
  auto* bodyBlock = &main.front();
  OpBuilder builder(main.getBody());

  // Create the required blocks
  auto* entryBlock = builder.createBlock(&main.getBody());
  // Move the entry block before the body block
  main.getBlocks().splice(Region::iterator(bodyBlock), main.getBlocks(),
                          entryBlock);
  Block* measurementsBlock = builder.createBlock(&main.getBody());
  Block* outputBlock = builder.createBlock(&main.getBody());

  state.entryBlock = entryBlock;
  state.measurementsBlock = measurementsBlock;
  state.outputBlock = outputBlock;

  auto& bodyBlockOps = bodyBlock->getOperations();
  auto& outputBlockOps = outputBlock->getOperations();

  // Move operations to appropriate blocks
  for (auto it = bodyBlock->begin(); it != bodyBlock->end();) {
    // Ensure iterator remains valid after potential move
    if (auto& op = *it++; isa<LLVM::ReturnOp>(op)) {
      // Move return to output block
      outputBlockOps.splice(outputBlock->end(), bodyBlockOps,
                            Block::iterator(op));
    } else if (op.hasTrait<OpTrait::ConstantLike>()) {
      // Move allocations and constant-like operations to entry block
      entryBlock->getOperations().splice(entryBlock->end(), bodyBlockOps,
                                         Block::iterator(op));
    }
    // All other operations (gates, etc.) stay in body block
  }

  // Add unconditional jumps between blocks
  builder.setInsertionPointToEnd(entryBlock);
  LLVM::BrOp::create(builder, main->getLoc(), bodyBlock);

  builder.setInsertionPointToEnd(bodyBlock);
  LLVM::BrOp::create(builder, main->getLoc(), measurementsBlock);

  builder.setInsertionPointToEnd(measurementsBlock);
  LLVM::BrOp::create(builder, main->getLoc(), outputBlock);
}

} // namespace base

//===----------------------------------------------------------------------===//
// Adaptive Profile
//===----------------------------------------------------------------------===//

namespace adaptive {

using namespace qc;
using namespace qir;

namespace {

constexpr uint32_t LOCAL_CBIT_REGISTER = 1U;
constexpr uint32_t RETURNED_CBIT_REGISTER = 2U;
constexpr uint32_t MIXED_CBIT_REGISTER =
    LOCAL_CBIT_REGISTER | RETURNED_CBIT_REGISTER;

} // namespace

static LogicalResult prepareCBitRegisterAccesses(Operation* moduleOp,
                                                 LoweringState& state) {
  // Output recording and fused stores need no register-access analysis.
  if (llvm::none_of(state.cregIndices, [&](const auto& entry) {
        const auto& reg = state.cregs[entry.second];
        return reg.record && !reg.booleanStorage &&
               llvm::any_of(entry.first->getUsers(), [&](Operation* user) {
                 return !user->hasAttr(QIR_FUSED_STORE_ATTR);
               });
      })) {
    return success();
  }
  DenseMap<Value, SmallVector<Value>> forwardedRegisters;
  moduleOp->walk([&](Operation* operation) {
    for (auto& region : operation->getRegions()) {
      for (auto& block : region) {
        for (auto argument : block.getArguments()) {
          if (!isa<cbit::RegisterType>(argument.getType())) {
            continue;
          }
          for (auto& edge : block.getUses()) {
            auto branch = dyn_cast<BranchOpInterface>(edge.getOwner());
            if (!branch) {
              continue;
            }
            auto operands =
                branch.getSuccessorOperands(edge.getOperandNumber());
            if (argument.getArgNumber() >= operands.size() ||
                operands.isOperandProduced(argument.getArgNumber())) {
              continue;
            }
            if (auto incoming = operands[argument.getArgNumber()]) {
              forwardedRegisters[incoming].push_back(argument);
            }
          }
        }
      }
    }
  });
  moduleOp->walk([&](arith::SelectOp selectOp) {
    if (isa<cbit::RegisterType>(selectOp.getType())) {
      forwardedRegisters[selectOp.getTrueValue()].push_back(
          selectOp.getResult());
      forwardedRegisters[selectOp.getFalseValue()].push_back(
          selectOp.getResult());
    }
  });

  DenseMap<Value, uint32_t> representations;
  SmallVector<Value> worklist;
  for (const auto& [allocation, index] : state.cregIndices) {
    auto value = allocation->getResult(0);
    const auto& reg = state.cregs[index];
    representations[value] = reg.record && !reg.booleanStorage
                                 ? RETURNED_CBIT_REGISTER
                                 : LOCAL_CBIT_REGISTER;
    worklist.push_back(value);
  }

  while (!worklist.empty()) {
    auto source = worklist.pop_back_val();
    const auto it = forwardedRegisters.find(source);
    if (it == forwardedRegisters.end()) {
      continue;
    }
    for (auto destination : it->second) {
      auto& representation = representations[destination];
      const auto merged = representation | representations.lookup(source);
      if (merged != representation) {
        representation = merged;
        worklist.push_back(destination);
      }
    }
  }

  bool hasInvalidAccess = false;
  const auto prepareRead = [&](Operation* operation, Value reg) {
    const auto representation = representations.lookup(reg);
    if (representation == MIXED_CBIT_REGISTER) {
      operation->emitOpError(
          "adaptive QIR conversion cannot merge returned and local CBit "
          "registers");
      hasInvalidAccess = true;
    } else if (representation == RETURNED_CBIT_REGISTER) {
      state.returnedCBitReads.insert(operation);
    }
  };
  moduleOp->walk(
      [&](cbit::LoadOp loadOp) { prepareRead(loadOp, loadOp.getReg()); });
  moduleOp->walk([&](cbit::StoreOp storeOp) {
    if (storeOp->hasAttr(QIR_FUSED_STORE_ATTR)) {
      return;
    }
    const auto representation = representations.lookup(storeOp.getReg());
    if (representation == MIXED_CBIT_REGISTER) {
      storeOp.emitOpError(
          "adaptive QIR conversion cannot merge returned and local CBit "
          "registers");
      hasInvalidAccess = true;
    } else if (representation == RETURNED_CBIT_REGISTER) {
      storeOp.emitOpError(
          "adaptive QIR conversion does not support non-measurement writes "
          "to returned CBit registers");
      hasInvalidAccess = true;
    }
  });
  return success(!hasInvalidAccess);
}

/// Return the result pointer that @p op writes to through its fused stores and
/// erase those stores, or return `nullptr` without fused stores.
static Value resolveRegisterMeasurement(LoweringState& state, MeasureOp op,
                                        ConversionPatternRewriter& rewriter) {
  const auto stores = getFusedStores(op);
  if (stores.empty()) {
    return nullptr;
  }
  auto store = stores.front();
  auto& reg = state.cregs[state.cregIndices.at(store.getReg().getDefiningOp())];
  assert(reg.array && "result array must be allocated");
  auto loc = op->getLoc();
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto index = rewriter.getRemappedValue(store.getIndex());
  auto elementptr = LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType,
                                        reg.array, ValueRange{index})
                        .getResult();
  for (auto fused : stores) {
    rewriter.eraseOp(fused);
  }
  return LLVM::LoadOp::create(rewriter, loc, ptrType, elementptr).getResult();
}

/// Converts qubit-register `memref.alloc` to `llvm.alloca`
static LogicalResult
convertQubitMemRefAllocOp(memref::AllocOp op, memref::AllocOp::Adaptor adaptor,
                          LoweringState& state,
                          ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto* ctx = op.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  auto fnSig = LLVM::LLVMFunctionType::get(
      voidType, {rewriter.getI64Type(), ptrType, ptrType});
  auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                              QIR_QUBIT_ARRAY_ALLOC, fnSig);

  Value size;
  if (op.getType().getShape()[0] == ShapedType::kDynamic) {
    size = adaptor.getDynamicSizes()[0];
  } else {
    size = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    op.getType().getShape()[0])
               .getResult();
  }
  auto array =
      LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, size).getResult();
  // Releases see the converted array, also through reference casts.
  state.qregSizes.try_emplace(array, size);
  auto zero = LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();
  LLVM::CallOp::create(rewriter, loc, fnDec, ValueRange{size, array, zero});

  rewriter.replaceOp(op, array);
  return success();
}

namespace {

struct ConvertCBitAllocOp final : StatefulOpConversionPattern<cbit::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto it = state.cregIndices.find(op.getOperation());
    if (it == state.cregIndices.end()) {
      return op.emitError(
          "QIR conversion requires CBit allocations in the entry function");
    }
    auto& reg = state.cregs[it->second];

    auto loc = op.getLoc();
    auto* ctx = op.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    auto size = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                         op.getResult().getType().getWidth())
                    .getResult();

    if (!reg.record || reg.booleanStorage) {
      auto i1Type = rewriter.getI1Type();
      auto storage =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, i1Type, size)
              .getResult();
      reg.array = storage;
      if (op.getInitialization() == cbit::Initialization::Zero) {
        rewriter.setInsertionPoint(op);
        auto zero = LLVM::ZeroOp::create(
            rewriter, loc,
            LLVM::LLVMArrayType::get(i1Type, op.getType().getWidth()));
        LLVM::StoreOp::create(rewriter, loc, zero, storage);
      }
      rewriter.replaceOp(op, storage);
      return success();
    }

    auto array = LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, size)
                     .getResult();
    for (int64_t index = 0; index < op.getType().getWidth(); ++index) {
      auto result =
          createPointerFromIndex(rewriter, loc, reg.resultOffset + index);
      auto position = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(index));
      auto element = LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, array,
                                         ValueRange{position});
      auto store = LLVM::StoreOp::create(rewriter, loc, result, element);
      store->setAttr(QIR_RESULT_STORE_ATTR, rewriter.getUnitAttr());
    }

    state.resultArrays.insert(array);
    reg.array = array;

    rewriter.replaceOp(op, array);
    return success();
  }
};

} // namespace

static Value loadCBit(Operation* op, Value reg, Value index,
                      ConversionPatternRewriter& rewriter,
                      bool returnedRegister) {
  const auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  if (!returnedRegister) {
    auto elementptr =
        LLVM::GEPOp::create(rewriter, op->getLoc(), ptrType,
                            rewriter.getI1Type(), reg, ValueRange{index});
    return LLVM::LoadOp::create(rewriter, op->getLoc(), rewriter.getI1Type(),
                                elementptr);
  }
  auto elementptr = LLVM::GEPOp::create(rewriter, op->getLoc(), ptrType,
                                        ptrType, reg, ValueRange{index});
  auto result =
      LLVM::LoadOp::create(rewriter, op->getLoc(), ptrType, elementptr);
  auto fnSig = LLVM::LLVMFunctionType::get(rewriter.getI1Type(), {ptrType});
  auto fnDec =
      getOrCreateFunctionDeclaration(rewriter, op, QIR_READ_RESULT, fnSig);
  return LLVM::CallOp::create(rewriter, op->getLoc(), fnDec, result.getResult())
      .getResult();
}

namespace {

struct ConvertCBitLoadOp final : StatefulOpConversionPattern<cbit::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto returnedRegister =
        getState().returnedCBitReads.contains(op.getOperation());
    rewriter.replaceOp(op, loadCBit(op, adaptor.getReg(), adaptor.getIndex(),
                                    rewriter, returnedRegister));
    return success();
  }
};

struct ConvertCBitStoreOp final : StatefulOpConversionPattern<cbit::StoreOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (getState().resultArrays.contains(adaptor.getReg())) {
      return op.emitError(
          "non-measurement stores to returned CBit registers are not "
          "supported by QIR conversion");
    }
    const auto ptrType = LLVM::LLVMPointerType::get(getContext());
    auto elementptr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), ptrType, rewriter.getI1Type(), adaptor.getReg(),
        ValueRange{adaptor.getIndex()});
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                          elementptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts `memref.alloc` to `llvm.alloca`
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getType().getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }
    return convertQubitMemRefAllocOp(op, adaptor, getState(), rewriter);
  }
};

/// Allocate local storage for qubit references without allocating qubits.
struct ConvertQubitAllocaOp final : OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto type = op.getType();
    if (type.getRank() != 1 || !isa<QubitType>(type.getElementType())) {
      return failure();
    }
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto size = type.hasStaticShape()
                    ? LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                               rewriter.getI64Type(),
                                               type.getNumElements())
                          .getResult()
                    : adaptor.getDynamicSizes().front();
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, ptrType, ptrType, size);
    return success();
  }
};

/// Shape casts retain the same reference buffer and quantum allocation size.
struct ConvertQubitCastOp final : OpConversionPattern<memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto type = dyn_cast<MemRefType>(op.getType());
    if (!type || type.getRank() != 1 ||
        !isa<QubitType>(type.getElementType())) {
      return failure();
    }
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// Store a qubit reference in a register slot.
struct ConvertQubitStoreOp final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto type = op.getMemref().getType();
    if (type.getRank() != 1 || !isa<QubitType>(type.getElementType())) {
      return failure();
    }
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto address =
        LLVM::GEPOp::create(rewriter, op.getLoc(), ptrType, ptrType,
                            adaptor.getMemref(), adaptor.getIndices()[0]);
    auto store = rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getValue(), address);
    store->setAttr(QIR_QUBIT_STORE_ATTR, rewriter.getUnitAttr());
    return success();
  }
};

/// Converts `memref.load` to `llvm.load`
///
/// @par Example:
/// ```mlir
/// %q = memref.load %memref[%c1] : memref<3x!qc.qubit>
/// ```
/// is converted to
/// ```mlir
/// %ptr = llvm.getelementptr %alloca[c1] : !llvm.ptr -> !llvm.ptr
/// %q = llvm.load %ptr : !llvm.ptr -> !llvm.ptr
/// ```
struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto memrefType = op.getMemref().getType();
    if (memrefType.getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }

    auto loc = op.getLoc();
    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto elementptr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType,
                            adaptor.getMemref(), adaptor.getIndices()[0])
            .getResult();
    auto result =
        LLVM::LoadOp::create(rewriter, loc, ptrType, elementptr).getResult();

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Converts memref.dealloc to QIR qubit-array release
///
/// @par Example:
/// ```mlir
/// memref.dealloc %memref : memref<3x!qc.qubit>
/// ```
/// is converted to
/// ```mlir
/// llvm.call @"@__quantum__rt__qubit_array_release"(%c3, %alloca) : (i64,
/// !llvm.ptr) -> ()
/// ```
struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (auto shape = op.getMemref().getType().getShape(); shape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }
    auto& state = getState();
    auto* ctx = getContext();
    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto size = state.qregSizes.lookup(adaptor.getMemref());
    if (!size) {
      return rewriter.notifyMatchFailure(op, "unknown qubit register");
    }

    auto fnSig = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                             {i64Type, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                                QIR_QUBIT_ARRAY_RELEASE, fnSig);

    // Create the release call
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{size, adaptor.getMemref()});
    rewriter.eraseOp(op);

    return success();
  }
};

/// Converts qc.alloc to QIR qubit allocation
///
/// @par Example:
/// ```mlir
/// %q = qc.alloc : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// %zero = llvm.mlir.zero : !llvm.ptr
/// %q = llvm.call @"@__quantum__rt__qubit_allocate"(%zero) : !llvm.ptr ->
/// !llvm.ptr
/// ```
struct ConvertQCAllocOp final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fnSig = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_QUBIT_ALLOC, fnSig);

    auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDec, zero.getResult());

    return success();
  }
};

/// Converts qc.dealloc to QIR qubit release
///
/// @par Example:
/// ```mlir
/// qc.dealloc %q : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @"@__quantum__rt__qubit_release"(%q) : !llvm.ptr -> ()
/// ```
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fnSig =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_QUBIT_RELEASE, fnSig);

    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec, adaptor.getQubit());
    rewriter.eraseOp(op);

    return success();
  }
};

/// Converts qc.reset operation to QIR reset
///
/// Converts qubit reset to a call to the QIR __quantum__qis__reset__body
/// function, which resets a qubit to the |0⟩ state.
///
/// @par Example:
/// ```mlir
/// qc.reset %q : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__reset__body(%q) : !llvm.ptr -> ()
/// ```
struct ConvertQCResetOp final : StatefulOpConversionPattern<ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Declare QIR function
    const auto fnSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
    auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_RESET, fnSignature);

    // Replace operation with CallOp
    auto call = LLVM::CallOp::create(rewriter, op.getLoc(), fnDecl,
                                     adaptor.getOperands());
    if (failed(getState().nameNativeCall(call, "reset", adaptor.getQubit()))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowers measurements to static result IDs.
///
/// Returned measurements use scalar IDs or pointers loaded from result arrays.
/// Non-output measurements share a scratch result. Remaining SSA users receive
/// the measured bit through a read-result call.
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    auto returnedResult = state.measuredResults.find(op);
    auto result = resolveRegisterMeasurement(state, op, rewriter);
    if (!result) {
      if (returnedResult != state.measuredResults.end()) {
        result = getResultPtr(state, op.getOperation(), rewriter);
      } else {
        // Non-output results are read into SSA values before the next measure.
        if (!state.scratchResult) {
          state.scratchResult =
              getResultPtr(state, op.getOperation(), rewriter);
        }
        result = state.scratchResult;
      }
    }

    if (returnedResult != state.measuredResults.end()) {
      returnedResult->second.value = result;
    }

    // Create measure call
    auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSig);

    auto call = LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                                     ValueRange{adaptor.getQubit(), result});
    if (failed(state.nameNativeCall(call, "measure", adaptor.getQubit()))) {
      return failure();
    }

    // Create read-result call if the result has uses besides fused stores
    if (llvm::all_of(op->getUsers(), [](Operation* user) {
          return user->hasAttr(QIR_FUSED_STORE_ATTR);
        })) {
      rewriter.eraseOp(op);
    } else {
      auto fnSig = LLVM::LLVMFunctionType::get(rewriter.getI1Type(), {ptrType});
      auto fnDec =
          getOrCreateFunctionDeclaration(rewriter, op, QIR_READ_RESULT, fnSig);

      auto readResult =
          LLVM::CallOp::create(rewriter, op.getLoc(), fnDec, result);
      rewriter.replaceOp(op, readResult.getResult());
    }

    return success();
  }
};
} // namespace

/// Populates the Adaptive Profile conversion patterns.
static void populatePatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
  patterns.add<ConvertQubitAllocaOp, ConvertQubitStoreOp, ConvertQubitCastOp>(
      typeConverter, ctx);
  patterns.add<ConvertCBitAllocOp, ConvertCBitLoadOp, ConvertCBitStoreOp,
               ConvertMemRefAllocOp, ConvertMemRefLoadOp,
               ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCDeallocOp,
               ConvertQCMeasureOp, ConvertQCResetOp>(typeConverter, ctx,
                                                     &state);
}

/// Add an entry block for constants and initialization and an output block
/// for output recording around the existing control flow.
///
/// Quantum releases remain at their source locations.
static void ensureBlocks(LLVM::LLVMFuncOp main, LoweringState& state) {
  OpBuilder builder(main.getBody());
  auto* firstBlock = &main.front();
  LLVM::ReturnOp returnOp;
  main.walk([&](LLVM::ReturnOp op) { returnOp = op; });
  auto* returnBlock = returnOp->getBlock();

  auto* entryBlock = builder.createBlock(&main.getBody());
  main.getBlocks().splice(Region::iterator(firstBlock), main.getBlocks(),
                          entryBlock);
  Block* outputBlock = builder.createBlock(&main.getBody());

  state.entryBlock = entryBlock;
  state.outputBlock = outputBlock;

  builder.setInsertionPointToEnd(entryBlock);
  LLVM::BrOp::create(builder, main->getLoc(), firstBlock);
  returnOp->moveBefore(outputBlock, outputBlock->end());

  builder.setInsertionPointToEnd(returnBlock);
  LLVM::BrOp::create(builder, main->getLoc(), outputBlock);

  // Move up all constants to the beginning
  auto& entryOps = entryBlock->getOperations();
  for (auto& block : main.getBlocks()) {
    if (&block == entryBlock || &block == outputBlock) {
      continue;
    }
    for (auto it = block.begin(); it != block.end();) {
      if (auto& op = *it++; op.hasTrait<OpTrait::ConstantLike>()) {
        entryOps.splice(entryBlock->getTerminator()->getIterator(),
                        block.getOperations(), op.getIterator());
      }
    }
  }
}

/// Reserve static result IDs for returned result registers.
static LogicalResult reserveRegisterResults(ModuleOp moduleOp,
                                            LoweringState& state) {
  for (auto& reg : state.cregs) {
    if (!reg.record || reg.booleanStorage) {
      continue;
    }
    reg.resultOffset = state.staticRegisterResults;
    const auto width = std::get<int64_t>(reg.size);
    if (width >
        std::numeric_limits<int64_t>::max() - state.staticRegisterResults) {
      return moduleOp.emitError("static result IDs exceed the supported range");
    }
    state.staticRegisterResults += width;
  }
  return success();
}

/// Collect structured operations so that parents precede their nested
/// operations and later operations of a block precede earlier ones.
static void collectStructuredOperations(Region& region,
                                        SmallVectorImpl<Operation*>& ops) {
  for (auto& block : region) {
    for (auto& op : llvm::reverse(block)) {
      if (isa<scf::SCFDialect>(op.getDialect()) &&
          !op.hasTrait<OpTrait::IsTerminator>()) {
        ops.push_back(&op);
      }
      for (auto& nested : op.getRegions()) {
        collectStructuredOperations(nested, ops);
      }
    }
  }
}

/// Lower structured control flow to the cf dialect before function conversion.
///
/// Lowering a parent first keeps its region terminators in place. Lowering the
/// last construct of a block first means each block split moves only the
/// operations up to the next construct; a dialect conversion would record every
/// moved operation, which is quadratic in the number of constructs per block.
static LogicalResult lowerStructuredControlFlow(ModuleOp moduleOp,
                                                ConversionTarget& target) {
  RewritePatternSet patterns(moduleOp.getContext());
  target.addIllegalDialect<scf::SCFDialect>();
  target.addLegalDialect<cf::ControlFlowDialect, arith::ArithDialect>();
  populateSCFToControlFlowConversionPatterns(patterns);
  const FrozenRewritePatternSet frozen(std::move(patterns));
  PatternApplicator applicator(frozen);
  applicator.applyDefaultCostModel();
  SmallVector<Operation*> ops;
  collectStructuredOperations(moduleOp.getBodyRegion(), ops);
  PatternRewriter rewriter(moduleOp.getContext());
  for (auto* op : ops) {
    rewriter.setInsertionPoint(op);
    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      return op->emitOpError("cannot be lowered to unstructured control flow");
    }
  }
  return success();
}

} // namespace adaptive

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

/// Lower supported QC operations to QIR runtime calls and LLVM IR.
///
/// Matching target payloads are checked before and after lowering.
struct QCToQIR final : impl::QCToQIRBase<QCToQIR> {
  using QCToQIRBase::QCToQIRBase;
  QCToQIR(const QCToQIROptions& options, const TargetEnvironment& target)
      : QCToQIRBase(options), environment(target) {}
  std::optional<TargetEnvironment> environment;

  void getDependentDialects(DialectRegistry& registry) const override {
    QCToQIRBase::getDependentDialects(registry);
    registerQIRClassicalTensorDialects(registry);
    mqt::createPromoteClassicalMemory()->getDependentDialects(registry);
    mqt::createLegalizeControlFlow()->getDependentDialects(registry);
  }

protected:
  void runOnOperation() override {
    if (failed(lower())) {
      signalPassFailure();
    }
  }

private:
  LogicalResult lower() {
    MLIRContext* ctx = &getContext();
    auto moduleOp = getOperation();
    const bool adaptive = profile == QIRProfile::Adaptive;
    // QIR has no qubit layout metadata.
    moduleOp->removeAttr(mqt::kLayoutAttr);
    if (failed(mqt::verifyQuantumAllocations(moduleOp))) {
      return failure();
    }
    auto entryPoint = mqt::getEntryPoint(moduleOp);
    if (!entryPoint) {
      return moduleOp->emitError("no main function with mqt.entry_point found");
    }
    if (entryPoint.getNumArguments() != 0) {
      return entryPoint.emitError(
          "QIR entry points do not support arguments; bind program parameters "
          "before lowering");
    }
    if (!adaptive && !entryPoint.getBody().hasOneBlock()) {
      return entryPoint.emitError(
          "QIR Base Profile requires a single-block entry function");
    }
    auto entryPointName = entryPoint.getSymNameAttr();
    if (failed(mqt::normalizeGlobalPhases(moduleOp))) {
      return failure();
    }
    const auto run = [&](OpPassManager& pm) {
      return runPipeline(pm, moduleOp);
    };
    LoweringState state;
    auto payload = prepareQIRTarget(moduleOp, profile, state, run,
                                    environment ? &*environment : nullptr);
    if (failed(payload)) {
      return failure();
    }
    ConversionTarget target(*ctx);
    QCToQIRTypeConverter typeConverter(ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (adaptive) {
      RewritePatternSet patterns(ctx);
      cbit::populateCBitDecompositionPatterns(patterns);
      walkAndApplyPatterns(moduleOp,
                           FrozenRewritePatternSet(std::move(patterns)));
    }
    if (failed(prepareClassicalResults(moduleOp, state,
                                       /*allowComputedOutputs=*/adaptive))) {
      return failure();
    }
    if (adaptive &&
        (failed(adaptive::reserveRegisterResults(moduleOp, state)) ||
         failed(adaptive::lowerStructuredControlFlow(moduleOp, target)) ||
         failed(adaptive::prepareCBitRegisterAccesses(moduleOp, state)))) {
      return failure();
    }

    {
      RewritePatternSet patterns(ctx);
      target.addIllegalDialect<func::FuncDialect>();
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);
      if (failed(
              applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        return failure();
      }
    }

    auto main = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(entryPointName);
    if (!main) {
      return moduleOp->emitError("no main function with mqt.entry_point found");
    }
    main.setPassthroughAttr(
        ArrayAttr::get(ctx, {StringAttr::get(ctx, ::qir::ENTRY_POINT_ATTR)}));
    mqt::removeEntryPoint(main);
    if (adaptive) {
      adaptive::ensureBlocks(main, state);
    } else {
      base::ensureBlocks(main, state);
    }
    addInitialize(main, ctx, state);

    {
      RewritePatternSet patterns(ctx);
      target.addIllegalDialect<cbit::CBitDialect, qc::QCDialect,
                               scf::SCFDialect>();
      // Defer classical memory and index types to final MemRef-to-LLVM
      // lowering.
      target.addDynamicallyLegalDialect<memref::MemRefDialect>(
          [&](Operation* op) {
            return llvm::all_of(
                llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()),
                [&](Type type) {
                  return !isa<MemRefType>(type) || typeConverter.isLegal(type);
                });
          });
      populateQCToQIRPatterns(patterns, typeConverter, ctx, state);
      if (adaptive) {
        adaptive::populatePatterns(patterns, typeConverter, ctx, state);
      } else {
        base::populatePatterns(patterns, typeConverter, ctx, state);
      }
      if (failed(
              applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        return failure();
      }
    }
    addOutputRecording(main, ctx, state);

    if (failed(finalizeQIRConversion(moduleOp, target, typeConverter))) {
      return failure();
    }
    if (!adaptive) {
      // Reconcile argument casts before comparing canonical qubit pointers.
      auto& body = *std::next(main.getBody().begin());
      if (failed(
              qir::moveTerminalMeasurements(body, *state.measurementsBlock))) {
        return failure();
      }
    }
    return finalizeQIRMetadata(moduleOp, profile, *payload, run);
  }
};

} // namespace

std::unique_ptr<Pass> createQCToQIR(const QCToQIROptions& options,
                                    const TargetEnvironment& environment) {
  return std::make_unique<QCToQIR>(options, environment);
}

} // namespace mlir
