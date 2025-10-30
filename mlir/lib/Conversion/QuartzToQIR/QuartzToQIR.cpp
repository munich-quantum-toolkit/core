/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h"

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <utility>

namespace mlir {

using namespace quartz;
using namespace qir;

#define GEN_PASS_DEF_QUARTZTOQIR
#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h.inc"

namespace {

/**
 * @brief State object for tracking lowering information during QIR conversion
 *
 * @details
 * This struct maintains state during the conversion of Quartz dialect
 * operations to QIR (Quantum Intermediate Representation). It tracks:
 * - Qubit and result counts for QIR metadata
 * - Pointer value caching for reuse
 * - Whether dynamic memory management is needed
 * - Sequence of measurements for output recording
 */
struct LoweringState : QIRMetadata {
  /// Map from qubit index to pointer value for reuse
  DenseMap<int64_t, Value> ptrMap;

  /// Map from classical result index to pointer value for reuse
  DenseMap<int64_t, Value> resultPtrMap;

  /// Map from (register_name, register_index) to result pointer
  /// This allows caching result pointers for measurements with register info
  DenseMap<std::pair<StringRef, int64_t>, Value> registerResultMap;
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which tracks qubit/result counts and caches values across multiple
 * pattern applications.
 *
 * @tparam OpType The operation type to convert
 */
template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter, MLIRContext* ctx,
                              LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, ctx), state_(state) {}

  /// Returns the shared lowering state object
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

} // namespace

/**
 * @brief Type converter for lowering Quartz dialect types to LLVM types
 *
 * @details
 * Converts Quartz dialect types to their LLVM equivalents for QIR emission.
 *
 * Type conversions:
 * - `!quartz.qubit` -> `!llvm.ptr` (opaque pointer to qubit in QIR)
 */
struct QuartzToQIRTypeConverter final : LLVMTypeConverter {
  explicit QuartzToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // Convert QubitType to LLVM pointer (QIR uses opaque pointers for qubits)
    addConversion(
        [ctx](QubitType /*type*/) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

namespace {

/**
 * @brief Converts quartz.alloc operation to QIR qubit_allocate
 *
 * @details
 * Converts dynamic qubit allocation to a call to the QIR
 * __quantum__rt__qubit_allocate function.
 *
 * Register metadata (register_name, register_size, register_index) is ignored
 * during QIR conversion as it is used for analysis and readability only.
 *
 * @par Example:
 * ```mlir
 * %q = quartz.alloc : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * %q = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
 * ```
 */
struct ConvertQuartzAllocQIR final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: () -> ptr
    const auto qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx), {});

    // Get or create function declaration
    const auto fnDecl = getOrCreateFunctionDeclaration(
        rewriter, op, QIR_QUBIT_ALLOCATE, qirSignature);

    // Replace with call to qubit_allocate
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, ValueRange{});

    // Track qubit count and mark as using dynamic allocation
    getState().numQubits++;
    getState().useDynamicQubit = true;

    return success();
  }
};

/**
 * @brief Converts quartz.dealloc operation to QIR qubit_release
 *
 * @details
 * Converts dynamic qubit deallocation to a call to the QIR
 * __quantum__rt__qubit_release function, which releases a dynamically
 * allocated qubit.
 *
 * @par Example:
 * ```mlir
 * quartz.dealloc %q : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * llvm.call @__quantum__rt__qubit_release(%q) : (!llvm.ptr) -> ()
 * ```
 */
struct ConvertQuartzDeallocQIR final : OpConversionPattern<DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (ptr) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // Get or create function declaration
    const auto fnDecl = getOrCreateFunctionDeclaration(
        rewriter, op, QIR_QUBIT_RELEASE, qirSignature);

    // Replace with call to qubit_release
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

/**
 * @brief Converts quartz.static operation to QIR inttoptr
 *
 * @details
 * Converts a static qubit reference to an LLVM pointer by creating a constant
 * with the qubit index and converting it to a pointer. The pointer is cached
 * in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q0 = quartz.static 0 : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQuartzStaticQIR final : StatefulOpConversionPattern<StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    const auto index = op.getIndex();

    // Get or create a pointer to the qubit
    if (getState().ptrMap.contains(index)) {
      // Reuse existing pointer
      rewriter.replaceOp(op, getState().ptrMap.at(index));
    } else {
      // Create constant and inttoptr operations
      const auto constantOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(index));
      const auto intToPtrOp = rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
          op, LLVM::LLVMPointerType::get(ctx), constantOp->getResult(0));

      // Cache for reuse
      getState().ptrMap.try_emplace(index, intToPtrOp->getResult(0));
    }

    // Track maximum qubit index
    if (index >= getState().numQubits) {
      getState().numQubits = index + 1;
    }

    return success();
  }
};

/**
 * @brief Converts quartz.measure operation to QIR measurement
 *
 * @details
 * Converts qubit measurement to a QIR call to `__quantum__qis__mz__body`.
 * Unlike the previous implementation, this does NOT immediately record output.
 * Instead, it tracks measurements in the lowering state for deferred output
 * recording in a separate output block, as required by the QIR Base Profile.
 *
 * For measurements with register information, the result pointer is mapped
 * to (register_name, register_index) for later retrieval. For measurements
 * without register information, a sequential result pointer is assigned.
 *
 * @par Example (with register):
 * ```mlir
 * %result = quartz.measure("c", 2, 0) %q : !quartz.qubit -> i1
 * ```
 * becomes:
 * ```mlir
 * %c0_i64 = llvm.mlir.constant(0 : i64) : i64
 * %result_ptr = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr
 * llvm.call @__quantum__qis__mz__body(%q, %result_ptr) : (!llvm.ptr, !llvm.ptr)
 * -> ()
 * ```
 */
struct ConvertQuartzMeasureQIR final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    const auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto& state = getState();
    const auto numResults = static_cast<int64_t>(state.numResults);
    auto& resultPtrMap = state.resultPtrMap;
    auto& registerResultMap = state.registerResultMap;

    // Get or create result pointer value
    Value resultValue;
    if (op.getRegisterName() && op.getRegisterSize() && op.getRegisterIndex()) {
      const auto registerName = op.getRegisterName().value();
      const auto registerIndex = op.getRegisterIndex().value();
      const auto key = std::make_pair(registerName, registerIndex);

      if (const auto it = registerResultMap.find(key);
          it != registerResultMap.end()) {
        resultValue = it->second;
      } else {
        const auto constantOp = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(),
            rewriter.getI64IntegerAttr(
                static_cast<int64_t>(numResults))); // Sequential result index
        resultValue = rewriter
                          .create<LLVM::IntToPtrOp>(op.getLoc(), ptrType,
                                                    constantOp->getResult(0))
                          .getResult();
        resultPtrMap[numResults] = resultValue;
        registerResultMap.insert({key, resultValue});
        state.numResults++;
      }
    } else {
      // No register info - assign sequential result pointer
      const auto constantOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(),
          rewriter.getI64IntegerAttr(numResults)); // Sequential result index
      resultValue = rewriter
                        .create<LLVM::IntToPtrOp>(op.getLoc(), ptrType,
                                                  constantOp->getResult(0))
                        .getResult();
      resultPtrMap[numResults] = resultValue;
      registerResultMap.insert({{"c", numResults}, resultValue});
      state.numResults++;
    }

    // Create mz (measure) call: mz(qubit, result)
    const auto mzSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto mzDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, mzSignature);
    rewriter.create<LLVM::CallOp>(op.getLoc(), mzDecl,
                                  ValueRange{adaptor.getQubit(), resultValue});

    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts quartz.reset operation to QIR reset
 *
 * @details
 * Converts qubit reset to a call to the QIR __quantum__qis__reset__body
 * function, which resets a qubit to the |0⟩ state.
 *
 * @par Example:
 * ```mlir
 * quartz.reset %q : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * llvm.call @__quantum__qis__reset__body(%q) : (!llvm.ptr) -> ()
 * ```
 */
struct ConvertQuartzResetQIR final : OpConversionPattern<ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (ptr) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // Get or create function declaration
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_RESET, qirSignature);

    // Replace with call to reset
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

// Temporary implementation of XOp conversion
struct ConvertQuartzXQIR final : StatefulOpConversionPattern<XOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(XOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (ptr) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // Get or create function declaration
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_X, qirSignature);

    // Replace with call to X
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

// Temporary implementation of RXOp conversion
struct ConvertQuartzRXQIR final : StatefulOpConversionPattern<RXOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(RXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (!llvm.ptr, f64) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), Float64Type::get(ctx)});

    // Get or create function declaration
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_RX, qirSignature);

    Value thetaDyn;
    if (op.getTheta().has_value()) {
      const auto& theta = op.getThetaAttr();
      auto constantOp = rewriter.create<LLVM::ConstantOp>(op.getLoc(), theta);
      thetaDyn = constantOp.getResult();
    } else {
      thetaDyn = op.getThetaDyn();
    }

    // Replace with call to RX
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fnDecl, ValueRange{adaptor.getQubit(), thetaDyn});
    return success();
  }
};

// Temporary implementation of U2Op conversion
struct ConvertQuartzU2QIR final : StatefulOpConversionPattern<U2Op> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(U2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (!llvm.ptr, f64, f64) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), Float64Type::get(ctx),
         Float64Type::get(ctx)});

    // Get or create function declaration
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_U2, qirSignature);

    Value phiDyn;
    if (op.getPhi().has_value()) {
      const auto& phi = op.getPhiAttr();
      auto constantOp = rewriter.create<LLVM::ConstantOp>(op.getLoc(), phi);
      phiDyn = constantOp.getResult();
    } else {
      phiDyn = op.getPhiDyn();
    }

    Value lambdaDyn;
    if (op.getLambda().has_value()) {
      const auto& lambda = op.getLambdaAttr();
      auto constantOp = rewriter.create<LLVM::ConstantOp>(op.getLoc(), lambda);
      lambdaDyn = constantOp.getResult();
    } else {
      lambdaDyn = op.getLambdaDyn();
    }

    // Replace with call to U2
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, fnDecl, ValueRange{adaptor.getQubit(), phiDyn, lambdaDyn});
    return success();
  }
};

// Temporary implementation of SWAPOp conversion
struct ConvertQuartzSWAPQIR final : StatefulOpConversionPattern<SWAPOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(SWAPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (ptr, ptr) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx)});

    // Get or create function declaration
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_SWAP, qirSignature);

    // Replace with call to SWAP
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

} // namespace

/**
 * @brief Pass for converting Quartz dialect operations to QIR
 *
 * @details
 * This pass converts Quartz dialect quantum operations to QIR (Quantum
 * Intermediate Representation) by lowering them to LLVM dialect operations
 * that call QIR runtime functions.
 *
 * Conversion stages:
 * 1. Convert func dialect to LLVM
 * 2. Ensure proper block structure for QIR base profile and add initialization
 * 3. Convert Quartz operations to QIR calls
 * 4. Set QIR metadata attributes
 * 5. Convert arith and cf dialects to LLVM
 * 6. Reconcile unrealized casts
 *
 * Current scope:
 * - Quartz operations: static, alloc, dealloc, measure, reset
 * - Gate operations will be added as the dialect expands
 * - Supports both static and dynamic qubit management
 */
struct QuartzToQIR final : impl::QuartzToQIRBase<QuartzToQIR> {
  using QuartzToQIRBase::QuartzToQIRBase;

  /**
   * @brief Ensures proper block structure for QIR base profile
   *
   * @details
   * The QIR base profile requires a specific 4-block structure:
   * 1. **Entry block**: Contains constant operations and initialization
   * 2. **Body block**: Contains reversible quantum operations (gates)
   * 3. **Measurements block**: Contains irreversible operations (measure,
   *    reset, dealloc)
   * 4. **Output block**: Contains output recording calls
   *
   * Blocks are connected with unconditional jumps (entry → body →
   * measurements → output). This structure ensures proper QIR Base
   * Profile semantics.
   *
   * If the function already has multiple blocks, this function does nothing.
   *
   * @param main The main LLVM function to restructure
   */
  static void ensureBlocks(LLVM::LLVMFuncOp& main) {
    // Return if there are already multiple blocks
    if (main.getBlocks().size() > 1) {
      return;
    }

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

    auto& bodyBlockOps = bodyBlock->getOperations();
    auto& outputBlockOps = outputBlock->getOperations();
    auto& measurementsBlockOps = measurementsBlock->getOperations();

    // Move operations to appropriate blocks
    for (auto it = bodyBlock->begin(); it != bodyBlock->end();) {
      // Ensure iterator remains valid after potential move
      if (auto& op = *it++;
          isa<DeallocOp>(op) || isa<ResetOp>(op) || isa<MeasureOp>(op)) {
        // Move irreversible quantum operations to measurements block
        measurementsBlockOps.splice(measurementsBlock->end(), bodyBlockOps,
                                    Block::iterator(op));
      } else if (isa<LLVM::ReturnOp>(op)) {
        // Move return to output block
        outputBlockOps.splice(outputBlock->end(), bodyBlockOps,
                              Block::iterator(op));
      } else if (op.hasTrait<OpTrait::ConstantLike>()) {
        // Move constant like operations to the entry block
        entryBlock->getOperations().splice(entryBlock->end(), bodyBlockOps,
                                           Block::iterator(op));
      }
      // All other operations (gates, etc.) stay in body block
    }

    // Add unconditional jumps between blocks
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), bodyBlock);

    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), measurementsBlock);

    builder.setInsertionPointToEnd(measurementsBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), outputBlock);
  }

  /**
   * @brief Adds QIR initialization call to the entry block
   *
   * @details
   * Inserts a call to `__quantum__rt__initialize` at the end of the entry
   * block (before the jump to main block). This QIR runtime function
   * initializes the quantum execution environment and takes a null pointer as
   * argument.
   *
   * @param main The main LLVM function
   * @param ctx The MLIR context
   */
  static void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                            LoweringState* /*state*/) {
    auto moduleOp = main->getParentOfType<ModuleOp>();
    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());

    // Create a zero (null) pointer for the initialize call
    builder.setInsertionPointToStart(&firstBlock);
    auto zeroOp = builder.create<LLVM::ZeroOp>(main->getLoc(),
                                               LLVM::LLVMPointerType::get(ctx));

    // Insert the initialize call before the jump to main block
    const auto insertPoint = std::prev(firstBlock.getOperations().end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // Get or create the initialize function declaration
    auto* fnDecl = SymbolTable::lookupNearestSymbolFrom(
        main, builder.getStringAttr(QIR_INITIALIZE));
    if (fnDecl == nullptr) {
      const PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToEnd(moduleOp.getBody());
      auto fnSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
      fnDecl = builder.create<LLVM::LLVMFuncOp>(main->getLoc(), QIR_INITIALIZE,
                                                fnSignature);
    }

    // Create the initialization call
    builder.create<LLVM::CallOp>(main->getLoc(), cast<LLVM::LLVMFuncOp>(fnDecl),
                                 ValueRange{zeroOp->getResult(0)});
  }

  /**
   * @brief Adds output recording calls to the output block
   *
   * @details
   * Generates output recording calls in the output block based on the
   * measurements tracked during conversion. Follows the QIR Base Profile
   * specification for labeled output schema.
   *
   * For each classical register, creates:
   * 1. An array_record_output call with the register size and label
   * 2. Individual result_record_output calls for each measurement in the
   * register
   *
   * Labels follow the format: "{registerName}{resultIndex}r"
   * - registerName: Name of the classical register (e.g., "c")
   * - resultIndex: Index within the array
   * - 'r' suffix: Indicates this is a result record
   *
   * Example output:
   * ```
   * @0 = internal constant [3 x i8] c"c\00"
   * @1 = internal constant [5 x i8] c"c0r\00"
   * @2 = internal constant [5 x i8] c"c1r\00"
   * call void @__quantum__rt__array_record_output(i64 2, ptr @0)
   * call void @__quantum__rt__result_record_output(ptr %result0, ptr @1)
   * call void @__quantum__rt__result_record_output(ptr %result1, ptr @2)
   * ```
   *
   * Any output recording calls that are not part of registers (i.e.,
   * measurements without register info) are grouped under a default label
   * "c" and recorded similarly.
   *
   * @param main The main LLVM function
   * @param ctx The MLIR context
   * @param state The lowering state containing measurement information
   */
  static void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                                 LoweringState* state) {
    if (state->registerResultMap.empty()) {
      return; // No measurements to record
    }

    OpBuilder builder(ctx);
    const auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Find the output block (4th block: entry, body, measurements, output, end)
    auto& outputBlock = main.getBlocks().back();

    // Insert before the branch to end block
    builder.setInsertionPoint(&outputBlock.back());

    // Group measurements by register
    llvm::StringMap<SmallVector<std::pair<int64_t, Value>>> registerGroups;
    for (const auto& [key, resultPtr] : state->registerResultMap) {
      const auto& [registerName, registerIndex] = key;
      registerGroups[registerName].emplace_back(registerIndex, resultPtr);
    }

    // Sort registers by name for deterministic output
    SmallVector<std::pair<StringRef, SmallVector<std::pair<int64_t, Value>>>>
        sortedRegisters;
    for (auto& [name, measurements] : registerGroups) {
      sortedRegisters.emplace_back(name, std::move(measurements));
    }
    llvm::sort(sortedRegisters,
               [](const auto& a, const auto& b) { return a.first < b.first; });

    // create function declarations for output recording
    const auto arrayRecordSig = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {builder.getI64Type(), ptrType});
    const auto arrayRecordDecl = getOrCreateFunctionDeclaration(
        builder, main, QIR_ARRAY_RECORD_OUTPUT, arrayRecordSig);

    const auto resultRecordSig = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto resultRecordDecl = getOrCreateFunctionDeclaration(
        builder, main, QIR_RECORD_OUTPUT, resultRecordSig);

    // Generate output recording for each register
    for (auto& [registerName, measurements] : sortedRegisters) {
      // Sort measurements by register index
      llvm::sort(measurements, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

      const auto arraySize = measurements.size();
      auto arrayLabelOp = createResultLabel(builder, main, registerName);
      auto arraySizeConst = builder.create<LLVM::ConstantOp>(
          main->getLoc(),
          builder.getI64IntegerAttr(static_cast<int64_t>(arraySize)));

      builder.create<LLVM::CallOp>(
          main->getLoc(), arrayRecordDecl,
          ValueRange{arraySizeConst.getResult(), arrayLabelOp.getResult()});

      // Create result_record_output calls for each measurement
      for (auto [regIdx, resultPtr] : measurements) {
        // Create label for result: "{arrayCounter+1+i}_{registerName}{i}r"
        const std::string resultLabel =
            registerName.str() + std::to_string(regIdx) + "r";
        auto resultLabelOp = createResultLabel(builder, main, resultLabel);

        builder.create<LLVM::CallOp>(
            main->getLoc(), resultRecordDecl,
            ValueRange{resultPtr, resultLabelOp.getResult()});
      }
    }
  }

  /**
   * @brief Executes the Quartz to QIR conversion pass
   *
   * @details
   * Performs the conversion in six stages:
   *
   * **Stage 1: Func to LLVM**
   * Convert func dialect operations (main function) to LLVM dialect
   * equivalents.
   *
   * **Stage 2: Block structure and initialization**
   * Create proper 4-block structure for QIR base profile (entry, main,
   * irreversible, end blocks) and insert the `__quantum__rt__initialize` call
   * in the entry block.
   *
   * **Stage 3: Quartz to LLVM**
   * Convert Quartz dialect operations to QIR calls (static, alloc, dealloc,
   * measure, reset) and add output recording to the output block.
   *
   * **Stage 4: QIR attributes**
   * Add QIR base profile metadata to the main function, including qubit/result
   * counts and version information.
   *
   * **Stage 5: Standard dialects to LLVM**
   * Convert arith and control flow dialects to LLVM (for index arithmetic and
   * function control flow).
   *
   * **Stage 6: Reconcile casts**
   * Clean up any unrealized cast operations introduced during type conversion.
   */
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto* moduleOp = getOperation();
    ConversionTarget target(*ctx);
    QuartzToQIRTypeConverter typeConverter(ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();

    // Stage 1: Convert func dialect to LLVM
    {
      RewritePatternSet funcPatterns(ctx);
      target.addIllegalDialect<func::FuncDialect>();
      populateFuncToLLVMConversionPatterns(typeConverter, funcPatterns);

      if (applyPartialConversion(moduleOp, target, std::move(funcPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    // Stage 2: Ensure proper block structure and add initialization
    auto main = getMainFunction(moduleOp);
    if (!main) {
      moduleOp->emitError("No main function with entry_point attribute found");
      signalPassFailure();
      return;
    }

    ensureBlocks(main);
    LoweringState state;
    addInitialize(main, ctx, &state);

    // Stage 3: Convert Quartz dialect to LLVM (QIR calls)
    {
      RewritePatternSet quartzPatterns(ctx);
      target.addIllegalDialect<QuartzDialect>();

      // Add conversion patterns for Quartz operations
      quartzPatterns.add<ConvertQuartzAllocQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzDeallocQIR>(typeConverter, ctx);
      quartzPatterns.add<ConvertQuartzStaticQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzMeasureQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzResetQIR>(typeConverter, ctx);
      quartzPatterns.add<ConvertQuartzXQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzRXQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzU2QIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzSWAPQIR>(typeConverter, ctx, &state);

      // Gate operations will be added here as the dialect expands

      if (applyPartialConversion(moduleOp, target, std::move(quartzPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      addOutputRecording(main, ctx, &state);
    }

    // Stage 4: Set QIR metadata attributes
    setQIRAttributes(main, state);

    // Stage 5: Convert standard dialects to LLVM
    {
      RewritePatternSet stdPatterns(ctx);
      target.addIllegalDialect<arith::ArithDialect>();
      target.addIllegalDialect<cf::ControlFlowDialect>();

      cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                      stdPatterns);
      arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);

      if (applyPartialConversion(moduleOp, target, std::move(stdPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    // Stage 6: Reconcile unrealized casts
    PassManager passManager(ctx);
    passManager.addPass(createReconcileUnrealizedCastsPass());
    if (passManager.run(moduleOp).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
