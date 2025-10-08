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

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>

namespace mlir {

using namespace mlir::quartz;

#define GEN_PASS_DEF_QUARTZTOQIR
#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h.inc"

namespace {

/**
 * @brief Looks up or creates a QIR function declaration
 *
 * @details
 * Searches for an existing function declaration in the symbol table. If not
 * found, creates a new function declaration at the end of the module.
 *
 * For QIR functions that are irreversible (measurement, reset, deallocation),
 * the "irreversible" attribute is added automatically.
 *
 * @param rewriter The pattern rewriter to use
 * @param op The operation requesting the function
 * @param fnName The name of the QIR function
 * @param fnType The LLVM function type signature
 * @return The LLVM function declaration
 */
LLVM::LLVMFuncOp getFunctionDeclaration(PatternRewriter& rewriter,
                                        Operation* op, StringRef fnName,
                                        Type fnType) {
  // Check if the function already exists
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnName));

  if (fnDecl == nullptr) {
    // Create the declaration at the end of the module
    const PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(module.getBody());

    fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnName, fnType);

    // Add irreversible attribute to irreversible quantum operations
    if (fnName == "__quantum__qis__mz__body" ||
        fnName == "__quantum__rt__qubit_release" ||
        fnName == "__quantum__qis__reset__body") {
      fnDecl->setAttr("passthrough",
                      rewriter.getStrArrayAttr({"irreversible"}));
    }
  }

  return cast<LLVM::LLVMFuncOp>(fnDecl);
}

/**
 * @brief State object for tracking lowering information during QIR conversion
 *
 * @details
 * This struct maintains state during the conversion of Quartz dialect
 * operations to QIR (Quantum Intermediate Representation). It tracks:
 * - Qubit and result counts for QIR metadata
 * - Pointer value caching for reuse
 * - Result output mapping
 * - Whether dynamic memory management is needed
 */
struct LoweringState {
  /// Map from qubit index to pointer value for reuse
  DenseMap<size_t, Value> ptrMap;
  /// Map from result index to addressOf operation for output recording
  DenseMap<size_t, Operation*> outputMap;
  /// Index for the next measure operation label
  size_t index{};
  /// Number of qubits used in the module
  size_t numQubits{};
  /// Number of measurement results stored in the module
  size_t numResults{};
  /// Whether the module uses dynamic qubit management (true when `quartz.alloc`
  /// is used, false when only `quartz.static` is used)
  bool useDynamicQubit{};
  /// Whether the module uses dynamic result management (expected: false for
  /// Quartz at the moment)
  bool useDynamicResult{};
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
    const auto index = static_cast<size_t>(op.getIndex());

    // Get or create a pointer to the qubit
    if (getState().ptrMap.contains(index)) {
      // Reuse existing pointer
      rewriter.replaceOp(op, getState().ptrMap.at(index));
    } else {
      // Create constant and inttoptr operations
      const auto constantOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
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

  static constexpr StringLiteral FN_NAME_ALLOCATE =
      "__quantum__rt__qubit_allocate";

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: () -> ptr
    const auto qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx), {});

    // Get or create function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, FN_NAME_ALLOCATE, qirSignature);

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

  static constexpr StringLiteral FN_NAME_QUBIT_RELEASE =
      "__quantum__rt__qubit_release";

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (ptr) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // Get or create function declaration
    const auto fnDecl = getFunctionDeclaration(
        rewriter, op, FN_NAME_QUBIT_RELEASE, qirSignature);

    // Replace with call to qubit_release
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
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

  static constexpr StringLiteral FN_NAME_RESET = "__quantum__qis__reset__body";

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Create QIR function signature: (ptr) -> void
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // Get or create function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, FN_NAME_RESET, qirSignature);

    // Replace with call to reset
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

/**
 * @brief Converts quartz.measure operation to QIR measurement and output
 * recording
 *
 * @details
 * Converts qubit measurement to two QIR calls:
 * 1. `__quantum__qis__mz__body`: Performs the measurement
 * 2. `__quantum__rt__result_record_output`: Records the result for output
 *
 * The converter examines the users of the measurement result to find
 * memref.store operations, extracting the classical register index from them.
 * This ensures faithful translation of which classical bit receives the
 * measurement result, including cases where multiple measurements target the
 * same bit.
 *
 * The result pointer is created using inttoptr and cached for reuse. A global
 * constant is created for the result label (r0, r1, etc.) based on the
 * classical register index.
 *
 * @par Example:
 * ```mlir
 * %result = quartz.measure %q : !quartz.qubit -> i1
 * %c0 = arith.constant 0 : index
 * memref.store %result, %creg[%c0] : memref<2xi1>
 * ```
 * becomes:
 * ```mlir
 * %c0_i64 = llvm.mlir.constant(0 : i64) : i64
 * %result_ptr = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr
 * llvm.call @__quantum__qis__mz__body(%q, %result_ptr) : (!llvm.ptr, !llvm.ptr)
 * -> () llvm.call @__quantum__rt__result_record_output(%result_ptr, %label0) :
 * (!llvm.ptr, !llvm.ptr) -> ()
 * ```
 */
struct ConvertQuartzMeasureQIR final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  static constexpr StringLiteral FN_NAME_MEASURE = "__quantum__qis__mz__body";
  static constexpr StringLiteral FN_NAME_RECORD_OUTPUT =
      "__quantum__rt__result_record_output";

  /**
   * @brief Creates a global constant and addressOf for result labeling
   *
   * @details
   * Creates a global string constant at module level with the label "rN\0"
   * where N is the index, and returns an addressOf operation pointing to it.
   * The addressOf is placed at the start of the main function.
   *
   * @param op The measure operation
   * @param rewriter The rewriter
   * @param state The lowering state
   * @return The addressOf operation for the global constant
   */
  static Operation* getAddressOfOp(Operation* op,
                                   ConversionPatternRewriter& rewriter,
                                   LoweringState& state) {
    // Create global at module level
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(module.getBody());

    // Calculate string length for array size
    auto num = state.index;
    auto digits = 1;
    while (num >= 10) {
      num /= 10;
      ++digits;
    }

    // Create global with label "r0\0", "r1\0", etc.
    const auto symbolName = rewriter.getStringAttr(
        "mlir.llvm.nameless_global_" + std::to_string(state.index));
    const auto llvmArrayType =
        LLVM::LLVMArrayType::get(rewriter.getIntegerType(8), digits + 2);
    const auto stringInitializer =
        rewriter.getStringAttr("r" + std::to_string(state.index) + '\0');

    auto globalOp = rewriter.create<LLVM::GlobalOp>(
        op->getLoc(), llvmArrayType, /*isConstant=*/true,
        LLVM::Linkage::Internal, symbolName, stringInitializer);
    globalOp->setAttr("addr_space", rewriter.getI32IntegerAttr(0));
    globalOp->setAttr("dso_local", rewriter.getUnitAttr());

    // Create addressOf at start of main function
    auto main = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto& firstBlock = *(main.getBlocks().begin());
    rewriter.setInsertionPointToStart(&firstBlock);
    const auto addressOfOp = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()),
        symbolName);

    // Restore insertion point
    rewriter.setInsertionPoint(op);
    state.index++;

    return addressOfOp;
  }

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    auto& ptrMap = getState().ptrMap;
    auto& outputMap = getState().outputMap;
    const auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Get or create result pointer
    Value resultValue = nullptr;
    if (ptrMap.contains(getState().numResults)) {
      resultValue = ptrMap.at(getState().numResults);
    } else {
      auto constantOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(
                           static_cast<int64_t>(getState().numResults)));
      resultValue = rewriter
                        .create<LLVM::IntToPtrOp>(op.getLoc(), ptrType,
                                                  constantOp->getResult(0))
                        .getResult();
      ptrMap.try_emplace(getState().numResults, resultValue);
    }

    // Create mz (measure) call: mz(qubit, result)
    const auto mzSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto mzDecl =
        getFunctionDeclaration(rewriter, op, FN_NAME_MEASURE, mzSignature);
    rewriter.create<LLVM::CallOp>(op.getLoc(), mzDecl,
                                  ValueRange{adaptor.getQubit(), resultValue});

    // Find the classical register index by examining store operations
    size_t index = 0;
    for (const auto* user : op->getResult(0).getUsers()) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
        // Extract the index from the store operation
        auto constantOp =
            storeOp.getIndices()[0].getDefiningOp<arith::ConstantOp>();
        if (!constantOp) {
          op.emitError("Measurement index could not be resolved. Index is not "
                       "a ConstantOp.");
          return failure();
        }

        auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!integerAttr) {
          op.emitError("Measurement index could not be resolved. Index cannot "
                       "be cast to IntegerAttr.");
          return failure();
        }
        index = integerAttr.getInt();

        const auto allocaOp = storeOp.getMemref();

        // Clean up the store operation and related ops
        storeOp->dropAllReferences();
        rewriter.eraseOp(storeOp);

        // Erase the alloca if all stores are removed
        if (allocaOp.use_empty()) {
          rewriter.eraseOp(allocaOp.getDefiningOp<memref::AllocaOp>());
        }

        // Delete the constant if there are no users left
        if (constantOp->use_empty()) {
          rewriter.eraseOp(constantOp);
        }
        break;
      }
    }

    // Create result_record_output call
    const auto recordSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto recordDecl = getFunctionDeclaration(
        rewriter, op, FN_NAME_RECORD_OUTPUT, recordSignature);

    // Get or create output label addressOf for this index
    Operation* labelOp = nullptr;
    if (outputMap.contains(index)) {
      // Reuse existing label for this index (handles multiple measurements to
      // same bit)
      labelOp = outputMap.at(index);
    } else {
      // Create new label
      labelOp = getAddressOfOp(op, rewriter, getState());
      outputMap.try_emplace(index, labelOp);
      // Only increment result count for new labels
      getState().numResults++;
    }

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), recordDecl,
        ValueRange{resultValue, labelOp->getResult(0)});

    // Remove the original operation
    rewriter.eraseOp(op);

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

  /// QIR runtime initialization function name
  static constexpr StringLiteral FN_NAME_INITIALIZE =
      "__quantum__rt__initialize";

  /**
   * @brief Finds the main entry point function in the module
   *
   * @details
   * Searches for the LLVM function marked with the "entry_point" attribute in
   * the passthrough attributes. This is the main entry point created by the
   * QuartzProgramBuilder.
   *
   * @param op The module operation to search in
   * @return The main LLVM function, or nullptr if not found
   */
  static LLVM::LLVMFuncOp getMainFunction(Operation* op) {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
      return nullptr;
    }

    // Search for function with entry_point attribute
    for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
      auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough");
      if (!passthrough) {
        continue;
      }
      if (llvm::any_of(passthrough, [](Attribute attr) {
            auto strAttr = dyn_cast<StringAttr>(attr);
            return strAttr && strAttr.getValue() == "entry_point";
          })) {
        return funcOp;
      }
    }
    return nullptr;
  }

  /**
   * @brief Ensures proper block structure for QIR base profile
   *
   * @details
   * The QIR base profile requires a specific 4-block structure:
   * 1. **Entry block**: Contains constant operations and initialization
   * 2. **Main block**: Contains reversible quantum operations (gates)
   * 3. **Irreversible block**: Contains irreversible operations (measure,
   *    reset, dealloc)
   * 4. **End block**: Contains return operation
   *
   * Blocks are connected with unconditional jumps (entry → main →
   * irreversible → end). This structure ensures proper QIR semantics and
   * enables optimizations.
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
    auto* mainBlock = &main.front();
    OpBuilder builder(main.getBody());

    // Create the required blocks
    auto* entryBlock = builder.createBlock(&main.getBody());
    // Move the entry block before the main block
    main.getBlocks().splice(Region::iterator(mainBlock), main.getBlocks(),
                            entryBlock);
    Block* irreversibleBlock = builder.createBlock(&main.getBody());
    Block* endBlock = builder.createBlock(&main.getBody());

    auto& mainBlockOps = mainBlock->getOperations();
    auto& endBlockOps = endBlock->getOperations();
    auto& irreversibleBlockOps = irreversibleBlock->getOperations();

    // Move operations to appropriate blocks
    for (auto it = mainBlock->begin(); it != mainBlock->end();) {
      // Ensure iterator remains valid after potential move
      auto& op = *it++;

      // Check for irreversible operations
      if (op.getDialect()->getNamespace() == "memref") {
        // Keep memref operations for classical bits in place (they're not
        // quantum operations)
        continue;
      }
      if (isa<DeallocOp>(op) || isa<ResetOp>(op) || isa<MeasureOp>(op)) {
        // Move irreversible quantum operations to irreversible block
        irreversibleBlockOps.splice(irreversibleBlock->end(), mainBlockOps,
                                    Block::iterator(op));
      } else if (isa<LLVM::ReturnOp>(op)) {
        // Move return to end block
        endBlockOps.splice(endBlock->end(), mainBlockOps, Block::iterator(op));
      }
      // All other operations (gates, etc.) stay in main block
    }

    // Add unconditional jumps between blocks
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), mainBlock);

    builder.setInsertionPointToEnd(mainBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), irreversibleBlock);

    builder.setInsertionPointToEnd(irreversibleBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), endBlock);
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
   * @param state The lowering state (unused but kept for consistency)
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
        main, builder.getStringAttr(FN_NAME_INITIALIZE));
    if (fnDecl == nullptr) {
      const PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToEnd(moduleOp.getBody());
      auto fnSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
      fnDecl = builder.create<LLVM::LLVMFuncOp>(
          main->getLoc(), FN_NAME_INITIALIZE, fnSignature);
    }

    // Create the initialization call
    builder.create<LLVM::CallOp>(main->getLoc(), cast<LLVM::LLVMFuncOp>(fnDecl),
                                 ValueRange{zeroOp->getResult(0)});
  }

  /**
   * @brief Sets QIR base profile metadata attributes on the main function
   *
   * @details
   * Adds the required metadata attributes for QIR base profile compliance:
   * - `entry_point`: Marks the main entry point function
   * - `output_labeling_schema`: schema_id
   * - `qir_profiles`: base_profile
   * - `required_num_qubits`: Number of qubits used
   * - `required_num_results`: Number of measurement results
   * - `qir_major_version`: 1
   * - `qir_minor_version`: 0
   * - `dynamic_qubit_management`: true/false
   * - `dynamic_result_management`: true/false
   *
   * These attributes are required by the QIR specification and inform QIR
   * consumers about the module's resource requirements and capabilities.
   *
   * @param main The main LLVM function to annotate
   * @param state The lowering state containing qubit/result counts
   */
  static void setAttributes(LLVM::LLVMFuncOp& main, LoweringState* state) {
    OpBuilder builder(main.getBody());
    SmallVector<Attribute> attributes;

    // Core QIR attributes
    attributes.emplace_back(builder.getStringAttr("entry_point"));
    attributes.emplace_back(
        builder.getStrArrayAttr({"output_labeling_schema", "schema_id"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"qir_profiles", "base_profile"}));

    // Resource requirements
    attributes.emplace_back(builder.getStrArrayAttr(
        {"required_num_qubits", std::to_string(state->numQubits)}));
    attributes.emplace_back(builder.getStrArrayAttr(
        {"required_num_results", std::to_string(state->numResults)}));

    // QIR version
    attributes.emplace_back(
        builder.getStrArrayAttr({"qir_major_version", "1"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"qir_minor_version", "0"}));

    // Management model
    attributes.emplace_back(
        builder.getStrArrayAttr({"dynamic_qubit_management",
                                 state->useDynamicQubit ? "true" : "false"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"dynamic_result_management",
                                 state->useDynamicResult ? "true" : "false"}));

    main->setAttr("passthrough", builder.getArrayAttr(attributes));
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
   * measure, reset).
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
      quartzPatterns.add<ConvertQuartzStaticQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzAllocQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzDeallocQIR>(typeConverter, ctx);
      quartzPatterns.add<ConvertQuartzResetQIR>(typeConverter, ctx);
      quartzPatterns.add<ConvertQuartzMeasureQIR>(typeConverter, ctx, &state);

      // Gate operations will be added here as the dialect expands

      if (applyPartialConversion(moduleOp, target, std::move(quartzPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    // Stage 4: Set QIR metadata attributes
    setAttributes(main, &state);

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
