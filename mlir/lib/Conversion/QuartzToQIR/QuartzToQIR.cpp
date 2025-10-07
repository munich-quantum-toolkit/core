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
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <utility>

namespace mlir {

using namespace mlir::quartz;

#define GEN_PASS_DEF_QUARTZTOQIR
#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h.inc"

namespace {

struct LoweringState {
  // map a given index to a pointer value, to reuse the value instead of
  // creating a new one every time
  DenseMap<size_t, Value> ptrMap;
  // map a given index to an address to record the classical output
  DenseMap<size_t, Operation*> outputMap;
  // Index for the next measure operation
  size_t index{};
  // number of stored results in the module
  size_t numResults{};
  // number of qubits in the module
  size_t numQubits{};
  // boolean to check if the module uses dynamically addressed qubits
  bool useDynamicQubit{};
  // boolean to check if the module uses dynamically addressed results
  bool useDynamicResult{};
};

template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter, MLIRContext* ctx,
                              LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, ctx), state_(state) {}

  /// @brief Return the state object as reference.
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

} // namespace

struct QuartzToQIRTypeConverter final : LLVMTypeConverter {
  explicit QuartzToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // QubitType conversion
    addConversion(
        [ctx](QubitType /*type*/) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

struct QuartzToQIR final : impl::QuartzToQIRBase<QuartzToQIR> {
  using QuartzToQIRBase::QuartzToQIRBase;

  static constexpr StringLiteral FN_NAME_INITIALIZE =
      "__quantum__rt__initialize";

  /**
   * @brief Finds the main function in the module
   *
   * @param op The module operation that holds all operations.
   * @return The main function.
   */
  static LLVM::LLVMFuncOp getMainFunction(Operation* op) {
    auto module = dyn_cast<ModuleOp>(op);
    // find the main function
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
   * @brief Adds the initialize operation to the first block of the main
   * function.
   *
   * @param main The main function of the module.
   * @param ctx The context of the module.
   */
  static void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                            LoweringState* state) {
    auto moduleOp = main->getParentOfType<ModuleOp>();

    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());

    // create the zero op
    builder.setInsertionPointToStart(&firstBlock);
    auto zeroOperation = builder.create<LLVM::ZeroOp>(
        main->getLoc(), LLVM::LLVMPointerType::get(ctx));

    // add the zero operation to the pointerMap
    state->ptrMap.try_emplace(0, zeroOperation->getResult(0));

    // create the initialize operation as the 2nd last operation in the first
    // block after all constant operations and before the last jump operation
    const auto insertPoint = std::prev(firstBlock.getOperations().end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // get the function declaration of initialize otherwise create one
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
    // create and insert the initialize operation
    builder.create<LLVM::CallOp>(main->getLoc(),
                                 static_cast<LLVM::LLVMFuncOp>(fnDecl),
                                 ValueRange{zeroOperation->getResult(0)});
  }
  /**
   * @brief Sets the necessary attributes to the main function for the QIR base
   * profile. The required module flags are also set as attributes.
   *
   * @param main The main function of the module.
   * @param state The lowering state of the conversion pass.
   */
  static void setAttributes(LLVM::LLVMFuncOp& main, LoweringState* state) {
    OpBuilder builder(main.getBody());
    SmallVector<Attribute> attributes;
    attributes.emplace_back(builder.getStringAttr("entry_point"));
    attributes.emplace_back(
        builder.getStrArrayAttr({"output_labeling_schema", "schema_id"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"qir_profiles", "base_profile"}));
    attributes.emplace_back(builder.getStrArrayAttr(
        {"required_num_qubits", std::to_string(state->numQubits)}));
    attributes.emplace_back(builder.getStrArrayAttr(
        {"required_num_results", std::to_string(state->numResults)}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"qir_major_version", "1"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"qir_minor_version", "0"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"dynamic_qubit_management",
                                 state->useDynamicQubit ? "true" : "false"}));
    attributes.emplace_back(
        builder.getStrArrayAttr({"dynamic_result_management",
                                 state->useDynamicResult ? "true" : "false"}));

    main->setAttr("passthrough", builder.getArrayAttr(attributes));
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto* moduleOp = getOperation();
    ConversionTarget target(*ctx);
    RewritePatternSet funcPatterns(ctx);
    RewritePatternSet mqtPatterns(ctx);
    RewritePatternSet stdPatterns(ctx);
    QuartzToQIRTypeConverter typeConverter(ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();

    // convert func to LLVM
    target.addIllegalDialect<func::FuncDialect>();

    populateFuncToLLVMConversionPatterns(typeConverter, funcPatterns);

    if (applyPartialConversion(moduleOp, target, std::move(funcPatterns))
            .failed()) {
      signalPassFailure();
    }

    // convert Quartz to LLVM
    auto main = getMainFunction(moduleOp);
    // ensureBlocks(main);
    LoweringState state;
    addInitialize(main, ctx, &state);

    target.addIllegalDialect<QuartzDialect>();

    if (applyPartialConversion(moduleOp, target, std::move(mqtPatterns))
            .failed()) {
      signalPassFailure();
    }

    setAttributes(main, &state);

    // convert arith and cf to LLVM
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<cf::ControlFlowDialect>();

    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, stdPatterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);

    if (applyPartialConversion(moduleOp, target, std::move(stdPatterns))
            .failed()) {
      signalPassFailure();
    }

    PassManager passManager(ctx);
    passManager.addPass(createReconcileUnrealizedCastsPass());
    if (passManager.run(moduleOp).failed()) {
      signalPassFailure();
    }
  };
};

} // namespace mlir
