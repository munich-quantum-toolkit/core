/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any dyn gate operation to a llvm
// call operation that adheres to the qir specification
#define ADD_CONVERT_PATTERN(gate)                                              \
  mqtPatterns.add<ConvertMQTDynGateOpQIR<dyn::gate>>(typeConverter, context);

#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <utility>
namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTDYNTOQIR
#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h.inc"

namespace {
// add function declaration at the end if it does not exist already and
// return the function
LLVM::LLVMFuncOp getFunctionDeclaration(PatternRewriter& rewriter,
                                        Operation* op, StringRef fnName,
                                        Type fnType) {
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnName));

  if (fnDecl == nullptr) {
    const PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(module.getBody());

    fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnName, fnType);
  }

  return static_cast<LLVM::LLVMFuncOp>(fnDecl);
}
struct LoweringState {

  SmallVector<Operation*> measureConstants;
  Operation* constantOp{};
  size_t index{};
};

template <typename OpType>
class StatefulOpConversionPattern : public mlir::OpConversionPattern<OpType> {
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(mlir::TypeConverter& typeConverter,
                              mlir::MLIRContext* context, LoweringState* state)
      : mlir::OpConversionPattern<OpType>(typeConverter, context),
        state_(state) {}

  /// @brief Return the state object as reference.
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};
} // namespace

struct MQTDynToQIRTypeConverter final : public LLVMTypeConverter {
  explicit MQTDynToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // QubitType conversion
    addConversion([ctx](dyn::QubitType /*type*/) {
      return LLVM::LLVMPointerType::get(ctx);
    });
    // QregType Conversion
    addConversion([ctx](dyn::QubitRegisterType /*type*/) {
      return LLVM::LLVMPointerType::get(ctx);
    });
  }
};

struct ConvertMQTDynAllocQIR final : OpConversionPattern<dyn::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_allocate_array";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 64));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // create a constantOp if the size is an attribute
    auto size = adaptor.getSize();
    if (!size) {
      size =
          rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getSizeAttrAttr());
    }

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, size);
    return success();
  }
};
struct ConvertMQTDynDeallocQIR final : OpConversionPattern<dyn::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_release_array";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

struct ConvertMQTDynResetQIR final : OpConversionPattern<dyn::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__qis__reset__body";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};
struct ConvertMQTDynExtractQIR final : OpConversionPattern<dyn::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__array_get_element_ptr_1d";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 64)});

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // create a constantOp if the index is an attribute
    auto index = adaptor.getIndex();
    if (!index) {
      index =
          rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getIndexAttrAttr());
    }

    // create the new callOp
    const auto elemPtr =
        rewriter
            .create<LLVM::CallOp>(op.getLoc(), fnDecl,
                                  ValueRange{adaptor.getInQreg(), index})
            .getResult();

    // replace the old operation with a loadOp
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, LLVM::LLVMPointerType::get(ctx), elemPtr);

    return success();
  }
};

template <typename MQTDynGateOp>
struct ConvertMQTDynGateOpQIR final : OpConversionPattern<MQTDynGateOp> {
  using OpConversionPattern<MQTDynGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTDynGateOp op, typename MQTDynGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // get all the values
    const auto& params = adaptor.getParams();
    const auto& inQubits = adaptor.getInQubits();
    const auto& posCtrlQubits = adaptor.getPosCtrlInQubits();
    const auto& negCtrlQubits = adaptor.getNegCtrlInQubits();

    // concatenate all the types
    SmallVector<Type> types;
    types.reserve(params.size() + inQubits.size() + posCtrlQubits.size() +
                  negCtrlQubits.size());
    types.append(params.getTypes().begin(), params.getTypes().end());
    types.append(inQubits.getTypes().begin(), inQubits.getTypes().end());
    types.append(posCtrlQubits.getTypes().begin(),
                 posCtrlQubits.getTypes().end());
    types.append(negCtrlQubits.getTypes().begin(),
                 negCtrlQubits.getTypes().end());

    // concatenate all the values
    SmallVector<Value> operands;
    operands.reserve(params.size() + inQubits.size() + posCtrlQubits.size() +
                     negCtrlQubits.size());
    operands.append(params.begin(), params.end());
    operands.append(inQubits.begin(), inQubits.end());
    operands.append(posCtrlQubits.begin(), posCtrlQubits.end());
    operands.append(negCtrlQubits.begin(), negCtrlQubits.end());

    // get the name of the gate
    const StringRef name = op->getName().getStringRef().split('.').second;
    std::string fnName;

    // add leading c's depending on the number of control qubits
    const auto ctrQubitsCount = posCtrlQubits.size() + negCtrlQubits.size();
    fnName.insert(0, ctrQubitsCount, 'c');

    // check if it is a cnot gate
    if (name == "x" && ctrQubitsCount == 1) {
      fnName = ("__quantum__qis__" + fnName + "not__body");
    } else {
      fnName = ("__quantum__qis__" + fnName + name + "__body").str();
    }

    // create the signature of the function
    const auto qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types);

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with a callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);

    return success();
  }
};

struct ConvertMQTDynMeasureQIR final
    : StatefulOpConversionPattern<dyn::MeasureOp> {
  using StatefulOpConversionPattern<
      dyn::MeasureOp>::StatefulOpConversionPattern;

  /**
   * @brief returns the next addressOfOp for a global constant to store the
   * results of the measure operation.
   *
   * @param op The current measure operation that is converted.
   * @param rewriter The PatternRewriter to use.
   * @param state The LoweringState of the current conversion pass.
   * @return The addressOfOp of the next global constant.
   */
  static Operation* getAddressOfOp(Operation* op,
                                   ConversionPatternRewriter& rewriter,
                                   LoweringState& state) {
    Operation* addressOfOp = nullptr;
    // get the next addressOfOp from the vector if it exists
    if (!state.measureConstants.empty()) {
      addressOfOp = state.measureConstants.front();
      // pop the next addressOfOp from the vector
      state.measureConstants.erase(&state.measureConstants.front());
    }
    // otherwise create a new globalOp and a addressOfOp
    else {
      // check how many digits the next index has for the array allocation
      auto num = state.index;
      int64_t digits = 1;
      while (num >= 10) {
        num /= 10;
        ++digits;
      }
      // set the insertionpoint to the beginning of the module
      auto module = op->getParentOfType<ModuleOp>();
      rewriter.setInsertionPointToStart(module.getBody());

      // create the necessary names and types for the global operation
      // symbol name should be mlir.llvm.nameless_global_0,
      // mlir.llvm.nameless_global_1 etc.
      const auto symbolName = rewriter.getStringAttr(
          "mlir.llvm.nameless_global_" + std::to_string(state.index));
      const auto llvmArrayType =
          LLVM::LLVMArrayType::get(rewriter.getIntegerType(8), digits + 2);
      // initializer name should be r0\00, r1\00 etc.
      const auto stringInitializer =
          rewriter.getStringAttr("r" + std::to_string(state.index) + '\0');

      // create the global operation
      auto globalOp = rewriter.create<LLVM::GlobalOp>(
          op->getLoc(), llvmArrayType,
          /*isConstant=*/true, LLVM::Linkage::Internal, symbolName,
          stringInitializer);
      globalOp->setAttr("addr_space", rewriter.getI32IntegerAttr(0));
      globalOp->setAttr("dso_local", rewriter.getUnitAttr());

      // get the first block of the main function
      auto main = op->getParentOfType<LLVM::LLVMFuncOp>();
      auto& firstBlock = *(main.getBlocks().begin());

      // insert the addressOfOp of the newly created global op at the beginning
      // of the block
      rewriter.setInsertionPointToStart(&firstBlock);
      addressOfOp = rewriter.create<LLVM::AddressOfOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()),
          symbolName);

      // reset the insertionpoint to the initial operation again
      rewriter.setInsertionPoint(op);
    }
    // increment the index of the next addressOfOp
    state.index++;

    return addressOfOp;
  }

  LogicalResult
  matchAndRewrite(dyn::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    StringRef fnName;
    LLVM::LLVMFunctionType qirSignature;
    LLVM::LLVMFuncOp fnDecl;

    // create measure operation
    fnName = "__quantum__qis__m__body";
    qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx),
                                               LLVM::LLVMPointerType::get(ctx));
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), fnDecl,
                                               adaptor.getInQubit());

    // create record result output
    fnName = "__quantum__rt__result_record_output";
    qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    Operation* addressOfOp = getAddressOfOp(op, rewriter, getState());
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{newOp->getResult(0), addressOfOp->getResult(0)});

    // create record update reference count
    fnName = "__quantum__rt__result_update_reference_count";
    qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 32)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{newOp->getResult(0), getState().constantOp->getResult(0)});

    // create read result op and replace the old result with new result
    fnName = "__quantum__rt__read_result";
    qirSignature = LLVM::LLVMFunctionType::get(
        IntegerType::get(ctx, 1), {LLVM::LLVMPointerType::get(ctx)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    auto resultOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), fnDecl, ValueRange{newOp->getResult(0)});
    op->replaceUsesOfWith(op->getResult(0), resultOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

struct MQTDynToQIR final : impl::MQTDynToQIRBase<MQTDynToQIR> {
  using MQTDynToQIRBase::MQTDynToQIRBase;

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
   * @brief Makes sure that the different blocks for the base profile of QIR
   * exist
   *
   * The first block should only contain constant operations for the initialize
   * operation. The second block contains all the quantum operation. The final
   * block should only contain the return operation. The blocks are connected
   * with an unconditional jump operation to the next block.
   *
   * @param main The main function of the module.
   */
  static void ensureBlocks(LLVM::LLVMFuncOp& main) {
    // return if there are more blocks already
    if (main.getBlocks().size() > 1) {
      return;
    }
    // get the existing block
    auto* entryBlock = &main.front();
    OpBuilder builder(main.getBody());

    // create the main and the endblock
    Block* mainBlock = builder.createBlock(&main.getBody());
    Block* endBlock = builder.createBlock(&main.getBody());

    // add jump from main to endBlock
    builder.setInsertionPointToEnd(mainBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), endBlock);

    // move the returnOp from the entryBlock to the endBlock
    builder.setInsertionPointToEnd(endBlock);
    auto& entryOperations = entryBlock->getOperations();
    auto& endOperations = endBlock->getOperations();
    auto lastOperation = std::prev(entryOperations.end());
    endOperations.splice(endOperations.end(), entryOperations, lastOperation);

    // add jump from entryBlock to mainBlock
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), mainBlock);

    // move every operation from the entry block except the jump operation to
    // the main block
    if (!entryBlock->empty()) {
      mainBlock->getOperations().splice(mainBlock->begin(), entryOperations,
                                        entryOperations.begin(),
                                        std::prev(entryOperations.end()));
    }
  }

  /**
   * @brief Collects the existing operations for the measure operation
   * conversion
   *
   * @param main The main function of the module.
   * @param operations The vector to store the found addressOf operations.
   * @param ctx The context of the module.
   * @return The LLVM constant operation with the value -1.
   */
  static Operation* collectMeasureConstants(LLVM::LLVMFuncOp& main,
                                            SmallVector<Operation*>& operations,
                                            MLIRContext* ctx) {
    // get the first block in the main function
    auto& firstBlock = *(main.getBlocks().begin());
    Operation* result = nullptr;
    // walk through the block and collect all addressOfOp and get the -1
    // constant value
    firstBlock.walk([&](Operation* op) {
      if (auto addressOfOp = dyn_cast<LLVM::AddressOfOp>(op)) {
        operations.emplace_back(addressOfOp);
      }

      if (auto constantOp = dyn_cast<LLVM::ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
          // check if the value is -1
          if (intAttr.getInt() == -1) {
            result = op;
          }
        }
      }
    });
    // create constant -1 value if there is none
    if (result == nullptr) {
      OpBuilder builder(main.getBody());
      builder.setInsertionPointToStart(&firstBlock);
      result = builder.create<LLVM::ConstantOp>(main->getLoc(),
                                                IntegerType::get(ctx, 32),
                                                builder.getI32IntegerAttr(-1));
    }
    return result;
  }

  /**
   * @brief Adds the initialize operation to the first block of the main
   * function.
   *
   * @param main The main function of the module.
   * @param ctx The context of the module.
   */
  static void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx) {
    auto module = main->getParentOfType<ModuleOp>();

    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());
    Operation* zeroOperation = nullptr;

    // find the zeroOp or create one
    firstBlock.walk([&](Operation* op) {
      if (auto zeroOp = dyn_cast<LLVM::ZeroOp>(op)) {
        zeroOperation = zeroOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (zeroOperation == nullptr) {
      builder.setInsertionPointToStart(&firstBlock);
      zeroOperation = builder.create<LLVM::ZeroOp>(
          main->getLoc(), LLVM::LLVMPointerType::get(ctx));
    }

    // create the initialize operation as the 2nd last operation in the first
    // block after all constant operations and before the last jump operation
    auto& ops = firstBlock.getOperations();
    const auto insertPoint = std::prev(ops.end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // get the function declaration of initialize otherwise create one
    const StringRef fnName = "__quantum__rt__initialize";
    auto* fnDecl = SymbolTable::lookupNearestSymbolFrom(
        main, builder.getStringAttr(fnName));
    if (fnDecl == nullptr) {
      const PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      auto fnSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
      fnDecl =
          builder.create<LLVM::LLVMFuncOp>(main->getLoc(), fnName, fnSignature);
    }
    // create and insert the initialize operation
    builder.create<LLVM::CallOp>(main->getLoc(),
                                 static_cast<LLVM::LLVMFuncOp>(fnDecl),
                                 ValueRange{zeroOperation->getResult(0)});
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ConversionTarget target(*context);
    RewritePatternSet stdPatterns(context);
    RewritePatternSet mqtPatterns(context);
    MQTDynToQIRTypeConverter typeConverter(context);

    // transform the default dialects first
    // maybe need to add more?
    target.addLegalDialect<LLVM::LLVMDialect>();
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, stdPatterns);
    populateFuncToLLVMConversionPatterns(typeConverter, stdPatterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<cf::ControlFlowDialect>();
    if (failed(
            applyPartialConversion(module, target, std::move(stdPatterns)))) {
      signalPassFailure();
    }

    // get the main function of the module
    auto main = getMainFunction(module);
    // make sure that the blocks for the QIR base profile exist
    ensureBlocks(main);
    // get the operations for measure conversions and store them in the
    // loweringstate struct
    LoweringState state;
    state.constantOp =
        collectMeasureConstants(main, state.measureConstants, context);
    // add the initialize operation
    addInitialize(main, context);

    target.addIllegalDialect<dyn::MQTDynDialect>();
    mqtPatterns.add<ConvertMQTDynAllocQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynDeallocQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynExtractQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynResetQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynMeasureQIR>(typeConverter, context, &state);

    ADD_CONVERT_PATTERN(GPhaseOp)
    ADD_CONVERT_PATTERN(IOp)
    ADD_CONVERT_PATTERN(BarrierOp)
    ADD_CONVERT_PATTERN(HOp)
    ADD_CONVERT_PATTERN(XOp)
    ADD_CONVERT_PATTERN(YOp)
    ADD_CONVERT_PATTERN(ZOp)
    ADD_CONVERT_PATTERN(SOp)
    ADD_CONVERT_PATTERN(SdgOp)
    ADD_CONVERT_PATTERN(TOp)
    ADD_CONVERT_PATTERN(TdgOp)
    ADD_CONVERT_PATTERN(VOp)
    ADD_CONVERT_PATTERN(VdgOp)
    ADD_CONVERT_PATTERN(UOp)
    ADD_CONVERT_PATTERN(U2Op)
    ADD_CONVERT_PATTERN(POp)
    ADD_CONVERT_PATTERN(SXOp)
    ADD_CONVERT_PATTERN(SXdgOp)
    ADD_CONVERT_PATTERN(RXOp)
    ADD_CONVERT_PATTERN(RYOp)
    ADD_CONVERT_PATTERN(RZOp)
    ADD_CONVERT_PATTERN(SWAPOp)
    ADD_CONVERT_PATTERN(iSWAPOp)
    ADD_CONVERT_PATTERN(iSWAPdgOp)
    ADD_CONVERT_PATTERN(PeresOp)
    ADD_CONVERT_PATTERN(PeresdgOp)
    ADD_CONVERT_PATTERN(DCXOp)
    ADD_CONVERT_PATTERN(ECROp)
    ADD_CONVERT_PATTERN(RXXOp)
    ADD_CONVERT_PATTERN(RYYOp)
    ADD_CONVERT_PATTERN(RZZOp)
    ADD_CONVERT_PATTERN(RZXOp)
    ADD_CONVERT_PATTERN(XXminusYY)
    ADD_CONVERT_PATTERN(XXplusYY)

    if (failed(
            applyPartialConversion(module, target, std::move(mqtPatterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
