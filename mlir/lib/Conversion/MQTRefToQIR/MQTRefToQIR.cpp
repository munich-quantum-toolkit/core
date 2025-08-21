/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any ref gate operation to a llvm
// call operation that adheres to the QIR specification
#define ADD_CONVERT_PATTERN(gate)                                              \
  mqtPatterns.add<ConvertMQTRefGateOpQIR<ref::gate>>(typeConverter, context);

#include "mlir/Conversion/MQTRefToQIR/MQTRefToQIR.h"

#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

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

#define GEN_PASS_DEF_MQTREFTOQIR
#include "mlir/Conversion/MQTRefToQIR/MQTRefToQIR.h.inc"

namespace {
/**
   * @brief Look up the function declaration with a given name. If it does not
   exist create one and return it.
   *
   * @param rewriter The PatternRewriter to use.
   * @param op The operation that is matched.
   * @param fnName The name of the function.
   * @param fnType The type signature of the function.
   * @return The LLVM funcOp declaration with the requested name and signature.
   */
LLVM::LLVMFuncOp getFunctionDeclaration(PatternRewriter& rewriter,
                                        Operation* op, StringRef fnName,
                                        Type fnType) {
  // check if the function already exists
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnName));

  if (fnDecl == nullptr) {
    // if not create the declaration at the end of the module
    const PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(module.getBody());

    fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnName, fnType);

    // add the irreversible attribute to irreversible quantum operations
    if (fnName == "__quantum__qis__mz__body" ||
        fnName == "__quantum__rt__qubit_release_array" ||
        fnName == "__quantum__rt__qubit_release") {
      fnDecl->setAttr("passthrough",
                      rewriter.getStrArrayAttr({"irreversible"}));
    }
  }

  return static_cast<LLVM::LLVMFuncOp>(fnDecl);
}
struct LoweringState {
  // map a given index to a pointer value, to reuse the value instead of
  // creating a new one every time
  DenseMap<size_t, Value> ptrMap;
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

struct MQTRefToQIRTypeConverter final : public LLVMTypeConverter {
  explicit MQTRefToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // QubitType conversion
    addConversion([ctx](ref::QubitType /*type*/) {
      return LLVM::LLVMPointerType::get(ctx);
    });
    // QregType Conversion
    addConversion([ctx](ref::QubitRegisterType /*type*/) {
      return LLVM::LLVMPointerType::get(ctx);
    });
  }
};

struct ConvertMQTRefAllocQIR final : StatefulOpConversionPattern<ref::AllocOp> {
  using StatefulOpConversionPattern<ref::AllocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::AllocOp op, OpAdaptor adaptor,
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
    // get the int value of the operation and increase the number of required
    // qubits
    auto constantOp = size.getDefiningOp<LLVM::ConstantOp>();
    const auto value = dyn_cast<IntegerAttr>(constantOp.getValue()).getInt();
    getState().numQubits += value;
    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, size);

    getState().useDynamicQubit = true;

    return success();
  }
};

struct ConvertMQTRefQubitQIR final : StatefulOpConversionPattern<ref::QubitOp> {
  using StatefulOpConversionPattern<ref::QubitOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::QubitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    const auto index = adaptor.getIndex();
    // get a pointer to qubit
    if (getState().ptrMap.contains(index)) {
      // check if the pointer already exist, if yes reuse them
      rewriter.replaceOp(op, getState().ptrMap.at(index));
    } else {
      // otherwise create a constant operation and a intToPtr operation
      auto constantOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
      auto intToPtrOp = rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
          op, LLVM::LLVMPointerType::get(ctx), constantOp->getResult(0));

      // store them in the map to reuse them later
      getState().ptrMap.try_emplace(index, intToPtrOp->getResult(0));
    }

    // increase the number of required qubits
    getState().numQubits++;
    return success();
  }
};

struct ConvertMQTRefAllocQubitQIR final
    : StatefulOpConversionPattern<ref::AllocQubitOp> {
  using StatefulOpConversionPattern<
      ref::AllocQubitOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(const ref::AllocQubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_allocate";
    const auto qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx), {});

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, ValueRange{});

    // increase the number of required qubits
    getState().numQubits++;

    getState().useDynamicQubit = true;

    return success();
  }
};

struct ConvertMQTRefDeallocQubitQIR final
    : OpConversionPattern<ref::DeallocQubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(const ref::DeallocQubitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_release";
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

struct ConvertMQTRefDeallocQIR final : OpConversionPattern<ref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::DeallocOp op, OpAdaptor adaptor,
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

struct ConvertMQTRefResetQIR final : OpConversionPattern<ref::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::ResetOp op, OpAdaptor adaptor,
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
struct ConvertMQTRefExtractQIR final : OpConversionPattern<ref::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::ExtractOp op, OpAdaptor adaptor,
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

template <typename MQTRefGateOp>
struct ConvertMQTRefGateOpQIR final : OpConversionPattern<MQTRefGateOp> {
  using OpConversionPattern<MQTRefGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTRefGateOp op, typename MQTRefGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // get all the values
    const auto& params = adaptor.getParams();
    const auto& inQubits = adaptor.getInQubits();
    const auto& posCtrlQubits = adaptor.getPosCtrlInQubits();
    const auto& negCtrlQubits = adaptor.getNegCtrlInQubits();
    auto staticParams = op.getStaticParams()
                            ? DenseF64ArrayAttr::get(rewriter.getContext(),
                                                     *op.getStaticParams())
                            : DenseF64ArrayAttr{};
    auto paramMask = op.getParamsMask()
                         ? DenseBoolArrayAttr::get(rewriter.getContext(),
                                                   *op.getParamsMask())
                         : DenseBoolArrayAttr{};

    SmallVector<Value> newParams;
    // check for static parameters
    if (staticParams) {
      SmallVector<Value> staticParamValues;
      // set the insertionpoint to the beginning of the first block
      auto funcOp = op->template getParentOfType<LLVM::LLVMFuncOp>();
      auto& firstBlock = *(funcOp.getBlocks().begin());
      rewriter.setInsertionPointToStart(&firstBlock);

      // create constant operations for every static parameter
      auto createF64Const = [&](double v) {
        auto fAttr = rewriter.getF64FloatAttr(v);
        return rewriter.create<LLVM::ConstantOp>(op.getLoc(), fAttr);
      };
      staticParamValues = llvm::to_vector_of<Value>(
          llvm::map_range(staticParams.asArrayRef(),
                          [&](double v) { return createF64Const(v); }));

      // reset the insertionpoint after creating the constants
      rewriter.setInsertionPoint(op);

      // merge parameters with static parameters
      if (!paramMask) {
        // assuming every value is true when the paramMask does not exist but
        // staticParameters do
        newParams = staticParamValues;
      } else {
        auto staticParamIndex = 0;
        auto paramIndex = 0;
        // merge the parameter values depending on the paramMask
        for (auto isStaticParam : paramMask.asArrayRef()) {
          // push_back the values as Value is a wrapper
          if (isStaticParam) {
            newParams.push_back(staticParamValues[staticParamIndex++]);
          } else {
            newParams.push_back(params[paramIndex++]);
          }
        }
      }
    } else {
      newParams = params;
    }

    // concatenate all the values
    SmallVector<Value> operands;
    operands.reserve(newParams.size() + inQubits.size() + posCtrlQubits.size() +
                     negCtrlQubits.size());

    operands.append(posCtrlQubits.begin(), posCtrlQubits.end());
    operands.append(negCtrlQubits.begin(), negCtrlQubits.end());
    operands.append(inQubits.begin(), inQubits.end());
    operands.append(newParams.begin(), newParams.end());

    // check for negative controlled qubits
    // for any negative controlled qubits place an NOT operation before and
    // after the matched gate operation
    if (negCtrlQubits.size() > 0) {
      // create name and signature of the NOT operation
      const StringRef notGate = "__quantum__qis__x__body";
      const auto notGateSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

      // get the function declaration
      const auto notGateDecl =
          getFunctionDeclaration(rewriter, op, notGate, notGateSignature);

      // place a NOT operation before and after the operation for each negative
      // controlled qubit
      for (const auto negCtrlQubit : negCtrlQubits) {
        rewriter.setInsertionPoint(op);
        rewriter.create<LLVM::CallOp>(op.getLoc(), notGateDecl, negCtrlQubit);
        rewriter.setInsertionPointAfter(op);
        rewriter.create<LLVM::CallOp>(op.getLoc(), notGateDecl, negCtrlQubit);
      }
      // reset the insertionpoint
      rewriter.setInsertionPoint(op);
    }
    // get the types of the values
    const SmallVector<Type> types =
        llvm::to_vector(ValueRange(operands).getTypes());

    // get the name of the gate
    const StringRef name = op->getName().getStringRef().split('.').second;

    // add leading c's to the function name depending on the number of control
    // qubits
    const auto ctrQubitsCount = posCtrlQubits.size() + negCtrlQubits.size();
    std::string fnName(ctrQubitsCount, 'c');

    // check if it is a u gate
    if (name == "u") {
      fnName = ("__quantum__qis__" + fnName + "u3__body");
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

struct ConvertMQTRefMeasureQIR final
    : StatefulOpConversionPattern<ref::MeasureOp> {
  using StatefulOpConversionPattern<
      ref::MeasureOp>::StatefulOpConversionPattern;

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

    // create a new globalOp and a addressOfOp
    // set the insertionpoint to the beginning of the module
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(module.getBody());

    // check how many digits the next index has for the array allocation
    auto num = state.index;
    auto digits = 1;
    while (num >= 10) {
      num /= 10;
      ++digits;
    }

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

    // increment the index of the next addressOfOp
    state.index++;

    return addressOfOp;
  }

  LogicalResult
  matchAndRewrite(ref::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // match and replace the measure operation to the following operations
    //  call void @__quantum__qis__mz__body(ptr %qubit , ptr %result)
    //  call void @__quantum__rt__result_record_output(ptr %result , ptr
    //  @constant)
    StringRef fnName;
    LLVM::LLVMFunctionType qirSignature;
    LLVM::LLVMFuncOp fnDecl;
    auto const ptrType = LLVM::LLVMPointerType::get(ctx);
    auto& ptrMap = getState().ptrMap;

    Value resultValue = nullptr;
    // get a pointer to result
    if (ptrMap.contains(getState().numResults)) {
      // check if the pointer already exist, if yes reuse them
      resultValue = ptrMap.at(getState().numResults);
    } else {
      // otherwise create a constant operation and a intToPtr operation
      auto constantOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(
                           static_cast<int64_t>(getState().numResults)));
      resultValue = rewriter
                        .create<LLVM::IntToPtrOp>(op.getLoc(), ptrType,
                                                  constantOp->getResult(0))
                        .getResult();

      // store them in the map to reuse them later
      ptrMap.try_emplace(getState().numResults, resultValue);
    }
    // create the measure operation
    fnName = "__quantum__qis__mz__body";
    qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                               {ptrType, ptrType});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), fnDecl, ValueRange{adaptor.getInQubit(), resultValue});

    // create the record output operation
    fnName = "__quantum__rt__result_record_output";
    qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                               {ptrType, ptrType});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // get the pointer to the internal constant
    auto* addressOfOp = getAddressOfOp(op, rewriter, getState());

    // create record result output in the last block of the main function
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    rewriter.setInsertionPointToStart(&func.back());

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{resultValue, addressOfOp->getResult(0)});

    // erase the old operation
    rewriter.eraseOp(op);

    // increase the number of required results
    getState().numResults++;

    return success();
  }
};

struct MQTRefToQIR final : impl::MQTRefToQIRBase<MQTRefToQIR> {
  using MQTRefToQIRBase::MQTRefToQIRBase;

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
   * operation. The second block contains the reversible quantum operations. The
   * third block contains the irreversible quantum operations like the measure
   * operation or the dealloc operation. The final block contains the
   * return operation and the record output operation. The blocks are connected
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

    // create the main blocks and the endblock
    Block* mainBlock = builder.createBlock(&main.getBody());
    Block* irreversibleMainBlock = builder.createBlock(&main.getBody());
    Block* endBlock = builder.createBlock(&main.getBody());

    // move the returnOp from the entryBlock to the endBlock
    auto& entryOperations = entryBlock->getOperations();
    auto& endOperations = endBlock->getOperations();
    endOperations.splice(endOperations.end(), entryOperations,
                         std::prev(entryOperations.end()));

    // get all quantum operations until the measure operation
    SmallVector<Operation*> opsToMove;
    for (auto& op : entryOperations) {
      if (dyn_cast<ref::MeasureOp>(op)) {
        break;
      }
      opsToMove.emplace_back(&op);
    }
    // move them to the 2nd block
    for (Operation* op : opsToMove) {
      op->moveBefore(mainBlock, mainBlock->end());
    }

    // move the remaining operations to the third block
    if (!entryBlock->empty()) {
      irreversibleMainBlock->getOperations().splice(
          irreversibleMainBlock->begin(), entryOperations,
          entryOperations.begin(), entryOperations.end());
    }

    // add jump from entryBlock to mainBlock
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), mainBlock);

    // add jump from main to irreversibleMain
    builder.setInsertionPointToEnd(mainBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), irreversibleMainBlock);

    // add jump from irreversibleMain to endBlock
    builder.setInsertionPointToEnd(irreversibleMainBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), endBlock);
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
    auto module = main->getParentOfType<ModuleOp>();

    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());

    // find the zeroOp or create one
    Operation* zeroOperation = nullptr;
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
    // add the zero operation to the pointerMap
    state->ptrMap.try_emplace(0, zeroOperation->getResult(0));

    // create the initialize operation as the 2nd last operation in the first
    // block after all constant operations and before the last jump operation
    const auto insertPoint = std::prev(firstBlock.getOperations().end(), 1);
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
        builder.getStrArrayAttr({"dynamic_result_management", "false"}));

    main->setAttr("passthrough", builder.getArrayAttr(attributes));
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ConversionTarget target(*context);
    RewritePatternSet stdPatterns(context);
    RewritePatternSet mqtPatterns(context);
    MQTRefToQIRTypeConverter typeConverter(context);

    // transform the default dialects first
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
    LoweringState state;
    // add the initialize operation
    addInitialize(main, context, &state);

    target.addIllegalDialect<ref::MQTRefDialect>();
    mqtPatterns.add<ConvertMQTRefAllocQIR>(typeConverter, context, &state);
    mqtPatterns.add<ConvertMQTRefDeallocQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTRefAllocQubitQIR>(typeConverter, context, &state);
    mqtPatterns.add<ConvertMQTRefDeallocQubitQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTRefExtractQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTRefResetQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTRefQubitQIR>(typeConverter, context, &state);
    mqtPatterns.add<ConvertMQTRefMeasureQIR>(typeConverter, context, &state);

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
    ADD_CONVERT_PATTERN(XXminusYYOp)
    ADD_CONVERT_PATTERN(XXplusYYOp)

    if (failed(
            applyPartialConversion(module, target, std::move(mqtPatterns)))) {
      signalPassFailure();
    }
    setAttributes(main, &state);
  };
};

} // namespace mqt::ir
