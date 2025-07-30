/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to match and replace single qubit gates
#define ADD_CONVERT_SIMPLE_GATE(gate)                                          \
  else if (gateName == #gate) {                                                \
    rewriter.replaceOpWithNewOp<dyn::gate##Op>(                                \
        op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, ValueRange{}, qubits,   \
        ctrlQubits, ValueRange{});                                             \
  }

#include "mlir/Conversion/QIRToMQTDyn/QIRToMQTDyn.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <cctype>
#include <iterator>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_QIRTOMQTDYN
#include "mlir/Conversion/QIRToMQTDyn/QIRToMQTDyn.h.inc"

class QIRToMQTDynTypeConverter final : public TypeConverter {
public:
  explicit QIRToMQTDynTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });
    addConversion(
        [](LLVM::LLVMPointerType type, SmallVectorImpl<Type>& results) {
          results.push_back(type);
          return success();
        });
  }
};
namespace {
// struct to store information for alloc register and qubits
struct AllocRegister {
  // dyn register
  mqt::ir::dyn::AllocOp qReg;
  // next index
  int64_t index{};
};
} // namespace

struct ConvertQIRLoad final : OpConversionPattern<LLVM::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // erase the operation and use the operands as results
    rewriter.replaceOp(op, adaptor.getOperands().front());
    return success();
  }
};
struct ConvertQIRCall final : OpConversionPattern<LLVM::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  llvm::DenseMap<Value, Value>* operandMap;
  AllocRegister* allocOp;
  explicit ConvertQIRCall(TypeConverter& typeConverter, MLIRContext* context,
                          llvm::DenseMap<Value, Value>& operandMap,
                          AllocRegister& allocOp)
      : OpConversionPattern(typeConverter, context), operandMap(&operandMap),
        allocOp(&allocOp) {}

  // match and replace simple qubit gates
  static bool convertSimpleGates(SmallVector<Value>& qubits,
                                 SmallVector<Value>& ctrlQubits,
                                 LLVM::CallOp& op,
                                 ConversionPatternRewriter& rewriter,
                                 StringRef& name) {
    static const std::map<std::string, std::string> GATE_NAMES = {
        {"x", "X"},
        {"not", "X"},
        {"h", "H"},
        {"i", "I"},
        {"y", "Y"},
        {"z", "Z"},
        {"s", "S"},
        {"sdg", "Sdg"},
        {"t", "Tdg"},
        {"v", "V"},
        {"sx", "SX"},
        {"sxdg", "SXdg"},
        {"swap", "SWAP"},
        {"iswap", "iSWAP"},
        {"iswapdg", "iSWAPdg"},
        {"peres", "Peres"},
        {"peresdg", "Peresdg"},
        {"dcx", "DCX"},
        {"ecr", "ECR"},
    };
    if (GATE_NAMES.find(name.str()) == GATE_NAMES.end()) {
      return false;
    }
    const auto gateName = GATE_NAMES.at(name.str());
    // swap iswap iswapdg peres peresdg dcx ecr
    //  match and replace the fitting gate
    if (gateName == "X" || gateName == "Not") {
      rewriter.replaceOpWithNewOp<dyn::XOp>(op, DenseF64ArrayAttr{},
                                            DenseBoolArrayAttr{}, ValueRange{},
                                            qubits, ctrlQubits, ValueRange{});
    }
    ADD_CONVERT_SIMPLE_GATE(H)
    ADD_CONVERT_SIMPLE_GATE(I)
    ADD_CONVERT_SIMPLE_GATE(Y)
    ADD_CONVERT_SIMPLE_GATE(Z)
    ADD_CONVERT_SIMPLE_GATE(S)
    ADD_CONVERT_SIMPLE_GATE(Sdg)
    ADD_CONVERT_SIMPLE_GATE(T)
    ADD_CONVERT_SIMPLE_GATE(Tdg)
    ADD_CONVERT_SIMPLE_GATE(V)
    ADD_CONVERT_SIMPLE_GATE(SX)
    ADD_CONVERT_SIMPLE_GATE(SWAP)
    ADD_CONVERT_SIMPLE_GATE(iSWAP)
    ADD_CONVERT_SIMPLE_GATE(iSWAPdg)
    ADD_CONVERT_SIMPLE_GATE(Peres)
    ADD_CONVERT_SIMPLE_GATE(Peresdg)
    ADD_CONVERT_SIMPLE_GATE(DCX)
    ADD_CONVERT_SIMPLE_GATE(ECR)
    return true;
  }
  // count how many control qubits are used
  static size_t countControlQubits(const StringRef& str) {
    size_t count = 0;
    for (char ch : str) {
      if (ch == 'c') {
        ++count;
      } else {
        break;
      }
    }
    return count;
  }

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto fnName = op.getCallee();
    auto qubitType = dyn::QubitType::get(rewriter.getContext());
    auto qregType = dyn::QubitRegisterType::get(rewriter.getContext());
    auto& map = *operandMap;
    auto operands = adaptor.getOperands();

    // get the new operands
    SmallVector<Value> newOperands;
    newOperands.reserve(operands.size());
    for (auto const& val : operands) {
      if (map.contains(val)) {
        newOperands.emplace_back(map[val]);
      } else {
        newOperands.emplace_back(val);
      }
    }

    // match alloc register
    if (fnName == "__quantum__rt__qubit_allocate_array") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::AllocOp>(
          op, qregType, adaptor.getOperands());
      operandMap->try_emplace(op->getResult(0), newOp->getResult(0));
    }
    // match alloc qubit
    else if (fnName == "__quantum__rt__qubit_allocate") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, allocOp->qReg, Value{},
          IntegerAttr::get(rewriter.getIntegerType(64), allocOp->index++));
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));

    }
    // match extract qubit from register
    else if (fnName == "__catalyst__rt__array_get_element_ptr_1d") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, newOperands);
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    } else if (fnName == "__quantum__qis__m__body") {
      SmallVector<Type> newBits(adaptor.getOperands().size(),
                                IntegerType::get(rewriter.getContext(), 1));
      auto newOp =
          rewriter.replaceOpWithNewOp<dyn::MeasureOp>(op, newBits, newOperands);
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    }
    // match dealloc register
    else if (fnName == "__quantum__rt__qubit_release_array") {
      rewriter.replaceOpWithNewOp<dyn::DeallocOp>(op, newOperands.front());
    }
    // get the gate name and the number of control qubits
    auto gateName(fnName->substr(16).drop_back(6));
    const size_t ctrlQubitCount = countControlQubits(gateName);
    gateName = gateName.substr(ctrlQubitCount);

    // extract the controlqubits from the operand list
    SmallVector<Value> ctrlQubits;
    ctrlQubits.reserve(ctrlQubitCount);
    ctrlQubits.insert(
        ctrlQubits.end(),
        std::make_move_iterator(newOperands.end() - ctrlQubitCount),
        std::make_move_iterator(newOperands.end()));
    newOperands.resize(newOperands.size() - ctrlQubitCount);

    if (convertSimpleGates(newOperands, ctrlQubits, op, rewriter, gateName)) {
      return success();
    }

    return success();
  }
};

struct ConvertQIRFunc final : OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // erase the operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct QIRToMQTDyn final : impl::QIRToMQTDynBase<QIRToMQTDyn> {
  using QIRToMQTDynBase::QIRToMQTDynBase;

  // create and return an alloc register if it does not exist already
  static std::optional<mqt::ir::dyn::AllocOp>
  ensureRegisterAllocation(Operation* op) {
    int64_t num = 0;
    // walk through the IR and count all qubit allocate operations
    auto result = op->walk([&](Operation* op) {
      if (auto call = llvm::dyn_cast<LLVM::CallOp>(op)) {
        auto name = call.getCallee();

        // stop if there is an allocate array operation
        if (name == "__quantum__rt__qubit_allocate_array") {
          return WalkResult::interrupt();
        }
        if (name == "__quantum__rt__qubit_allocate") {
          num++;
        }
      }
      return mlir::WalkResult::advance();
    });
    // if there was no alloc register registration, create one
    if (!result.wasInterrupted() && num != 0) {
      // find the 2nd block of the main function
      auto module = llvm::dyn_cast<ModuleOp>(op);
      auto func = module.lookupSymbol<func::FuncOp>("main");
      auto& secondBlock = *(++func.getBlocks().begin());
      OpBuilder builder(func.getBody());

      // create the alloc register operation at the start of the block
      builder.setInsertionPointToStart(&secondBlock);
      auto allocOp = builder.create<mqt::ir::dyn::AllocOp>(
          func->getLoc(),
          mqt::ir::dyn::QubitRegisterType::get(module->getContext()), Value{},
          builder.getI64IntegerAttr(num));

      // create the dealloc operation at the end of the block
      auto& ops = secondBlock.getOperations();
      auto insertPoint = std::prev(ops.end(), 1);
      builder.setInsertionPoint(&*insertPoint);
      builder.create<dyn::DeallocOp>(func->getLoc(), allocOp);

      return allocOp;
    }
    // otherwise return null
    return std::nullopt;
  }
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    // map each initial dyn qubit to its latest opt qubit
    llvm::DenseMap<Value, Value> operandMap;

    auto alloc = ensureRegisterAllocation(module).value_or(nullptr);

    AllocRegister registerInfo{.qReg = alloc, .index = 0};
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QIRToMQTDynTypeConverter typeConverter(context);
    target.addLegalDialect<dyn::MQTDynDialect>();
    target.addIllegalOp<LLVM::CallOp>();
    target.addIllegalOp<LLVM::LoadOp>();
    target.addLegalOp<LLVM::ConstantOp>();
    target.addLegalOp<LLVM::GlobalOp>();
    target.addLegalOp<LLVM::AddressOfOp>();
    target.addLegalOp<LLVM::LLVMFuncOp>();
    target.addLegalOp<LLVM::ZeroOp>();
    // patterns.add<ConvertQIRFunc>(typeConverter, context);

    patterns.add<ConvertQIRLoad>(typeConverter, context);
    patterns.add<ConvertQIRCall>(typeConverter, context, operandMap,
                                 registerInfo);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
