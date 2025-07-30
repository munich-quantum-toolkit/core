/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to match and replace simple qubit gates
#define ADD_CONVERT_SIMPLE_GATE(gate)                                          \
  else if (gateName == #gate) {                                                \
    rewriter.replaceOpWithNewOp<dyn::gate>(op, DenseF64ArrayAttr{},            \
                                           DenseBoolArrayAttr{}, ValueRange{}, \
                                           qubits, ctrlQubits, ValueRange{});  \
  }
// macro to match and replace rotation gates
#define ADD_CONVERT_ROTATION_GATE(gate)                                        \
  else if (gateName == #gate) {                                                \
    rewriter.replaceOpWithNewOp<dyn::gate>(                                    \
        op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, rotationDegrees,        \
        operands, ctrlQubits, ValueRange{});                                   \
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
        {"x", "XOp"},
        {"not", "XOp"},
        {"h", "HOp"},
        {"i", "IOp"},
        {"y", "YOp"},
        {"z", "ZOp"},
        {"s", "SOp"},
        {"sdg", "SdgOp"},
        {"t", "TdgOp"},
        {"v", "VOp"},
        {"sx", "SXOp"},
        {"sxdg", "SXdgOp"},
        {"swap", "SWAPOp"},
        {"iswap", "iSWAPOp"},
        {"iswapdg", "iSWAPdgOp"},
        {"peres", "PeresOp"},
        {"peresdg", "PeresdgOp"},
        {"dcx", "DCXOp"},
        {"ecr", "ECROp"},
    };
    // check if it is a simple gate
    if (GATE_NAMES.find(name.str()) == GATE_NAMES.end()) {
      return false;
    }
    const auto gateName = GATE_NAMES.at(name.str());

    //  match and replace the fitting gate
    if (gateName == "XOp") {
      rewriter.replaceOpWithNewOp<dyn::XOp>(op, DenseF64ArrayAttr{},
                                            DenseBoolArrayAttr{}, ValueRange{},
                                            qubits, ctrlQubits, ValueRange{});
    }
    ADD_CONVERT_SIMPLE_GATE(HOp)
    ADD_CONVERT_SIMPLE_GATE(IOp)
    ADD_CONVERT_SIMPLE_GATE(YOp)
    ADD_CONVERT_SIMPLE_GATE(ZOp)
    ADD_CONVERT_SIMPLE_GATE(SOp)
    ADD_CONVERT_SIMPLE_GATE(SdgOp)
    ADD_CONVERT_SIMPLE_GATE(TOp)
    ADD_CONVERT_SIMPLE_GATE(TdgOp)
    ADD_CONVERT_SIMPLE_GATE(VOp)
    ADD_CONVERT_SIMPLE_GATE(SXOp)
    ADD_CONVERT_SIMPLE_GATE(SWAPOp)
    ADD_CONVERT_SIMPLE_GATE(iSWAPOp)
    ADD_CONVERT_SIMPLE_GATE(iSWAPdgOp)
    ADD_CONVERT_SIMPLE_GATE(PeresOp)
    ADD_CONVERT_SIMPLE_GATE(PeresdgOp)
    ADD_CONVERT_SIMPLE_GATE(DCXOp)
    ADD_CONVERT_SIMPLE_GATE(ECROp)
    return true;
  }

  // match and replace singlue qubit gates
  static bool convertRotationGates(SmallVector<Value>& operands,
                                   SmallVector<Value>& ctrlQubits,
                                   LLVM::CallOp& op,
                                   ConversionPatternRewriter& rewriter,
                                   StringRef& name) {
    static std::map<std::string, std::string> singleRotationGates = {
        {"p", "POp"},     {"rx", "RXOp"},   {"ry", "RYOp"},
        {"rz", "RZOp"},   {"rxx", "RXXOp"}, {"ryy", "RYYOp"},
        {"rzz", "RZZOp"}, {"rzx", "RZXOp"}

    };
    static std::map<std::string, std::string> doubleRotationGates = {
        {"u2", "U2Op"}, {"xxminusyy", "XXminusYY"}, {"XXPLUSYY", "XXplusYY"}

    };
    std::string gateName;
    size_t rotationCount = 0;
    // check if it is rotation gate and get the number of degrees
    if (singleRotationGates.find(name.str()) != singleRotationGates.end()) {
      gateName = singleRotationGates.at(name.str());

      rotationCount = 1;
    } else if (doubleRotationGates.find(name.str()) !=
               doubleRotationGates.end()) {
      gateName = doubleRotationGates.at(name.str());
      rotationCount = 2;
    } else if (name.str() == "u") {
      gateName = "UOp";
      rotationCount = 3;
    } else {
      return false;
    }

    // extract the degrees from the operand list
    SmallVector<Value> rotationDegrees;
    rotationDegrees.insert(
        rotationDegrees.end(), std::make_move_iterator(operands.begin()),
        std::make_move_iterator(operands.begin() + rotationCount));
    operands.erase(operands.begin(), operands.begin() + rotationCount);

    //  match and replace the fitting gate
    if (gateName == "POp") {
      rewriter.replaceOpWithNewOp<dyn::POp>(
          op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, rotationDegrees,
          operands, ctrlQubits, ValueRange{});
    }
    ADD_CONVERT_ROTATION_GATE(UOp)
    ADD_CONVERT_ROTATION_GATE(U2Op)
    ADD_CONVERT_ROTATION_GATE(RXOp)
    ADD_CONVERT_ROTATION_GATE(RYOp)
    ADD_CONVERT_ROTATION_GATE(RZOp)
    ADD_CONVERT_ROTATION_GATE(RXXOp)
    ADD_CONVERT_ROTATION_GATE(RYYOp)
    ADD_CONVERT_ROTATION_GATE(RZZOp)
    ADD_CONVERT_ROTATION_GATE(RZXOp)
    ADD_CONVERT_ROTATION_GATE(XXminusYY)
    ADD_CONVERT_ROTATION_GATE(XXplusYY)
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
    // get the name of the operation and prepare the return types
    const auto fnName = op.getCallee();
    const auto qubitType = dyn::QubitType::get(rewriter.getContext());
    const auto qregType = dyn::QubitRegisterType::get(rewriter.getContext());
    auto& map = *operandMap;
    const auto operands = adaptor.getOperands();

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

      // update the operand list
      operandMap->try_emplace(op->getResult(0), newOp->getResult(0));

    }
    // match alloc qubit
    else if (fnName == "__quantum__rt__qubit_allocate") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, allocOp->qReg, Value{},
          IntegerAttr::get(rewriter.getIntegerType(64), allocOp->index++));

      // update the operand list
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    }
    // match extract qubit from register
    else if (fnName == "__catalyst__rt__array_get_element_ptr_1d") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, newOperands);
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    }
    // match measure operation
    else if (fnName == "__quantum__qis__m__body") {
      SmallVector<Type> newBits(adaptor.getOperands().size(),
                                IntegerType::get(rewriter.getContext(), 1));
      const auto newOp =
          rewriter.replaceOpWithNewOp<dyn::MeasureOp>(op, newBits, newOperands);
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    }
    // match dealloc register
    else if (fnName == "__quantum__rt__qubit_release_array") {
      rewriter.replaceOpWithNewOp<dyn::DeallocOp>(op, newOperands.front());
    } else {
      // get the gate name and the number of control qubits
      auto gateName(fnName->substr(16).drop_back(6));
      const size_t ctrlQubitCount = countControlQubits(gateName);

      // remove the control qubits from the name
      gateName = gateName.substr(ctrlQubitCount);

      // extract the controlqubits from the operand list
      SmallVector<Value> ctrlQubits;
      ctrlQubits.reserve(ctrlQubitCount);
      ctrlQubits.insert(
          ctrlQubits.end(),
          std::make_move_iterator(newOperands.end() - ctrlQubitCount),
          std::make_move_iterator(newOperands.end()));
      newOperands.resize(newOperands.size() - ctrlQubitCount);

      // check and match simple gate operations
      if (convertSimpleGates(newOperands, ctrlQubits, op, rewriter, gateName)) {
        return success();
      }
      // check and match rotation gate operations
      if (convertRotationGates(newOperands, ctrlQubits, op, rewriter,
                               gateName)) {
        return success();
      }
      // otherwise erase the operation
      rewriter.eraseOp(op);
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
    const auto result = op->walk([&](Operation* op) {
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
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
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
      const auto insertPoint = std::prev(ops.end(), 1);
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

    patterns.add<ConvertQIRLoad>(typeConverter, context);
    patterns.add<ConvertQIRCall>(typeConverter, context, operandMap,
                                 registerInfo);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
