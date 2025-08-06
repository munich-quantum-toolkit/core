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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <llvm/Support/Casting.h>
#include <map>
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
#include <optional>
#include <string>
#include <utility>
namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_QIRTOMQTDYN
#include "mlir/Conversion/QIRToMQTDyn/QIRToMQTDyn.h.inc"

class QIRToMQTDynTypeConverter final : public TypeConverter {
public:
  explicit QIRToMQTDynTypeConverter(MLIRContext* /*ctx*/) {
    // Identity conversion
    addConversion([](Type type) { return type; });
  }
};
namespace {
// struct to store information for alloc register and qubits
struct AllocRegisterData {
  // dyn register
  dyn::AllocOp qReg;
  // next free index
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
  AllocRegisterData* allocOp;
  explicit ConvertQIRCall(TypeConverter& typeConverter, MLIRContext* context,
                          llvm::DenseMap<Value, Value>& operandMap,
                          AllocRegisterData& allocOp)
      : OpConversionPattern(typeConverter, context), operandMap(&operandMap),
        allocOp(&allocOp) {}

  // match and replace simple qubit gates
  static bool convertSimpleGates(const SmallVector<Value>& qubits,
                                 const SmallVector<Value>& ctrlQubits,
                                 LLVM::CallOp& op,
                                 ConversionPatternRewriter& rewriter,
                                 const StringRef& name) {
    static const std::map<std::string, std::string> GATE_NAMES = {
        {"x", "XOp"},
        {"not", "XOp"},
        {"h", "HOp"},
        {"i", "IOp"},
        {"y", "YOp"},
        {"z", "ZOp"},
        {"s", "SOp"},
        {"sdg", "SdgOp"},
        {"t", "TOp"},
        {"tdg", "TdgOp"},
        {"v", "VOp"},
        {"vdg", "VdgOp"},
        {"sx", "SXOp"},
        {"sxdg", "SXdgOp"},
        {"swap", "SWAPOp"},
        {"iswap", "iSWAPOp"},
        {"iswapdg", "iSWAPdgOp"},
        {"peres", "PeresOp"},
        {"peresdg", "PeresdgOp"},
        {"dcx", "DCXOp"},
        {"ecr", "ECROp"}};
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
    ADD_CONVERT_SIMPLE_GATE(VdgOp)
    ADD_CONVERT_SIMPLE_GATE(SXOp)
    ADD_CONVERT_SIMPLE_GATE(SXdgOp)
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
                                   const SmallVector<Value>& ctrlQubits,
                                   LLVM::CallOp& op,
                                   ConversionPatternRewriter& rewriter,
                                   const StringRef& name) {
    static const std::map<std::string, std::string> SINGLE_ROTATION_GATES = {
        {"p", "POp"},     {"rx", "RXOp"},   {"ry", "RYOp"},
        {"rz", "RZOp"},   {"rxx", "RXXOp"}, {"ryy", "RYYOp"},
        {"rzz", "RZZOp"}, {"rzx", "RZXOp"}

    };
    static const std::map<std::string, std::string> DOUBLE_ROTATION_GATES = {
        {"u2", "U2Op"}, {"xxminusyy", "XXminusYY"}, {"xxplusyy", "XXplusYY"}

    };
    std::string gateName;
    size_t rotationCount = 0;
    // check if it is rotation gate and get the number of degrees
    if (SINGLE_ROTATION_GATES.find(name.str()) != SINGLE_ROTATION_GATES.end()) {
      gateName = SINGLE_ROTATION_GATES.at(name.str());
      rotationCount = 1;
    } else if (DOUBLE_ROTATION_GATES.find(name.str()) !=
               DOUBLE_ROTATION_GATES.end()) {
      gateName = DOUBLE_ROTATION_GATES.at(name.str());
      rotationCount = 2;
    } else if (name.str() == "u") {
      gateName = "UOp";
      rotationCount = 3;
    } else {
      return false;
    }
    // reminder: not matched u1 and u3 gates
    // u3 == UOp?

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

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get the name of the operation and prepare the return types
    const auto fnName = op.getCallee();
    const auto qubitType = dyn::QubitType::get(rewriter.getContext());
    const auto qregType = dyn::QubitRegisterType::get(rewriter.getContext());
    auto& operandMapRef = *operandMap;
    const auto operands = adaptor.getOperands();

    // get the new operands from the operandMap
    // workaround to avoid unrealized conversion; there is probably a better
    // solution
    SmallVector<Value> newOperands;
    newOperands.reserve(operands.size());
    for (auto const& val : operands) {
      if (operandMapRef.contains(val)) {
        newOperands.emplace_back(operandMapRef[val]);
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
    // match alloc qubit using the given allocOp and the index
    else if (fnName == "__quantum__rt__qubit_allocate") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, allocOp->qReg, Value{},
          IntegerAttr::get(rewriter.getIntegerType(64), allocOp->index++));

      // update the operand list
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    }
    // match extract qubit from register
    else if (fnName == "__quantum__rt__array_get_element_ptr_1d") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, newOperands);
      operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
    }
    // match measure operation
    else if (fnName == "__quantum__qis__m__body") {
      const SmallVector<Type> newBits(
          adaptor.getOperands().size(),
          IntegerType::get(rewriter.getContext(), 1));

      bool foundUser = false;
      for (auto* user : op->getUsers()) {
        if (auto callOp = dyn_cast<LLVM::CallOp>(user)) {
          if (callOp.getCallee() == "__quantum__rt__read_result") {
            const auto newOp = rewriter.replaceOpWithNewOp<dyn::MeasureOp>(
                callOp, newBits, newOperands);
            operandMap->try_emplace(callOp->getResult(0),
                                    newOp->getOpResult(0));
            foundUser = true;
            rewriter.eraseOp(op);
            break;
          }
        }
      }
      if (!foundUser) {
        const auto newOp = rewriter.replaceOpWithNewOp<dyn::MeasureOp>(
            op, newBits, newOperands);
        operandMap->try_emplace(op->getResult(0), newOp->getOpResult(0));
      }
    }
    // match dealloc register
    else if (fnName == "__quantum__rt__qubit_release_array") {
      rewriter.replaceOpWithNewOp<dyn::DeallocOp>(op, newOperands.front());
    } else {
      // remove the prefix and the suffix of the gate name
      auto gateName(fnName->substr(16).drop_back(6));

      // check how many control qubits are used by counting the number of
      // leading c's
      const size_t ctrlQubitCount =
          std::ranges::find_if(gateName, [](char ch) { return ch != 'c'; }) -
          gateName.begin();

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

      // try to match and replace gate operations
      if (gateName == "gphase") {
        rewriter.replaceOpWithNewOp<dyn::GPhaseOp>(
            op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, newOperands,
            ValueRange{}, ctrlQubits, ValueRange{});
        return success();
      }
      if (gateName == "barrier") {
        rewriter.replaceOpWithNewOp<dyn::BarrierOp>(
            op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, ValueRange{},
            newOperands, ctrlQubits, ValueRange{});
        return success();
      }
      if (convertSimpleGates(newOperands, ctrlQubits, op, rewriter, gateName) ||
          convertRotationGates(newOperands, ctrlQubits, op, rewriter,
                               gateName)) {
        return success();
      }

      // otherwise erase the operation
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct QIRToMQTDyn final : impl::QIRToMQTDynBase<QIRToMQTDyn> {
  using QIRToMQTDynBase::QIRToMQTDynBase;

  // create and return an alloc register if it does not exist already
  static std::optional<dyn::AllocOp> ensureRegisterAllocation(Operation* op) {
    int64_t requiredQubits = 0;
    auto module = llvm::dyn_cast<ModuleOp>(op);

    // walk through the IR and count all qubit allocate operations
    const auto result = module->walk([&](Operation* op) {
      if (auto call = llvm::dyn_cast<LLVM::CallOp>(op)) {
        auto name = call.getCallee();

        // stop if there is an allocate array operation
        if (name == "__quantum__rt__qubit_allocate_array") {
          return WalkResult::interrupt();
        }
        // increment the counter for each allocate qubit operation
        if (name == "__quantum__rt__qubit_allocate") {
          requiredQubits++;
        }
      }
      return WalkResult::advance();
    });
    // if there was no alloc register registration, create one
    if (!result.wasInterrupted() && requiredQubits != 0) {

      LLVM::LLVMFuncOp main;
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
          main = funcOp;
          break;
        }
      }

      // get the 2nd block of the main function
      auto& secondBlock = *(++main.getBlocks().begin());
      OpBuilder builder(main.getBody());

      // create the alloc register operation at the start of the block
      builder.setInsertionPointToStart(&secondBlock);
      auto allocOp = builder.create<dyn::AllocOp>(
          main->getLoc(), dyn::QubitRegisterType::get(module->getContext()),
          Value{}, builder.getI64IntegerAttr(requiredQubits));

      // create the dealloc operation at the end of the block
      auto& ops = secondBlock.getOperations();
      const auto insertPoint = std::prev(ops.end(), 1);
      builder.setInsertionPoint(&*insertPoint);
      builder.create<dyn::DeallocOp>(main->getLoc(), allocOp);
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
    // create an allocOp if necessary
    auto alloc = ensureRegisterAllocation(module).value_or(nullptr);
    AllocRegisterData registerInfo{.qReg = alloc, .index = 0};

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QIRToMQTDynTypeConverter typeConverter(context);
    target.addLegalDialect<dyn::MQTDynDialect>();

    // only convert the call and load operations for now
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
