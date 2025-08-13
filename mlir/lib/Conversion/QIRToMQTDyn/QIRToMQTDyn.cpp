/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to check the gatename and to replace the simple gate
#define ADD_CONVERT_SIMPLE_GATE(gate)                                          \
  else if (gateName == #gate) {                                                \
    rewriter.replaceOpWithNewOp<dyn::gate>(op, DenseF64ArrayAttr{},            \
                                           DenseBoolArrayAttr{}, ValueRange{}, \
                                           qubits, ctrlQubits, ValueRange{});  \
  }

// macro to check the gateName and to replace the rotation gate
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
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
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
#include <unordered_map>
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

struct LoweringState {
  // map each llvm qir result to the new mqtdyn result
  DenseMap<Value, Value> operandMap;
  // the newly created alloc register if it does not exist already, otherwise
  // this is a nullptr and will not be used
  dyn::AllocOp qReg;
  // next free allocate index
  int64_t index{};
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

  inline static const std::unordered_map<std::string, std::string>
      SINGLE_ROTATION_GATES = {
          {"p", "POp"},     {"rx", "RXOp"},   {"ry", "RYOp"},
          {"rz", "RZOp"},   {"rxx", "RXXOp"}, {"ryy", "RYYOp"},
          {"rzz", "RZZOp"}, {"rzx", "RZXOp"}, {"u1", "POp"}};

  inline static const std::unordered_map<std::string, std::string>
      SIMPLE_GATES = {{"x", "XOp"},
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

  inline static const std::unordered_map<std::string, std::string>
      DOUBLE_ROTATION_GATES = {
          {"u2", "U2Op"}, {"xxminusyy", "XXminusYY"}, {"xxplusyy", "XXplusYY"}};

private:
  LoweringState* state_;
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
struct ConvertQIRCall final : StatefulOpConversionPattern<LLVM::CallOp> {
  using StatefulOpConversionPattern<LLVM::CallOp>::StatefulOpConversionPattern;

  /**
   * @brief Replaces the call operation with a matching simple gate operation
   * from the mqtdyn dialect
   *
   * @param op The call operation that is replaced.
   * @param gateName The name of the gate.
   * @param qubits The Qubits of the given operation.
   * @param ctrlQubits The control Qubits of the given operation.
   * @param rewriter The PatternRewriter to use.
   */
  static void convertSimpleGate(LLVM::CallOp& op, const std::string& gateName,
                                const SmallVector<Value>& qubits,
                                const SmallVector<Value>& ctrlQubits,
                                ConversionPatternRewriter& rewriter) {

    // match and replace the fitting gate
    // macro contains the else if statements to check the gateName in addition
    // to the replacement, therefore the first operation in the chain is
    // manually replaced
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
  }

  /**
   * @brief Replaces the call operation with a matching rotation gate operation
   * from the mqtdyn dialect
   *
   * @param op The call operation that is replaced.
   * @param gateName The name of the gate.
   * @param operands The operands of the given operation.
   * @param ctrlQubits The control Qubits of the given operation.
   * @param rotationCount The number of rotation degrees.
   * @param rewriter The PatternRewriter to use.
   */
  static void convertRotationGate(LLVM::CallOp& op, const std::string& gateName,
                                  SmallVector<Value>& operands,
                                  const SmallVector<Value>& ctrlQubits,
                                  const size_t rotationCount,
                                  ConversionPatternRewriter& rewriter

  ) {

    // extract the degrees from the operand list
    SmallVector<Value> rotationDegrees;
    rotationDegrees.insert(
        rotationDegrees.end(), std::make_move_iterator(operands.begin()),
        std::make_move_iterator(operands.begin() + rotationCount));
    operands.erase(operands.begin(), operands.begin() + rotationCount);

    //  match and replace the fitting gate
    // macro contains the else if statements to check the gateName in addition
    // to the replacement, therefore the first operation in the chain is
    // manually replaced
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
  }

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get the name of the operation and prepare the return types
    const auto fnName = op.getCallee();
    const auto qubitType = dyn::QubitType::get(rewriter.getContext());
    const auto qregType = dyn::QubitRegisterType::get(rewriter.getContext());
    const auto operands = adaptor.getOperands();

    // get the new operands from the operandMap
    // workaround to avoid unrealized conversion; there is probably a better
    // solution
    SmallVector<Value> newOperands;
    newOperands.reserve(operands.size());
    for (auto const& val : operands) {
      if (getState().operandMap.contains(val)) {
        newOperands.emplace_back(getState().operandMap.at(val));
      } else {
        newOperands.emplace_back(val);
      }
    }

    // match alloc register
    if (fnName == "__quantum__rt__qubit_allocate_array") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::AllocOp>(
          op, qregType, adaptor.getOperands());

      // update the operand list
      getState().operandMap.try_emplace(op->getResult(0), newOp->getResult(0));

    }
    // match alloc qubit using the given allocOp and the index
    else if (fnName == "__quantum__rt__qubit_allocate") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, getState().qReg, Value{},
          IntegerAttr::get(rewriter.getIntegerType(64), getState().index++));

      // update the operand list
      getState().operandMap.try_emplace(op->getResult(0),
                                        newOp->getOpResult(0));
    }
    // match extract qubit from register
    else if (fnName == "__quantum__rt__array_get_element_ptr_1d") {
      const auto newOp = rewriter.replaceOpWithNewOp<dyn::ExtractOp>(
          op, qubitType, newOperands);

      // update the operand list
      getState().operandMap.try_emplace(op->getResult(0),
                                        newOp->getOpResult(0));
    }
    // match measure operation
    else if (fnName == "__quantum__qis__m__body") {
      const auto bitType = IntegerType::get(rewriter.getContext(), 1);

      bool foundUser = false;
      for (auto* user : op->getUsers()) {
        if (auto callOp = dyn_cast<LLVM::CallOp>(user)) {
          // check if there is read result operation to replace the result uses
          if (callOp.getCallee() == "__quantum__rt__read_result") {
            rewriter.replaceOpWithNewOp<dyn::MeasureOp>(callOp, bitType,
                                                        newOperands);
            foundUser = true;
            rewriter.eraseOp(op);
            break;
          }
        }
      }
      if (!foundUser) {
        // otherwise just create the measure operation
        rewriter.replaceOpWithNewOp<dyn::MeasureOp>(op, bitType, newOperands);
      }
    }
    // match dealloc register
    else if (fnName == "__quantum__rt__qubit_release_array") {
      rewriter.replaceOpWithNewOp<dyn::DeallocOp>(op, newOperands.front());
    }
    // match reset operation
    else if (fnName == "__quantum__qis__reset__body") {
      rewriter.replaceOpWithNewOp<dyn::ResetOp>(op, newOperands.front());
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
      if (SIMPLE_GATES.contains(gateName.str())) {
        const auto name = SIMPLE_GATES.at(gateName.str());
        convertSimpleGate(op, name, newOperands, ctrlQubits, rewriter);
        return success();
      }
      if (SINGLE_ROTATION_GATES.contains(gateName.str())) {
        const auto name = SINGLE_ROTATION_GATES.at(gateName.str());
        convertRotationGate(op, name, newOperands, ctrlQubits, 1, rewriter);
        return success();
      }
      if (DOUBLE_ROTATION_GATES.contains(gateName.str())) {
        const auto name = DOUBLE_ROTATION_GATES.at(gateName.str());
        convertRotationGate(op, name, newOperands, ctrlQubits, 2, rewriter);
        return success();
      }
      if (gateName == "u3") {
        convertRotationGate(op, "UOp", newOperands, ctrlQubits, 3, rewriter);
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
   * @brief Creates the Allocate register operation if it does not exist
   * already.
   *
   * @param op The module operation that holds all operations.
   * @return An optional that either contains the newly created allocate
   * operation or std::nullopt.
   */
  static dyn::AllocOp ensureRegisterAllocation(Operation* op) {
    int64_t requiredQubits = 0;
    auto module = dyn_cast<ModuleOp>(op);

    // walk through the IR and count all qubit allocate operations
    const auto result = module->walk([&](Operation* op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
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

      LLVM::LLVMFuncOp main = getMainFunction(op);

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
    // otherwise return nullptr
    return nullptr;
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    LoweringState state;
    // create an allocOp and store it in the loweringState if necessary
    state.qReg = ensureRegisterAllocation(module);

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QIRToMQTDynTypeConverter typeConverter(context);
    target.addLegalDialect<dyn::MQTDynDialect>();

    // only convert the call and load operations for now
    target.addIllegalOp<LLVM::CallOp>();
    target.addIllegalOp<LLVM::LoadOp>();

    patterns.add<ConvertQIRLoad>(typeConverter, context);
    patterns.add<ConvertQIRCall>(typeConverter, context, &state);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
