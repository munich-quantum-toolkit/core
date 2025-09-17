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
  if (gateName == #gate) {                                                     \
    rewriter.replaceOpWithNewOp<ref::gate>(op, DenseF64ArrayAttr{},            \
                                           DenseBoolArrayAttr{}, ValueRange{}, \
                                           qubits, ctrlQubits, ValueRange{});  \
    return;                                                                    \
  }

// macro to check the gateName and to replace the rotation gate
#define ADD_CONVERT_ROTATION_GATE(gate)                                        \
  if (gateName == #gate) {                                                     \
    rewriter.replaceOpWithNewOp<ref::gate>(                                    \
        op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, rotationDegrees,        \
        operands, ctrlQubits, ValueRange{});                                   \
    return;                                                                    \
  }

#include "mlir/Conversion/QIRToMQTRef/QIRToMQTRef.h"

#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_QIRTOMQTREF
#include "mlir/Conversion/QIRToMQTRef/QIRToMQTRef.h.inc"

class QIRToMQTRefTypeConverter final : public TypeConverter {
public:
  explicit QIRToMQTRefTypeConverter(MLIRContext* /*ctx*/) {
    // Identity conversion
    addConversion([](Type type) { return type; });
  }
};

namespace {

struct LoweringState {
  // map each llvm QIR result to the new mqtref result
  DenseMap<Value, Value> operandMap;
  // memref operation to store the classical results
  Operation* allocaOp{};
  // map a llvm result ptr to a classical result bit
  DenseMap<Value, Value> resultMap;
  // next free allocate index
  int64_t index{};
};

template <typename OpType>
class StatefulOpConversionPattern : public mlir::OpConversionPattern<OpType> {
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(mlir::TypeConverter& typeConverter,
                              mlir::MLIRContext* ctx, LoweringState* state)
      : mlir::OpConversionPattern<OpType>(typeConverter, ctx), state_(state) {}

  /// @brief Return the state object as reference.
  [[nodiscard]] LoweringState& getState() const { return *state_; }

  inline static const llvm::StringMap<StringRef> SIMPLE_GATES = {
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

  inline static const llvm::StringMap<StringRef> SINGLE_ROTATION_GATES = {
      {"p", "POp"},     {"rx", "RXOp"},   {"ry", "RYOp"},
      {"rz", "RZOp"},   {"rxx", "RXXOp"}, {"ryy", "RYYOp"},
      {"rzz", "RZZOp"}, {"rzx", "RZXOp"}, {"u1", "POp"}};

  inline static const llvm::StringMap<StringRef> DOUBLE_ROTATION_GATES = {
      {"u2", "U2Op"},
      {"xx_minus_yy", "XXminusYYOp"},
      {"xx_plus_yy", "XXplusYYOp"}};

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

struct ConvertQIRZero final : StatefulOpConversionPattern<LLVM::ZeroOp> {
  using StatefulOpConversionPattern<LLVM::ZeroOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ZeroOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto isQubitPtr = false;

    // check if the zero op is used in any functions other than for result
    // recording and the initialize operation
    for (auto* userOp : op->getResult(0).getUsers()) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(userOp)) {
        const auto fnName = callOp.getCallee();
        if (fnName != "__quantum__rt__result_record_output" &&
            fnName != "__quantum__qis__mz__body" &&
            fnName != "__quantum__rt__initialize") {
          isQubitPtr = true;
          break;
        }
        if (fnName == "__quantum__qis__mz__body" &&
            op->getResult(0) == callOp->getOperand(0)) {
          isQubitPtr = true;
          break;
        }
      }
    }

    // if yes the IntToPtr op should be converted to a static qubit
    if (isQubitPtr) {
      // replace the zero operation with a static qubit with index 0
      const auto newOp = rewriter.replaceOpWithNewOp<ref::QubitOp>(
          op, ref::QubitType::get(rewriter.getContext()), 0);

      // update the operand map
      getState().operandMap.try_emplace(op->getResult(0),
                                        newOp->getOpResult(0));
    }
    // otherwise delete the operation
    else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct ConvertQIRIntToPtr final
    : StatefulOpConversionPattern<LLVM::IntToPtrOp> {
  using StatefulOpConversionPattern<
      LLVM::IntToPtrOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::IntToPtrOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto constantOp = op->getOperand(0).getDefiningOp<LLVM::ConstantOp>();
    auto isQubitPtr = false;

    // check if the ptr is used in any functions other than for result
    // recording
    for (auto* userOp : op->getResult(0).getUsers()) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(userOp)) {
        const auto fnName = callOp.getCallee();
        if (fnName != "__quantum__rt__result_record_output" &&
            fnName != "__quantum__qis__mz__body") {
          isQubitPtr = true;
          break;
        }
        if (fnName == "__quantum__qis__mz__body" &&
            op->getResult(0) == callOp->getOperand(0)) {
          isQubitPtr = true;
          break;
        }
      }
    }

    // if yes the IntToPtr op should be converted to a static qubit
    if (isQubitPtr) {
      // get the int value of the operation
      const auto value = dyn_cast<IntegerAttr>(constantOp.getValue()).getInt();

      // replace the int to ptr operation with static qubit
      const auto newOp = rewriter.replaceOpWithNewOp<ref::QubitOp>(
          op, ref::QubitType::get(rewriter.getContext()), value);

      // update the operand map
      getState().operandMap.try_emplace(op->getResult(0),
                                        newOp->getOpResult(0));
    }
    // otherwise delete the operation
    else {
      rewriter.eraseOp(op);
    }

    // erase the constantOp for the operation if op was the only user
    if (constantOp->getResult(0).hasOneUse()) {
      rewriter.eraseOp(constantOp);
    }

    return success();
  }
};

struct ConvertQIRAddressOf final : OpConversionPattern<LLVM::AddressOfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AddressOfOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // erase the operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQIRGlobal final : OpConversionPattern<LLVM::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // erase the operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQIRCall final : StatefulOpConversionPattern<LLVM::CallOp> {
  using StatefulOpConversionPattern<LLVM::CallOp>::StatefulOpConversionPattern;

  // constants to trim the function name to get the name of the gate
  constexpr static size_t QIS_OPERATION_PREFIX_LENGTH = 16;
  constexpr static size_t QIS_OPERATION_SUFFIX_LENGTH = 6;
  // constant to trim the global constant name to get the index of the global
  // constant
  constexpr static size_t GLOBAL_CONSTANT_PREFIX_LENGTH = 26;

  /**
   * @brief Replaces the call operation with a matching simple gate operation
   * from the mqtref dialect
   *
   * @param op The call operation that is replaced.
   * @param gateName The name of the gate.
   * @param qubits The Qubits of the given operation.
   * @param ctrlQubits The control Qubits of the given operation.
   * @param rewriter The PatternRewriter to use.
   */
  static void convertSimpleGate(LLVM::CallOp& op, const StringRef gateName,
                                const SmallVector<Value>& qubits,
                                const SmallVector<Value>& ctrlQubits,
                                ConversionPatternRewriter& rewriter) {
    // match and replace the fitting gate
    ADD_CONVERT_SIMPLE_GATE(XOp)
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
   * from the mqtref dialect
   *
   * @param op The call operation that is replaced.
   * @param gateName The name of the gate.
   * @param operands The operands of the given operation.
   * @param ctrlQubits The control Qubits of the given operation.
   * @param rotationCount The number of rotation degrees.
   * @param rewriter The PatternRewriter to use.
   */
  static void convertRotationGate(LLVM::CallOp& op, const StringRef gateName,
                                  SmallVector<Value>& operands,
                                  const SmallVector<Value>& ctrlQubits,
                                  const size_t rotationCount,
                                  ConversionPatternRewriter& rewriter) {
    // extract the degrees from the operand list
    SmallVector<Value> rotationDegrees;
    rotationDegrees.reserve(rotationCount);
    rotationDegrees.insert(
        rotationDegrees.end(),
        std::make_move_iterator(std::prev(
            operands.end(), static_cast<std::ptrdiff_t>(rotationCount))),
        std::make_move_iterator(operands.end()));
    operands.resize(operands.size() - rotationCount);

    // match and replace the fitting gate
    ADD_CONVERT_ROTATION_GATE(POp)
    ADD_CONVERT_ROTATION_GATE(UOp)
    ADD_CONVERT_ROTATION_GATE(U2Op)
    ADD_CONVERT_ROTATION_GATE(RXOp)
    ADD_CONVERT_ROTATION_GATE(RYOp)
    ADD_CONVERT_ROTATION_GATE(RZOp)
    ADD_CONVERT_ROTATION_GATE(RXXOp)
    ADD_CONVERT_ROTATION_GATE(RYYOp)
    ADD_CONVERT_ROTATION_GATE(RZZOp)
    ADD_CONVERT_ROTATION_GATE(RZXOp)
    ADD_CONVERT_ROTATION_GATE(XXminusYYOp)
    ADD_CONVERT_ROTATION_GATE(XXplusYYOp)
  }

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // get the name of the operation and prepare the return types
    const auto fnName = op.getCallee();
    const auto qubitType = ref::QubitType::get(ctx);
    const auto qregType = ref::QubitRegisterType::get(ctx);
    const auto operands = adaptor.getOperands();

    // get the new operands from the operandMap
    // workaround as we need to convert !llvm.ptr to either !mqtref.Qubit or
    // !mqtref.QubitRegister type depending on the operation and this is not
    // doable with the typeConverter
    SmallVector<Value> newOperands;
    newOperands.reserve(operands.size());
    for (auto const& val : operands) {
      if (getState().operandMap.contains(val)) {
        newOperands.emplace_back(getState().operandMap.at(val));
      } else {
        newOperands.emplace_back(val);
      }
    }

    // match initialize operation
    if (fnName == "__quantum__rt__initialize") {
      op.erase();
      return success();
    }
    // match alloc register
    if (fnName == "__quantum__rt__qubit_allocate_array") {
      const auto newOp = rewriter.replaceOpWithNewOp<ref::AllocOp>(
          op, qregType, adaptor.getOperands());

      // update the operand list
      getState().operandMap.try_emplace(op->getResult(0), newOp->getResult(0));
      return success();
    }
    // match alloc qubit
    if (fnName == "__quantum__rt__qubit_allocate") {
      const auto newOp =
          rewriter.replaceOpWithNewOp<ref::AllocQubitOp>(op, qubitType);

      // update the operand list
      getState().operandMap.try_emplace(op->getResult(0), newOp->getResult(0));
      return success();
    }
    // match qubit release operation
    if (fnName == "__quantum__rt__qubit_release") {
      rewriter.replaceOpWithNewOp<ref::DeallocQubitOp>(op, newOperands.front());
      return success();
    }
    // match dealloc register
    if (fnName == "__quantum__rt__qubit_release_array") {
      rewriter.replaceOpWithNewOp<ref::DeallocOp>(op, newOperands.front());
      return success();
    }
    // match reset operation
    if (fnName == "__quantum__qis__reset__body") {
      rewriter.replaceOpWithNewOp<ref::ResetOp>(op, newOperands.front());
      return success();
    }
    // match extract qubit from register
    if (fnName == "__quantum__rt__array_get_element_ptr_1d") {
      const auto newOp = rewriter.replaceOpWithNewOp<ref::ExtractOp>(
          op, qubitType, newOperands);

      // update the operand list
      getState().operandMap.try_emplace(op->getResult(0),
                                        newOp->getOpResult(0));
      return success();
    }
    // match measure operation
    if (fnName == "__quantum__qis__mz__body") {
      const auto bitType = IntegerType::get(ctx, 1);

      // create the measure operation
      auto measureOp = rewriter.create<ref::MeasureOp>(op.getLoc(), bitType,
                                                       newOperands.front());

      // store the result pointer and the measurement result in the map
      getState().resultMap.try_emplace(newOperands[1], measureOp->getResult(0));

      rewriter.eraseOp(op);
      return success();
    }
    // match record result output operation
    if (fnName == "__quantum__rt__result_record_output") {

      if (getState().resultMap.contains(newOperands[0])) {
        // get the matching measurement result
        auto measureResult = getState().resultMap.at(newOperands[0]);
        // get the index where the measurement is stored from the name of the
        // global op
        auto addressOfOp = op->getOperand(1).getDefiningOp<LLVM::AddressOfOp>();
        if (op->getOperand(1).getDefiningOp<LLVM::AddressOfOp>()) {
          auto const index = stoi(addressOfOp.getGlobalName()
                                      .substr(GLOBAL_CONSTANT_PREFIX_LENGTH)
                                      .str());
          auto constantIndexOp =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), index);

          // create the store operation for the classical result
          rewriter.create<memref::StoreOp>(
              op.getLoc(), measureResult, getState().allocaOp->getResult(0),
              ValueRange{constantIndexOp->getResult(0)});
        }
      }
      rewriter.eraseOp(op);
      return success();
    }

    // remove the prefix and the suffix of the gate name
    auto gateName(fnName->substr(QIS_OPERATION_PREFIX_LENGTH)
                      .drop_back(QIS_OPERATION_SUFFIX_LENGTH));

    // check how many control qubits are used by counting the number of
    // leading c's
    const size_t ctrlQubitCount =
        std::ranges::find_if(gateName, [](char ch) { return ch != 'c'; }) -
        gateName.begin();

    // remove the control qubits from the name
    gateName = gateName.substr(ctrlQubitCount);

    // extract the controlqubits from the operand list
    SmallVector<Value> ctrlQubits;
    auto* const firstCtrl = newOperands.begin();
    auto* const lastCtrl =
        std::next(firstCtrl, static_cast<std::ptrdiff_t>(ctrlQubitCount));
    ctrlQubits.reserve(ctrlQubitCount);
    ctrlQubits.insert(ctrlQubits.end(), std::make_move_iterator(firstCtrl),
                      std::make_move_iterator(lastCtrl));
    newOperands.erase(firstCtrl, lastCtrl);

    // try to match and replace gate operations
    if (gateName == "gphase") {
      rewriter.replaceOpWithNewOp<ref::GPhaseOp>(
          op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, newOperands,
          ValueRange{}, ctrlQubits, ValueRange{});
      return success();
    }
    if (gateName == "barrier") {
      rewriter.replaceOpWithNewOp<ref::BarrierOp>(
          op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, ValueRange{},
          newOperands, ctrlQubits, ValueRange{});
      return success();
    }
    if (SIMPLE_GATES.contains(gateName.str())) {
      convertSimpleGate(op, SIMPLE_GATES.at(gateName.str()), newOperands,
                        ctrlQubits, rewriter);
      return success();
    }
    if (SINGLE_ROTATION_GATES.contains(gateName.str())) {
      convertRotationGate(op, SINGLE_ROTATION_GATES.at(gateName.str()),
                          newOperands, ctrlQubits, 1, rewriter);
      return success();
    }
    if (DOUBLE_ROTATION_GATES.contains(gateName.str())) {
      convertRotationGate(op, DOUBLE_ROTATION_GATES.at(gateName.str()),
                          newOperands, ctrlQubits, 2, rewriter);
      return success();
    }
    if (gateName == "u3" || gateName == "u") {
      convertRotationGate(op, "UOp", newOperands, ctrlQubits, 3, rewriter);
      return success();
    }

    return failure();
  }
};

struct QIRToMQTRef final : impl::QIRToMQTRefBase<QIRToMQTRef> {
  using QIRToMQTRefBase::QIRToMQTRefBase;

  /**
   * @brief Finds the main function in the module
   *
   * @param op The module operation that holds all operations.
   * @return The main function.
   */
  static LLVM::LLVMFuncOp getMainFunction(ModuleOp op) {

    // find the main function
    for (auto funcOp : op.getOps<LLVM::LLVMFuncOp>()) {
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
   * @brief Finds the number of required results from an array of attributes.
   *
   * @param passthrough The ArrayAttr that holds all attributes
   * @return The number of result bits.
   */
  static int64_t getRequiredNumResults(ArrayAttr passthrough) {
    for (const auto attr : passthrough) {
      const auto innerArray = dyn_cast<ArrayAttr>(attr);
      if (!innerArray || innerArray.size() < 2) {
        continue;
      }

      if (const auto key = dyn_cast<StringAttr>(innerArray[0])) {
        if (key.getValue() == "required_num_results") {
          if (const auto value = dyn_cast<StringAttr>(innerArray[1])) {
            return std::stoi(value.str());
          }
        }
      }
    }
    return 0;
  }

  /**
   * @brief create the allocaOp to store classical results
   *
   * @param op The module operation that holds all operations.
   * @param state The lowering state of the conversion pass.
   */
  static void createAllocaOp(Operation* op, LoweringState* state) {
    auto mod = dyn_cast<ModuleOp>(op);

    // get the main function
    LLVM::LLVMFuncOp main = getMainFunction(mod);
    if (!main) {
      return;
    }

    auto passthrough = main->getAttrOfType<ArrayAttr>("passthrough");

    // get the number of results
    auto numResults = getRequiredNumResults(passthrough);
    // return if no results are needed
    if (numResults == 0) {
      return;
    }

    // create the allocaOp at the beginning of the first block
    auto& entryBlock = main.getBody().front();
    OpBuilder builder(&main.getBody());
    builder.setInsertionPointToStart(&entryBlock);

    const auto memrefType = MemRefType::get({numResults}, builder.getI1Type());
    // set the allocaOp in the loweringState
    state->allocaOp =
        builder.create<memref::AllocaOp>(op->getLoc(), memrefType);
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto* moduleOp = getOperation();
    LoweringState state;

    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    QIRToMQTRefTypeConverter typeConverter(ctx);
    target.addLegalDialect<ref::MQTRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    createAllocaOp(moduleOp, &state);
    target.addIllegalOp<LLVM::CallOp>();
    target.addIllegalOp<LLVM::LoadOp>();
    target.addIllegalOp<LLVM::AddressOfOp>();
    target.addIllegalOp<LLVM::GlobalOp>();
    target.addIllegalOp<LLVM::IntToPtrOp>();
    target.addIllegalOp<LLVM::ZeroOp>();
    patterns.add<ConvertQIRCall>(typeConverter, ctx, &state);
    patterns.add<ConvertQIRLoad>(typeConverter, ctx);
    patterns.add<ConvertQIRAddressOf>(typeConverter, ctx);
    patterns.add<ConvertQIRGlobal>(typeConverter, ctx);
    patterns.add<ConvertQIRIntToPtr>(typeConverter, ctx, &state);
    patterns.add<ConvertQIRZero>(typeConverter, ctx, &state);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
