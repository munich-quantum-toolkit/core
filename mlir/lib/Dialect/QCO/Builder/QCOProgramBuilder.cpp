/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

using namespace mlir::utils;

namespace mlir::qco {

QCOProgramBuilder::QCOProgramBuilder(MLIRContext* context)
    : ImplicitLocOpBuilder(
          FileLineColLoc::get(context, "<qco-program-builder>", 1, 1), context),
      ctx(context), module(ModuleOp::create(*this)) {
  ctx->loadDialect<QCODialect, qtensor::QTensorDialect>();
}

void QCOProgramBuilder::initialize() {
  // Set insertion point to the module body
  setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = getFunctionType({}, {getI64Type()});
  auto mainFunc = func::FuncOp::create(*this, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = getStringAttr("entry_point");
  mainFunc->setAttr("passthrough", getArrayAttr({entryPointAttr}));

  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

Value QCOProgramBuilder::intConstant(const int64_t value) {
  checkFinalized();
  return arith::ConstantOp::create(*this, getI64IntegerAttr(value)).getResult();
}

Value QCOProgramBuilder::allocQubit() {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Dynamic);

  auto allocOp = AllocOp::create(*this);
  auto qubit = allocOp.getResult();

  // Track the allocated qubit as valid
  validQubits.try_emplace(qubit, QubitInfo{});

  return qubit;
}

Value QCOProgramBuilder::staticQubit(const uint64_t index) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Static);

  auto staticOp = StaticOp::create(*this, index);
  const auto qubit = staticOp.getQubit();

  // Track the static qubit as valid
  validQubits.try_emplace(qubit, QubitInfo{});

  return qubit;
}

QCOProgramBuilder::QubitRegister
QCOProgramBuilder::allocQubitRegister(const int64_t size) {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  auto qtensor = qtensorAlloc(size);

  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    auto [qtensorOut, qubit] = qtensorExtract(qtensor, i);
    qtensor = qtensorOut;
    qubits.emplace_back(qubit);
  }

  return {.value = qtensor, .qubits = std::move(qubits)};
}

QCOProgramBuilder::ClassicalRegister
QCOProgramBuilder::allocClassicalBitRegister(const int64_t size,
                                             std::string name) const {
  checkFinalized();

  if (size <= 0) {
    llvm::reportFatalUsageError("Size must be positive");
  }

  return {.name = std::move(name), .size = size};
}

//===----------------------------------------------------------------------===//
// Linear Type Tracking Helpers
//===----------------------------------------------------------------------===//

void QCOProgramBuilder::validateQubitValue(Value qubit) const {
  if (!validQubits.contains(qubit)) {
    llvm::errs() << "Attempting to use an invalid qubit SSA value. "
                 << "The value may have been consumed by a previous operation "
                 << "or was never created through this builder.\n";
    llvm::reportFatalUsageError(
        "Invalid qubit value used (either consumed or not tracked)");
  }
}

void QCOProgramBuilder::updateQubitTracking(Value inputQubit,
                                            Value outputQubit) {
  // Validate the input qubit
  validateQubitValue(inputQubit);

  auto it = validQubits.find(inputQubit);
  auto info = it->second;

  // Remove the input (consumed) value from tracking
  validQubits.erase(it);

  // Add the output (new) value to tracking
  validQubits.try_emplace(outputQubit, info);
}

void QCOProgramBuilder::validateTensorValue(Value tensor) const {
  if (!validTensors.contains(tensor)) {
    llvm::errs() << "Attempting to use an invalid tensor SSA value. "
                 << "The value may have been consumed by a previous operation "
                 << "or was never created through this builder.\n";
    llvm::reportFatalUsageError(
        "Invalid tensor value used (either consumed or not tracked)");
  }

  auto tensorType = llvm::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || tensorType.getRank() != 1) {
    llvm::reportFatalUsageError("Tensor must be of 1-D RankedTensorType!");
  }
  if (!llvm::isa<QubitType>(tensorType.getElementType())) {
    llvm::reportFatalUsageError("Elements must be of QubitType!");
  }
}

void QCOProgramBuilder::updateTensorTracking(Value inputTensor,
                                             Value outputTensor) {
  // Validate the input tensor
  validateTensorValue(inputTensor);

  auto it = validTensors.find(inputTensor);
  auto info = it->second;

  // Remove the input (consumed) value from tracking
  validTensors.erase(it);

  // Add the output (new) value to tracking
  validTensors.try_emplace(outputTensor, info);
}

llvm::SmallVector<Value>
QCOProgramBuilder::insertExtractedQubits(ValueRange initArgs) {
  llvm::SmallVector<Value> updatedArgs;
  updatedArgs.reserve(initArgs.size());

  // Iterate through the initial values and add the latest value to the updated
  // arguments
  for (auto initArg : initArgs) {
    TypeSwitch<Type>(initArg.getType())
        .Case<QubitType>([&](auto) {
          // Directly insert qubits
          updatedArgs.emplace_back(initArg);
        })
        .Case<RankedTensorType>([&](auto) {
          validateTensorValue(initArg);
          // For tensors check if you have to insert qubits first
          const auto regId = validTensors[initArg].regId;
          auto currentTensor = initArg;
          // Iterate through the validQubits and find the qubits that were
          // extracted from this tensor
          for (auto it = validQubits.begin(); it != validQubits.end();) {
            auto& [qubit, qubitInfo] = *it;
            if (qubitInfo.regId == regId) {
              // Create an InsertOp for the qubit
              auto newTensor =
                  qtensor::InsertOp::create(*this, qubit, currentTensor,
                                            qubitInfo.regIndex)
                      .getResult();
              // Update the tensor tracking
              updateTensorTracking(currentTensor, newTensor);
              currentTensor = newTensor;
              validQubits.erase(it++);
            } else {
              ++it;
            }
          }
          updatedArgs.emplace_back(currentTensor);
        })
        .Default([&](auto) {
          llvm::reportFatalUsageError("Elements must be qubit values");
        });
  }
  return updatedArgs;
}

void QCOProgramBuilder::removeQubitValueTracking(ValueRange values) {
  for (auto value : values) {
    if (llvm::isa<QubitType>(value.getType())) {
      validateQubitValue(value);
      validQubits.erase(value);
    } else {
      validateTensorValue(value);
      validTensors.erase(value);
    }
  }
}

/** @brief Helper function to check if every value is a qubit value*/
static void checkQubitType(ValueRange values) {
  for (Type type : values.getTypes()) {
    auto isQubitType = TypeSwitch<Type, bool>(type)
                           .Case<QubitType>([](auto) { return true; })
                           .Case<RankedTensorType>([](RankedTensorType t) {
                             return llvm::isa<QubitType>(t.getElementType());
                           })
                           .Default([](Type) { return false; });

    if (!isQubitType) {
      llvm::reportFatalUsageError("Elements must be qubit values");
    }
  }
}

//===----------------------------------------------------------------------===//
// QTensor Operations
//===----------------------------------------------------------------------===//

Value QCOProgramBuilder::qtensorAlloc(
    const std::variant<int64_t, Value>& size) {
  checkFinalized();
  ensureAllocationMode(AllocationMode::Dynamic);

  auto sizeValue = variantToValue(*this, getLoc(), size);
  auto allocOp = qtensor::AllocOp::create(*this, sizeValue);

  auto result = allocOp.getResult();
  validTensors.try_emplace(result, TensorInfo{tensorCounter++});

  return result;
}

Value QCOProgramBuilder::qtensorFromElements(ValueRange elements) {
  checkFinalized();

  if (elements.empty()) {
    llvm::reportFatalUsageError("Elements must contain at least one qubit");
  }

  for (auto element : elements) {
    if (!llvm::isa<QubitType>(element.getType())) {
      llvm::reportFatalUsageError("Elements must be QubitType!");
    }
    validateQubitValue(element);
    validQubits.erase(element);
  }

  auto fromElementsOp = qtensor::FromElementsOp::create(*this, elements);
  auto result = fromElementsOp.getResult();
  validTensors.try_emplace(result, TensorInfo{tensorCounter++});
  return result;
}

std::pair<Value, Value>
QCOProgramBuilder::qtensorExtract(Value tensor,
                                  const std::variant<int64_t, Value>& index) {
  checkFinalized();

  auto indexValue = variantToValue(*this, getLoc(), index);
  auto extractOp = qtensor::ExtractOp::create(*this, tensor, indexValue);
  auto qubit = extractOp.getResult();
  auto outTensor = extractOp.getOutTensor();

  validateTensorValue(tensor);
  const auto regId = validTensors[tensor].regId;

  validQubits.try_emplace(qubit,
                          QubitInfo{.regId = regId, .regIndex = indexValue});
  updateTensorTracking(tensor, outTensor);

  return {outTensor, qubit};
}

Value QCOProgramBuilder::qtensorInsert(
    Value scalar, Value tensor, const std::variant<int64_t, Value>& index) {
  checkFinalized();

  auto indexValue = variantToValue(*this, getLoc(), index);
  auto insertOp = qtensor::InsertOp::create(*this, scalar, tensor, indexValue);

  auto outTensor = insertOp.getResult();

  validateQubitValue(scalar);
  validQubits.erase(scalar);
  updateTensorTracking(tensor, outTensor);

  return outTensor;
}

QCOProgramBuilder& QCOProgramBuilder::qtensorDealloc(Value tensor) {
  checkFinalized();

  validateTensorValue(tensor);
  validTensors.erase(tensor);

  qtensor::DeallocOp::create(*this, tensor);

  return *this;
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

std::pair<Value, Value> QCOProgramBuilder::measure(Value qubit) {
  checkFinalized();

  auto measureOp = MeasureOp::create(*this, qubit);
  auto qubitOut = measureOp.getQubitOut();
  auto result = measureOp.getResult();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return {qubitOut, result};
}

Value QCOProgramBuilder::measure(Value qubit, const Bit& bit) {
  checkFinalized();

  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  auto measureOp =
      MeasureOp::create(*this, qubit, nameAttr, sizeAttr, indexAttr);
  auto qubitOut = measureOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

Value QCOProgramBuilder::reset(Value qubit) {
  checkFinalized();

  auto resetOp = ResetOp::create(*this, qubit);
  auto qubitOut = resetOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  void QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM)) { \
    checkFinalized();                                                          \
    OP_CLASS::create(*this, PARAM);                                            \
  }                                                                            \
  Value QCOProgramBuilder::c##OP_NAME(                                         \
      const std::variant<double, Value>&(PARAM), Value control) {              \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    const auto controlsOut =                                                   \
        ctrl(control, {},                                                      \
             [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {         \
               OP_NAME(param);                                                 \
               return {};                                                      \
             })                                                                \
            .first;                                                            \
    return controlsOut[0];                                                     \
  }                                                                            \
  ValueRange QCOProgramBuilder::mc##OP_NAME(                                   \
      const std::variant<double, Value>&(PARAM), ValueRange controls) {        \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    const auto controlsOut =                                                   \
        ctrl(controls, {},                                                     \
             [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {         \
               OP_NAME(param);                                                 \
               return {};                                                      \
             })                                                                \
            .first;                                                            \
    return controlsOut;                                                        \
  }

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  Value QCOProgramBuilder::OP_NAME(Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit);                                  \
    auto qubitOut = op.getQubitOut();                                          \
    updateQubitTracking(qubit, qubitOut);                                      \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(Value control,         \
                                                        Value target) {        \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(targets[0])};                                        \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      ValueRange controls, Value target) {                                     \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(targets[0])};                                   \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, id)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXOp, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  Value QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM),  \
                                   Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit, PARAM);                           \
    auto qubitOut = op.getQubitOut();                                          \
    updateQubitTracking(qubit, qubitOut);                                      \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(                       \
      const std::variant<double, Value>&(PARAM), Value control,                \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(param, targets[0])};                                 \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(param, targets[0])};                            \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, phi)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  Value QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit, PARAM1, PARAM2);                  \
    auto qubitOut = op.getQubitOut();                                          \
    updateQubitTracking(qubit, qubitOut);                                      \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(                       \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(param1, param2, targets[0])};                        \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(param1, param2, targets[0])};                   \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  Value QCOProgramBuilder::OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   const std::variant<double, Value>&(PARAM3), \
                                   Value qubit) {                              \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit, PARAM1, PARAM2, PARAM3);          \
    auto qubitOut = op.getQubitOut();                                          \
    updateQubitTracking(qubit, qubitOut);                                      \
    return qubitOut;                                                           \
  }                                                                            \
  std::pair<Value, Value> QCOProgramBuilder::c##OP_NAME(                       \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    auto param3 = variantToValue(*this, getLoc(), PARAM3);                     \
    const auto [controlsOut, targetsOut] = ctrl(                               \
        control, target, [&](ValueRange targets) -> llvm::SmallVector<Value> { \
          return {OP_NAME(param1, param2, param3, targets[0])};                \
        });                                                                    \
    return {controlsOut[0], targetsOut[0]};                                    \
  }                                                                            \
  std::pair<ValueRange, Value> QCOProgramBuilder::mc##OP_NAME(                 \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    auto param3 = variantToValue(*this, getLoc(), PARAM3);                     \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, target,                                                 \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               return {OP_NAME(param1, param2, param3, targets[0])};           \
             });                                                               \
    return {controlsOut, targetsOut[0]};                                       \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  std::pair<Value, Value> QCOProgramBuilder::OP_NAME(Value qubit0,             \
                                                     Value qubit1) {           \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit0, qubit1);                         \
    auto qubit0Out = op.getQubit0Out();                                        \
    auto qubit1Out = op.getQubit1Out();                                        \
    updateQubitTracking(qubit0, qubit0Out);                                    \
    updateQubitTracking(qubit1, qubit1Out);                                    \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> QCOProgramBuilder::c##OP_NAME(     \
      Value control, Value qubit0, Value qubit1) {                             \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(targets[0], targets[1]);                \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      QCOProgramBuilder::mc##OP_NAME(ValueRange controls, Value qubit0,        \
                                     Value qubit1) {                           \
    checkFinalized();                                                          \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(targets[0], targets[1]);                \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
  }

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  std::pair<Value, Value> QCOProgramBuilder::OP_NAME(                          \
      const std::variant<double, Value>&(PARAM), Value qubit0, Value qubit1) { \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit0, qubit1, PARAM);                  \
    auto qubit0Out = op.getQubit0Out();                                        \
    auto qubit1Out = op.getQubit1Out();                                        \
    updateQubitTracking(qubit0, qubit0Out);                                    \
    updateQubitTracking(qubit1, qubit1Out);                                    \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> QCOProgramBuilder::c##OP_NAME(     \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(param, targets[0], targets[1]);         \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      QCOProgramBuilder::mc##OP_NAME(                                          \
          const std::variant<double, Value>&(PARAM), ValueRange controls,      \
          Value qubit0, Value qubit1) {                                        \
    checkFinalized();                                                          \
    auto param = variantToValue(*this, getLoc(), PARAM);                       \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] = OP_NAME(param, targets[0], targets[1]);         \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
  }

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  std::pair<Value, Value> QCOProgramBuilder::OP_NAME(                          \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value qubit0,                \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    auto op = OP_CLASS::create(*this, qubit0, qubit1, PARAM1, PARAM2);         \
    auto qubit0Out = op.getQubit0Out();                                        \
    auto qubit1Out = op.getQubit1Out();                                        \
    updateQubitTracking(qubit0, qubit0Out);                                    \
    updateQubitTracking(qubit1, qubit1Out);                                    \
    return {qubit0Out, qubit1Out};                                             \
  }                                                                            \
  std::pair<Value, std::pair<Value, Value>> QCOProgramBuilder::c##OP_NAME(     \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1) {                                                          \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(control, {qubit0, qubit1},                                        \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] =                                                 \
                   OP_NAME(param1, param2, targets[0], targets[1]);            \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut[0], {targetsOut[0], targetsOut[1]}};                   \
  }                                                                            \
  std::pair<ValueRange, std::pair<Value, Value>>                               \
      QCOProgramBuilder::mc##OP_NAME(                                          \
          const std::variant<double, Value>&(PARAM1),                          \
          const std::variant<double, Value>&(PARAM2), ValueRange controls,     \
          Value qubit0, Value qubit1) {                                        \
    checkFinalized();                                                          \
    auto param1 = variantToValue(*this, getLoc(), PARAM1);                     \
    auto param2 = variantToValue(*this, getLoc(), PARAM2);                     \
    const auto [controlsOut, targetsOut] =                                     \
        ctrl(controls, {qubit0, qubit1},                                       \
             [&](ValueRange targets) -> llvm::SmallVector<Value> {             \
               auto [q0, q1] =                                                 \
                   OP_NAME(param1, param2, targets[0], targets[1]);            \
               return {q0, q1};                                                \
             });                                                               \
    return {controlsOut, {targetsOut[0], targetsOut[1]}};                      \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

ValueRange QCOProgramBuilder::barrier(ValueRange qubits) {
  checkFinalized();

  auto op = BarrierOp::create(*this, qubits);
  auto qubitsOut = op.getQubitsOut();
  for (const auto& [inputQubit, outputQubit] : llvm::zip(qubits, qubitsOut)) {
    updateQubitTracking(inputQubit, outputQubit);
  }
  return qubitsOut;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

std::pair<ValueRange, ValueRange> QCOProgramBuilder::ctrl(
    ValueRange controls, ValueRange targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> body) {
  checkFinalized();

  auto ctrlOp = CtrlOp::create(*this, controls, targets);
  auto& block = ctrlOp.getBodyRegion().emplaceBlock();
  auto qubitType = QubitType::get(getContext());
  for (const auto target : targets) {
    const auto arg = block.addArgument(qubitType, getLoc());
    updateQubitTracking(target, arg);
  }
  const InsertionGuard guard(*this);
  setInsertionPointToStart(&block);
  const auto innerTargetsOut = body(block.getArguments());
  YieldOp::create(*this, innerTargetsOut);

  if (innerTargetsOut.size() != targets.size()) {
    llvm::reportFatalUsageError(
        "Ctrl body must return exactly one output qubit per target");
  }

  // Update tracking
  const auto& controlsOut = ctrlOp.getControlsOut();
  for (const auto& [control, controlOut] :
       llvm::zip_equal(controls, controlsOut)) {
    updateQubitTracking(control, controlOut);
  }
  const auto& targetsOut = ctrlOp.getTargetsOut();
  for (const auto& [target, targetOut] :
       llvm::zip_equal(innerTargetsOut, targetsOut)) {
    updateQubitTracking(target, targetOut);
  }

  return {controlsOut, targetsOut};
}

ValueRange QCOProgramBuilder::inv(
    ValueRange qubits,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> body) {
  checkFinalized();

  auto invOp = InvOp::create(*this, qubits);

  // Add block arguments for all qubits
  auto& block = invOp.getBodyRegion().emplaceBlock();
  auto qubitType = QubitType::get(getContext());
  for (auto qubit : qubits) {
    const auto arg = block.addArgument(qubitType, getLoc());
    updateQubitTracking(qubit, arg);
  }

  // Create the final yield operation
  const InsertionGuard guard(*this);
  setInsertionPointToStart(&block);
  const auto innerTargetsOut = body(block.getArguments());
  YieldOp::create(*this, innerTargetsOut);

  if (innerTargetsOut.size() != qubits.size()) {
    llvm::reportFatalUsageError(
        "Inv body must return exactly one output qubit per target");
  }

  // Update tracking
  const auto& targetsOut = invOp.getQubitsOut();
  for (const auto& [target, targetOut] :
       llvm::zip_equal(innerTargetsOut, targetsOut)) {
    updateQubitTracking(target, targetOut);
  }

  return targetsOut;
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

QCOProgramBuilder& QCOProgramBuilder::sink(Value qubit) {
  checkFinalized();

  validateQubitValue(qubit);
  validQubits.erase(qubit);

  SinkOp::create(*this, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// SCF Operations
//===----------------------------------------------------------------------===//

ValueRange QCOProgramBuilder::scfFor(
    const std::variant<int64_t, Value>& lowerbound,
    const std::variant<int64_t, Value>& upperbound,
    const std::variant<int64_t, Value>& step, ValueRange initArgs,
    llvm::function_ref<llvm::SmallVector<Value>(Value, ValueRange)> body) {
  checkFinalized();
  checkQubitType(initArgs);

  const auto loc = getLoc();
  const auto lb = utils::variantToValue(*this, loc, lowerbound);
  const auto ub = utils::variantToValue(*this, loc, upperbound);
  const auto stepSize = utils::variantToValue(*this, loc, step);
  // Get the updated arguments after inserting the extracted qubits
  auto updatedArgs = insertExtractedQubits(initArgs);

  // Create the empty for operation
  auto forOp = scf::ForOp::create(*this, lb, ub, stepSize, updatedArgs);

  auto* forBody = forOp.getBody();
  auto iv = forBody->getArgument(0);
  auto loopArgs = forBody->getArguments().drop_front();
  // Set the insertionpoint
  const OpBuilder::InsertionGuard guard(*this);
  setInsertionPointToStart(forBody);

  // Add the iterArgs as valid qubit values
  for (const auto& arg : loopArgs) {
    if (llvm::isa<QubitType>(arg.getType())) {
      validQubits.try_emplace(arg, QubitInfo{});
    } else {
      validTensors.try_emplace(arg, TensorInfo{tensorCounter++});
    }
  }

  // Build the body
  const auto bodyResults = body(iv, loopArgs);
  scf::YieldOp::create(*this, bodyResults);

  if (bodyResults.size() != initArgs.size()) {
    llvm::reportFatalUsageError(
        "scf.for body must return exactly one value per iter arg");
  }

  // Remove the bodyResults as valid qubit values
  removeQubitValueTracking(bodyResults);

  // Update the qubit tracking
  for (auto [arg, result] : llvm::zip_equal(updatedArgs, forOp->getResults())) {
    if (arg.getType() != result.getType()) {
      llvm::reportFatalUsageError("Result types must match input types");
    }
    if (llvm::isa<QubitType>(arg.getType())) {
      updateQubitTracking(arg, result);
    } else {
      updateTensorTracking(arg, result);
    }
  }

  return forOp->getResults();
}

ValueRange QCOProgramBuilder::scfWhile(
    ValueRange initArgs,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> beforeBody,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> afterBody) {
  checkFinalized();
  checkQubitType(initArgs);

  // Get the updated arguments after inserting the extracted qubits
  auto updatedArgs = insertExtractedQubits(initArgs);
  // Create the empty while operation
  auto whileOp = scf::WhileOp::create(*this, initArgs.getTypes(), updatedArgs);
  const llvm::SmallVector<Location> locs(initArgs.size(), getLoc());

  const OpBuilder::InsertionGuard guard(*this);

  // Construct the blocks
  auto* beforeBlock =
      createBlock(&whileOp.getBefore(), {}, initArgs.getTypes(), locs);
  auto* afterBlock =
      createBlock(&whileOp.getAfter(), {}, initArgs.getTypes(), locs);

  // Create the body regions
  auto createBody =
      [&](Block* block,
          llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> body,
          bool createYield) {
        auto args = block->getArguments();
        auto* region = block->getParent();
        setInsertionPointToStart(block);

        // Add the args to the valid qubits/tensors
        for (auto arg : args) {
          if (llvm::isa<QubitType>(arg.getType())) {
            validQubits.try_emplace(arg, QubitInfo{});
          } else {
            validTensors.try_emplace(arg, TensorInfo{tensorCounter++});
          }
        }
        // Construct the body
        const auto& results = body(args);

        if (results.size() != initArgs.size()) {
          llvm::reportFatalUsageError(
              "scf.while body must return exactly one value per iter arg");
        }

        // Create the terminator operation and erase the yielded values if
        // required
        if (createYield) {
          scf::YieldOp::create(*this, results);
          removeQubitValueTracking(results);
        }
      };

  createBody(beforeBlock, beforeBody, false);
  createBody(afterBlock, afterBody, true);

  // Update the qubit tracking
  for (auto [arg, result] :
       llvm::zip_equal(updatedArgs, whileOp->getResults())) {
    if (arg.getType() != result.getType()) {
      llvm::reportFatalUsageError("Result types must match input types");
    }
    if (llvm::isa<QubitType>(arg.getType())) {
      updateQubitTracking(arg, result);
    } else {
      updateTensorTracking(arg, result);
    }
  }

  return whileOp->getResults();
}

ValueRange QCOProgramBuilder::qcoIf(
    const std::variant<bool, Value>& condition, ValueRange initArgs,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> thenBody,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> elseBody) {
  checkFinalized();
  checkQubitType(initArgs);

  auto conditionValue = variantToValue(*this, getLoc(), condition);
  auto updatedArgs = insertExtractedQubits(initArgs);
  auto ifOp = IfOp::create(*this, conditionValue, updatedArgs);

  // Create the then and else block
  auto& thenBlock = ifOp->getRegion(0).emplaceBlock();
  auto& elseBlock = ifOp->getRegion(1).emplaceBlock();

  // Create the block arguments and add them as valid qubits
  for (auto qubitType : initArgs.getTypes()) {
    auto thenArg = thenBlock.addArgument(qubitType, getLoc());
    auto elseArg = elseBlock.addArgument(qubitType, getLoc());
    if (llvm::isa<QubitType>(qubitType)) {
      validQubits.try_emplace(thenArg, QubitInfo{});
      validQubits.try_emplace(elseArg, QubitInfo{});
    } else {
      validTensors.try_emplace(thenArg, TensorInfo{tensorCounter++});
      validTensors.try_emplace(elseArg, TensorInfo{tensorCounter++});
    }
  }

  // Construct the bodies of the regions
  const InsertionGuard guard(*this);
  setInsertionPointToStart(&thenBlock);
  const auto thenResult = thenBody(thenBlock.getArguments());
  YieldOp::create(*this, thenResult);
  setInsertionPointToStart(&elseBlock);
  llvm::SmallVector<Value> elseResult;
  if (elseBody) {
    elseResult = elseBody(elseBlock.getArguments());
    YieldOp::create(*this, elseResult);
  } else {
    elseResult.assign(elseBlock.getArguments().begin(),
                      elseBlock.getArguments().end());
    YieldOp::create(*this, elseBlock.getArguments());
  }

  if (thenResult.size() != initArgs.size() ||
      thenResult.size() != elseResult.size()) {
    llvm::reportFatalUsageError(
        "Then and else body must return the same amount of qubits as the "
        "number of input qubits!");
  }

  for (auto [arg, result] : llvm::zip_equal(updatedArgs, ifOp->getResults())) {
    if (arg.getType() != result.getType()) {
      llvm::reportFatalUsageError("Result types must match input types");
    }
    if (llvm::isa<QubitType>(arg.getType())) {
      updateQubitTracking(arg, result);
    } else {
      updateTensorTracking(arg, result);
    }
  }

  // Remove the inner qubit values as valid qubit values
  removeQubitValueTracking(thenResult);
  removeQubitValueTracking(elseResult);

  return ifOp->getResults();
}

QCOProgramBuilder& QCOProgramBuilder::scfCondition(Value condition,
                                                   ValueRange yieldedValues) {
  checkFinalized();
  checkQubitType(yieldedValues);

  // Erase the yieldedValues from tracking
  removeQubitValueTracking(yieldedValues);

  scf::ConditionOp::create(*this, condition, yieldedValues);

  return *this;
}
//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void QCOProgramBuilder::checkFinalized() const {
  if (ctx == nullptr) {
    llvm::reportFatalUsageError(
        "QCOProgramBuilder instance has been finalized");
  }
}

void QCOProgramBuilder::ensureAllocationMode(
    const AllocationMode requestedMode) {
  if (allocationMode == AllocationMode::Unset) {
    allocationMode = requestedMode;
    return;
  }
  if (allocationMode == requestedMode) {
    return;
  }

  const char* const existingName =
      allocationMode == AllocationMode::Static ? "static" : "dynamic";
  const char* const requestedName =
      requestedMode == AllocationMode::Static ? "static" : "dynamic";

  const std::string message =
      llvm::formatv("Cannot mix {0} and {1} qubit allocation modes in "
                    "QCOProgramBuilder",
                    existingName, requestedName)
          .str();
  llvm::reportFatalUsageError(message.c_str());
}

OwningOpRef<ModuleOp> QCOProgramBuilder::finalize() {
  checkFinalized();

  // Ensure that main function exists and insertion point is valid
  auto* insertionBlock = getInsertionBlock();
  func::FuncOp mainFunc = nullptr;
  for (auto op : module.getOps<func::FuncOp>()) {
    if (op.getName() == "main") {
      mainFunc = op;
      break;
    }
  }
  if (!mainFunc) {
    llvm::reportFatalUsageError("Could not find main function");
  }
  if ((insertionBlock == nullptr) ||
      insertionBlock != &mainFunc.getBody().front()) {
    llvm::reportFatalUsageError(
        "Insertion point is not in entry block of main function");
  }

  llvm::DenseSet<int64_t> validTensorIds;
  for (const auto& [tensor, info] : validTensors) {
    validTensorIds.insert(info.regId);
  }

  llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<Value, QubitInfo>>>
      qubitsByRegister;
  for (auto [qubit, info] : validQubits) {
    if (info.regId == -1 || !validTensorIds.contains(info.regId)) {
      // Automatically deallocate all still-allocated qubits
      SinkOp::create(*this, qubit);
    } else {
      qubitsByRegister[info.regId].emplace_back(qubit, info);
    }
  }

  // Automatically deallocate all still-allocated tensors
  for (auto& [tensor, tensorInfo] : validTensors) {
    auto currentTensor = tensor;
    // Filter out qubits belonging to this tensor
    for (auto& [qubit, qubitInfo] : qubitsByRegister[tensorInfo.regId]) {
      currentTensor = qtensor::InsertOp::create(*this, qubit, currentTensor,
                                                qubitInfo.regIndex)
                          .getResult();
    }
    // Deallocate tensor
    qtensor::DeallocOp::create(*this, currentTensor);
  }
  validQubits.clear();
  validTensors.clear();

  // Create constant 0 for successful exit code
  auto exitCode = intConstant(0);

  // Add return statement with exit code 0 to the main function
  func::ReturnOp::create(*this, exitCode);

  // Invalidate context to prevent use-after-finalize
  ctx = nullptr;

  return module;
}

OwningOpRef<ModuleOp> QCOProgramBuilder::build(
    MLIRContext* context,
    const llvm::function_ref<void(QCOProgramBuilder&)>& buildFunc) {
  QCOProgramBuilder builder(context);
  builder.initialize();
  buildFunc(builder);
  return builder.finalize();
}

} // namespace mlir::qco
