/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/Builder/FluxProgramBuilder.h"

#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <utility>
#include <variant>

namespace mlir::flux {

FluxProgramBuilder::FluxProgramBuilder(MLIRContext* context)
    : OpBuilder(context), ctx(context), loc(getUnknownLoc()),
      module(create<ModuleOp>(loc)) {}

void FluxProgramBuilder::initialize() {
  // Ensure the Flux dialect is loaded
  ctx->loadDialect<FluxDialect>();

  // Set insertion point to the module body
  setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = getFunctionType({}, {getI64Type()});
  auto mainFunc = create<func::FuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = getStringAttr("entry_point");
  mainFunc->setAttr("passthrough", getArrayAttr({entryPointAttr}));

  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  setInsertionPointToStart(&entryBlock);
}

Value FluxProgramBuilder::allocQubit() {
  auto allocOp = create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit as valid
  validQubits.insert(qubit);

  return qubit;
}

Value FluxProgramBuilder::staticQubit(const int64_t index) {
  auto indexAttr = getI64IntegerAttr(index);
  auto staticOp = create<StaticOp>(loc, indexAttr);
  const auto qubit = staticOp.getQubit();

  // Track the static qubit as valid
  validQubits.insert(qubit);

  return qubit;
}

llvm::SmallVector<Value>
FluxProgramBuilder::allocQubitRegister(const int64_t size,
                                       const StringRef name) {
  llvm::SmallVector<Value> qubits;
  qubits.reserve(static_cast<size_t>(size));

  auto nameAttr = getStringAttr(name);
  auto sizeAttr = getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = getI64IntegerAttr(i);
    auto allocOp = create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
    const auto& qubit = qubits.emplace_back(allocOp.getResult());
    // Track the allocated qubit as valid
    validQubits.insert(qubit);
  }

  return qubits;
}

FluxProgramBuilder::ClassicalRegister&
FluxProgramBuilder::allocClassicalBitRegister(int64_t size, StringRef name) {
  return allocatedClassicalRegisters.emplace_back(name, size);
}

//===----------------------------------------------------------------------===//
// Linear Type Tracking Helpers
//===----------------------------------------------------------------------===//

void FluxProgramBuilder::validateQubitValue(const Value qubit) const {
  if (!validQubits.contains(qubit)) {
    llvm::errs() << "Error: Attempting to use an invalid qubit SSA value. "
                 << "The value may have been consumed by a previous operation "
                 << "or was never created through this \n";
    llvm::reportFatalUsageError(
        "Invalid qubit value used (either consumed or not tracked)");
  }
}

void FluxProgramBuilder::updateQubitTracking(const Value inputQubit,
                                             const Value outputQubit) {
  // Validate the input qubit
  validateQubitValue(inputQubit);

  // Remove the input (consumed) value from tracking
  validQubits.erase(inputQubit);

  // Add the output (new) value to tracking
  validQubits.insert(outputQubit);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

std::pair<Value, Value> FluxProgramBuilder::measure(Value qubit) {
  auto measureOp = create<MeasureOp>(loc, qubit);
  auto qubitOut = measureOp.getQubitOut();
  auto result = measureOp.getResult();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return {qubitOut, result};
}

Value FluxProgramBuilder::measure(Value qubit, const Bit& bit) {
  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  auto measureOp = create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  const auto qubitOut = measureOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

Value FluxProgramBuilder::reset(Value qubit) {
  auto resetOp = create<ResetOp>(loc, qubit);
  const auto qubitOut = resetOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// OneTargetZeroParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetZeroParameter(const Value qubit) {
  auto op = create<OpType>(loc, qubit);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetZeroParameter(const Value control,
                                                           const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0]);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetZeroParameter(
    const ValueRange controls, const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0]);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// OneTargetOneParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const Value qubit) {
  auto op = create<OpType>(loc, qubit, parameter);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const Value control,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], parameter);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const ValueRange controls,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], parameter);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// OneTargetTwoParameter helpers

template <typename OpType>
Value FluxProgramBuilder::createOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const Value qubit) {
  auto op = create<OpType>(loc, qubit, parameter1, parameter2);
  const auto& qubitOut = op.getQubitOut();
  updateQubitTracking(qubit, qubitOut);
  return qubitOut;
}

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createControlledOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const Value control,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op =
                 b.create<OpType>(loc, targets[0], parameter1, parameter2);
             return op->getResults();
           });
  return {controlsOut[0], targetsOut[0]};
}

template <typename OpType>
std::pair<ValueRange, Value>
FluxProgramBuilder::createMultiControlledOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const ValueRange controls,
    const Value target) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, target,
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op =
                 b.create<OpType>(loc, targets[0], parameter1, parameter2);
             return op->getResults();
           });
  return {controlsOut, targetsOut[0]};
}

// TwoTargetZeroParameter helpers

template <typename OpType>
std::pair<Value, Value>
FluxProgramBuilder::createTwoTargetZeroParameter(const Value qubit0,
                                                 const Value qubit1) {
  auto op = create<OpType>(loc, qubit0, qubit1);
  const auto& qubit0Out = op.getQubit0Out();
  const auto& qubit1Out = op.getQubit1Out();
  updateQubitTracking(qubit0, qubit0Out);
  updateQubitTracking(qubit1, qubit1Out);
  return {qubit0Out, qubit1Out};
}

template <typename OpType>
std::pair<Value, std::pair<Value, Value>>
FluxProgramBuilder::createControlledTwoTargetZeroParameter(const Value control,
                                                           const Value qubit0,
                                                           const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(control, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], targets[1]);
             return op->getResults();
           });
  return {controlsOut[0], {targetsOut[0], targetsOut[1]}};
}

template <typename OpType>
std::pair<ValueRange, std::pair<Value, Value>>
FluxProgramBuilder::createMultiControlledTwoTargetZeroParameter(
    const ValueRange controls, const Value qubit0, const Value qubit1) {
  const auto [controlsOut, targetsOut] =
      ctrl(controls, {qubit0, qubit1},
           [&](OpBuilder& b, const ValueRange targets) -> ValueRange {
             const auto op = b.create<OpType>(loc, targets[0], targets[1]);
             return op->getResults();
           });
  return {controlsOut, {targetsOut[0], targetsOut[1]}};
}

// IdOp

Value FluxProgramBuilder::id(const Value qubit) {
  return createOneTargetZeroParameter<IdOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::cid(const Value control,
                                                const Value target) {
  return createControlledOneTargetZeroParameter<IdOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mcid(const ValueRange controls,
                                                      const Value target) {
  return createMultiControlledOneTargetZeroParameter<IdOp>(controls, target);
}

// XOp

Value FluxProgramBuilder::x(const Value qubit) {
  return createOneTargetZeroParameter<XOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::cx(const Value control,
                                               const Value target) {
  return createControlledOneTargetZeroParameter<XOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mcx(const ValueRange controls,
                                                     const Value target) {
  return createMultiControlledOneTargetZeroParameter<XOp>(controls, target);
}

// YOp

Value FluxProgramBuilder::y(const Value qubit) {
  return createOneTargetZeroParameter<YOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::cy(const Value control,
                                               const Value target) {
  return createControlledOneTargetZeroParameter<YOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mcy(const ValueRange controls,
                                                     const Value target) {
  return createMultiControlledOneTargetZeroParameter<YOp>(controls, target);
}

// ZOp

Value FluxProgramBuilder::z(const Value qubit) {
  return createOneTargetZeroParameter<ZOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::cz(const Value control,
                                               const Value target) {
  return createControlledOneTargetZeroParameter<ZOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mcz(const ValueRange controls,
                                                     const Value target) {
  return createMultiControlledOneTargetZeroParameter<ZOp>(controls, target);
}

// HOp

Value FluxProgramBuilder::h(const Value qubit) {
  return createOneTargetZeroParameter<HOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::ch(const Value control,
                                               const Value target) {
  return createControlledOneTargetZeroParameter<HOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mch(const ValueRange controls,
                                                     const Value target) {
  return createMultiControlledOneTargetZeroParameter<HOp>(controls, target);
}

// SOp

Value FluxProgramBuilder::s(const Value qubit) {
  return createOneTargetZeroParameter<SOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::cs(const Value control,
                                               const Value target) {
  return createControlledOneTargetZeroParameter<SOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mcs(const ValueRange controls,
                                                     const Value target) {
  return createMultiControlledOneTargetZeroParameter<SOp>(controls, target);
}

// SdgOp

Value FluxProgramBuilder::sdg(const Value qubit) {
  return createOneTargetZeroParameter<SdgOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::csdg(const Value control,
                                                 const Value target) {
  return createControlledOneTargetZeroParameter<SdgOp>(control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcsdg(const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetZeroParameter<SdgOp>(controls, target);
}

// TOp

Value FluxProgramBuilder::t(const Value qubit) {
  return createOneTargetZeroParameter<TOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::ct(const Value control,
                                               const Value target) {
  return createControlledOneTargetZeroParameter<TOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mct(const ValueRange controls,
                                                     const Value target) {
  return createMultiControlledOneTargetZeroParameter<TOp>(controls, target);
}

// TdgOp

Value FluxProgramBuilder::tdg(const Value qubit) {
  return createOneTargetZeroParameter<TdgOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::ctdg(const Value control,
                                                 const Value target) {
  return createControlledOneTargetZeroParameter<TdgOp>(control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mctdg(const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetZeroParameter<TdgOp>(controls, target);
}

// SXOp

Value FluxProgramBuilder::sx(const Value qubit) {
  return createOneTargetZeroParameter<SXOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::csx(const Value control,
                                                const Value target) {
  return createControlledOneTargetZeroParameter<SXOp>(control, target);
}

std::pair<ValueRange, Value> FluxProgramBuilder::mcsx(const ValueRange controls,
                                                      const Value target) {
  return createMultiControlledOneTargetZeroParameter<SXOp>(controls, target);
}

// SXdgOp

Value FluxProgramBuilder::sxdg(const Value qubit) {
  return createOneTargetZeroParameter<SXdgOp>(qubit);
}

std::pair<Value, Value> FluxProgramBuilder::csxdg(const Value control,
                                                  const Value target) {
  return createControlledOneTargetZeroParameter<SXdgOp>(control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcsxdg(const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetZeroParameter<SXdgOp>(controls, target);
}

// RXOp

Value FluxProgramBuilder::rx(const std::variant<double, Value>& theta,
                             const Value qubit) {
  return createOneTargetOneParameter<RXOp>(theta, qubit);
}

std::pair<Value, Value>
FluxProgramBuilder::crx(const std::variant<double, Value>& theta,
                        const Value control, const Value target) {
  return createControlledOneTargetOneParameter<RXOp>(theta, control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcrx(const std::variant<double, Value>& theta,
                         const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetOneParameter<RXOp>(theta, controls,
                                                          target);
}

// RYOp

Value FluxProgramBuilder::ry(const std::variant<double, Value>& theta,
                             const Value qubit) {
  return createOneTargetOneParameter<RYOp>(theta, qubit);
}

std::pair<Value, Value>
FluxProgramBuilder::cry(const std::variant<double, Value>& theta,
                        const Value control, const Value target) {
  return createControlledOneTargetOneParameter<RYOp>(theta, control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcry(const std::variant<double, Value>& theta,
                         const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetOneParameter<RYOp>(theta, controls,
                                                          target);
}

// RZOp

Value FluxProgramBuilder::rz(const std::variant<double, Value>& theta,
                             const Value qubit) {
  return createOneTargetOneParameter<RZOp>(theta, qubit);
}

std::pair<Value, Value>
FluxProgramBuilder::crz(const std::variant<double, Value>& theta,
                        const Value control, const Value target) {
  return createControlledOneTargetOneParameter<RZOp>(theta, control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcrz(const std::variant<double, Value>& theta,
                         const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetOneParameter<RZOp>(theta, controls,
                                                          target);
}

// POp

Value FluxProgramBuilder::p(const std::variant<double, Value>& theta,
                            const Value qubit) {
  return createOneTargetOneParameter<POp>(theta, qubit);
}

std::pair<Value, Value>
FluxProgramBuilder::cp(const std::variant<double, Value>& theta,
                       const Value control, const Value target) {
  return createControlledOneTargetOneParameter<POp>(theta, control, target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcp(const std::variant<double, Value>& theta,
                        const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetOneParameter<POp>(theta, controls,
                                                         target);
}

// ROp

Value FluxProgramBuilder::r(const std::variant<double, Value>& theta,
                            const std::variant<double, Value>& phi,
                            const Value qubit) {
  return createOneTargetTwoParameter<ROp>(theta, phi, qubit);
}

std::pair<Value, Value>
FluxProgramBuilder::cr(const std::variant<double, Value>& theta,
                       const std::variant<double, Value>& phi,
                       const Value control, const Value target) {
  return createControlledOneTargetTwoParameter<ROp>(theta, phi, control,
                                                    target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcr(const std::variant<double, Value>& theta,
                        const std::variant<double, Value>& phi,
                        const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetTwoParameter<ROp>(theta, phi, controls,
                                                         target);
}

// U2Op

Value FluxProgramBuilder::u2(const std::variant<double, Value>& phi,
                             const std::variant<double, Value>& lambda,
                             Value qubit) {
  return createOneTargetTwoParameter<U2Op>(phi, lambda, qubit);
}

std::pair<Value, Value>
FluxProgramBuilder::cu2(const std::variant<double, Value>& phi,
                        const std::variant<double, Value>& lambda,
                        const Value control, const Value target) {
  return createControlledOneTargetTwoParameter<U2Op>(phi, lambda, control,
                                                     target);
}

std::pair<ValueRange, Value>
FluxProgramBuilder::mcu2(const std::variant<double, Value>& phi,
                         const std::variant<double, Value>& lambda,
                         const ValueRange controls, const Value target) {
  return createMultiControlledOneTargetTwoParameter<U2Op>(phi, lambda, controls,
                                                          target);
}

// SWAPOp

std::pair<Value, Value> FluxProgramBuilder::swap(Value qubit0, Value qubit1) {
  return createTwoTargetZeroParameter<SWAPOp>(qubit0, qubit1);
}

std::pair<Value, std::pair<Value, Value>>
FluxProgramBuilder::cswap(const Value control, Value qubit0, Value qubit1) {
  return createControlledTwoTargetZeroParameter<SWAPOp>(control, qubit0,
                                                        qubit1);
}

std::pair<ValueRange, std::pair<Value, Value>>
FluxProgramBuilder::mcswap(const ValueRange controls, Value qubit0,
                           Value qubit1) {
  return createMultiControlledTwoTargetZeroParameter<SWAPOp>(controls, qubit0,
                                                             qubit1);
}

// iSWAPOp

std::pair<Value, Value> FluxProgramBuilder::iswap(Value qubit0, Value qubit1) {
  return createTwoTargetZeroParameter<iSWAPOp>(qubit0, qubit1);
}

std::pair<Value, std::pair<Value, Value>>
FluxProgramBuilder::ciswap(const Value control, Value qubit0, Value qubit1) {
  return createControlledTwoTargetZeroParameter<iSWAPOp>(control, qubit0,
                                                         qubit1);
}

std::pair<ValueRange, std::pair<Value, Value>>
FluxProgramBuilder::mciswap(const ValueRange controls, Value qubit0,
                            Value qubit1) {
  return createMultiControlledTwoTargetZeroParameter<iSWAPOp>(controls, qubit0,
                                                              qubit1);
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

std::pair<ValueRange, ValueRange> FluxProgramBuilder::ctrl(
    const ValueRange controls, const ValueRange targets,
    const std::function<ValueRange(OpBuilder&, ValueRange)>& body) {
  auto ctrlOp = create<CtrlOp>(loc, controls, targets, body);

  // Update tracking
  const auto& controlsOut = ctrlOp.getControlsOut();
  for (const auto& [control, controlOut] : llvm::zip(controls, controlsOut)) {
    updateQubitTracking(control, controlOut);
  }
  const auto& targetsOut = ctrlOp.getTargetsOut();
  for (const auto& [target, targetOut] : llvm::zip(targets, targetsOut)) {
    updateQubitTracking(target, targetOut);
  }

  return {controlsOut, targetsOut};
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

FluxProgramBuilder& FluxProgramBuilder::dealloc(Value qubit) {
  validateQubitValue(qubit);
  validQubits.erase(qubit);

  create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> FluxProgramBuilder::finalize() {
  // Automatically deallocate all remaining valid qubits
  for (const auto qubit : validQubits) {
    create<DeallocOp>(loc, qubit);
  }

  validQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode = create<arith::ConstantOp>(loc, getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  create<func::ReturnOp>(loc, ValueRange{exitCode});

  return module;
}

} // namespace mlir::flux
