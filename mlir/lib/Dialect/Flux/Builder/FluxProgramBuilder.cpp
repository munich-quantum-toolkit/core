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
#include <utility>

namespace mlir::flux {

FluxProgramBuilder::FluxProgramBuilder(MLIRContext* context)
    : builder(context),
      module(builder.create<ModuleOp>(UnknownLoc::get(context))),
      loc(UnknownLoc::get(context)) {}

void FluxProgramBuilder::initialize() {
  // Ensure the Flux dialect is loaded
  builder.getContext()->loadDialect<FluxDialect>();

  // Set insertion point to the module body
  builder.setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = builder.getFunctionType({}, {builder.getI64Type()});
  auto mainFunc = builder.create<func::FuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(builder.getContext(), "entry_point");
  mainFunc->setAttr("passthrough",
                    ArrayAttr::get(builder.getContext(), {entryPointAttr}));

  // Create entry block and set insertion point
  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);
}

Value FluxProgramBuilder::allocQubit() {
  auto allocOp = builder.create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit as valid
  validQubits.insert(qubit);

  return qubit;
}

Value FluxProgramBuilder::staticQubit(const int64_t index) {
  auto indexAttr = builder.getI64IntegerAttr(index);
  auto staticOp = builder.create<StaticOp>(loc, indexAttr);
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

  auto nameAttr = builder.getStringAttr(name);
  auto sizeAttr = builder.getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = builder.getI64IntegerAttr(i);
    auto allocOp = builder.create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
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
                 << "or was never created through this builder.\n";
    llvm_unreachable(
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
  auto measureOp = builder.create<MeasureOp>(loc, qubit);
  auto qubitOut = measureOp.getQubitOut();
  auto result = measureOp.getResult();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return {qubitOut, result};
}

Value FluxProgramBuilder::measure(Value qubit, const Bit& bit) {
  auto nameAttr = builder.getStringAttr(bit.registerName);
  auto sizeAttr = builder.getI64IntegerAttr(bit.registerSize);
  auto indexAttr = builder.getI64IntegerAttr(bit.registerIndex);
  auto measureOp =
      builder.create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  const auto qubitOut = measureOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

Value FluxProgramBuilder::reset(Value qubit) {
  auto resetOp = builder.create<ResetOp>(loc, qubit);
  const auto qubitOut = resetOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

Value FluxProgramBuilder::x(Value qubit) {
  auto xOp = builder.create<XOp>(loc, qubit);
  const auto qubitOut = xOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

Value FluxProgramBuilder::rx(Value angle, Value qubit) {
  auto rxOp = builder.create<RXOp>(loc, qubit, angle);
  const auto qubitOut = rxOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

Value FluxProgramBuilder::rx(double angle, Value qubit) {
  auto rxOp = builder.create<RXOp>(loc, qubit, angle);
  const auto qubitOut = rxOp.getQubitOut();

  // Update tracking
  updateQubitTracking(qubit, qubitOut);

  return qubitOut;
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

FluxProgramBuilder& FluxProgramBuilder::dealloc(Value qubit) {
  validateQubitValue(qubit);
  validQubits.erase(qubit);

  builder.create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> FluxProgramBuilder::finalize() {
  // Automatically deallocate all remaining valid qubits
  for (const auto qubit : validQubits) {
    builder.create<DeallocOp>(loc, qubit);
  }

  validQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode =
      builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  builder.create<func::ReturnOp>(loc, ValueRange{exitCode});

  return module;
}

} // namespace mlir::flux
