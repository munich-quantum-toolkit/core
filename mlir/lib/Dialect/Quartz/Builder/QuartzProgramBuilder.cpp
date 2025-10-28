/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

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

namespace mlir::quartz {

QuartzProgramBuilder::QuartzProgramBuilder(MLIRContext* context)
    : builder(context),
      module(builder.create<ModuleOp>(UnknownLoc::get(context))),
      loc(UnknownLoc::get(context)) {}

void QuartzProgramBuilder::initialize() {
  // Ensure the Quartz dialect is loaded
  builder.getContext()->loadDialect<QuartzDialect>();

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

Value QuartzProgramBuilder::allocQubit() {
  // Create the AllocOp without register metadata
  auto allocOp = builder.create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit for automatic deallocation
  allocatedQubits.insert(qubit);

  return qubit;
}

Value QuartzProgramBuilder::staticQubit(const int64_t index) {
  // Create the StaticOp with the given index
  auto indexAttr = builder.getI64IntegerAttr(index);
  auto staticOp = builder.create<StaticOp>(loc, indexAttr);
  return staticOp.getQubit();
}

llvm::SmallVector<Value>
QuartzProgramBuilder::allocQubitRegister(const int64_t size,
                                         const StringRef name) {
  // Allocate a sequence of qubits with register metadata
  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);

  auto nameAttr = builder.getStringAttr(name);
  auto sizeAttr = builder.getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = builder.getI64IntegerAttr(i);
    auto allocOp = builder.create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
    const auto& qubit = qubits.emplace_back(allocOp.getResult());
    // Track the allocated qubit for automatic deallocation
    allocatedQubits.insert(qubit);
  }

  return qubits;
}

QuartzProgramBuilder::ClassicalRegister&
QuartzProgramBuilder::allocClassicalBitRegister(int64_t size, StringRef name) {
  return allocatedClassicalRegisters.emplace_back(name, size);
}

//===----------------------------------------------------------------------===//
// Measurement and Reset
//===----------------------------------------------------------------------===//

Value QuartzProgramBuilder::measure(Value qubit) {
  auto measureOp = builder.create<MeasureOp>(loc, qubit);
  return measureOp.getResult();
}

QuartzProgramBuilder& QuartzProgramBuilder::measure(Value qubit,
                                                    const Bit& bit) {
  auto nameAttr = builder.getStringAttr(bit.registerName);
  auto sizeAttr = builder.getI64IntegerAttr(bit.registerSize);
  auto indexAttr = builder.getI64IntegerAttr(bit.registerIndex);
  builder.create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::reset(Value qubit) {
  builder.create<ResetOp>(loc, qubit);
  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

QuartzProgramBuilder& QuartzProgramBuilder::x(Value qubit) {
  builder.create<XOp>(loc, qubit);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::rx(double theta, Value qubit) {
  builder.create<RXOp>(loc, qubit, theta);
  return *this;
}
QuartzProgramBuilder& QuartzProgramBuilder::rx(Value theta, Value qubit) {
  builder.create<RXOp>(loc, qubit, theta);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::u2(double phi, double lambda,
                                               Value qubit) {
  builder.create<U2Op>(loc, qubit, phi, lambda);
  return *this;
}
QuartzProgramBuilder& QuartzProgramBuilder::u2(double phi, Value lambda,
                                               Value qubit) {
  builder.create<U2Op>(loc, qubit, phi, lambda);
  return *this;
}
QuartzProgramBuilder& QuartzProgramBuilder::u2(Value phi, double lambda,
                                               Value qubit) {
  builder.create<U2Op>(loc, qubit, phi, lambda);
  return *this;
}
QuartzProgramBuilder& QuartzProgramBuilder::u2(Value phi, Value lambda,
                                               Value qubit) {
  builder.create<U2Op>(loc, qubit, phi, lambda);
  return *this;
}

//===----------------------------------------------------------------------===//
// Deallocation
//===----------------------------------------------------------------------===//

QuartzProgramBuilder& QuartzProgramBuilder::dealloc(Value qubit) {
  // Check if the qubit is in the tracking set
  if (!allocatedQubits.erase(qubit)) {
    // Qubit was not found in the set - either never allocated or already
    // deallocated
    llvm::errs() << "Error: Attempting to deallocate a qubit that was not "
                    "allocated or has already been deallocated\n";
    llvm_unreachable("Double deallocation or invalid qubit deallocation");
  }

  // Create the DeallocOp
  builder.create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> QuartzProgramBuilder::finalize() {
  // Automatically deallocate all remaining allocated qubits
  for (const Value qubit : allocatedQubits) {
    builder.create<DeallocOp>(loc, qubit);
  }

  // Clear the tracking set
  allocatedQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode =
      builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  builder.create<func::ReturnOp>(loc, ValueRange{exitCode});

  // Transfer ownership to the caller
  return module;
}

} // namespace mlir::quartz
