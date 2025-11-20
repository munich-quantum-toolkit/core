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
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <variant>

namespace mlir::quartz {

QuartzProgramBuilder::QuartzProgramBuilder(MLIRContext* context)
    : OpBuilder(context), ctx(context), loc(getUnknownLoc()),
      module(create<ModuleOp>(loc)) {}

void QuartzProgramBuilder::initialize() {
  // Ensure the Quartz dialect is loaded
  ctx->loadDialect<QuartzDialect>();

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

Value QuartzProgramBuilder::allocQubit() {
  // Create the AllocOp without register metadata
  auto allocOp = create<AllocOp>(loc);
  const auto qubit = allocOp.getResult();

  // Track the allocated qubit for automatic deallocation
  allocatedQubits.insert(qubit);

  return qubit;
}

Value QuartzProgramBuilder::staticQubit(const int64_t index) {
  // Create the StaticOp with the given index
  auto indexAttr = getI64IntegerAttr(index);
  auto staticOp = create<StaticOp>(loc, indexAttr);
  return staticOp.getQubit();
}

llvm::SmallVector<Value>
QuartzProgramBuilder::allocQubitRegister(const int64_t size,
                                         const StringRef name) {
  // Allocate a sequence of qubits with register metadata
  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);

  auto nameAttr = getStringAttr(name);
  auto sizeAttr = getI64IntegerAttr(size);

  for (int64_t i = 0; i < size; ++i) {
    auto indexAttr = getI64IntegerAttr(i);
    auto allocOp = create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
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
  auto measureOp = create<MeasureOp>(loc, qubit);
  return measureOp.getResult();
}

QuartzProgramBuilder& QuartzProgramBuilder::measure(Value qubit,
                                                    const Bit& bit) {
  auto nameAttr = getStringAttr(bit.registerName);
  auto sizeAttr = getI64IntegerAttr(bit.registerSize);
  auto indexAttr = getI64IntegerAttr(bit.registerIndex);
  create<MeasureOp>(loc, qubit, nameAttr, sizeAttr, indexAttr);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::reset(Value qubit) {
  create<ResetOp>(loc, qubit);
  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// XOp

QuartzProgramBuilder& QuartzProgramBuilder::x(Value qubit) {
  create<XOp>(loc, qubit);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::cx(Value control, Value target) {
  return mcx({control}, target);
}

QuartzProgramBuilder& QuartzProgramBuilder::mcx(ValueRange controls,
                                                Value target) {
  create<CtrlOp>(loc, controls,
                 [&](OpBuilder& b) { b.create<XOp>(loc, target); });
  return *this;
}

// SOp

QuartzProgramBuilder& QuartzProgramBuilder::s(Value qubit) {
  create<SOp>(loc, qubit);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::cs(Value control, Value target) {
  return mcs({control}, target);
}

QuartzProgramBuilder& QuartzProgramBuilder::mcs(ValueRange controls,
                                                Value target) {
  create<CtrlOp>(loc, controls,
                 [&](OpBuilder& b) { b.create<SOp>(loc, target); });
  return *this;
}

// SdgOp

QuartzProgramBuilder& QuartzProgramBuilder::sdg(Value qubit) {
  create<SdgOp>(loc, qubit);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::csdg(Value control, Value target) {
  return mcsdg({control}, target);
}

QuartzProgramBuilder& QuartzProgramBuilder::mcsdg(ValueRange controls,
                                                  Value target) {
  create<CtrlOp>(loc, controls,
                 [&](OpBuilder& b) { b.create<SdgOp>(loc, target); });
  return *this;
}

// RXOp

QuartzProgramBuilder&
QuartzProgramBuilder::rx(const std::variant<double, Value>& theta,
                         Value qubit) {
  create<RXOp>(loc, qubit, theta);
  return *this;
}

QuartzProgramBuilder&
QuartzProgramBuilder::crx(const std::variant<double, Value>& theta,
                          Value control, const Value target) {
  return mcrx(theta, {control}, target);
}

QuartzProgramBuilder&
QuartzProgramBuilder::mcrx(const std::variant<double, Value>& theta,
                           ValueRange controls, Value target) {
  create<CtrlOp>(loc, controls,
                 [&](OpBuilder& b) { b.create<RXOp>(loc, target, theta); });
  return *this;
}

// U2Op

QuartzProgramBuilder&
QuartzProgramBuilder::u2(const std::variant<double, Value>& phi,
                         const std::variant<double, Value>& lambda,
                         Value qubit) {
  create<U2Op>(loc, qubit, phi, lambda);
  return *this;
}

QuartzProgramBuilder&
QuartzProgramBuilder::cu2(const std::variant<double, Value>& phi,
                          const std::variant<double, Value>& lambda,
                          Value control, const Value target) {
  return mcu2(phi, lambda, {control}, target);
}

QuartzProgramBuilder&
QuartzProgramBuilder::mcu2(const std::variant<double, Value>& phi,
                           const std::variant<double, Value>& lambda,
                           ValueRange controls, Value target) {
  create<CtrlOp>(loc, controls, [&](OpBuilder& b) {
    b.create<U2Op>(loc, target, phi, lambda);
  });
  return *this;
}

// SWAPOp

QuartzProgramBuilder& QuartzProgramBuilder::swap(Value qubit0, Value qubit1) {
  create<SWAPOp>(loc, qubit0, qubit1);
  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::cswap(Value control,
                                                  const Value qubit0,
                                                  const Value qubit1) {
  return mcswap({control}, qubit0, qubit1);
}

QuartzProgramBuilder& QuartzProgramBuilder::mcswap(ValueRange controls,
                                                   Value qubit0, Value qubit1) {
  create<CtrlOp>(loc, controls,
                 [&](OpBuilder& b) { b.create<SWAPOp>(loc, qubit0, qubit1); });
  return *this;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

QuartzProgramBuilder&
QuartzProgramBuilder::ctrl(ValueRange controls,
                           const std::function<void(OpBuilder&)>& body) {
  create<CtrlOp>(loc, controls, body);
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
    llvm::reportFatalUsageError(
        "Double deallocation or invalid qubit deallocation");
  }

  // Create the DeallocOp
  create<DeallocOp>(loc, qubit);

  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> QuartzProgramBuilder::finalize() {
  // Automatically deallocate all remaining allocated qubits
  for (const Value qubit : allocatedQubits) {
    create<DeallocOp>(loc, qubit);
  }

  // Clear the tracking set
  allocatedQubits.clear();

  // Create constant 0 for successful exit code
  auto exitCode = create<arith::ConstantOp>(loc, getI64IntegerAttr(0));

  // Add return statement with exit code 0 to the main function
  create<func::ReturnOp>(loc, ValueRange{exitCode});

  // Transfer ownership to the caller
  return module;
}

} // namespace mlir::quartz
