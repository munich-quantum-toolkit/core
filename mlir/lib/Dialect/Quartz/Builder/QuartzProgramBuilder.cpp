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

#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

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
  auto funcType = builder.getFunctionType({}, {});
  auto mainFunc = builder.create<mlir::func::FuncOp>(loc, "main", funcType);

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
  return allocOp.getResult();
}

Value QuartzProgramBuilder::staticQubit(size_t index) {
  // Create the StaticOp with the given index
  auto indexAttr = builder.getI64IntegerAttr(static_cast<int64_t>(index));
  auto staticOp = builder.create<StaticOp>(loc, indexAttr);
  return staticOp.getQubit();
}

SmallVector<Value> QuartzProgramBuilder::allocQubitRegister(size_t size,
                                                            StringRef name) {
  // Allocate a sequence of qubits with register metadata
  SmallVector<Value> qubits;
  qubits.reserve(size);

  auto nameAttr = builder.getStringAttr(name);
  auto sizeAttr = builder.getI64IntegerAttr(static_cast<int64_t>(size));

  for (size_t i = 0; i < size; ++i) {
    auto indexAttr = builder.getI64IntegerAttr(static_cast<int64_t>(i));
    auto allocOp = builder.create<AllocOp>(loc, nameAttr, sizeAttr, indexAttr);
    qubits.push_back(allocOp.getResult());
  }

  return qubits;
}

Value QuartzProgramBuilder::allocClassicalBitRegister(size_t size,
                                                      StringRef name) {
  // Create memref type
  auto memrefType =
      MemRefType::get({static_cast<int64_t>(size)}, builder.getI1Type());

  // Allocate the memref
  auto allocOp = builder.create<memref::AllocOp>(loc, memrefType);

  allocOp->setAttr("sym_name", builder.getStringAttr(name));

  return allocOp.getResult();
}

Value QuartzProgramBuilder::measure(Value qubit) {
  auto measureOp = builder.create<MeasureOp>(loc, qubit);
  return measureOp.getResult();
}

QuartzProgramBuilder& QuartzProgramBuilder::measure(Value qubit, Value memref,
                                                    size_t index) {
  // Measure the qubit
  auto result = measure(qubit);

  // Create constant index for the store operation
  auto indexValue = builder.create<mlir::arith::ConstantIndexOp>(
      loc, static_cast<int64_t>(index));

  // Store the result in the memref at the given index
  builder.create<memref::StoreOp>(loc, result, memref,
                                  ValueRange{indexValue.getResult()});

  return *this;
}

QuartzProgramBuilder& QuartzProgramBuilder::reset(Value qubit) {
  builder.create<ResetOp>(loc, qubit);
  return *this;
}

OwningOpRef<ModuleOp> QuartzProgramBuilder::finalize() {
  // Add return statement to the main function
  builder.create<mlir::func::ReturnOp>(loc);

  // Transfer ownership to the caller
  return module;
}

} // namespace mlir::quartz
