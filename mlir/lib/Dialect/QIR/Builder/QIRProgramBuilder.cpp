/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qir {

QIRProgramBuilder::QIRProgramBuilder(MLIRContext* context)
    : builder(context),
      module(builder.create<ModuleOp>(UnknownLoc::get(context))),
      loc(UnknownLoc::get(context)) {}

void QIRProgramBuilder::initialize() {
  // Ensure LLVM dialect is loaded
  builder.getContext()->loadDialect<LLVM::LLVMDialect>();

  // Set insertion point to the module body
  builder.setInsertionPointToStart(module.getBody());

  // Create main function: () -> i64
  auto funcType = LLVM::LLVMFunctionType::get(builder.getI64Type(), {});
  mainFunc = builder.create<LLVM::LLVMFuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  auto entryPointAttr = StringAttr::get(builder.getContext(), "entry_point");
  mainFunc->setAttr("passthrough",
                    ArrayAttr::get(builder.getContext(), {entryPointAttr}));

  // Create the 4-block structure for QIR Base Profile
  entryBlock = mainFunc.addEntryBlock(builder);
  bodyBlock = mainFunc.addBlock();
  measurementsBlock = mainFunc.addBlock();
  outputBlock = mainFunc.addBlock();

  // Create exit code constant in entry block (where constants belong) and add
  // QIR initialization call in entry block (after exit code constant)
  builder.setInsertionPointToStart(entryBlock);
  auto zeroOp = builder.create<LLVM::ZeroOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));
  exitCode =
      builder.create<LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  const auto initType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto initFunc =
      getOrCreateFunctionDeclaration(builder, module, QIR_INITIALIZE, initType);
  builder.create<LLVM::CallOp>(loc, initFunc, ValueRange{zeroOp.getResult()});

  // Add unconditional branches between blocks
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<LLVM::BrOp>(loc, bodyBlock);

  builder.setInsertionPointToEnd(bodyBlock);
  builder.create<LLVM::BrOp>(loc, measurementsBlock);

  builder.setInsertionPointToEnd(measurementsBlock);
  builder.create<LLVM::BrOp>(loc, outputBlock);

  // Return the exit code (success) in output block
  builder.setInsertionPointToEnd(outputBlock);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{exitCode.getResult()});

  // Set insertion point to body block for user operations
  builder.setInsertionPointToStart(bodyBlock);
}

Value QIRProgramBuilder::staticQubit(const int64_t index) {
  // Check cache
  Value val{};
  if (const auto it = ptrCache.find(index); it != ptrCache.end()) {
    val = it->second;
  } else {
    val = createPointerFromIndex(builder, loc, index);
    // Cache for reuse
    ptrCache[index] = val;
  }

  // Update qubit count
  if (std::cmp_greater_equal(index, metadata_.numQubits)) {
    metadata_.numQubits = static_cast<size_t>(index) + 1;
  }

  return val;
}

llvm::SmallVector<Value>
QIRProgramBuilder::allocQubitRegister(const int64_t size) {
  llvm::SmallVector<Value> qubits;
  qubits.reserve(size);

  for (int64_t i = 0; i < size; ++i) {
    qubits.push_back(staticQubit(static_cast<int64_t>(metadata_.numQubits)));
  }

  return qubits;
}

QIRProgramBuilder::ClassicalRegister&
QIRProgramBuilder::allocClassicalBitRegister(const int64_t size,
                                             StringRef name) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  const auto numResults = static_cast<int64_t>(metadata_.numResults);
  auto& reg = allocatedClassicalRegisters.emplace_back(name, size);
  for (int64_t i = 0; i < size; ++i) {
    Value val{};
    if (const auto it = ptrCache.find(numResults + i); it != ptrCache.end()) {
      val = it->second;
    } else {
      val = createPointerFromIndex(builder, loc, numResults + i);
      // Cache for reuse
      ptrCache[numResults + i] = val;
    }
    registerResultMap.insert({{name, i}, val});
  }
  metadata_.numResults += size;
  return reg;
}

Value QIRProgramBuilder::measure(const Value qubit, const int64_t resultIndex) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  const auto key = std::make_pair("c", resultIndex);
  if (const auto it = registerResultMap.find(key);
      it != registerResultMap.end()) {
    return it->second;
  }

  Value resultValue{};
  if (const auto it = ptrCache.find(resultIndex); it != ptrCache.end()) {
    resultValue = it->second;
  } else {
    resultValue = createPointerFromIndex(builder, loc, resultIndex);
    ptrCache[resultIndex] = resultValue;
    registerResultMap.insert({key, resultValue});
  }

  // Update result count
  if (std::cmp_greater_equal(resultIndex, metadata_.numResults)) {
    metadata_.numResults = static_cast<size_t>(resultIndex) + 1;
  }

  // Create mz call
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto mzSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_MEASURE, mzSignature);
  builder.create<LLVM::CallOp>(loc, mzDecl, ValueRange{qubit, resultValue});

  return resultValue;
}

QIRProgramBuilder& QIRProgramBuilder::measure(const Value qubit,
                                              const Bit& bit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  // Check if we already have a result pointer for this register slot
  const auto& registerName = bit.registerName;
  const auto registerIndex = bit.registerIndex;
  const auto key = std::make_pair(registerName, registerIndex);
  assert(registerResultMap.contains(key));
  const auto resultValue = registerResultMap.at(key);

  // Create mz call
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto mzSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  auto mzDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_MEASURE, mzSignature);
  builder.create<LLVM::CallOp>(loc, mzDecl, ValueRange{qubit, resultValue});

  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::reset(const Value qubit) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in measurements block (before branch)
  builder.setInsertionPoint(measurementsBlock->getTerminator());

  // Create reset call
  const auto qirSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      LLVM::LLVMPointerType::get(builder.getContext()));
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, QIR_RESET, qirSignature);
  builder.create<LLVM::CallOp>(loc, fnDecl, ValueRange{qubit});

  return *this;
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// Helper methods

void QIRProgramBuilder::createOneTargetZeroParameter(const ValueRange controls,
                                                     const Value target,
                                                     StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 1);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 1);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createOneTargetOneParameter(
    const std::variant<double, Value>& parameter, const ValueRange controls,
    const Value target, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameterOperand;
  if (std::holds_alternative<double>(parameter)) {
    parameterOperand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter)))
            .getResult();
  } else {
    parameterOperand = std::get<Value>(parameter);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 2);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter type
  argumentTypes.push_back(Float64Type::get(builder.getContext()));

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 2);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);
  operands.push_back(parameterOperand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createOneTargetTwoParameter(
    const std::variant<double, Value>& parameter1,
    const std::variant<double, Value>& parameter2, const ValueRange controls,
    const Value target, StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard entryGuard(builder);

  // Insert constants in entry block
  builder.setInsertionPointToEnd(entryBlock);

  Value parameter1Operand;
  if (std::holds_alternative<double>(parameter1)) {
    parameter1Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter1)))
            .getResult();
  } else {
    parameter1Operand = std::get<Value>(parameter1);
  }

  Value parameter2Operand;
  if (std::holds_alternative<double>(parameter2)) {
    parameter2Operand =
        builder
            .create<LLVM::ConstantOp>(
                loc, builder.getF64FloatAttr(std::get<double>(parameter2)))
            .getResult();
  } else {
    parameter2Operand = std::get<Value>(parameter2);
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard bodyGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 3);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter types
  argumentTypes.push_back(Float64Type::get(builder.getContext()));
  argumentTypes.push_back(Float64Type::get(builder.getContext()));

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 3);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target);
  operands.push_back(parameter1Operand);
  operands.push_back(parameter2Operand);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

void QIRProgramBuilder::createTwoTargetZeroParameter(const ValueRange controls,
                                                     const Value target0,
                                                     const Value target1,
                                                     StringRef fnName) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in body block (before branch)
  builder.setInsertionPoint(bodyBlock->getTerminator());

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(controls.size() + 2);
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  // Add control pointers
  for (size_t i = 0; i < controls.size(); ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  argumentTypes.push_back(ptrType);
  argumentTypes.push_back(ptrType);

  // Define function signature
  const auto fnSignature = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(builder, module, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(controls.size() + 2);
  operands.append(controls.begin(), controls.end());
  operands.push_back(target0);
  operands.push_back(target1);

  builder.create<LLVM::CallOp>(loc, fnDecl, operands);
}

// IdOp

QIRProgramBuilder& QIRProgramBuilder::id(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_ID);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cid(const Value control,
                                          const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CID);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcid(const ValueRange controls,
                                           const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CID;
  } else if (controls.size() == 2) {
    fnName = QIR_CCID;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCID;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// XOp

QIRProgramBuilder& QIRProgramBuilder::x(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_CX);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cx(const Value control,
                                         const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CX);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcx(const ValueRange controls,
                                          const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CX;
  } else if (controls.size() == 2) {
    fnName = QIR_CCX;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCX;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// YOp

QIRProgramBuilder& QIRProgramBuilder::y(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_Y);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cy(const Value control,
                                         const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CY);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcy(const ValueRange controls,
                                          const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CY;
  } else if (controls.size() == 2) {
    fnName = QIR_CCY;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCY;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// ZOp

QIRProgramBuilder& QIRProgramBuilder::z(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_Z);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cz(const Value control,
                                         const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CZ);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcz(const ValueRange controls,
                                          const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CZ;
  } else if (controls.size() == 2) {
    fnName = QIR_CCZ;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCZ;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// HOp

QIRProgramBuilder& QIRProgramBuilder::h(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_H);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::ch(const Value control,
                                         const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CH);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mch(const ValueRange controls,
                                          const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CH;
  } else if (controls.size() == 2) {
    fnName = QIR_CCH;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCH;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// SOp

QIRProgramBuilder& QIRProgramBuilder::s(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_S);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cs(const Value control,
                                         const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CS);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcs(const ValueRange controls,
                                          const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CS;
  } else if (controls.size() == 2) {
    fnName = QIR_CCS;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCS;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// SdgOp

QIRProgramBuilder& QIRProgramBuilder::sdg(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_SDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::csdg(const Value control,
                                           const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CSDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcsdg(const ValueRange controls,
                                            const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CSDG;
  } else if (controls.size() == 2) {
    fnName = QIR_CCSDG;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCSDG;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// TOp

QIRProgramBuilder& QIRProgramBuilder::t(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_T);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::ct(const Value control,
                                         const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CT);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mct(const ValueRange controls,
                                          const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CT;
  } else if (controls.size() == 2) {
    fnName = QIR_CCT;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCT;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// TdgOp

QIRProgramBuilder& QIRProgramBuilder::tdg(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_TDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::ctdg(const Value control,
                                           const Value target) {
  createOneTargetZeroParameter(control, target, QIR_CTDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mctdg(const ValueRange controls,
                                            const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CTDG;
  } else if (controls.size() == 2) {
    fnName = QIR_CCTDG;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCTDG;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// SXOp

QIRProgramBuilder& QIRProgramBuilder::sx(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_SX);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::csx(const Value control,
                                          const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CSX);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcsx(const ValueRange controls,
                                           const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CSX;
  } else if (controls.size() == 2) {
    fnName = QIR_CCSX;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCSX;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// SXdgOp

QIRProgramBuilder& QIRProgramBuilder::sxdg(const Value qubit) {
  createOneTargetZeroParameter({}, qubit, QIR_SXDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::csxdg(const Value control,
                                            const Value target) {
  createOneTargetZeroParameter({control}, target, QIR_CSXDG);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcsxdg(const ValueRange controls,
                                             const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CSXDG;
  } else if (controls.size() == 2) {
    fnName = QIR_CCSXDG;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCSXDG;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetZeroParameter(controls, target, fnName);
  return *this;
}

// RXOp

QIRProgramBuilder&
QIRProgramBuilder::rx(const std::variant<double, Value>& theta,
                      const Value qubit) {
  createOneTargetOneParameter(theta, {}, qubit, QIR_RX);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::crx(const std::variant<double, Value>& theta,
                       const Value control, const Value target) {
  createOneTargetOneParameter(theta, {control}, target, QIR_CRX);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::mcrx(const std::variant<double, Value>& theta,
                        const ValueRange controls, const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CRX;
  } else if (controls.size() == 2) {
    fnName = QIR_CCRX;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCRX;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetOneParameter(theta, controls, target, fnName);
  return *this;
}

// RYOp

QIRProgramBuilder&
QIRProgramBuilder::ry(const std::variant<double, Value>& theta,
                      const Value qubit) {
  createOneTargetOneParameter(theta, {}, qubit, QIR_RY);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::cry(const std::variant<double, Value>& theta,
                       const Value control, const Value target) {
  createOneTargetOneParameter(theta, {control}, target, QIR_CRY);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::mcry(const std::variant<double, Value>& theta,
                        const ValueRange controls, const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CRY;
  } else if (controls.size() == 2) {
    fnName = QIR_CCRY;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCRY;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetOneParameter(theta, controls, target, fnName);
  return *this;
}

// RZOp

QIRProgramBuilder&
QIRProgramBuilder::rz(const std::variant<double, Value>& theta,
                      const Value qubit) {
  createOneTargetOneParameter(theta, {}, qubit, QIR_RZ);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::crz(const std::variant<double, Value>& theta,
                       const Value control, const Value target) {
  createOneTargetOneParameter(theta, {control}, target, QIR_CRZ);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::mcrz(const std::variant<double, Value>& theta,
                        const ValueRange controls, const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CRZ;
  } else if (controls.size() == 2) {
    fnName = QIR_CCRZ;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCRZ;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetOneParameter(theta, controls, target, fnName);
  return *this;
}

// POp

QIRProgramBuilder&
QIRProgramBuilder::p(const std::variant<double, Value>& theta,
                     const Value qubit) {
  createOneTargetOneParameter(theta, {}, qubit, QIR_P);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::cp(const std::variant<double, Value>& theta,
                      const Value control, const Value target) {
  createOneTargetOneParameter(theta, {control}, target, QIR_CP);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::mcp(const std::variant<double, Value>& theta,
                       const ValueRange controls, const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CP;
  } else if (controls.size() == 2) {
    fnName = QIR_CCP;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCP;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetOneParameter(theta, controls, target, fnName);
  return *this;
}

// ROp

QIRProgramBuilder&
QIRProgramBuilder::r(const std::variant<double, Value>& theta,
                     const std::variant<double, Value>& phi,
                     const Value qubit) {
  createOneTargetTwoParameter(theta, phi, {}, qubit, QIR_R);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::cr(const std::variant<double, Value>& theta,
                      const std::variant<double, Value>& phi,
                      const Value control, const Value target) {
  createOneTargetTwoParameter(theta, phi, {control}, target, QIR_CR);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::mcr(const std::variant<double, Value>& theta,
                       const std::variant<double, Value>& phi,
                       const ValueRange controls, const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CR;
  } else if (controls.size() == 2) {
    fnName = QIR_CCR;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCR;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetTwoParameter(theta, phi, controls, target, fnName);
  return *this;
}

// U2Op

QIRProgramBuilder&
QIRProgramBuilder::u2(const std::variant<double, Value>& phi,
                      const std::variant<double, Value>& lambda,
                      const Value qubit) {
  createOneTargetTwoParameter(phi, lambda, {}, qubit, QIR_U2);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::cu2(const std::variant<double, Value>& phi,
                       const std::variant<double, Value>& lambda,
                       const Value control, const Value target) {
  createOneTargetTwoParameter(phi, lambda, {control}, target, QIR_CU2);
  return *this;
}

QIRProgramBuilder&
QIRProgramBuilder::mcu2(const std::variant<double, Value>& phi,
                        const std::variant<double, Value>& lambda,
                        const ValueRange controls, const Value target) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CU2;
  } else if (controls.size() == 2) {
    fnName = QIR_CCU2;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCU2;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createOneTargetTwoParameter(phi, lambda, controls, target, fnName);
  return *this;
}

// SWAPOp

QIRProgramBuilder& QIRProgramBuilder::swap(const Value qubit0,
                                           const Value qubit1) {
  createTwoTargetZeroParameter({}, qubit0, qubit1, QIR_SWAP);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::cswap(const Value control,
                                            const Value target0,
                                            const Value target1) {
  createTwoTargetZeroParameter({control}, target0, target1, QIR_CSWAP);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mcswap(const ValueRange controls,
                                             const Value target0,
                                             const Value target1) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CSWAP;
  } else if (controls.size() == 2) {
    fnName = QIR_CCSWAP;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCSWAP;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createTwoTargetZeroParameter(controls, target0, target1, fnName);
  return *this;
}

// iSWAPOp

QIRProgramBuilder& QIRProgramBuilder::iswap(const Value qubit0,
                                            const Value qubit1) {
  createTwoTargetZeroParameter({}, qubit0, qubit1, QIR_ISWAP);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::ciswap(const Value control,
                                             const Value target0,
                                             const Value target1) {
  createTwoTargetZeroParameter({control}, target0, target1, QIR_CISWAP);
  return *this;
}

QIRProgramBuilder& QIRProgramBuilder::mciswap(const ValueRange controls,
                                              const Value target0,
                                              const Value target1) {
  StringRef fnName;
  if (controls.size() == 1) {
    fnName = QIR_CISWAP;
  } else if (controls.size() == 2) {
    fnName = QIR_CCISWAP;
  } else if (controls.size() == 3) {
    fnName = QIR_CCCISWAP;
  } else {
    llvm::report_fatal_error("Multi-controlled with more than 3 controls are "
                             "currently not supported");
  }
  createTwoTargetZeroParameter(controls, target0, target1, fnName);
  return *this;
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void QIRProgramBuilder::generateOutputRecording() {
  if (registerResultMap.empty()) {
    return; // No measurements to record
  }

  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Insert in output block (before return)
  builder.setInsertionPoint(outputBlock->getTerminator());

  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  // Group measurements by register
  llvm::StringMap<llvm::SmallVector<std::pair<int64_t, Value>>> registerGroups;
  for (const auto& [key, resultPtr] : registerResultMap) {
    const auto& [regName, regIdx] = key;
    registerGroups[regName].emplace_back(regIdx, resultPtr);
  }

  // Sort registers by name for deterministic output
  llvm::SmallVector<
      std::pair<StringRef, llvm::SmallVector<std::pair<int64_t, Value>>>>
      sortedRegisters;
  for (auto& [name, measurements] : registerGroups) {
    sortedRegisters.emplace_back(name, std::move(measurements));
  }
  sort(sortedRegisters,
       [](const auto& a, const auto& b) { return a.first < b.first; });

  // Create array_record_output call
  const auto arrayRecordSig =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()),
                                  {builder.getI64Type(), ptrType});
  const auto arrayRecordDecl = getOrCreateFunctionDeclaration(
      builder, module, QIR_ARRAY_RECORD_OUTPUT, arrayRecordSig);

  // Create result_record_output calls for each measurement
  const auto resultRecordSig = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()), {ptrType, ptrType});
  const auto resultRecordDecl = getOrCreateFunctionDeclaration(
      builder, module, QIR_RECORD_OUTPUT, resultRecordSig);

  // Generate output recording for each register
  for (auto& [registerName, measurements] : sortedRegisters) {
    // Sort measurements by register index
    sort(measurements,
         [](const auto& a, const auto& b) { return a.first < b.first; });

    const auto arraySize = measurements.size();
    auto arrayLabelOp = createResultLabel(builder, module, registerName);
    auto arraySizeConst = builder.create<LLVM::ConstantOp>(
        loc, builder.getI64IntegerAttr(static_cast<int64_t>(arraySize)));

    builder.create<LLVM::CallOp>(
        loc, arrayRecordDecl,
        ValueRange{arraySizeConst.getResult(), arrayLabelOp.getResult()});

    for (const auto [regIdx, resultPtr] : measurements) {
      // Create label for result: "{registerName}{regIdx}r"
      const std::string resultLabel =
          registerName.str() + std::to_string(regIdx) + "r";
      auto resultLabelOp = createResultLabel(builder, module, resultLabel);

      builder.create<LLVM::CallOp>(
          loc, resultRecordDecl,
          ValueRange{resultPtr, resultLabelOp.getResult()});
    }
  }
}

OwningOpRef<ModuleOp> QIRProgramBuilder::finalize() {
  // Generate output recording in the output block
  generateOutputRecording();

  setQIRAttributes(mainFunc, metadata_);

  return module;
}

} // namespace mlir::qir
