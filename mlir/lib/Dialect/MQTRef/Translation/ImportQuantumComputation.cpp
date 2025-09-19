/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTRef/Translation/ImportQuantumComputation.h"

#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <ranges>
#include <utility>

namespace {

struct QregInfo {
  const qc::QuantumRegister* qregPtr;
  mlir::Value qreg;
  llvm::SmallVector<mlir::Value> qubits;
};

using BitMemInfo = std::pair<mlir::Value, std::size_t>; // (memref, localIdx)
using BitIndexVec = llvm::SmallVector<BitMemInfo>;

/**
 * @brief Allocates a quantum register in the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param numQubits The number of qubits to allocate in the register
 * @return mlir::Value The allocated quantum register value
 */
mlir::Value allocateQreg(mlir::OpBuilder& builder, mlir::MLIRContext* context,
                         const std::size_t numQubits) {
  const auto& qubitType = mqt::ir::ref::QubitType::get(context);
  auto memRefType =
      mlir::MemRefType::get({static_cast<int64_t>(numQubits)}, qubitType);
  auto memref = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(),
                                                      memRefType);
  return memref.getResult();
}

/**
 * @brief Extracts all qubits from a quantum register.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param qreg The quantum register from which to extract qubits
 * @param numQubits The number of qubits to extract
 * @return llvm::SmallVector<mlir::Value> Vector of extracted qubit values
 */
llvm::SmallVector<mlir::Value> extractQubits(mlir::OpBuilder& builder,
                                             mlir::Value qreg,
                                             const std::size_t numQubits) {
  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(numQubits);

  for (std::size_t qubit = 0; qubit < numQubits; ++qubit) {
    auto index = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), qubit);
    qubits.emplace_back(
        builder
            .create<mlir::memref::LoadOp>(builder.getUnknownLoc(), qreg,
                                          mlir::ValueRange{index})
            .getResult());
  }

  return qubits;
}

/**
 * @brief Allocates quantum registers and extracts qubits.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param quantumComputation The quantum computation to translate
 * @return llvm::SmallVector<QregInfo> Vector containing information about all
 * quantum registers
 */
llvm::SmallVector<QregInfo>
getQregs(mlir::OpBuilder& builder, mlir::MLIRContext* context,
         const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::QuantumRegister*> qregPtrs;
  qregPtrs.reserve(quantumComputation.getQuantumRegisters().size() +
                   quantumComputation.getAncillaRegisters().size());
  for (const auto& qreg :
       quantumComputation.getQuantumRegisters() | std::views::values) {
    qregPtrs.emplace_back(&qreg);
  }
  for (const auto& qreg :
       quantumComputation.getAncillaRegisters() | std::views::values) {
    qregPtrs.emplace_back(&qreg);
  }

  // Sort by start index
  std::ranges::sort(
      qregPtrs, [](const qc::QuantumRegister* a, const qc::QuantumRegister* b) {
        return a->getStartIndex() < b->getStartIndex();
      });

  // Allocate quantum registers and extract qubits
  llvm::SmallVector<QregInfo> qregs;
  for (const auto* qregPtr : qregPtrs) {
    const auto qreg = allocateQreg(builder, context, qregPtr->getSize());
    auto qubits = extractQubits(builder, qreg, qregPtr->getSize());
    qregs.emplace_back(qregPtr, qreg, std::move(qubits));
  }

  return qregs;
}

/**
 * @brief Builds a mapping from global qubit index to extracted qubit value.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param context The MLIR context in which types are created
 * @param qregs Vector containing information about all quantum registers
 * @return llvm::SmallVector<mlir::Value> Sorted vector of qubit values
 */
llvm::SmallVector<mlir::Value>
getQubits(const qc::QuantumComputation& quantumComputation,
          llvm::SmallVector<QregInfo>& qregs) {
  llvm::SmallVector<mlir::Value> flatQubits;
  const auto maxPhys = quantumComputation.getHighestPhysicalQubitIndex();
  flatQubits.resize(static_cast<size_t>(maxPhys) + 1);
  for (const auto& qreg : qregs) {
    for (std::size_t i = 0; i < qreg.qregPtr->getSize(); ++i) {
      const auto globalIdx =
          static_cast<size_t>(qreg.qregPtr->getStartIndex() + i);
      flatQubits[globalIdx] = qreg.qubits[i];
    }
  }

  return flatQubits;
}

/**
 * @brief Allocates a classical register in the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param numBits The number of bits to allocate in the register
 * @return mlir::Value The allocated classical register value
 */
mlir::Value allocateBits(mlir::OpBuilder& builder, int64_t numBits) {
  auto memRefType = mlir::MemRefType::get({numBits}, builder.getI1Type());
  auto memref = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(),
                                                       memRefType);
  return memref.getResult();
}

/**
 * @brief Builds a mapping from global bit index to (memref, localIdx).
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param numBits The number of bits to allocate in the register
 * @return mlir::Value The allocated classical register value
 */
BitIndexVec getBitMap(mlir::OpBuilder& builder,
                      const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::ClassicalRegister*> cregPtrs;
  cregPtrs.reserve(quantumComputation.getClassicalRegisters().size());
  for (const auto& [_, reg] : quantumComputation.getClassicalRegisters()) {
    cregPtrs.emplace_back(&reg);
  }

  // Sort by start index
  std::ranges::sort(cregPtrs, [](const qc::ClassicalRegister* a,
                                 const qc::ClassicalRegister* b) {
    return a->getStartIndex() < b->getStartIndex();
  });

  // Build mapping
  BitIndexVec bitMap;
  bitMap.resize(quantumComputation.getNcbits());
  for (const auto* reg : cregPtrs) {
    auto mem = allocateBits(builder, static_cast<int64_t>(reg->getSize()));
    for (std::size_t i = 0; i < reg->getSize(); ++i) {
      const auto globalIdx = static_cast<std::size_t>(reg->getStartIndex() + i);
      bitMap[globalIdx] = {mem, i};
    }
  }

  return bitMap;
}

/**
 * @brief Adds a single quantum operation to the MLIR module.
 *
 * @tparam OpType The type of the operation to create
 * @param builder The MLIR OpBuilder
 * @param operation The quantum operation to add
 * @param qubits The qubits of the quantum register
 */
template <typename OpType>
void addUnitaryOp(mlir::OpBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<mlir::Value>& qubits) {
  // Define operation parameters
  mlir::DenseF64ArrayAttr staticParamsAttr = nullptr;
  if (const auto& parameters = operation.getParameter(); !parameters.empty()) {
    staticParamsAttr = builder.getDenseF64ArrayAttr(parameters);
  }

  // Define input qubits
  const auto& targets = operation.getTargets();
  llvm::SmallVector<mlir::Value> inQubits;
  inQubits.reserve(targets.size());
  for (const auto& t : targets) {
    inQubits.emplace_back(qubits[t]);
  }

  // Define control qubits
  llvm::SmallVector<mlir::Value> posCtrlQubits;
  llvm::SmallVector<mlir::Value> negCtrlQubits;
  if (const auto& controls = operation.getControls(); !controls.empty()) {
    posCtrlQubits.reserve(controls.size());
    negCtrlQubits.reserve(controls.size());
    for (const auto& [qubit, type] : controls) {
      if (type == qc::Control::Type::Pos) {
        posCtrlQubits.emplace_back(qubits[qubit]);
      } else if (type == qc::Control::Type::Neg) {
        negCtrlQubits.emplace_back(qubits[qubit]);
      }
    }
  }

  // Create operation
  builder.create<OpType>(builder.getUnknownLoc(), staticParamsAttr, nullptr,
                         mlir::ValueRange{}, inQubits, posCtrlQubits,
                         negCtrlQubits);
}

/**
 * @brief Adds a measure operation to the MLIR module.
 *
 * @param builder The MLIR OpBuilder
 * @param operation The measure operation to add
 * @param qubits The qubits of the quantum register
 * @param bitMap The mapping from global classical bit index to (memref,
 * localIdx)
 */
void addMeasureOp(mlir::OpBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<mlir::Value>& qubits,
                  const BitIndexVec& bitMap) {
  const auto& measureOp =
      dynamic_cast<const qc::NonUnitaryOperation&>(operation);
  const auto& targets = measureOp.getTargets();
  const auto& classics = measureOp.getClassics();
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const auto& qubit = qubits[targets[i]];
    auto result =
        builder.create<mqt::ir::ref::MeasureOp>(builder.getUnknownLoc(), qubit);
    const auto bitIdx = static_cast<std::size_t>(classics[i]);
    assert(bitIdx < bitMap.size() && "Classical bit index out of range");
    const auto& [mem, localIdx] = bitMap[bitIdx];
    auto idxVal = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), static_cast<int64_t>(localIdx));
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), result, mem,
                                          mlir::ValueRange{idxVal});
  }
}

/**
 * @brief Adds a reset operation to the MLIR module.
 *
 * @param builder The MLIR OpBuilder
 * @param operation The reset operation to add
 * @param qubits The qubits of the quantum register
 */
void addResetOp(mlir::OpBuilder& builder, const qc::Operation& operation,
                const llvm::SmallVector<mlir::Value>& qubits) {
  for (const auto& target : operation.getTargets()) {
    const mlir::Value inQubit = qubits[target];
    builder.create<mqt::ir::ref::ResetOp>(builder.getUnknownLoc(), inQubit);
  }
}

// Forward declaration
llvm::LogicalResult addOperation(mlir::OpBuilder& builder,
                                 const qc::Operation& operation,
                                 const llvm::SmallVector<mlir::Value>& qubits,
                                 const BitIndexVec& bitMap);

/**
 * @brief Compute integer value from a classical register (memref<Nxi1>).
 *
 * @param builder The MLIR OpBuilder
 * @param bits The bits of the classical register
 * @return mlir::Value The integer value of the register
 */
mlir::Value getIntegerValueFromRegister(mlir::OpBuilder& builder,
                                        const mlir::Value bits) {
  const auto loc = builder.getUnknownLoc();

  // Extract length (assumed 1-D static memref<Nxi1>)
  const auto bitsType = mlir::cast<mlir::MemRefType>(bits.getType());
  const auto shape = bitsType.getShape();
  assert(shape.size() == 1 &&
         "Expected 1-D memref type in getIntegerValueFromRegister");
  const auto size = shape[0];

  // Loop constants
  auto lb = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto ub = builder.create<mlir::arith::ConstantIndexOp>(loc, size);
  auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  // Initial accumulator (i64 0)
  auto initial = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(0));

  // for (i = 0; i < size; ++i) acc += (bit_i << i)
  auto loop = builder.create<mlir::scf::ForOp>(
      loc, lb, ub, step, mlir::ValueRange{initial},
      [&builder, &bits](mlir::OpBuilder& b, const mlir::Location forLoc,
                        mlir::Value iv, const mlir::ValueRange iterArgs) {
        auto bit =
            b.create<mlir::memref::LoadOp>(forLoc, bits, mlir::ValueRange{iv});
        auto bitExt =
            b.create<mlir::arith::ExtUIOp>(forLoc, builder.getI64Type(), bit);

        // Cast loop index to i64 for shift amount
        auto shiftAmt = b.create<mlir::arith::IndexCastOp>(
            forLoc, builder.getI64Type(), iv);
        // shifted = bitExt << shiftAmt
        auto shifted = b.create<mlir::arith::ShLIOp>(forLoc, bitExt, shiftAmt);

        auto acc = b.create<mlir::arith::AddIOp>(forLoc, iterArgs[0], shifted);
        b.create<mlir::scf::YieldOp>(forLoc, acc.getResult());
      });

  return loop.getResult(0);
}

/**
 * @brief Adds an if-else operation to the MLIR module.
 *
 * @param builder The MLIR OpBuilder
 * @param op The if-else operation to add
 * @param qubits The qubits of the quantum register
 * @param bitMap The mapping from global classical bit index to (memref,
 * localIdx)
 * @return mlir::LogicalResult Success if all operations were added, failure
 * otherwise
 */
llvm::LogicalResult addIfElseOp(mlir::OpBuilder& builder,
                                const qc::Operation& op,
                                const llvm::SmallVector<mlir::Value>& qubits,
                                const BitIndexVec& bitMap) {
  const auto loc = builder.getUnknownLoc();
  const auto& ifElse = dynamic_cast<const qc::IfElseOperation&>(op);

  const auto* thenOp = ifElse.getThenOp();
  // Canonicalization should have removed empty then blocks
  assert(thenOp != nullptr);
  const auto* elseOp = ifElse.getElseOp();

  mlir::Value controlValue;
  mlir::Value expectedValue;
  if (ifElse.getControlRegister().has_value()) {
    const auto& ctrlReg = ifElse.getControlRegister().value();
    const auto startIdx = static_cast<std::size_t>(ctrlReg.getStartIndex());
    assert(startIdx < bitMap.size() &&
           "Control register start index out of range");
    const auto mem = bitMap[startIdx].first;
    controlValue = getIntegerValueFromRegister(builder, mem);

    expectedValue = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(
                 static_cast<int64_t>(ifElse.getExpectedValueRegister())));
  } else {
    const auto controlBit = ifElse.getControlBit();
    const auto bitIdx = static_cast<std::size_t>(*controlBit);
    assert(bitIdx < bitMap.size() && "Control bit index out of range");
    const auto& [mem, localIdx] = bitMap[bitIdx];
    auto indexValue = builder.create<mlir::arith::ConstantIndexOp>(
        loc, static_cast<int64_t>(localIdx));
    controlValue = builder.create<mlir::memref::LoadOp>(
        loc, mem, mlir::ValueRange{indexValue});
    expectedValue = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(
                 builder.getI1Type(),
                 static_cast<int64_t>(ifElse.getExpectedValueBit())));
  }

  // Define comparison predicate
  const auto comparisonKind = ifElse.getComparisonKind();
  auto predicate = mlir::arith::CmpIPredicate::eq;
  switch (comparisonKind) {
  case qc::ComparisonKind::Eq:
    predicate = mlir::arith::CmpIPredicate::eq;
    break;
  case qc::ComparisonKind::Neq:
    predicate = mlir::arith::CmpIPredicate::ne;
    break;
  case qc::ComparisonKind::Lt:
    predicate = mlir::arith::CmpIPredicate::ult;
    break;
  case qc::ComparisonKind::Leq:
    predicate = mlir::arith::CmpIPredicate::ule;
    break;
  case qc::ComparisonKind::Gt:
    predicate = mlir::arith::CmpIPredicate::ugt;
    break;
  case qc::ComparisonKind::Geq:
    predicate = mlir::arith::CmpIPredicate::uge;
    break;
  }

  // Define condition
  auto condition = builder.create<mlir::arith::CmpIOp>(
      loc, predicate, controlValue, expectedValue);

  // Define operation
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{},
                                              condition.getResult(), true);

  // Populate then block
  {
    const mlir::OpBuilder::InsertionGuard thenGuard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    if (addOperation(builder, *thenOp, qubits, bitMap).failed()) {
      return llvm::failure();
    }
  }

  // Populate else block
  if (elseOp != nullptr) {
    const mlir::OpBuilder::InsertionGuard elseGuard(builder);
    builder.setInsertionPointToStart(ifOp.elseBlock());
    if (addOperation(builder, *elseOp, qubits, bitMap).failed()) {
      return llvm::failure();
    }
  }

  return llvm::success();
}

#define ADD_OP_CASE(op)                                                        \
  case qc::OpType::op:                                                         \
    addUnitaryOp<mqt::ir::ref::op##Op>(builder, operation, qubits);            \
    return llvm::success();

/**
 * @brief Adds a single quantum operation to the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param operation The quantum operation to add
 * @param qubits The qubits of the quantum register
 * @param bitMap The mapping from global classical bit index to (memref,
 * localIdx)
 * @return mlir::LogicalResult Success if all operations were added, failure
 * otherwise
 */
llvm::LogicalResult addOperation(mlir::OpBuilder& builder,
                                 const qc::Operation& operation,
                                 const llvm::SmallVector<mlir::Value>& qubits,
                                 const BitIndexVec& bitMap) {
  switch (operation.getType()) {
    ADD_OP_CASE(I)
    ADD_OP_CASE(H)
    ADD_OP_CASE(X)
    ADD_OP_CASE(Y)
    ADD_OP_CASE(Z)
    ADD_OP_CASE(S)
    ADD_OP_CASE(Sdg)
    ADD_OP_CASE(T)
    ADD_OP_CASE(Tdg)
    ADD_OP_CASE(V)
    ADD_OP_CASE(Vdg)
    ADD_OP_CASE(U)
    ADD_OP_CASE(U2)
    ADD_OP_CASE(P)
    ADD_OP_CASE(SX)
    ADD_OP_CASE(SXdg)
    ADD_OP_CASE(RX)
    ADD_OP_CASE(RY)
    ADD_OP_CASE(RZ)
    ADD_OP_CASE(SWAP)
    ADD_OP_CASE(iSWAP)
    ADD_OP_CASE(iSWAPdg)
    ADD_OP_CASE(Peres)
    ADD_OP_CASE(Peresdg)
    ADD_OP_CASE(DCX)
    ADD_OP_CASE(ECR)
    ADD_OP_CASE(RXX)
    ADD_OP_CASE(RYY)
    ADD_OP_CASE(RZZ)
    ADD_OP_CASE(RZX)
    ADD_OP_CASE(XXminusYY)
    ADD_OP_CASE(XXplusYY)
  case qc::OpType::Measure:
    addMeasureOp(builder, operation, qubits, bitMap);
    return llvm::success();
  case qc::OpType::Reset:
    addResetOp(builder, operation, qubits);
    return llvm::success();
  case qc::OpType::IfElse:
    return addIfElseOp(builder, operation, qubits, bitMap);
  default:
    return llvm::failure();
  }
}

/**
 * @brief Adds quantum operations to the MLIR module.
 *
 * @param builder The MLIR OpBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @param qubits The qubits of the quantum register
 * @param bitMap The mapping from global classical bit index to (memref,
 * localIdx)
 * @return mlir::LogicalResult Success if all operations were added, failure
 * otherwise
 */
llvm::LogicalResult addOperations(
    mlir::OpBuilder& builder, const qc::QuantumComputation& quantumComputation,
    const llvm::SmallVector<mlir::Value>& qubits, const BitIndexVec& bitMap) {
  for (const auto& operation : quantumComputation) {
    if (const auto result = addOperation(builder, *operation, qubits, bitMap);
        result.failed()) {
      return result;
    }
  }
  return llvm::success();
}
} // namespace

/**
 * @brief Translates a QuantumComputation to an MLIR module with MQTRef
 * operations.
 *
 * This function takes a quantum computation and translates it into an MLIR
 * module containing MQTRef dialect operations. It creates a main function that
 * contains all quantum operations from the input computation.
 *
 * @param context The MLIR context in which the module will be created
 * @param quantumComputation The quantum computation to translate
 * @return mlir::OwningOpRef<mlir::ModuleOp> The translated MLIR module
 */
mlir::OwningOpRef<mlir::ModuleOp> translateQuantumComputationToMLIR(
    mlir::MLIRContext* context,
    const qc::QuantumComputation& quantumComputation) {
  mlir::OpBuilder builder(context);
  const auto loc = builder.getUnknownLoc();

  // Create module
  auto module = builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Create main function as entry point
  auto funcType = builder.getFunctionType({}, {});
  auto mainFunc = builder.create<mlir::func::FuncOp>(loc, "main", funcType);

  // Add entry_point attribute to identify the main function
  const auto entryPointAttr = mlir::StringAttr::get(context, "entry_point");
  mainFunc->setAttr("passthrough",
                    mlir::ArrayAttr::get(context, {entryPointAttr}));

  auto& entryBlock = mainFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Allocate quantum registers and extract qubits
  auto qregs = getQregs(builder, context, quantumComputation);
  auto qubits = getQubits(quantumComputation, qregs);

  // Allocate classical registers
  auto bitMap = getBitMap(builder, quantumComputation);

  // Add operations and handle potential failures
  if (addOperations(builder, quantumComputation, qubits, bitMap).failed()) {
    // Even if operations fail, return the module with what we could translate
    emitError(loc) << "Failed to translate some quantum operations";
  }
  // Create terminator
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}
