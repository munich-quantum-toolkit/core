/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir_programs.h"

#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Support/LLVM.h>

#include <numbers>

/**
 * @brief Creates a struct value using an `llvm.poison` operation with the given
 * types.
 * @param b The QIRProgramBuilder used to create the struct.
 * @param types The types of the elements in the struct.
 * @return The created struct value.
 */
mlir::Value createStruct(mlir::qir::QIRProgramBuilder& b,
                         mlir::SmallVector<mlir::Type> types) {
  auto structType =
      mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), types);
  mlir::Value structValue = mlir::LLVM::PoisonOp::create(b, structType);
  return structValue;
}

/**
 * @brief Collects measurement outcomes into a struct value.
 * @param b The QIRProgramBuilder used to create the struct.
 * @param outcomes The measurement outcomes to be read and collected.
 * @param structValue The struct value to insert the measurement outcomes into.
 * @return The struct value with the measurement outcomes inserted.
 */
mlir::Value
collectMeasurementOutcomesInStruct(mlir::qir::QIRProgramBuilder& b,
                                   mlir::SmallVector<mlir::Value> outcomes,
                                   mlir::Value structValue) {
  int64_t index = 0;
  for (auto bit : outcomes) {
    auto c = b.readResult(bit);
    auto insert = mlir::LLVM::InsertValueOp::create(
        b, structValue, c, mlir::ArrayRef<int64_t>(index++));
    structValue = insert.getResult();
  }
  return structValue;
}

/**
 * @brief Measures the given qubits and returns the measurement outcomes as a
 * struct value or a single `i1`.
 * @param b The QIRProgramBuilder used to perform the measurements and create
 * the struct.
 * @param qubits The qubits to be measured.
 * @param inRegister Whether to store the results in a classical result array or
 * not.
 * @return A pair containing the result value and its type.
 */
static std::pair<mlir::Value, mlir::Type>
measureAndReturn(mlir::qir::QIRProgramBuilder& b,
                 mlir::SmallVector<mlir::Value> qubits, bool inRegister) {

  if (qubits.empty()) {
    auto zeroConst = b.intConstant(0);
    return {zeroConst, b.getI64Type()};
  }
  mlir::qir::QIRProgramBuilder::ClassicalRegister resultArray;
  if (inRegister) {
    resultArray = b.allocClassicalBitRegister(qubits.size(), "meas");
  }

  if (qubits.size() == 1) {
    auto outcome = inRegister ? b.measure(qubits[0], resultArray[0])
                              : b.measure(qubits[0], 0);
    auto result = b.readResult(outcome);
    return {result, b.getI1Type()};
  }

  llvm::SmallVector<mlir::Type> elementTypes(qubits.size(), b.getI1Type());
  mlir::Value structValue = createStruct(b, elementTypes);

  for (auto i = 0L; i < qubits.size(); ++i) {
    auto outcome = inRegister ? b.measure(qubits[i], resultArray[i])
                              : b.measure(qubits[i], i);
    auto result = b.readResult(outcome);
    auto insert = mlir::LLVM::InsertValueOp::create(b, structValue, result, i);
    structValue = insert.getResult();
  }
  return {structValue, structValue.getType()};
}

/**
 * @brief Measures the given qubits and returns the measurement outcomes as a
 * struct value or a single `i1`.
 *
 * @detail The measurement outcomes are stored in a classical result array.
 * @param b The QIRProgramBuilder used to perform the measurements and create
 * the struct.
 * @param qubits The qubits to be measured.
 * @return The result value and its type as a pair.
 */
static std::pair<mlir::Value, mlir::Type>
measureAndReturn(mlir::qir::QIRProgramBuilder& b,
                 mlir::SmallVector<mlir::Value> qubits) {
  return measureAndReturn(b, qubits, true);
}

namespace mlir::qir {
std::pair<Value, Type> emptyQIR(QIRProgramBuilder& b) {
  return measureAndReturn(b, {});
}

std::pair<Value, Type> allocQubit(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  return measureAndReturn(b, {q});
}

std::pair<Value, Type> alloc1QubitRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> allocQubitRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> allocMultipleQubitRegisters(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  return measureAndReturn(b, {q0[0], q0[1], q1[0], q1[1], q1[2]});
}

std::pair<Value, Type>
allocMultipleQubitRegistersWithOps(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.allocQubitRegister(3);
  b.h(q0[0]);
  b.h(q0[1]);
  b.h(q1[0]);
  b.h(q1[1]);
  b.h(q1[2]);
  return measureAndReturn(b, {q0[0], q0[1], q1[0], q1[1], q1[2]});
}

std::pair<Value, Type> allocLargeRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(100);
  return measureAndReturn(b, {q[0], q[99]});
}

std::pair<Value, Type> staticQubits(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type> staticQubitsWithOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.h(q0);
  b.h(q1);
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type> staticQubitsWithParametricOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type> staticQubitsWithTwoTargetOps(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rzz(0.123, q0, q1);
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type> staticQubitsWithCtrl(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.cx(q0, q1);
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type> staticQubitsWithInv(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  b.tdg(q0);
  return measureAndReturn(b, {q0}, false);
}

std::pair<Value, Type> staticQubitsWithDuplicates(QIRProgramBuilder& b) {
  auto q0a = b.staticQubit(0);
  auto q1a = b.staticQubit(1);
  auto q0b = b.staticQubit(0);
  auto q1b = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0a);
  b.p(std::numbers::pi / 2., q1a);
  b.rzz(0.123, q0b, q1b);
  b.cx(q0b, q1b);
  b.tdg(q0a);
  return measureAndReturn(b, {q0b, q1b}, false);
}

std::pair<Value, Type> staticQubitsCanonical(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.staticQubit(1);
  b.rx(std::numbers::pi / 4., q0);
  b.p(std::numbers::pi / 2., q1);
  b.rzz(0.123, q0, q1);
  b.cx(q0, q1);
  b.tdg(q0);
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type> mixedStaticThenDynamicQubit(QIRProgramBuilder& b) {
  auto q0 = b.staticQubit(0);
  auto q1 = b.allocQubit();
  return measureAndReturn(b, {q0, q1}, false);
}

std::pair<Value, Type>
mixedDynamicRegisterThenStaticQubit(QIRProgramBuilder& b) {
  auto q0 = b.allocQubitRegister(2);
  auto q1 = b.staticQubit(0);
  return measureAndReturn(b, {q0[0], q0[1], q1}, false);
}

std::pair<Value, Type> singleMeasurementToSingleBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  const auto v = b.measure(q[0], c[0]);
  const auto read = b.readResult(v);
  return {read, b.getI1Type()};
}

std::pair<Value, Type> repeatedMeasurementToSameBit(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(1);
  b.measure(q[0], c[0]);
  b.measure(q[0], c[0]);
  auto b3 = b.measure(q[0], c[0]);
  auto c3 = b.readResult(b3);
  return {c3, b.getI1Type()};
}

std::pair<Value, Type>
repeatedMeasurementToDifferentBits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  const auto c = b.allocClassicalBitRegister(3);
  auto b1 = b.measure(q[0], c[0]);
  auto b2 = b.measure(q[0], c[1]);
  auto b3 = b.measure(q[0], c[2]);
  auto structValue =
      createStruct(b, {b.getI1Type(), b.getI1Type(), b.getI1Type()});
  auto filledStruct =
      collectMeasurementOutcomesInStruct(b, {b1, b2, b3}, structValue);
  return {filledStruct, filledStruct.getType()};
}

std::pair<Value, Type>
multipleClassicalRegistersAndMeasurements(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  const auto& c0 = b.allocClassicalBitRegister(1, "c0");
  const auto& c1 = b.allocClassicalBitRegister(2, "c1");
  auto b1 = b.measure(q[0], c0[0]);
  auto b2 = b.measure(q[1], c1[0]);
  auto b3 = b.measure(q[2], c1[1]);
  auto structValue =
      createStruct(b, {b.getI1Type(), b.getI1Type(), b.getI1Type()});
  auto filledStruct =
      collectMeasurementOutcomesInStruct(b, {b1, b2, b3}, structValue);
  return {filledStruct, filledStruct.getType()};
}

std::pair<Value, Type> measurementWithoutRegisters(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  auto bit = b.measure(q, 0);
  auto c = b.readResult(bit);
  return {c, b.getI1Type()};
}

std::pair<Value, Type> resetQubitWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> resetMultipleQubitsWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.reset(q[0]);
  b.reset(q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> repeatedResetWithoutOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> resetQubitAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> resetMultipleQubitsAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  b.reset(q[0]);
  b.h(q[1]);
  b.reset(q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> repeatedResetAfterSingleOp(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  b.reset(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> globalPhase(QIRProgramBuilder& b) {
  b.gphase(0.123);
  return measureAndReturn(b, {});
}

std::pair<Value, Type> identity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.id(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cid(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledIdentity(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcid({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> x(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.x(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledX(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcx({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> y(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.y(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cy(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcy({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> z(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.z(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cz(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledZ(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcz({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> h(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ch(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledH(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mch({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> hWithoutRegister(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  return measureAndReturn(b, {q}, false);
}

std::pair<Value, Type> s(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.s(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cs(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledS(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcs({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> sdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csdg(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledSdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> t_(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.t(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ct(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledT(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mct({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> tdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.tdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ctdg(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledTdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mctdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> sx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sx(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledSx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsx({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> sxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.sxdg(q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.csxdg(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledSxdg(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcsxdg({q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> rx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rx(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledRx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrx(0.123, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> ry(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.ry(0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cry(0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledRy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcry(0.456, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> rz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.rz(0.789, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.crz(0.789, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledRz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcrz(0.789, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> p(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.p(0.123, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cp(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledP(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcp(0.123, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> r(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.r(0.123, 0.456, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cr(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledR(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcr(0.123, 0.456, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> u2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u2(0.234, 0.567, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu2(0.234, 0.567, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledU2(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu2(0.234, 0.567, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> u(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.u(0.1, 0.2, 0.3, q[0]);
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> singleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.cu(0.1, 0.2, 0.3, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> multipleControlledU(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.mcu(0.1, 0.2, 0.3, {q[0], q[1]}, q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> swap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.swap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cswap(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledSwap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcswap({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> iswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.iswap(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.ciswap(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledIswap(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mciswap({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> dcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.dcx(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cdcx(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledDcx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcdcx({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> ecr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ecr(q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cecr(q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledEcr(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcecr({q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> rxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rxx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crxx(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> tripleControlledRxx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(5);
  b.mcrxx(0.123, {q[0], q[1], q[2]}, q[3], q[4]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3], q[4]});
}

std::pair<Value, Type> ryy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.ryy(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cryy(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledRyy(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcryy(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> rzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzx(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzx(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledRzx(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> rzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.rzz(0.123, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.crzz(0.123, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledRzz(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcrzz(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> xxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_plus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_plus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledXxPlusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_plus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> xxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.xx_minus_yy(0.123, 0.456, q[0], q[1]);
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> singleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  b.cxx_minus_yy(0.123, 0.456, q[0], q[1], q[2]);
  return measureAndReturn(b, {q[0], q[1], q[2]});
}

std::pair<Value, Type> multipleControlledXxMinusYY(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcxx_minus_yy(0.123, 0.456, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

std::pair<Value, Type> simpleIf(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0], 0);
  b.scfIf(cond, [&] { b.x(q[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> ifElse(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  b.h(q[0]);
  auto cond = b.measure(q[0], 0);
  b.scfIf(cond, [&] { b.x(q[0]); }, [&] { b.z(q[0]); });
  return measureAndReturn(b, {q[0]});
}

std::pair<Value, Type> ifTwoQubits(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  b.h(q[0]);
  auto cond = b.measure(q[0], 0);
  b.scfIf(cond, [&] {
    b.x(q[0]);
    b.x(q[1]);
  });
  return measureAndReturn(b, {q[0], q[1]});
}

std::pair<Value, Type> nestedIfOpForLoop(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto q0 = b.allocQubit();
  b.h(q0);
  auto cond = b.measure(q0, 0);
  b.scfIf(
      cond, [&] { b.h(q0); },
      [&] {
        b.scfFor(0, 3, 1, [&](Value iv) {
          auto q1 = b.load(reg.value, iv);
          b.h(q1);
        });
      });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], q0});
}

std::pair<Value, Type> simpleWhileReset(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.h(q);
  b.scfWhile(
      [&] {
        auto measureResult = b.measure(q, 0);
        return measureResult;
      },
      [&] { b.h(q); });
  return measureAndReturn(b, {q});
}

std::pair<Value, Type> simpleDoWhileReset(QIRProgramBuilder& b) {
  auto q = b.allocQubit();
  b.scfWhile([&] {
    b.h(q);
    auto measureResult = b.measure(q, 0);
    return measureResult;
  });
  return measureAndReturn(b, {q});
}

std::pair<Value, Type> simpleForLoop(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.load(reg.value, iv);
    b.h(q);
  });
  return measureAndReturn(b, {reg[0], reg[1]});
};

std::pair<Value, Type> nestedForLoopIfOp(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto qCond = b.allocQubit();
  b.scfFor(0, 2, 1, [&](Value iv) {
    b.h(qCond);
    auto cond = b.measure(qCond, 0);
    b.scfIf(cond, [&] {
      auto q = b.load(reg.value, iv);
      b.h(q);
    });
  });
  return measureAndReturn(b, {reg[0], reg[1], qCond});
}

std::pair<Value, Type> nestedForLoopWhileOp(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.load(reg.value, iv);
    b.h(q);
  });
  b.scfFor(0, 2, 1, [&](Value iv) {
    auto q = b.load(reg.value, iv);
    b.scfWhile(
        [&] {
          auto measureResult = b.measure(q, 0);
          return measureResult;
        },
        [&] { b.h(q); });
  });
  return measureAndReturn(b, {reg[0], reg[1]});
}

std::pair<Value, Type>
nestedForLoopCtrlOpWithSeparateQubit(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(3);
  auto control = b.allocQubit();
  b.h(control);
  b.scfFor(0, 3, 1, [&](Value iv) {
    auto q0 = b.load(reg.value, iv);
    b.h(q0);
    b.cx(control, q0);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], control});
}

std::pair<Value, Type>
nestedForLoopCtrlOpWithExtractedQubit(QIRProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  b.h(reg[0]);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto q0 = b.load(reg.value, iv);
    b.h(q0);
    b.cx(reg[0], q0);
  });
  return measureAndReturn(b, {reg[0], reg[1], reg[2], reg[3]});
}

std::pair<Value, Type> ctrlTwo(QIRProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  b.mcx({q[0], q[1]}, q[2]);
  b.mcrxx(0.123, {q[0], q[1]}, q[2], q[3]);
  return measureAndReturn(b, {q[0], q[1], q[2], q[3]});
}

} // namespace mlir::qir
