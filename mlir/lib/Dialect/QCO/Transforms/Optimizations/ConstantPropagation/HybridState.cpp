/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "HybridState.hpp"

#include "QuantumState.hpp"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
#include <optional>
#include <utility>

namespace mlir::qco {

/// @brief Truthiness of a resolved classical constant (non-zero == true), or
/// nullopt if attr is not an integer/index/bool/float constant.
static std::optional<bool> classicalTruth(Attribute attr) {
  if (const auto ia = dyn_cast_if_present<IntegerAttr>(attr)) {
    return !ia.getValue().isZero();
  }
  if (const auto fa = dyn_cast_if_present<FloatAttr>(attr)) {
    return !fa.getValue().isZero();
  }
  return std::nullopt;
}

/// @brief Numeric value of a resolved classical constant, or nullopt if attr is
/// not an integer/index/bool/float constant.
static std::optional<double> classicalDouble(Attribute attr) {
  if (const auto ia = dyn_cast_if_present<IntegerAttr>(attr)) {
    if (const std::optional<int64_t> v = ia.getValue().trySExtValue()) {
      return static_cast<double>(*v);
    }
    return {};
  }
  if (const auto fa = dyn_cast_if_present<FloatAttr>(attr)) {
    return fa.getValueAsDouble();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Observers
//===----------------------------------------------------------------------===//

std::optional<Attribute> HybridState::getClassical(Value v) const {
  const auto it = classical.find(v);
  if (it == classical.end()) {
    return std::nullopt;
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// Mutation
//===----------------------------------------------------------------------===//

void HybridState::setClassical(Value v, Attribute attr) { classical[v] = attr; }

void HybridState::forwardValue(Value from, Value to) {
  state.forwardQubit(from, to);
  const auto it = classical.find(from);
  if (it != classical.end()) {
    const Attribute attr = it->second;
    classical.erase(it);
    classical[to] = attr;
  }
}

void HybridState::markStateTop() { state.markTop(); }

void HybridState::intersectClassical(const HybridState& other) {
  SmallVector<Value> disagreeing;
  for (const auto& [v, attr] : classical) {
    const auto it = other.classical.find(v);
    if (it == other.classical.end() || it->second != attr) {
      disagreeing.push_back(v);
    }
  }
  for (Value v : disagreeing) {
    classical.erase(v);
  }
}

HybridState HybridState::tensor(const HybridState& other) const {
  HybridState result(state.unify(other.state), maxNonzeroAmplitudes,
                     probability * other.probability);
  result.globalPhase = globalPhase * other.globalPhase;
  result.classical = classical;
  for (const auto& [v, attr] : other.classical) {
    result.classical[v] = attr;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Classical-control handling
//===----------------------------------------------------------------------===//

FailureOr<bool> HybridState::classicalControlsHold(ArrayRef<Value> pos,
                                                   ArrayRef<Value> neg) const {
  for (Value p : pos) {
    const auto attr = getClassical(p);
    if (!attr) {
      return failure();
    }
    const auto truth = classicalTruth(*attr);
    if (!truth) {
      return failure();
    }
    if (!*truth) {
      return false;
    }
  }
  for (Value n : neg) {
    const auto attr = getClassical(n);
    if (!attr) {
      return failure();
    }
    const auto truth = classicalTruth(*attr);
    if (!truth) {
      return failure();
    }
    if (*truth) {
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Gate application
//===----------------------------------------------------------------------===//

LogicalResult HybridState::applyMatrix1Q(Value in, Value out,
                                         const Matrix2x2& matrix,
                                         ArrayRef<Value> quantumCtrlsIn,
                                         ArrayRef<Value> quantumCtrlsOut,
                                         ArrayRef<Value> posClassicalCtrls,
                                         ArrayRef<Value> negClassicalCtrls) {
  if (quantumCtrlsIn.size() != quantumCtrlsOut.size() || !state.contains(in)) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    return state.applyMatrix1Q(in, out, matrix, quantumCtrlsIn,
                               quantumCtrlsOut);
  }
  // Classical control false: only the identities thread on.
  state.forwardQubit(in, out);
  state.forwardQubits(quantumCtrlsIn, quantumCtrlsOut);
  return success();
}

LogicalResult HybridState::applyMatrix2Q(Value in0, Value in1, Value out0,
                                         Value out1, const Matrix4x4& matrix,
                                         ArrayRef<Value> quantumCtrlsIn,
                                         ArrayRef<Value> quantumCtrlsOut,
                                         ArrayRef<Value> posClassicalCtrls,
                                         ArrayRef<Value> negClassicalCtrls) {
  if (quantumCtrlsIn.size() != quantumCtrlsOut.size() || !state.contains(in0) ||
      !state.contains(in1)) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    return state.applyMatrix2Q(in0, in1, out0, out1, matrix, quantumCtrlsIn,
                               quantumCtrlsOut);
  }
  state.forwardQubit(in0, out0);
  state.forwardQubit(in1, out1);
  state.forwardQubits(quantumCtrlsIn, quantumCtrlsOut);
  return success();
}

LogicalResult HybridState::addGlobalPhase(Value theta,
                                          ArrayRef<Value> quantumCtrlsIn,
                                          ArrayRef<Value> quantumCtrlsOut,
                                          ArrayRef<Value> posClassicalCtrls,
                                          ArrayRef<Value> negClassicalCtrls) {
  if (quantumCtrlsIn.size() != quantumCtrlsOut.size()) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    const auto angle = classicalDouble(classical.lookup(theta));
    if (!angle) {
      return failure();
    }
    if (!quantumCtrlsIn.empty()) {
      return state.applyControlledPhase(*angle, quantumCtrlsIn,
                                        quantumCtrlsOut);
    }
    globalPhase *= std::exp(Complex{0.0, *angle});
    return success();
  }
  state.forwardQubits(quantumCtrlsIn, quantumCtrlsOut);
  return success();
}

void HybridState::propagateClassical(Operation* op) {
  SmallVector<Attribute> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    operands.push_back(classical.lookup(operand));
  }
  SmallVector<OpFoldResult> folded;
  if (failed(op->fold(operands, folded)) ||
      folded.size() != op->getNumResults()) {
    return;
  }
  for (const auto& [result, foldResult] : llvm::zip(op->getResults(), folded)) {
    if (const auto attr = dyn_cast<Attribute>(foldResult)) {
      setClassical(result, attr);
    }
  }
}

//===----------------------------------------------------------------------===//
// Measurement / reset
//===----------------------------------------------------------------------===//

LogicalResult HybridState::measureQubit(Value in, Value out,
                                        Value classicalResult,
                                        ArrayRef<Value> posClassicalCtrls,
                                        ArrayRef<Value> negClassicalCtrls) {
  if (!state.contains(in)) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    auto branches = state.measure(in, out);
    if (failed(branches)) {
      return failure();
    }
    if (branches->size() == 1) {
      const auto resultType = dyn_cast<IntegerType>(classicalResult.getType());
      if (!resultType) {
        return failure();
      }
      setClassical(classicalResult,
                   IntegerAttr::get(resultType, branches->front().bit));
      state = std::move(*branches->front().state);
      return success();
    }
    if (branches->size() == 2) {
      state.markTop(); // This will be handled in a later version
    }
    // branches->empty() => state was already top.
  }
  state.forwardQubit(in, out);
  return success();
}

LogicalResult HybridState::resetQubit(Value in, Value out,
                                      ArrayRef<Value> posClassicalCtrls,
                                      ArrayRef<Value> negClassicalCtrls) {
  if (!state.contains(in)) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    auto branches = state.reset(in, out);
    if (failed(branches)) {
      return failure();
    }
    // One outcome, or two that agree = `in` was unentangled: reset is exact and
    // the branch state (already named `out`) is the result. Two that disagree =
    // the reduced state after tracing out `in` is mixed.
    if (branches->size() == 1) {
      state = std::move(*branches->front().state);
      return success();
    }
    if (branches->size() == 2) {
      state.markTop();
    }
    // branches->empty() => state was already top.
  }
  state.forwardQubit(in, out);
  return success();
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

bool HybridState::isQubitAlwaysZero(Value q) const {
  return state.isAlwaysZero(q);
}

bool HybridState::isQubitAlwaysOne(Value q) const {
  return state.isAlwaysOne(q);
}

bool HybridState::isClassicalTrue(Value v) const {
  const auto attr = getClassical(v);
  return attr && classicalTruth(*attr).value_or(false);
}

bool HybridState::isClassicalFalse(Value v) const {
  const auto attr = getClassical(v);
  if (!attr) {
    return false;
  }
  const auto truth = classicalTruth(*attr);
  return truth && !*truth;
}

bool HybridState::areControlsSatisfiable(
    ArrayRef<Value> quantumCtrls, ArrayRef<Value> posClassicalCtrls,
    ArrayRef<Value> negClassicalCtrls) const {
  for (Value pc : posClassicalCtrls) {
    if (isClassicalFalse(pc)) {
      return false;
    }
  }
  for (Value nc : negClassicalCtrls) {
    if (isClassicalTrue(nc)) {
      return false;
    }
  }
  if (quantumCtrls.empty()) {
    return true;
  }
  SmallVector<std::pair<Value, bool>> assignment;
  for (Value qc : quantumCtrls) {
    if (!state.contains(qc)) {
      return false;
    }
    assignment.emplace_back(qc, true);
  }
  return !state.hasAlwaysZeroAmplitude(assignment);
}

//===----------------------------------------------------------------------===//
// Comparison / dump
//===----------------------------------------------------------------------===//

bool HybridState::sameConfiguration(const HybridState& other) const {
  if (std::abs(globalPhase - other.globalPhase) > MATRIX_TOLERANCE ||
      classical.size() != other.classical.size() || state != other.state) {
    return false;
  }
  return llvm::all_of(classical, [&](const auto& entry) {
    const auto it = other.classical.find(entry.first);
    return it != other.classical.end() && it->second == entry.second;
  });
}

bool HybridState::operator==(const HybridState& other) const {
  return std::abs(probability - other.probability) <= MATRIX_TOLERANCE &&
         sameConfiguration(other);
}

void HybridState::print(raw_ostream& os) const {
  os << "p=" << llvm::format("%.4f", probability);
  if (std::abs(globalPhase - Complex{1.0, 0.0}) > MATRIX_TOLERANCE) {
    os << " phase=(" << llvm::format("%.4f", globalPhase.real()) << ","
       << llvm::format("%.4f", globalPhase.imag()) << ")";
  }
  os << " [";
  state.print(os);
  os << "]";
  if (!classical.empty()) {
    os << " classical:";
    SmallVector<std::string> entries;
    entries.reserve(classical.size());
    for (const auto& [v, attr] : classical) {
      std::string entry;
      llvm::raw_string_ostream entryOs(entry);
      entryOs << v << "=" << attr;
      entries.push_back(std::move(entry));
    }
    llvm::sort(entries);
    for (const auto& entry : entries) {
      os << " " << entry;
    }
  }
}

} // namespace mlir::qco
