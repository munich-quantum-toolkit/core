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
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
#include <optional>
#include <utility>

namespace mlir::qco {

namespace {

/// @brief Whether ctrlsOut is a valid rename target for ctrlsIn: empty (no
/// rename), or the same length.
bool ctrlRenameOk(const ArrayRef<Value> ctrlsIn,
                  const ArrayRef<Value> ctrlsOut) {
  return ctrlsOut.empty() || ctrlsOut.size() == ctrlsIn.size();
}

/// @brief Truthiness of a resolved classical constant (non-zero == true), or
/// nullopt if attr is not an integer/index/bool/float constant.
std::optional<bool> classicalTruth(const Attribute attr) {
  if (const auto ia = dyn_cast<IntegerAttr>(attr)) {
    return !ia.getValue().isZero();
  }
  if (const auto fa = dyn_cast<FloatAttr>(attr)) {
    return !fa.getValue().isZero();
  }
  return std::nullopt;
}
} // namespace

//===----------------------------------------------------------------------===//
// Observers
//===----------------------------------------------------------------------===//

std::optional<Attribute> HybridState::getClassical(const Value v) const {
  const auto it = classical.find(v);
  if (it == classical.end()) {
    return std::nullopt;
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// Mutation
//===----------------------------------------------------------------------===//

void HybridState::setClassical(const Value v, const Attribute attr) {
  classical[v] = attr;
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

FailureOr<bool>
HybridState::classicalControlsHold(const ArrayRef<Value> pos,
                                   const ArrayRef<Value> neg) const {
  for (const Value p : pos) {
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
  for (const Value n : neg) {
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

LogicalResult HybridState::applyMatrix1Q(
    const Value in, const Value out, const Matrix2x2& matrix,
    const ArrayRef<Value> quantumCtrlsIn, const ArrayRef<Value> quantumCtrlsOut,
    const ArrayRef<Value> posClassicalCtrls,
    const ArrayRef<Value> negClassicalCtrls) {
  if (!ctrlRenameOk(quantumCtrlsIn, quantumCtrlsOut) || !state.contains(in)) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    return state.applyMatrix1Q(in, out, matrix, quantumCtrlsIn, quantumCtrlsOut);
  }
  // Classical control false: the gate is skipped, only the identities thread on.
  state.forwardQubit(in, out);
  state.forwardQubits(quantumCtrlsIn, quantumCtrlsOut);
  return success();
}

LogicalResult
HybridState::applyMatrix2Q(const Value in0, const Value in1, const Value out0,
                           const Value out1, const Matrix4x4& matrix,
                           const ArrayRef<Value> quantumCtrlsIn,
                           const ArrayRef<Value> quantumCtrlsOut,
                           const ArrayRef<Value> posClassicalCtrls,
                           const ArrayRef<Value> negClassicalCtrls) {
  if (!ctrlRenameOk(quantumCtrlsIn, quantumCtrlsOut) || !state.contains(in0) ||
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

LogicalResult
HybridState::addGlobalPhase(const double theta,
                            const ArrayRef<Value> quantumCtrlsIn,
                            const ArrayRef<Value> quantumCtrlsOut,
                            const ArrayRef<Value> posClassicalCtrls,
                            const ArrayRef<Value> negClassicalCtrls) {
  if (!ctrlRenameOk(quantumCtrlsIn, quantumCtrlsOut)) {
    return failure();
  }
  const auto hold = classicalControlsHold(posClassicalCtrls, negClassicalCtrls);
  if (failed(hold)) {
    return failure();
  }
  if (*hold) {
    if (!quantumCtrlsIn.empty()) {
      return state.applyControlledPhase(theta, quantumCtrlsIn, quantumCtrlsOut);
    }
    globalPhase *= std::exp(Complex{0.0, theta});
    return success();
  }
  state.forwardQubits(quantumCtrlsIn, quantumCtrlsOut);
  return success();
}

//===----------------------------------------------------------------------===//
// Measurement / reset
//===----------------------------------------------------------------------===//

LogicalResult
HybridState::measureQubit(const Value in, const Value out,
                          const Value classicalResult,
                          const ArrayRef<Value> posClassicalCtrls,
                          const ArrayRef<Value> negClassicalCtrls) {
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

LogicalResult HybridState::resetQubit(const Value in, const Value out,
                                      const ArrayRef<Value> posClassicalCtrls,
                                      const ArrayRef<Value> negClassicalCtrls) {
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

bool HybridState::isQubitAlwaysZero(const Value q) const {
  return state.isAlwaysZero(q);
}

bool HybridState::isQubitAlwaysOne(const Value q) const {
  return state.isAlwaysOne(q);
}

bool HybridState::isClassicalTrue(const Value v) const {
  const auto attr = getClassical(v);
  return attr && classicalTruth(*attr).value_or(false);
}

bool HybridState::isClassicalFalse(const Value v) const {
  const auto attr = getClassical(v);
  if (!attr) {
    return false;
  }
  const auto truth = classicalTruth(*attr);
  return truth && !*truth;
}

bool HybridState::areControlsSatisfiable(
    const ArrayRef<Value> quantumCtrls, const ArrayRef<Value> posClassicalCtrls,
    const ArrayRef<Value> negClassicalCtrls) const {
  for (const Value pc : posClassicalCtrls) {
    if (isClassicalFalse(pc)) {
      return false;
    }
  }
  for (const Value nc : negClassicalCtrls) {
    if (isClassicalTrue(nc)) {
      return false;
    }
  }
  if (quantumCtrls.empty()) {
    return true;
  }
  SmallVector<std::pair<Value, bool>> assignment;
  for (const Value qc : quantumCtrls) {
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

bool HybridState::operator==(const HybridState& other) const {
  if (std::abs(probability - other.probability) > MATRIX_TOLERANCE ||
      std::abs(globalPhase - other.globalPhase) > MATRIX_TOLERANCE ||
      classical.size() != other.classical.size() || state != other.state) {
    return false;
  }
  for (const auto& [v, attr] : classical) {
    const auto it = other.classical.find(v);
    if (it == other.classical.end() || it->second != attr) {
      return false;
    }
  }
  return true;
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
    for (const auto& [v, attr] : classical) {
      os << " " << attr;
    }
  }
}

} // namespace mlir::qco
