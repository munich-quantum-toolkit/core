/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagationLattice.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>

using namespace mlir;
namespace mlir::qco {

/**
 * Removes the bit at a specified position in a 64-bit unsigned integer.
 * The resulting value is effectively the input value with the bit at the given
 * position cleared or removed, shifting higher bits down by one position.
 *
 * @param value The 64-bit unsigned integer from which to clear a bit.
 * @param pos The zero-based position of the bit to remove.
 *            Must be less than 64; undefined behavior if out of bounds.
 * @return A new 64-bit unsigned integer with the specified bit removed.
 */
static uint64_t clearBit(const uint64_t value, const unsigned pos) {
  const uint64_t lowMask = pos == 0 ? 0 : (uint64_t{1} << pos) - 1;
  const uint64_t low = value & lowMask;
  const uint64_t high = value >> (pos + 1);
  return low | high << pos;
}

/**
 * Inserts a bit at a specified position in a 64-bit unsigned integer.
 * The resulting value includes the new bit at the given position, with
 * all higher bits shifted up by one position to make room for the insertion.
 *
 * @param value The 64-bit unsigned integer where the bit will be inserted.
 * @param pos The zero-based position at which the bit is to be inserted.
 *            Must be less than 64; undefined behavior if out of bounds.
 * @param bit The value of the bit to be inserted (true for 1, false for 0).
 * @return A new 64-bit unsigned integer with the specified bit inserted.
 */
static uint64_t insertBit(const uint64_t value, const unsigned pos,
                          const bool bit) {
  const uint64_t lowMask = pos == 0 ? 0 : (uint64_t{1} << pos) - 1;
  const uint64_t low = value & lowMask;
  const uint64_t high = value >> pos;
  return low | static_cast<uint64_t>(bit) << pos | high << (pos + 1);
}

bool isZeroAttribute(const Attribute attr) {
  if (!attr) {
    return false;
  }
  if (const auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getValue().isZero();
  }
  if (const auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValue().isZero();
  }
  if (const auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    return !boolAttr.getValue();
  }
  return false;
}

bool isTrueAttribute(const Attribute attr) {
  if (!attr) {
    return false;
  }
  if (const auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return !intAttr.getValue().isZero();
  }
  if (const auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return !floatAttr.getValue().isZero();
  }
  if (const auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    return boolAttr.getValue();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// QuantumState
//===----------------------------------------------------------------------===//

QuantumState
QuantumState::singletonZero(const unsigned int maxTrackedAmplitudes,
                            const Value qubit) {
  QuantumState c(std::min(maxTrackedAmplitudes, 64u));
  c.qubits.push_back(qubit);
  c.amplitudes[0] = Complex(1.0, 0.0);
  return c;
}

bool QuantumState::operator==(const QuantumState& other) const {
  if (isTop != other.isTop) {
    return false;
  }
  if (maxTrackedAmplitudes != other.maxTrackedAmplitudes) {
    return false;
  }
  if (qubits.size() != other.qubits.size()) {
    return false;
  }
  for (auto [a, b] : llvm::zip(qubits, other.qubits)) {
    if (a != b) {
      return false;
    }
  }
  if (isTop) {
    return other.isTop;
  }
  if (amplitudes.size() != other.amplitudes.size()) {
    return false;
  }
  for (const auto& it : amplitudes) {
    auto found = other.amplitudes.find(it.first);
    if (found == other.amplitudes.end()) {
      return false;
    }
    if (found->second != it.second) {
      return false;
    }
  }
  return true;
}

bool QuantumState::contains(const Value v) const {
  return llvm::is_contained(qubits, v);
}

std::optional<unsigned> QuantumState::indexOf(const Value v) const {
  for (auto [idx, q] : llvm::enumerate(qubits)) {
    if (q == v) {
      return idx;
    }
  }
  return {};
}

bool QuantumState::isAlwaysZero(const Value q) const {
  if (isTop) {
    return false;
  }
  const auto idx = indexOf(q);
  if (!idx) {
    return false;
  }
  return std::ranges::all_of(amplitudes, [&](const auto& it) {
    return (it.first >> *idx & 1ULL) == 0ULL;
  });
}

bool QuantumState::isAlwaysOne(const Value q) const {
  if (isTop) {
    return false;
  }
  const auto idx = indexOf(q);
  if (!idx) {
    return false;
  }
  return std::ranges::all_of(amplitudes, [&](const auto& it) {
    return (it.first >> *idx & 1ULL) != 0ULL;
  });
}

void QuantumState::markTop() {
  isTop = true;
  amplitudes.clear();
}

void QuantumState::forwardQubit(const Value from, const Value to) {
  const auto id = indexOf(from);
  if (!id) {
    return;
  }
  qubits[id.value()] = to;
}

QuantumState QuantumState::tensorProduct(const QuantumState& that) {
  QuantumState result(maxTrackedAmplitudes);
  result.qubits.append(qubits.begin(), qubits.end());
  result.qubits.append(that.qubits.begin(), that.qubits.end());

  if (isTop || that.isTop ||
      amplitudes.size() * that.amplitudes.size() > maxTrackedAmplitudes) {
    result.isTop = true;
    return result;
  }

  for (const auto& itA : amplitudes) {
    for (const auto& itB : that.amplitudes) {
      uint64_t basis = itA.first | itB.first << qubits.size();
      result.amplitudes[basis] += itA.second * itB.second;
    }
  }
  return result;
}
bool QuantumState::isStateTop() const { return isTop; }

void QuantumState::applyMatrix1Q(const Value input, const Value output,
                                 const Matrix2x2& matrix) {
  if (isTop) {
    for (Value& q : qubits) {
      if (q == input) {
        q = output;
        break;
      }
    }
    return;
  }

  const auto idxOpt = indexOf(input);
  if (!idxOpt) {
    return;
  }
  const unsigned idx = *idxOpt;

  llvm::DenseMap<uint64_t, Complex> result;

  // Group amplitudes by all bits except the target bit.
  llvm::DenseMap<uint64_t, std::array<Complex, 2>> grouped;
  for (const auto& it : amplitudes) {
    uint64_t reduced = clearBit(it.first, idx);
    const bool bit = (it.first >> idx & 1ULL) != 0ULL;
    grouped[reduced][bit ? 1 : 0] += it.second;
  }

  for (const auto& it : grouped) {
    Complex in0 = it.second[0];
    Complex in1 = it.second[1];
    Complex out0 = matrix[0][0] * in0 + matrix[0][1] * in1;
    Complex out1 = matrix[1][0] * in0 + matrix[1][1] * in1;
    if (out0 != Complex(0.0, 0.0)) {
      result[insertBit(it.first, idx, false)] += out0;
    }
    if (out1 != Complex(0.0, 0.0)) {
      result[insertBit(it.first, idx, true)] += out1;
    }
  }

  amplitudes = std::move(result);
  for (Value& q : qubits) {
    if (q == input) {
      q = output;
      break;
    }
  }
  if (amplitudes.size() > maxTrackedAmplitudes) {
    amplitudes.clear();
    isTop = true;
  }
}

void QuantumState::applyMatrix2Q(const Value input0, const Value input1,
                                 const Value output0, const Value output1,
                                 const Matrix4x4& matrix) {
  if (isTop) {
    for (Value& q : qubits) {
      if (q == input0) {
        q = output0;
      } else if (q == input1) {
        q = output1;
      }
    }
    return;
  }

  const auto idx0Opt = indexOf(input0);
  const auto idx1Opt = indexOf(input1);
  if (!idx0Opt || !idx1Opt || *idx0Opt == *idx1Opt) {
    return;
  }
  const unsigned idx0 = *idx0Opt;
  const unsigned idx1 = *idx1Opt;

  llvm::DenseMap<uint64_t, std::array<Complex, 4>> grouped;
  for (const auto& it : amplitudes) {
    const bool b0 = (it.first >> idx0 & 1ULL) != 0ULL;
    const bool b1 = (it.first >> idx1 & 1ULL) != 0ULL;
    const unsigned local = static_cast<unsigned>(b0) | static_cast<unsigned>(b1)
                                                           << 1u;
    uint64_t reduced = clearBit(clearBit(it.first, idx1), idx0);
    grouped[reduced][local] += it.second;
  }

  llvm::DenseMap<uint64_t, Complex> result;
  for (const auto& it : grouped) {
    std::array<Complex, 4> outVec{};
    for (unsigned row = 0; row < 4; ++row) {
      Complex sum(0.0, 0.0);
      for (unsigned col = 0; col < 4; ++col) {
        sum += matrix[row][col] * it.second[col];
      }
      outVec[row] = sum;
    }

    for (unsigned row = 0; row < 4; ++row) {
      if (outVec[row] == Complex(0.0, 0.0)) {
        continue;
      }
      bool b0 = (row & 1u) != 0u;
      bool b1 = (row & 2u) != 0u;
      uint64_t basis = insertBit(insertBit(it.first, idx0, b0), idx1, b1);
      result[basis] += outVec[row];
    }
  }

  amplitudes = std::move(result);
  for (Value& q : qubits) {
    if (q == input0) {
      q = output0;
    } else if (q == input1) {
      q = output1;
    }
  }
  if (amplitudes.size() > maxTrackedAmplitudes) {
    amplitudes.clear();
    isTop = true;
  }
}

LogicalResult QuantumState::applyUnitary(const ArrayRef<Value> inputs,
                                         const UnitaryMatrix& matrix,
                                         const ArrayRef<Value> outputs) {

  if (inputs.size() != outputs.size()) {
    return failure();
  }
  if (inputs.empty() || inputs.size() > 2) {
    return failure();
  }

  for (const auto& in : inputs) {
    if (!indexOf(in)) {
      return failure();
    }
  }

  if (isTop) {
    for (const auto& [in, out] : llvm::zip(inputs, outputs)) {
      forwardQubit(in, out);
    }
    return success();
  }

  if (inputs.size() == 1) {
    if (!std::holds_alternative<Matrix2x2>(matrix)) {
      return failure();
    }
    applyMatrix1Q(inputs[0], outputs[0], std::get<Matrix2x2>(matrix));

    return success();
  }

  if (!std::holds_alternative<Matrix4x4>(matrix)) {
    return failure();
  }
  applyMatrix2Q(inputs[0], inputs[1], outputs[0], outputs[1],
                std::get<Matrix4x4>(matrix));

  return success();
}

std::unordered_map<unsigned int, std::pair<QuantumState, double>>
QuantumState::measure(const Value inQubit, const Value outQubit,
                      MLIRContext* ctx) {

  if (isTop) {
    forwardQubit(inQubit, outQubit);
    return {};
  }

  const auto idxOpt = indexOf(inQubit);
  if (!idxOpt) {
    llvm::report_fatal_error("Called measure on a qubit not in the state");
  }
  unsigned idx = *idxOpt;

  double prob0 = 0.0;
  double prob1 = 0.0;
  for (const auto& it : amplitudes) {
    const double p = std::norm(it.second);
    if ((it.first >> idx & 1ULL) == 0ULL) {
      prob0 += p;
    } else {
      prob1 += p;
    }
  }

  auto makeSuccessor = [&](const bool bit, const double probability) {
    const double scaleFactor = 1.0 / std::sqrt(probability);
    auto c = QuantumState(maxTrackedAmplitudes);
    for (const auto& it : amplitudes) {
      const bool curBit = (it.first >> idx & 1ULL) != 0ULL;
      if (curBit == bit) {
        c.amplitudes[it.first] = it.second * scaleFactor;
      }
    }

    for (Value& q : qubits) {
      if (q == inQubit) {
        c.qubits.push_back(outQubit);
      } else {
        c.qubits.push_back(q);
      }
    }

    return std::pair{std::move(c), probability};
  };

  std::unordered_map<unsigned int, std::pair<QuantumState, double>> result;
  if (std::norm(prob0 - 0.0) >= 1e-10) {
    result.emplace(0, makeSuccessor(false, prob0));
  }
  if (std::norm(prob1 - 0.0) >= 1e-10) {
    result.emplace(1, makeSuccessor(true, prob1));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// HybridState
//===----------------------------------------------------------------------===//

bool HybridState::operator==(const HybridState& that) const {
  if (isTop || that.isTop) {
    return isTop == that.isTop;
  }
  if (std::norm(probability - that.probability) >= 1e-10) {
    return false;
  }
  if (maxTrackedAmplitudes != that.maxTrackedAmplitudes) {
    return false;
  }
  if (quantumState != that.quantumState) {
    return false;
  }
  if (classicalValues.size() != that.classicalValues.size()) {
    return false;
  }
  for (const auto& it : classicalValues) {
    auto found = that.classicalValues.find(it.first);
    if (found == that.classicalValues.end()) {
      return false;
    }
    if (it.second != found->second) {
      return false;
    }
  }
  return true;
}

std::optional<Attribute> HybridState::getClassical(const Value v) const {
  const auto it = classicalValues.find(v);
  if (it == classicalValues.end()) {
    return {};
  }
  return it->second;
}

void HybridState::setClassical(const Value v, const Attribute attr) {
  classicalValues[v] = attr;
}

bool HybridState::contains(const Value v) const {
  if (getClassical(v).has_value()) {
    return true;
  }
  return quantumState->contains(v);
}
bool HybridState::isStateTop() const { return isTop; }

bool HybridState::isAlwaysFalse(const Value v) const {
  const auto attr = getClassical(v);
  if (attr.has_value()) {
    return isZeroAttribute(attr.value());
  }
  return quantumState->isAlwaysZero(v);
}

bool HybridState::isAlwaysTrue(const Value v) const {
  const auto attr = getClassical(v);
  if (attr.has_value()) {
    return isTrueAttribute(attr.value());
  }
  return quantumState->isAlwaysOne(v);
}

HybridState HybridState::mergeStates(const HybridState& that) const {
  auto result = HybridState(maxTrackedAmplitudes);
  result.probability = probability * that.probability;

  if (isTop || that.isTop) {
    result.isTop = true;
    return result;
  }
  result.classicalValues = classicalValues;
  for (const auto& [v, a] : that.classicalValues) {
    result.classicalValues[v] = a;
  }
  auto qS = quantumState->tensorProduct(*that.quantumState);
  if (qS.isStateTop()) {
    result.isTop = true;
    return result;
  }
  result.quantumState = std::make_unique<QuantumState>(qS);
  return result;
}

//===----------------------------------------------------------------------===//
// HybridStateSet
//===----------------------------------------------------------------------===//

bool HybridStateSet::operator==(const HybridStateSet& that) const {
  if (isTop || that.isTop) {
    return isTop && that.isTop;
  }
  if (states.size() != that.states.size()) {
    return false;
  }
  if (maxTrackedAmplitudes != that.maxTrackedAmplitudes) {
    return false;
  }
  if (maxTrackedHybridStates != that.maxTrackedHybridStates) {
    return false;
  }
  for (const auto& s : states) {
    if (!llvm::is_contained(that.states, s)) {
      return false;
    }
  }
  return true;
}

void HybridStateSet::addState(HybridState state) {
  if (isTop) {
    return;
  }
  if (maxTrackedHybridStates == states.size()) {
    isTop = true;
    states.clear();
    return;
  }
  states.push_back(std::move(state));
}

void HybridStateSet::join(const HybridStateSet& other) {
  if (isTop || other.isTop) {
    isTop = true;
    states.clear();
    return;
  }
  llvm::append_range(states, other.states);
}

void HybridStateSet::enforceMaxStates() {
  if (isTop) {
    return;
  }
  if (states.size() > maxTrackedHybridStates) {
    isTop = true;
    states.clear();
  }
}
HybridStateSet HybridStateSet::mergeStates(const HybridStateSet& that) const {
  auto result = HybridStateSet(maxTrackedAmplitudes, maxTrackedHybridStates);
  if (isTop || that.isTop) {
    result.isTop = true;
    result.states.clear();
    return result;
  }
  bool allTop = true;
  SmallVector<HybridState> newStates;
  for (const auto& s : states) {
    for (const auto& thatS : that.states) {
      const auto newState = s.mergeStates(thatS);
      newStates.push_back(newState);
      allTop &= newState.isStateTop();
    }
  }
  if (allTop) {
    result.isTop = true;
    result.states.clear();
  } else {
    result.states = std::move(newStates);
  }
  return result;
}

bool HybridStateSet::areStatesTop() const { return isTop; }

bool HybridStateSet::isAlwaysFalse(const Value v) const {
  if (isTop) {
    return false;
  }
  for (const HybridState& state : states) {
    if (state.contains(v)) {
      if (!state.isAlwaysFalse(v)) {
        return false;
      }
    }
  }
  return true;
}

bool HybridStateSet::isAlwaysTrue(const Value v) const {
  if (isTop) {
    return false;
  }
  for (const HybridState& state : states) {
    if (state.contains(v)) {
      if (!state.isAlwaysTrue(v)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace mlir::qco