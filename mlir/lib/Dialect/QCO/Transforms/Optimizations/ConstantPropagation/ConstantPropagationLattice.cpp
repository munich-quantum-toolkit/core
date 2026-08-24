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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>

using namespace mlir;
namespace mlir::mqt::qco {

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

bool isOneAttribute(const Attribute attr) {
  if (!attr) {
    return false;
  }
  if (const auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getValue().isOne();
  }
  if (const auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValue().isExactlyValue(1.0);
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
  for (const auto& it : amplitudes) {
    if ((it.first >> *idx & 1ULL) != 0ULL) {
      return false;
    }
  }
  return true;
}

bool QuantumState::isAlwaysOne(const Value q) const {
  if (isTop) {
    return false;
  }
  const auto idx = indexOf(q);
  if (!idx) {
    return false;
  }
  for (const auto& it : amplitudes) {
    if ((it.first >> *idx & 1ULL) == 0ULL) {
      return false;
    }
  }
  return true;
}

void QuantumState::markTop() {
  isTop = true;
  amplitudes.clear();
}

void QuantumState::forwardQubit(const Value from, const Value to) {
  const auto id = indexOf(from);
  if (!id)
    return;
  qubits[id.value()] = to;
}

QuantumState QuantumState::tensorProduct(const QuantumState& that) {
  QuantumState result(maxTrackedAmplitudes);
  result.qubits.append(qubits.begin(), qubits.end());
  result.qubits.append(that.qubits.begin(), that.qubits.end());

  if (isTop || that.isTop) {
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
    if (out0 != Complex(0.0, 0.0))
      result[insertBit(it.first, idx, false)] += out0;
    if (out1 != Complex(0.0, 0.0))
      result[insertBit(it.first, idx, true)] += out1;
  }

  amplitudes = std::move(result);
  for (Value& q : qubits) {
    if (q == input) {
      q = output;
      break;
    }
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
      for (unsigned col = 0; col < 4; ++col)
        sum += matrix[row][col] * it.second[col];
      outVec[row] = sum;
    }

    for (unsigned row = 0; row < 4; ++row) {
      if (outVec[row] == Complex(0.0, 0.0))
        continue;
      bool b0 = (row & 1u) != 0u;
      bool b1 = (row & 2u) != 0u;
      uint64_t basis = insertBit(insertBit(it.first, idx0, b0), idx1, b1);
      result[basis] += outVec[row];
    }
  }

  out.amplitudes = std::move(result);
  for (Value& q : out.qubits) {
    if (q == input0)
      q = output0;
    else if (q == input1)
      q = output1;
  }
  out.enforceMaxAmplitudes(maxTrackedAmplitudes);
  return out;
}

LogicalResult QuantumState::applyUnitary(ArrayRef<Value> inputs,
                                         const UnitaryMatrix& matrix,
                                         ArrayRef<Value> outputs,
                                         unsigned maxTrackedAmplitudes) {
  if (inputs.size() != outputs.size())
    return failure();
  if (inputs.empty() || inputs.size() > 2)
    return failure();

  for (Value in : inputs) {
    if (!getComponentId(in))
      initializeQubit(in);
  }

  if (inputs.size() == 1) {
    auto id = getComponentId(inputs[0]);
    if (!id)
      return failure();

    QuantumState component = components[*id];
    if (!std::holds_alternative<Matrix2x2>(matrix)) {
      component.markTop();
    } else {
      component =
          applyMatrix1Q(component, inputs[0], outputs[0],
                        std::get<Matrix2x2>(matrix), maxTrackedAmplitudes);
    }

    unsigned newId = nextComponentId++;
    components.erase(*id);
    qubitToComponent.erase(inputs[0]);
    components[newId] = std::move(component);
    qubitToComponent[outputs[0]] = newId;
    return success();
  }

  if (failed(mergeComponents(inputs[0], inputs[1], maxTrackedAmplitudes)))
    return failure();

  auto mergedId = getComponentId(inputs[0]);
  if (!mergedId)
    return failure();

  QuantumState component = components[*mergedId];
  if (!std::holds_alternative<Matrix4x4>(matrix)) {
    component.markTop();
  } else {
    component =
        applyMatrix2Q(component, inputs[0], inputs[1], outputs[0], outputs[1],
                      std::get<Matrix4x4>(matrix), maxTrackedAmplitudes);
  }

  components.erase(*mergedId);
  qubitToComponent.erase(inputs[0]);
  qubitToComponent.erase(inputs[1]);

  unsigned newId = nextComponentId++;
  components[newId] = std::move(component);
  qubitToComponent[outputs[0]] = newId;
  qubitToComponent[outputs[1]] = newId;
  return success();
}

SmallVector<std::pair<QuantumState, Attribute>>
QuantumState::measure(Value inQubit, Value outQubit, MLIRContext* ctx) const {
  SmallVector<std::pair<QuantumState, Attribute>> successors;
  auto compId = getComponentId(inQubit);
  if (!compId) {
    QuantumState unknown = *this;
    auto i1 = IntegerType::get(ctx, 1);
    // Unknown classical result cannot be represented as an Attribute directly
    // here, so return no successors and let caller handle top/unknown.
    (void)i1;
    return successors;
  }

  const QuantumState& component = components.at(*compId);
  if (component.isTop)
    return successors;

  auto idxOpt = component.indexOf(inQubit);
  if (!idxOpt)
    return successors;
  unsigned idx = *idxOpt;

  double prob0 = 0.0;
  double prob1 = 0.0;
  for (const auto& it : component.amplitudes) {
    double p = std::norm(it.second);
    if (((it.first >> idx) & 1ULL) == 0ULL)
      prob0 += p;
    else
      prob1 += p;
  }

  auto makeSuccessor = [&](bool bit) {
    QuantumState next = *this;
    auto nextId = next.getComponentId(inQubit);
    if (!nextId)
      return std::pair<QuantumState, Attribute>{next, {}};

    QuantumState& c = next.components[*nextId];
    llvm::DenseMap<uint64_t, Complex> filtered;
    double norm = 0.0;
    for (const auto& it : c.amplitudes) {
      bool curBit = (((it.first >> idx) & 1ULL) != 0ULL);
      if (curBit == bit) {
        filtered[it.first] = it.second;
        norm += std::norm(it.second);
      }
    }

    if (norm == 0.0) {
      c.markTop();
    } else {
      double scale = 1.0 / std::sqrt(norm);
      for (auto& it : filtered)
        it.second *= scale;
      c.amplitudes = std::move(filtered);
    }

    for (Value& q : c.qubits) {
      if (q == inQubit) {
        q = outQubit;
        break;
      }
    }
    next.qubitToComponent.erase(inQubit);
    next.qubitToComponent[outQubit] = *nextId;

    auto i1 = IntegerType::get(ctx, 1);
    Attribute bitAttr = IntegerAttr::get(i1, bit ? 1 : 0);
    return std::pair<QuantumState, Attribute>{std::move(next), bitAttr};
  };

  if (prob0 > 0.0)
    successors.push_back(makeSuccessor(false));
  if (prob1 > 0.0)
    successors.push_back(makeSuccessor(true));
  return successors;
}

//===----------------------------------------------------------------------===//
// HybridState
//===----------------------------------------------------------------------===//

bool HybridState::operator==(const HybridState& other) const {
  if (probability != other.probability)
    return false;
  if (!(quantumState == other.quantumState))
    return false;
  if (classicalValues.size() != other.classicalValues.size())
    return false;
  for (const auto& it : classicalValues) {
    auto found = other.classicalValues.find(it.first);
    if (found == other.classicalValues.end())
      return false;
    if (!sameAttribute(it.second, found->second))
      return false;
  }
  return true;
}

std::optional<Attribute> HybridState::getClassical(Value v) const {
  auto it = classicalValues.find(v);
  if (it == classicalValues.end())
    return std::nullopt;
  return it->second;
}

void HybridState::setClassical(Value v, Attribute attr) {
  classicalValues[v] = attr;
}

//===----------------------------------------------------------------------===//
// HybridStateSet
//===----------------------------------------------------------------------===//

bool HybridStateSet::operator==(const HybridStateSet& other) const {
  if (isTop != other.isTop)
    return false;
  if (isTop)
    return true;
  if (states.size() != other.states.size())
    return false;
  for (const auto& s : states) {
    if (!llvm::is_contained(other.states, s))
      return false;
  }
  return true;
}

HybridStateSet HybridStateSet::top() {
  HybridStateSet s;
  s.isTop = true;
  return s;
}

HybridStateSet HybridStateSet::singletonInitial() {
  HybridStateSet s;
  s.states.push_back(HybridState{});
  return s;
}

void HybridStateSet::addState(HybridState state) {
  if (isTop)
    return;
  states.push_back(std::move(state));
}

void HybridStateSet::canonicalize() {
  if (isTop)
    return;

  SmallVector<HybridState> merged;
  for (HybridState& state : states) {
    bool found = false;
    for (HybridState& existing : merged) {
      HybridState lhs = state;
      HybridState rhs = existing;
      lhs.probability = 0.0;
      rhs.probability = 0.0;
      if (lhs == rhs) {
        existing.probability += state.probability;
        found = true;
        break;
      }
    }
    if (!found)
      merged.push_back(std::move(state));
  }
  states = std::move(merged);
}

void HybridStateSet::join(const HybridStateSet& other) {
  if (isTop || other.isTop) {
    isTop = true;
    states.clear();
    return;
  }
  states.append(other.states.begin(), other.states.end());
  canonicalize();
}

void HybridStateSet::enforceMaxStates(unsigned maxTrackedStates) {
  if (isTop)
    return;
  canonicalize();
  if (states.size() > maxTrackedStates) {
    isTop = true;
    states.clear();
  }
}

bool HybridStateSet::isAlwaysZero(Value v) const {
  if (isTop || states.empty())
    return false;
  for (const HybridState& state : states) {
    auto attr = state.getClassical(v);
    if (attr && isZeroAttribute(*attr))
      continue;
    if (state.quantumState.isAlwaysZero(v))
      continue;
    return false;
  }
  return true;
}

bool HybridStateSet::isAlwaysOne(Value v) const {
  if (isTop || states.empty())
    return false;
  for (const HybridState& state : states) {
    auto attr = state.getClassical(v);
    if (attr && isOneAttribute(*attr))
      continue;
    if (state.quantumState.isAlwaysOne(v))
      continue;
    return false;
  }
  return true;
}

std::optional<Attribute> HybridStateSet::getUniqueConstant(Value v) const {
  if (isTop || states.empty())
    return std::nullopt;
  std::optional<Attribute> candidate;
  for (const HybridState& state : states) {
    auto attr = state.getClassical(v);
    if (!attr)
      return std::nullopt;
    if (!candidate)
      candidate = attr;
    else if (*candidate != *attr)
      return std::nullopt;
  }
  return candidate;
}
} // namespace mlir::mqt::qco