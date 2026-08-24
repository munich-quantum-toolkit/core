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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::mqt::qco;

static uint64_t clearBit(uint64_t value, unsigned pos) {
  const uint64_t lowMask = (pos == 0) ? 0 : ((uint64_t{1} << pos) - 1);
  uint64_t low = value & lowMask;
  uint64_t high = value >> (pos + 1);
  return low | (high << pos);
}

static uint64_t insertBit(uint64_t value, unsigned pos, bool bit) {
  const uint64_t lowMask = (pos == 0) ? 0 : ((uint64_t{1} << pos) - 1);
  uint64_t low = value & lowMask;
  uint64_t high = value >> pos;
  return low | (uint64_t(bit) << pos) | (high << (pos + 1));
}

static bool sameAttribute(Attribute a, Attribute b) { return a == b; }

bool mlir::mqt::isZeroAttribute(Attribute attr) {
  if (!attr)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return !boolAttr.getValue();
  return false;
}

bool mlir::mqt::isOneAttribute(Attribute attr) {
  if (!attr)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isOne();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isExactlyValue(1.0);
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  return false;
}

//===----------------------------------------------------------------------===//
// QuantumComponent
//===----------------------------------------------------------------------===//

QuantumComponent QuantumComponent::singletonZero(Value qubit) {
  QuantumComponent c;
  c.qubits.push_back(qubit);
  c.amplitudes[0] = Complex(1.0, 0.0);
  return c;
}

QuantumComponent QuantumComponent::top(ArrayRef<Value> qs) {
  QuantumComponent c;
  c.isTop = true;
  c.qubits.append(qs.begin(), qs.end());
  return c;
}

bool QuantumComponent::operator==(const QuantumComponent& other) const {
  if (isTop != other.isTop)
    return false;
  if (qubits.size() != other.qubits.size())
    return false;
  for (auto [a, b] : llvm::zip(qubits, other.qubits)) {
    if (a != b)
      return false;
  }
  if (isTop)
    return true;
  if (amplitudes.size() != other.amplitudes.size())
    return false;
  for (const auto& it : amplitudes) {
    auto found = other.amplitudes.find(it.first);
    if (found == other.amplitudes.end())
      return false;
    if (found->second != it.second)
      return false;
  }
  return true;
}

bool QuantumComponent::contains(Value v) const {
  return llvm::is_contained(qubits, v);
}

std::optional<unsigned> QuantumComponent::indexOf(Value v) const {
  for (auto [idx, q] : llvm::enumerate(qubits)) {
    if (q == v)
      return idx;
  }
  return std::nullopt;
}

bool QuantumComponent::isAlwaysZero(Value q) const {
  if (isTop)
    return false;
  auto idx = indexOf(q);
  if (!idx)
    return false;
  for (const auto& it : amplitudes) {
    if (((it.first >> *idx) & 1ULL) != 0ULL)
      return false;
  }
  return true;
}

bool QuantumComponent::isAlwaysOne(Value q) const {
  if (isTop)
    return false;
  auto idx = indexOf(q);
  if (!idx)
    return false;
  for (const auto& it : amplitudes) {
    if (((it.first >> *idx) & 1ULL) == 0ULL)
      return false;
  }
  return true;
}

void QuantumComponent::markTop() {
  isTop = true;
  amplitudes.clear();
}

bool QuantumComponent::enforceMaxAmplitudes(unsigned maxTrackedAmplitudes) {
  if (isTop)
    return true;
  if (amplitudes.size() > maxTrackedAmplitudes) {
    markTop();
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// QuantumState
//===----------------------------------------------------------------------===//

bool QuantumState::operator==(const QuantumState& other) const {
  if (qubitToComponent.size() != other.qubitToComponent.size())
    return false;
  if (components.size() != other.components.size())
    return false;

  for (const auto& it : qubitToComponent) {
    auto found = other.qubitToComponent.find(it.first);
    if (found == other.qubitToComponent.end())
      return false;

    auto compA = components.find(it.second);
    auto compB = other.components.find(found->second);
    if (compA == components.end() || compB == other.components.end())
      return false;
    if (!(compA->second == compB->second))
      return false;
  }
  return true;
}

void QuantumState::initializeQubit(Value q) {
  if (qubitToComponent.count(q))
    return;
  assignFreshComponent(q, QuantumComponent::singletonZero(q));
}

void QuantumState::assignFreshComponent(Value q, QuantumComponent component) {
  unsigned id = nextComponentId++;
  qubitToComponent[q] = id;
  components[id] = std::move(component);
}

void QuantumState::forwardQubit(Value from, Value to) {
  auto id = getComponentId(from);
  if (!id)
    return;
  auto& component = components[*id];
  for (Value& q : component.qubits) {
    if (q == from) {
      q = to;
      break;
    }
  }
  qubitToComponent.erase(from);
  qubitToComponent[to] = *id;
}

std::optional<unsigned> QuantumState::getComponentId(Value q) const {
  auto it = qubitToComponent.find(q);
  if (it == qubitToComponent.end())
    return std::nullopt;
  return it->second;
}

QuantumComponent* QuantumState::getComponent(Value q) {
  auto id = getComponentId(q);
  if (!id)
    return nullptr;
  auto it = components.find(*id);
  if (it == components.end())
    return nullptr;
  return &it->second;
}

const QuantumComponent* QuantumState::getComponent(Value q) const {
  auto id = getComponentId(q);
  if (!id)
    return nullptr;
  auto it = components.find(*id);
  if (it == components.end())
    return nullptr;
  return &it->second;
}

void QuantumState::markTop(Value q) {
  if (auto* component = getComponent(q))
    component->markTop();
}

bool QuantumState::isAlwaysZero(Value q) const {
  if (const auto* component = getComponent(q))
    return component->isAlwaysZero(q);
  return false;
}

bool QuantumState::isAlwaysOne(Value q) const {
  if (const auto* component = getComponent(q))
    return component->isAlwaysOne(q);
  return false;
}

static QuantumComponent tensorProduct(const QuantumComponent& a,
                                      const QuantumComponent& b) {
  QuantumComponent result;
  result.qubits.append(a.qubits.begin(), a.qubits.end());
  result.qubits.append(b.qubits.begin(), b.qubits.end());

  if (a.isTop || b.isTop) {
    result.isTop = true;
    return result;
  }

  unsigned widthB = b.qubits.size();
  for (const auto& itA : a.amplitudes) {
    for (const auto& itB : b.amplitudes) {
      uint64_t basis = itA.first | (itB.first << a.qubits.size());
      result.amplitudes[basis] += itA.second * itB.second;
    }
  }
  return result;
}

LogicalResult QuantumState::mergeComponents(Value a, Value b,
                                            unsigned maxTrackedAmplitudes) {
  auto idA = getComponentId(a);
  auto idB = getComponentId(b);
  if (!idA || !idB)
    return failure();
  if (*idA == *idB)
    return success();

  QuantumComponent merged = tensorProduct(components[*idA], components[*idB]);
  merged.enforceMaxAmplitudes(maxTrackedAmplitudes);

  unsigned newId = nextComponentId++;
  components[newId] = std::move(merged);

  for (Value q : components[*idA].qubits)
    qubitToComponent[q] = newId;
  for (Value q : components[*idB].qubits)
    qubitToComponent[q] = newId;

  components.erase(*idA);
  components.erase(*idB);
  return success();
}

static QuantumComponent applyMatrix1Q(const QuantumComponent& component,
                                      Value input, Value output,
                                      const Matrix2x2& matrix,
                                      unsigned maxTrackedAmplitudes) {
  QuantumComponent out = component;
  if (out.isTop) {
    for (Value& q : out.qubits) {
      if (q == input) {
        q = output;
        break;
      }
    }
    return out;
  }

  auto idxOpt = out.indexOf(input);
  if (!idxOpt) {
    out.markTop();
    return out;
  }
  unsigned idx = *idxOpt;

  llvm::DenseMap<uint64_t, Complex> result;
  llvm::DenseMap<uint64_t, Complex> inputAmps = out.amplitudes;

  // Group amplitudes by all bits except target bit.
  llvm::DenseMap<uint64_t, std::array<Complex, 2>> grouped;
  for (const auto& it : inputAmps) {
    uint64_t reduced = clearBit(it.first, idx);
    bool bit = ((it.first >> idx) & 1ULL) != 0ULL;
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

  out.amplitudes = std::move(result);
  for (Value& q : out.qubits) {
    if (q == input) {
      q = output;
      break;
    }
  }
  out.enforceMaxAmplitudes(maxTrackedAmplitudes);
  return out;
}

static QuantumComponent applyMatrix2Q(const QuantumComponent& component,
                                      Value input0, Value input1, Value output0,
                                      Value output1, const Matrix4x4& matrix,
                                      unsigned maxTrackedAmplitudes) {
  QuantumComponent out = component;
  if (out.isTop) {
    for (Value& q : out.qubits) {
      if (q == input0)
        q = output0;
      else if (q == input1)
        q = output1;
    }
    return out;
  }

  auto idx0Opt = out.indexOf(input0);
  auto idx1Opt = out.indexOf(input1);
  if (!idx0Opt || !idx1Opt || *idx0Opt == *idx1Opt) {
    out.markTop();
    return out;
  }
  unsigned idx0 = *idx0Opt;
  unsigned idx1 = *idx1Opt;
  if (idx0 > idx1)
    std::swap(idx0, idx1);

  llvm::DenseMap<uint64_t, std::array<Complex, 4>> grouped;
  for (const auto& it : out.amplitudes) {
    bool b0 = ((it.first >> idx0) & 1ULL) != 0ULL;
    bool b1 = ((it.first >> idx1) & 1ULL) != 0ULL;
    unsigned local = unsigned(b0) | (unsigned(b1) << 1u);
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

    QuantumComponent component = components[*id];
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

  QuantumComponent component = components[*mergedId];
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

  const QuantumComponent& component = components.at(*compId);
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

    QuantumComponent& c = next.components[*nextId];
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