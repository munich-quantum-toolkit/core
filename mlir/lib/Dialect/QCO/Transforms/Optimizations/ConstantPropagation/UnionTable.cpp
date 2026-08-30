/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "UnionTable.hpp"

#include "HybridState.hpp"
#include "QuantumState.hpp"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>
#include <utility>
#include <vector>

namespace mlir::qco {

/// The qubit set of a slot, order-normalized, so two slots (from sibling
/// control-flow paths) can be matched.
static std::vector<const void*> qubitKey(const UnionTable::Slot& slot) {
  std::vector<const void*> key;
  key.reserve(slot.front().getQubits().size());
  for (Value q : slot.front().getQubits()) {
    key.push_back(q.getAsOpaquePointer());
  }
  llvm::sort(key);
  return key;
}

//===----------------------------------------------------------------------===//
// Partition helpers
//===----------------------------------------------------------------------===//

std::optional<unsigned> UnionTable::slotIndexContaining(Value v) const {
  for (const auto& [i, slot] : llvm::enumerate(slots)) {
    if (slot.front().hasQubit(v)) {
      return static_cast<unsigned>(i);
    }
    for (const auto& hs : slot) {
      if (hs.getClassical(v).has_value()) {
        return static_cast<unsigned>(i);
      }
    }
  }
  return std::nullopt;
}

SmallVector<unsigned> UnionTable::slotsTouchedBy(ArrayRef<Value> values) const {
  SmallVector<unsigned> result;
  for (Value v : values) {
    if (const auto i = slotIndexContaining(v)) {
      if (!llvm::is_contained(result, *i)) {
        result.push_back(*i);
      }
    }
  }
  llvm::sort(result);
  return result;
}

HybridState UnionTable::reducedRepresentative(const Slot& slot) {
  HybridState representative = slot.front();
  for (size_t j = 1; j < slot.size(); ++j) {
    representative.intersectClassical(slot[j]);
  }
  return representative;
}

void UnionTable::mergeSlots(ArrayRef<Value> values) {
  if (allTop) {
    return;
  }
  const auto touched = slotsTouchedBy(values);
  if (touched.size() <= 1) {
    return;
  }

  size_t product = 1;
  bool overflow = false;
  for (const unsigned i : touched) {
    if (product > maxHybridStates / slots[i].size()) {
      overflow = true;
      break;
    }
    product *= slots[i].size();
  }

  Slot fused;
  if (overflow) {
    const auto toppedRepresentative = [](const Slot& slot) {
      HybridState representative = reducedRepresentative(slot);
      representative.markStateTop();
      return representative;
    };
    HybridState top = toppedRepresentative(slots[touched.front()]);
    for (size_t k = 1; k < touched.size(); ++k) {
      top = top.tensor(toppedRepresentative(slots[touched[k]]));
    }
    top.setProbability(1.0);
    fused.push_back(std::move(top));
  } else {
    fused = slots[touched.front()];
    for (size_t k = 1; k < touched.size(); ++k) {
      const Slot& next = slots[touched[k]];
      Slot combined;
      combined.reserve(fused.size() * next.size());
      for (const auto& a : fused) {
        for (const auto& b : next) {
          combined.push_back(a.tensor(b));
        }
      }
      fused = std::move(combined);
    }
  }

  // Erase the merged slots high-index-first, then append the fused one.
  for (size_t k = touched.size(); k-- > 0;) {
    slots.erase(slots.begin() + touched[k]);
  }
  slots.push_back(std::move(fused));
}

UnionTable::Slot UnionTable::mergeAlternatives(const Slot& a, const Slot& b) {
  Slot merged;
  const auto absorb = [&merged](const Slot& side) {
    for (const auto& hs : side) {
      HybridState* match = nullptr;
      for (auto& candidate : merged) {
        if (candidate.sameConfiguration(hs)) {
          match = &candidate;
          break;
        }
      }
      if (match != nullptr) {
        match->setProbability(match->getProbability() + hs.getProbability());
      } else {
        merged.push_back(hs);
      }
    }
  };
  absorb(a);
  absorb(b);

  double sum = 0.0;
  for (const auto& hs : merged) {
    sum += hs.getProbability();
  }
  if (sum > MATRIX_TOLERANCE) {
    for (auto& hs : merged) {
      hs.scaleProbability(1.0 / sum);
    }
  }
  return merged;
}

bool UnionTable::sameSlot(const Slot& a, const Slot& b) {
  if (a.size() != b.size()) {
    return false;
  }
  SmallVector<bool> used(b.size(), false);
  for (const auto& lhs : a) {
    bool matched = false;
    for (unsigned j = 0; j < b.size(); ++j) {
      if (!used[j] && lhs == b[j]) {
        used[j] = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Seeding
//===----------------------------------------------------------------------===//

void UnionTable::seedQubit(Value qubit) {
  if (allTop || isTracked(qubit)) {
    return;
  }
  Slot slot;
  slot.emplace_back(QuantumState::singletonZero(qubit, maxNonzeroAmplitudes),
                    maxNonzeroAmplitudes, 1.0);
  slots.push_back(std::move(slot));
}

void UnionTable::seedClassical(Value value, Attribute attr) {
  if (allTop) {
    return;
  }
  if (const auto i = slotIndexContaining(value)) {
    for (auto& hs : slots[*i]) {
      hs.setClassical(value, attr);
    }
    return;
  }
  Slot slot;
  slot.emplace_back(QuantumState(ArrayRef<Value>{}, maxNonzeroAmplitudes),
                    maxNonzeroAmplitudes, 1.0);
  slot.back().setClassical(value, attr);
  slots.push_back(std::move(slot));
}

bool UnionTable::isTracked(Value v) const {
  return slotIndexContaining(v).has_value();
}

//===----------------------------------------------------------------------===//
// SSA forwarding
//===----------------------------------------------------------------------===//

void UnionTable::forwardValue(Value from, Value to) {
  for (auto& slot : slots) {
    for (auto& hs : slot) {
      hs.forwardValue(from, to);
    }
  }
}

void UnionTable::forwardValues(ArrayRef<Value> from, ArrayRef<Value> to) {
  for (const auto [f, t] : llvm::zip(from, to)) {
    forwardValue(f, t);
  }
}

//===----------------------------------------------------------------------===//
// Operation propagation
//===----------------------------------------------------------------------===//

LogicalResult UnionTable::applyMatrix1Q(Value in, Value out,
                                        const Matrix2x2& matrix,
                                        ArrayRef<Value> quantumCtrlsIn,
                                        ArrayRef<Value> quantumCtrlsOut,
                                        ArrayRef<Value> posClassicalCtrls,
                                        ArrayRef<Value> negClassicalCtrls) {
  if (allTop) {
    return success();
  }
  if (!isTracked(in)) {
    return failure();
  }

  SmallVector<Value> touched{in};
  touched.append(quantumCtrlsIn.begin(), quantumCtrlsIn.end());
  touched.append(posClassicalCtrls.begin(), posClassicalCtrls.end());
  touched.append(negClassicalCtrls.begin(), negClassicalCtrls.end());
  mergeSlots(touched);
  if (allTop) {
    return success();
  }

  for (auto& hs : slots[*slotIndexContaining(in)]) {
    if (failed(hs.applyMatrix1Q(in, out, matrix, quantumCtrlsIn,
                                quantumCtrlsOut, posClassicalCtrls,
                                negClassicalCtrls))) {
      return failure();
    }
  }
  return success();
}

LogicalResult UnionTable::applyMatrix2Q(Value in0, Value in1, Value out0,
                                        Value out1, const Matrix4x4& matrix,
                                        ArrayRef<Value> quantumCtrlsIn,
                                        ArrayRef<Value> quantumCtrlsOut,
                                        ArrayRef<Value> posClassicalCtrls,
                                        ArrayRef<Value> negClassicalCtrls) {
  if (allTop) {
    return success();
  }
  if (!isTracked(in0) || !isTracked(in1)) {
    return failure();
  }

  SmallVector<Value> touched{in0, in1};
  touched.append(quantumCtrlsIn.begin(), quantumCtrlsIn.end());
  touched.append(posClassicalCtrls.begin(), posClassicalCtrls.end());
  touched.append(negClassicalCtrls.begin(), negClassicalCtrls.end());
  mergeSlots(touched);
  if (allTop) {
    return success();
  }

  for (auto& hs : slots[*slotIndexContaining(in0)]) {
    if (failed(hs.applyMatrix2Q(in0, in1, out0, out1, matrix, quantumCtrlsIn,
                                quantumCtrlsOut, posClassicalCtrls,
                                negClassicalCtrls))) {
      return failure();
    }
  }
  return success();
}

LogicalResult UnionTable::addGlobalPhase(Value theta,
                                         ArrayRef<Value> quantumCtrlsIn,
                                         ArrayRef<Value> quantumCtrlsOut,
                                         ArrayRef<Value> posClassicalCtrls,
                                         ArrayRef<Value> negClassicalCtrls) {
  if (allTop) {
    return success();
  }
  if ((quantumCtrlsIn.empty() && !quantumCtrlsOut.empty()) ||
      !isTracked(theta)) {
    return failure();
  }

  SmallVector<Value> touched{theta};
  touched.append(quantumCtrlsIn.begin(), quantumCtrlsIn.end());
  touched.append(posClassicalCtrls.begin(), posClassicalCtrls.end());
  touched.append(negClassicalCtrls.begin(), negClassicalCtrls.end());
  mergeSlots(touched);
  if (allTop) {
    return success();
  }

  Value anchor = theta;
  if (!quantumCtrlsIn.empty()) {
    anchor = quantumCtrlsIn.front();
  } else if (!posClassicalCtrls.empty()) {
    anchor = posClassicalCtrls.front();
  } else if (!negClassicalCtrls.empty()) {
    anchor = negClassicalCtrls.front();
  }

  const auto slot = slotIndexContaining(anchor);
  if (!slot) {
    return failure();
  }
  for (auto& hs : slots[*slot]) {
    if (failed(hs.addGlobalPhase(theta, quantumCtrlsIn, quantumCtrlsOut,
                                 posClassicalCtrls, negClassicalCtrls))) {
      return failure();
    }
  }
  return success();
}

void UnionTable::propagateClassical(Operation* op) {
  if (allTop) {
    return;
  }
  const SmallVector<Value> operands(op->getOperands().begin(),
                                    op->getOperands().end());
  mergeSlots(operands);
  if (allTop) {
    return;
  }
  for (Value operand : operands) {
    if (const auto slot = slotIndexContaining(operand)) {
      for (auto& hs : slots[*slot]) {
        hs.propagateClassical(op);
      }
      return;
    }
  }
}

LogicalResult UnionTable::measureQubit(Value in, Value out,
                                       Value classicalResult,
                                       ArrayRef<Value> posClassicalCtrls,
                                       ArrayRef<Value> negClassicalCtrls) {
  if (allTop) {
    return success();
  }
  if (!isTracked(in)) {
    return failure();
  }

  SmallVector<Value> touched{in};
  touched.push_back(classicalResult);
  touched.append(posClassicalCtrls.begin(), posClassicalCtrls.end());
  touched.append(negClassicalCtrls.begin(), negClassicalCtrls.end());
  mergeSlots(touched);
  if (allTop) {
    return success();
  }

  for (auto& hs : slots[*slotIndexContaining(in)]) {
    if (failed(hs.measureQubit(in, out, classicalResult, posClassicalCtrls,
                               negClassicalCtrls))) {
      return failure();
    }
  }
  return success();
}

LogicalResult UnionTable::resetQubit(Value in, Value out,
                                     ArrayRef<Value> posClassicalCtrls,
                                     ArrayRef<Value> negClassicalCtrls) {
  if (allTop) {
    return success();
  }
  if (!isTracked(in)) {
    return failure();
  }

  SmallVector<Value> touched{in};
  touched.append(posClassicalCtrls.begin(), posClassicalCtrls.end());
  touched.append(negClassicalCtrls.begin(), negClassicalCtrls.end());
  mergeSlots(touched);
  if (allTop) {
    return success();
  }

  for (auto& hs : slots[*slotIndexContaining(in)]) {
    if (failed(hs.resetQubit(in, out, posClassicalCtrls, negClassicalCtrls))) {
      return failure();
    }
  }
  return success();
}

void UnionTable::markQubitsTop(ArrayRef<Value> qubits) {
  if (allTop) {
    return;
  }
  llvm::DenseSet<unsigned> done;
  for (Value q : qubits) {
    const auto i = slotIndexContaining(q);
    if (i && done.insert(*i).second) {
      for (auto& hs : slots[*i]) {
        hs.markStateTop();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

bool UnionTable::isQubitAlwaysOne(Value q) const {
  if (allTop) {
    return false;
  }
  const auto i = slotIndexContaining(q);
  return i && llvm::all_of(slots[*i], [&](const HybridState& hs) {
           return hs.isQubitAlwaysOne(q);
         });
}

bool UnionTable::isQubitAlwaysZero(Value q) const {
  if (allTop) {
    return false;
  }
  const auto i = slotIndexContaining(q);
  return i && llvm::all_of(slots[*i], [&](const HybridState& hs) {
           return hs.isQubitAlwaysZero(q);
         });
}

bool UnionTable::isClassicalAlwaysTrue(Value v) const {
  if (allTop) {
    return false;
  }
  const auto i = slotIndexContaining(v);
  return i && llvm::all_of(slots[*i], [&](const HybridState& hs) {
           return hs.isClassicalTrue(v);
         });
}

bool UnionTable::isClassicalAlwaysFalse(Value v) const {
  if (allTop) {
    return false;
  }
  const auto i = slotIndexContaining(v);
  return i && llvm::all_of(slots[*i], [&](const HybridState& hs) {
           return hs.isClassicalFalse(v);
         });
}

bool UnionTable::areControlsSatisfiable(
    ArrayRef<Value> quantumCtrls, ArrayRef<Value> posClassicalCtrls,
    ArrayRef<Value> negClassicalCtrls) const {
  if (allTop) {
    return true;
  }

  SmallVector<Value> all(quantumCtrls.begin(), quantumCtrls.end());
  all.append(posClassicalCtrls.begin(), posClassicalCtrls.end());
  all.append(negClassicalCtrls.begin(), negClassicalCtrls.end());

  for (const unsigned si : slotsTouchedBy(all)) {
    const Slot& slot = slots[si];
    const auto hasClassical = [&](Value c) {
      return llvm::any_of(slot, [&](const HybridState& hs) {
        return hs.getClassical(c).has_value();
      });
    };

    SmallVector<Value> quantum;
    for (Value c : quantumCtrls) {
      if (slot.front().hasQubit(c)) {
        quantum.push_back(c);
      }
    }
    SmallVector<Value> pos;
    for (Value c : posClassicalCtrls) {
      if (hasClassical(c)) {
        pos.push_back(c);
      }
    }
    SmallVector<Value> neg;
    for (Value c : negClassicalCtrls) {
      if (hasClassical(c)) {
        neg.push_back(c);
      }
    }
    if (quantum.empty() && pos.empty() && neg.empty()) {
      continue;
    }
    const bool anySatisfiable = llvm::any_of(slot, [&](const HybridState& hs) {
      return hs.areControlsSatisfiable(quantum, pos, neg);
    });
    if (!anySatisfiable) {
      return false;
    }
  }
  return true;
}

SuperfluousResult
UnionTable::getSuperfluousControls(ArrayRef<Value> quantumCtrls,
                                   ArrayRef<Value> posClassicalCtrls,
                                   ArrayRef<Value> negClassicalCtrls) const {
  SuperfluousResult result;
  if (!areControlsSatisfiable(quantumCtrls, posClassicalCtrls,
                              negClassicalCtrls)) {
    result.completelySuperfluous = true;
    return result;
  }
  for (Value q : quantumCtrls) {
    if (isQubitAlwaysOne(q)) {
      result.superfluousQubits.insert(q);
    }
  }
  for (Value p : posClassicalCtrls) {
    if (isClassicalAlwaysTrue(p)) {
      result.superfluousClassicalValues.insert(p);
    }
  }
  for (Value n : negClassicalCtrls) {
    if (isClassicalAlwaysFalse(n)) {
      result.superfluousClassicalValues.insert(n);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Lattice support
//===----------------------------------------------------------------------===//

void UnionTable::join(const UnionTable& other) {
  if (allTop || other.allTop) {
    markAllTop();
    return;
  }

  SmallVector<unsigned> myQuantum;
  SmallVector<unsigned> myClassical;
  for (const auto& [i, slot] : llvm::enumerate(slots)) {
    (slot.front().getQubits().empty() ? myClassical : myQuantum)
        .push_back(static_cast<unsigned>(i));
  }
  SmallVector<unsigned> theirQuantum;
  SmallVector<unsigned> theirClassical;
  for (const auto& [i, slot] : llvm::enumerate(other.slots)) {
    (slot.front().getQubits().empty() ? theirClassical : theirQuantum)
        .push_back(static_cast<unsigned>(i));
  }

  if (myQuantum.size() != theirQuantum.size()) {
    markAllTop(); // different entanglement structure
    return;
  }

  SmallVector<Slot> merged;

  for (const unsigned mi : myQuantum) {
    const auto key = qubitKey(slots[mi]);
    const Slot* theirs = nullptr;
    for (const unsigned ti : theirQuantum) {
      if (qubitKey(other.slots[ti]) == key) {
        theirs = &other.slots[ti];
        break;
      }
    }
    if (theirs == nullptr) {
      markAllTop();
      return;
    }

    Slot combined = mergeAlternatives(slots[mi], *theirs);
    if (combined.size() > maxHybridStates) {
      // Too many alternatives for this factor: collapse just this slot to top.
      HybridState top = reducedRepresentative(combined);
      top.markStateTop();
      top.setProbability(1.0);
      combined.clear();
      combined.push_back(std::move(top));
    }
    merged.push_back(std::move(combined));
  }

  // A purely classical fact survives only if the other branch asserts the same
  // one; otherwise it becomes unknown (it is simply dropped).
  for (const unsigned mi : myClassical) {
    const bool inBoth = llvm::any_of(theirClassical, [&](unsigned ti) {
      return llvm::any_of(other.slots[ti], [&](const HybridState& theirHs) {
        return slots[mi].front().sameConfiguration(theirHs);
      });
    });
    if (inBoth) {
      Slot slot;
      slot.push_back(slots[mi].front());
      slot.back().setProbability(1.0);
      merged.push_back(std::move(slot));
    }
  }

  slots = std::move(merged);
}

void UnionTable::markAllTop() {
  allTop = true;
  slots.clear();
}

bool UnionTable::areStatesAllTop() const {
  if (allTop) {
    return true;
  }
  bool sawQuantum = false;
  for (const auto& slot : slots) {
    if (slot.front().getQubits().empty()) {
      continue;
    }
    sawQuantum = true;
    for (const auto& hs : slot) {
      if (!hs.isTop()) {
        return false;
      }
    }
  }
  return sawQuantum;
}

bool UnionTable::operator==(const UnionTable& other) const {
  if (allTop || other.allTop) {
    return allTop == other.allTop;
  }
  if (slots.size() != other.slots.size()) {
    return false;
  }
  SmallVector<bool> used(other.slots.size(), false);
  for (const auto& mine : slots) {
    bool matched = false;
    for (unsigned j = 0; j < other.slots.size(); ++j) {
      if (!used[j] && sameSlot(mine, other.slots[j])) {
        used[j] = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

void UnionTable::print(raw_ostream& os) const {
  if (allTop) {
    os << "<all top>";
    return;
  }
  if (slots.empty()) {
    os << "<empty>";
    return;
  }
  bool firstSlot = true;
  for (const auto& slot : slots) {
    if (!firstSlot) {
      os << "\n---\n";
    }
    firstSlot = false;
    bool firstAlt = true;
    for (const auto& hs : slot) {
      if (!firstAlt) {
        os << "\n";
      }
      firstAlt = false;
      hs.print(os);
    }
  }
}

} // namespace mlir::qco
