/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

namespace mlir::qco {

#define GEN_PASS_DEF_RESYNTHESIZECNOTPHASE
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

using Parity = uint64_t;

static Parity parityBit(unsigned index) { return Parity{1} << index; }

static bool isCNOT(Operation* operation) {
  auto control = dyn_cast<CtrlOp>(operation);
  if (!control || control.getNumControls() != 1 ||
      control.getNumTargets() != 1) {
    return false;
  }
  // Do not discard supporting classical operations or additional modifiers.
  auto& body = *control.getBody();
  return llvm::hasSingleElement(body.without_terminator()) &&
         isa<XOp>(body.front());
}

static bool isCNOTPhaseGate(Operation* operation) {
  return isa<RZOp, POp, ZOp, SOp, SdgOp, TOp, TdgOp>(operation) ||
         isCNOT(operation);
}

namespace {

/// Each phase gate keeps its original kind and angle, including RZ's global
/// phase. Its argument is an XOR of block inputs, never an affine complement.
struct Phase {
  Operation* operation;
  Parity parity;
};

/// A CNOT, or a phase gate identified by its index in the block's phase list.
struct Step {
  unsigned target;
  unsigned control = 0;
  unsigned phase = std::numeric_limits<unsigned>::max();

  [[nodiscard]] bool isPhase() const {
    return phase != std::numeric_limits<unsigned>::max();
  }
};

/// Word-sized GraySynth (Algorithm 1 of arXiv:1712.01859), independently
/// implemented from its parity-coordinate updates. No angle algebra is needed.
class ParitySynthesis {
  SmallVector<Parity> masks_;
  SmallVector<Step> steps_;
  size_t budget_;
  size_t cnots_ = 0;

  bool cnot(unsigned control, unsigned target) {
    if (++cnots_ >= budget_) {
      return false;
    }
    steps_.push_back({.target = target, .control = control});
    // y_target ^= y_control changes coordinates of every pending parity.
    for (auto& mask : masks_) {
      if ((mask & parityBit(target)) != 0) {
        mask ^= parityBit(control);
      }
    }
    return true;
  }

  bool partition(ArrayRef<unsigned> terms, Parity columns, int target = -1) {
    if (terms.empty()) {
      return true;
    }
    if (target >= 0) {
      const auto wire = static_cast<unsigned>(target);
      auto common = ~parityBit(wire);
      for (const auto term : terms) {
        common &= masks_[term];
      }
      while (common != 0) {
        const auto control = static_cast<unsigned>(std::countr_zero(common));
        common &= common - 1;
        if (!cnot(control, wire)) {
          return false;
        }
      }
    }
    if (columns == 0) {
      if (target < 0) {
        return false;
      }
      for (const auto term : terms) {
        steps_.push_back(
            {.target = static_cast<unsigned>(target), .phase = term});
      }
      return true;
    }

    unsigned split = 0;
    size_t best = 0;
    for (auto remaining = columns; remaining != 0; remaining &= remaining - 1) {
      const auto column = static_cast<unsigned>(std::countr_zero(remaining));
      const auto ones = static_cast<size_t>(llvm::count_if(
          terms, [&](auto i) { return (masks_[i] & parityBit(column)) != 0; }));
      const auto score = std::max(ones, terms.size() - ones);
      if (score > best) {
        best = score;
        split = column;
      }
    }
    SmallVector<unsigned> zero;
    SmallVector<unsigned> one;
    for (const auto term : terms) {
      ((masks_[term] & parityBit(split)) == 0 ? zero : one).push_back(term);
    }
    columns &= ~parityBit(split);
    // The zero branch is visited first, as in the paper's explicit stack.
    return partition(zero, columns, target) &&
           partition(one, columns,
                     target < 0 ? static_cast<int>(split) : target);
  }

  bool restoreLinearMap(ArrayRef<Parity> desired) {
    const auto width = static_cast<unsigned>(desired.size());
    SmallVector<Parity> residual(desired);
    // B = L_m ... L_1, so A B^-1 = A L_1 ... L_m. Right multiplication
    // by CNOT(c,t) toggles column c using column t, in forward gate order.
    for (const auto& step : steps_) {
      if (step.isPhase()) {
        continue;
      }
      for (auto& row : residual) {
        if ((row & parityBit(step.target)) != 0) {
          row ^= parityBit(step.control);
        }
      }
    }
    SmallVector<Step> elimination;
    const auto addRow = [&](unsigned control, unsigned target) {
      residual[target] ^= residual[control];
      elimination.push_back({.target = target, .control = control});
      return cnots_ + elimination.size() < budget_;
    };
    // ponytail: bounded Gaussian elimination; add PMH only if measurements
    // justify its extra machinery for these at-most-64-qubit blocks.
    for (unsigned column = 0; column < width; ++column) {
      if ((residual[column] & parityBit(column)) == 0) {
        auto pivot = column + 1;
        while (pivot < width && (residual[pivot] & parityBit(column)) == 0) {
          ++pivot;
        }
        if (pivot == width || !addRow(pivot, column)) {
          return false;
        }
      }
      for (unsigned row = 0; row < width; ++row) {
        if (row != column && (residual[row] & parityBit(column)) != 0 &&
            !addRow(column, row)) {
          return false;
        }
      }
    }
    for (const auto& step : llvm::reverse(elimination)) {
      steps_.push_back(step);
    }
    return true;
  }

public:
  ParitySynthesis(ArrayRef<Phase> phases, size_t budget) : budget_(budget) {
    for (const auto& phase : phases) {
      masks_.push_back(phase.parity);
    }
  }

  bool synthesize(ArrayRef<Parity> desired) {
    SmallVector<unsigned> terms(masks_.size());
    std::iota(terms.begin(), terms.end(), 0U);
    const auto columns =
        desired.size() == 64 ? ~Parity{0} : parityBit(desired.size()) - 1;
    return partition(terms, columns) && restoreLinearMap(desired);
  }

  ArrayRef<Step> steps() { return steps_; }
};

} // namespace

/// Replay against the original specification, independently of the heuristic's
/// changing coordinates. Check every phase once and the full output map.
static bool checkCandidate(ArrayRef<Step> steps, ArrayRef<Phase> phases,
                           ArrayRef<Parity> desired) {
  SmallVector<Parity> rows;
  for (unsigned i = 0; i < desired.size(); ++i) {
    rows.push_back(parityBit(i));
  }
  SmallVector<bool> seen(phases.size(), false);
  for (const auto& step : steps) {
    if (step.target >= rows.size()) {
      return false;
    }
    if (!step.isPhase()) {
      if (step.control >= rows.size() || step.control == step.target) {
        return false;
      }
      rows[step.target] ^= rows[step.control];
    } else {
      const auto index = static_cast<size_t>(step.phase);
      if (index >= phases.size() || seen[index] ||
          rows[step.target] != phases[index].parity) {
        return false;
      }
      seen[index] = true;
    }
  }
  return ArrayRef<Parity>(rows) == desired &&
         llvm::all_of(seen, [](bool value) { return value; });
}

namespace {

class CNOTPhaseBlock {
  DenseMap<Value, unsigned> wires_;
  SmallVector<Value> inputs_;
  SmallVector<Value> outputs_;
  SmallVector<Parity> rows_;
  SmallVector<Operation*> operations_;
  SmallVector<Phase> phases_;
  size_t cnots_ = 0;

public:
  bool fits(UnitaryOpInterface gate, unsigned maxQubits, unsigned maxGates) {
    const auto newWires =
        llvm::count_if(gate.getInputQubits(),
                       [&](Value value) { return !wires_.contains(value); });
    return inputs_.size() + newWires <= maxQubits &&
           operations_.size() < maxGates;
  }

  bool touches(Operation* operation) {
    return llvm::any_of(operation->getOperands(),
                        [&](Value value) { return wires_.contains(value); });
  }

  void append(UnitaryOpInterface gate) {
    SmallVector<unsigned, 2> indices;
    for (auto input : gate.getInputQubits()) {
      auto [it, inserted] =
          wires_.try_emplace(input, static_cast<unsigned>(inputs_.size()));
      const auto index = it->second;
      if (inserted) {
        inputs_.push_back(input);
        outputs_.push_back(input);
        rows_.push_back(parityBit(index));
      }
      indices.push_back(index);
      outputs_[index] = gate.getOutputForInput(input);
      wires_.erase(input);
      wires_.try_emplace(outputs_[index], index);
    }
    operations_.push_back(gate.getOperation());
    if (isCNOT(gate.getOperation())) {
      rows_[indices[1]] ^= rows_[indices[0]];
      ++cnots_;
    } else {
      phases_.push_back(
          {.operation = gate.getOperation(), .parity = rows_[indices[0]]});
    }
  }

  void rewrite(Operation* boundary) {
    if (cnots_ == 0) {
      return;
    }
    ParitySynthesis synthesis(phases_, cnots_);
    if (!synthesis.synthesize(rows_) ||
        !checkCandidate(synthesis.steps(), phases_, rows_)) {
      return;
    }
    OpBuilder builder(boundary);
    auto values = inputs_;
    for (const auto& step : synthesis.steps()) {
      if (!step.isPhase()) {
        auto gate = CtrlOp::create(
            builder, operations_.front()->getLoc(), values[step.control],
            values[step.target], [&](Value target) -> Value {
              return XOp::create(builder, operations_.front()->getLoc(), target)
                  .getOutputQubit(0);
            });
        values[step.control] = gate.getOutputControl(0);
        values[step.target] = gate.getOutputTarget(0);
      } else {
        auto gate = cast<UnitaryOpInterface>(phases_[step.phase].operation);
        IRMapping mapping;
        mapping.map(gate.getInputQubit(0), values[step.target]);
        auto clone = cast<UnitaryOpInterface>(
            builder.clone(*gate.getOperation(), mapping));
        values[step.target] = clone.getOutputQubit(0);
      }
    }
    for (auto [oldValue, newValue] : llvm::zip_equal(outputs_, values)) {
      oldValue.replaceAllUsesWith(newValue);
    }
    for (auto* operation : llvm::reverse(operations_)) {
      operation->erase();
    }
  }
};

struct ResynthesizeCNOTPhasePass final
    : impl::ResynthesizeCNOTPhaseBase<ResynthesizeCNOTPhasePass> {
  using ResynthesizeCNOTPhaseBase::ResynthesizeCNOTPhaseBase;

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (maxQubits < 1 || maxQubits > 64 || maxGates < 1) {
      moduleOp.emitError("resynthesize-cnot-phase requires max-qubits in [1, "
                         "64] and positive max-gates");
      return signalPassFailure();
    }
    if (failed(verifyLinearity(moduleOp))) {
      return signalPassFailure();
    }
    moduleOp.walk([&](Block* block) {
      CNOTPhaseBlock region;
      for (auto& operation : llvm::make_early_inc_range(*block)) {
        auto* op = &operation;
        if (isCNOTPhaseGate(op)) {
          auto gate = cast<UnitaryOpInterface>(op);
          if (!region.fits(gate, maxQubits, maxGates)) {
            region.rewrite(op);
            region = CNOTPhaseBlock{};
          }
          if (region.fits(gate, maxQubits, maxGates)) {
            region.append(gate);
          }
        } else if (op->getNumRegions() != 0 || !isMemoryEffectFree(op) ||
                   op->hasTrait<OpTrait::IsTerminator>() ||
                   region.touches(op)) {
          region.rewrite(op);
          region = CNOTPhaseBlock{};
        }
      }
    });
    if (failed(verifyLinearity(moduleOp))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::qco
