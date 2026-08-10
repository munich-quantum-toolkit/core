/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/Utils/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>

#define DEBUG_TYPE "unroll-modifiers"

namespace mlir::mqt {

#define GEN_PASS_DEF_UNROLLMODIFIERS
#include "mlir/Dialect/Utils/Transforms/Passes.h.inc"

/// Return the distinct qubit operands of @p op in operand order.
template <typename QubitType>
static SmallVector<Value> getQubitOperands(Operation* op) {
  SmallVector<Value> qubits;
  for (auto operand : op->getOperands()) {
    if (isa<QubitType>(operand.getType()) &&
        !llvm::is_contained(qubits, operand)) {
      qubits.push_back(operand);
    }
  }
  return qubits;
}

/**
 *@brief Move the classical operations of @p body in front of @p modifier.
 *
 * @details Fails if a classical operation is impure or depends on values
 * defined in @p body.
 */
template <typename UnitaryOpInterface>
static LogicalResult hoistClassicalOps(Block& body, Operation* modifier,
                                       RewriterBase& rewriter) {
  const auto isClassical = [](Operation& op) {
    return !isa<UnitaryOpInterface>(op) &&
           !op.hasTrait<OpTrait::IsTerminator>();
  };
  for (auto& op : body) {
    if (isClassical(op) &&
        (!isPure(&op) || llvm::any_of(op.getOperands(), [&](Value operand) {
          return operand.getParentBlock() == &body;
        }))) {
      return failure();
    }
  }
  for (auto& op : llvm::make_early_inc_range(body)) {
    if (isClassical(op)) {
      rewriter.moveOpBefore(&op, modifier);
    }
  }
  return success();
}

/// Check whether the exponent of @p op is a compile-time known integer.
template <typename PowOp> static bool hasIntegerExponent(PowOp op) {
  const auto exponent = op.getExponentValue();
  return exponent && utils::isIntegerExponent(*exponent);
}

//===----------------------------------------------------------------------===//
// QC
//===----------------------------------------------------------------------===//

/// Clone @p unitary into the body of a new `qc` modifier, replacing the qubits
/// of the original body @p qubits with the block arguments @p args.
static void cloneIntoBody(Operation* unitary, ValueRange qubits,
                          ValueRange args, RewriterBase& rewriter) {
  IRMapping mapping;
  mapping.map(qubits, args);
  rewriter.clone(*unitary, mapping);
}

/// Unroll a `qc.ctrl` modifier with more than one body unitary,
/// or fail if it cannot be unrolled.
static LogicalResult unrollModifier(qc::CtrlOp op, RewriterBase& rewriter) {
  if (op.getNumBodyUnitaries() < 2) {
    return success();
  }
  auto* body = op.getBody();
  if (failed(hoistClassicalOps<qc::UnitaryOpInterface>(*body, op, rewriter))) {
    return failure();
  }

  rewriter.setInsertionPoint(op);
  for (auto unitary : body->getOps<qc::UnitaryOpInterface>()) {
    qc::CtrlOp::create(rewriter, op.getLoc(), op.getControls(), op.getTargets(),
                       [&](ValueRange args) {
                         cloneIntoBody(unitary, body->getArguments(), args,
                                       rewriter);
                       });
  }
  rewriter.eraseOp(op);
  return success();
}

/// Unroll a `qc.inv` modifier with more than one body unitary,
/// or fail if it cannot be unrolled.
static LogicalResult unrollModifier(qc::InvOp op, RewriterBase& rewriter) {
  if (op.getNumBodyUnitaries() < 2) {
    return success();
  }
  auto* body = op.getBody();
  if (failed(hoistClassicalOps<qc::UnitaryOpInterface>(*body, op, rewriter))) {
    return failure();
  }

  rewriter.setInsertionPoint(op);
  // (a b)^-1 = b^-1 a^-1, so the operations are inverted in reverse order.
  for (auto unitary : llvm::reverse(body->getOps<qc::UnitaryOpInterface>())) {
    qc::InvOp::create(
        rewriter, op.getLoc(), op.getQubits(), [&](ValueRange args) {
          cloneIntoBody(unitary, body->getArguments(), args, rewriter);
        });
  }
  rewriter.eraseOp(op);
  return success();
}

/// Check that the unitary operations in @p body act on disjoint qubits.
static bool hasDisjointBodyQubits(Block& body) {
  DenseSet<Value> qubits;
  for (auto unitary : body.getOps<qc::UnitaryOpInterface>()) {
    for (auto qubit : getQubitOperands<qc::QubitType>(unitary)) {
      if (!qubits.insert(qubit).second) {
        return false;
      }
    }
  }
  return true;
}

/// Unroll a `qc.pow` modifier with more than one body unitary,
/// or fail if it cannot be unrolled.
static LogicalResult unrollModifier(qc::PowOp op, RewriterBase& rewriter) {
  if (op.getNumBodyUnitaries() < 2) {
    return success();
  }
  auto* body = op.getBody();
  if (!hasIntegerExponent(op) || !hasDisjointBodyQubits(*body)) {
    return failure();
  }
  if (failed(hoistClassicalOps<qc::UnitaryOpInterface>(*body, op, rewriter))) {
    return failure();
  }

  rewriter.setInsertionPoint(op);
  for (auto unitary : body->getOps<qc::UnitaryOpInterface>()) {
    qc::PowOp::create(rewriter, op.getLoc(), op.getExponent(), op.getQubits(),
                      [&](ValueRange args) {
                        cloneIntoBody(unitary, body->getArguments(), args,
                                      rewriter);
                      });
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// QCO
//===----------------------------------------------------------------------===//

/// Clone @p unitary into the body of a new `qco` modifier, replacing the qubits
/// it acts on with the block arguments @p args of the wires @p wires, and
/// return the qubits that the new body yields.
static SmallVector<Value> cloneIntoBody(Operation* unitary,
                                        ArrayRef<size_t> wires, ValueRange args,
                                        RewriterBase& rewriter) {
  IRMapping mapping;
  for (auto [qubit, wire] :
       llvm::zip_equal(getQubitOperands<qco::QubitType>(unitary), wires)) {
    mapping.map(qubit, args[wire]);
  }

  SmallVector<Value> yielded(args);
  auto* clone = rewriter.clone(*unitary, mapping);
  for (auto [result, wire] : llvm::zip_equal(clone->getResults(), wires)) {
    yielded[wire] = result;
  }
  return yielded;
}

/// Unroll a `qco.ctrl` modifier with more than one body unitary,
/// or fail if it cannot be unrolled.
static LogicalResult unrollModifier(qco::CtrlOp op, RewriterBase& rewriter) {
  auto* body = op.getBody();
  if (op.getNumBodyUnitaries() < 2) {
    return success();
  }
  if (failed(hoistClassicalOps<qco::UnitaryOpInterface>(*body, op, rewriter))) {
    return failure();
  }

  // Maps the qubits of the body to the wires they belong to. The inputs of the
  // modifier enter the body at its block arguments.
  DenseMap<Value, size_t> wires;
  for (auto [index, arg] : llvm::enumerate(body->getArguments())) {
    wires.try_emplace(arg, index);
  }

  rewriter.setInsertionPoint(op);
  SmallVector<Value> controls(op.getControlsIn());
  SmallVector<Value> targets(op.getTargetsIn());
  for (auto unitary : body->getOps<qco::UnitaryOpInterface>()) {
    const auto operands = getQubitOperands<qco::QubitType>(unitary);
    const auto indices = llvm::map_to_vector(
        operands, [&](Value qubit) { return wires.at(qubit); });
    auto ctrlOp = qco::CtrlOp::create(
        rewriter, op.getLoc(), controls, targets,
        [&](ValueRange args) -> SmallVector<Value> {
          return cloneIntoBody(unitary, indices, args, rewriter);
        });
    controls.assign(ctrlOp.getControlsOut().begin(),
                    ctrlOp.getControlsOut().end());
    targets.assign(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    for (auto [result, index] :
         llvm::zip_equal(unitary->getResults(), indices)) {
      wires[result] = index;
    }
  }

  SmallVector<Value> results(controls);
  for (auto yielded : body->getTerminator()->getOperands()) {
    results.push_back(targets[wires.at(yielded)]);
  }
  rewriter.replaceOp(op, results);
  return success();
}

/// Unroll a `qco.inv` modifier with more than one body unitary,
/// or fail if it cannot be unrolled.
static LogicalResult unrollModifier(qco::InvOp op, RewriterBase& rewriter) {
  auto* body = op.getBody();
  if (op.getNumBodyUnitaries() < 2) {
    return success();
  }
  if (failed(hoistClassicalOps<qco::UnitaryOpInterface>(*body, op, rewriter))) {
    return failure();
  }

  // Maps the qubits of the body to the wires they belong to. (a b)^-1 =
  // b^-1 a^-1, so the operations are inverted in reverse order and the inputs
  // of the modifier enter the body at the qubits it yields.
  DenseMap<Value, size_t> wires;
  for (auto [index, yielded] :
       llvm::enumerate(body->getTerminator()->getOperands())) {
    wires.try_emplace(yielded, index);
  }

  rewriter.setInsertionPoint(op);
  SmallVector<Value> qubits(op.getQubitsIn());
  auto unitaries = llvm::to_vector(body->getOps<qco::UnitaryOpInterface>());
  for (auto unitary : llvm::reverse(unitaries)) {
    const auto indices = llvm::map_to_vector(
        unitary->getResults(), [&](Value qubit) { return wires.at(qubit); });
    auto invOp = qco::InvOp::create(rewriter, op.getLoc(), qubits,
                                    [&](ValueRange args) -> SmallVector<Value> {
                                      return cloneIntoBody(unitary, indices,
                                                           args, rewriter);
                                    });
    qubits.assign(invOp.getResults().begin(), invOp.getResults().end());
    for (auto [operand, index] :
         llvm::zip_equal(getQubitOperands<qco::QubitType>(unitary), indices)) {
      wires[operand] = index;
    }
  }

  rewriter.replaceOp(op,
                     llvm::map_to_vector(body->getArguments(), [&](Value arg) {
                       return qubits[wires.at(arg)];
                     }));
  return success();
}

/// Check that the unitary operations in @p body act on disjoint wires.
static bool hasDisjointBodyWires(Block& body) {
  DenseMap<Value, size_t> wires;
  for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
    wires.try_emplace(arg, index);
  }

  DenseSet<size_t> used;
  for (auto unitary : body.getOps<qco::UnitaryOpInterface>()) {
    const auto qubits = getQubitOperands<qco::QubitType>(unitary);
    for (auto [qubit, result] :
         llvm::zip_equal(qubits, unitary->getResults())) {
      const auto it = wires.find(qubit);
      if (it == wires.end()) {
        return false;
      }
      const auto wire = it->second;
      if (!used.insert(wire).second) {
        return false;
      }
      wires.try_emplace(result, wire);
    }
  }
  return true;
}

/// Unroll a `qco.pow` modifier with more than one body unitary,
/// or fail if it cannot be unrolled.
static LogicalResult unrollModifier(qco::PowOp op, RewriterBase& rewriter) {
  if (op.getNumBodyUnitaries() < 2) {
    return success();
  }
  auto* body = op.getBody();
  if (!hasIntegerExponent(op) || !hasDisjointBodyWires(*body)) {
    return failure();
  }
  if (failed(hoistClassicalOps<qco::UnitaryOpInterface>(*body, op, rewriter))) {
    return failure();
  }

  // Maps the qubits of the body to the wires they belong to. The inputs of the
  // modifier enter the body at its block arguments.
  DenseMap<Value, size_t> wires;
  for (auto [index, arg] : llvm::enumerate(body->getArguments())) {
    wires.try_emplace(arg, index);
  }

  rewriter.setInsertionPoint(op);
  SmallVector<Value> qubits(op.getQubitsIn());
  for (auto unitary : body->getOps<qco::UnitaryOpInterface>()) {
    const auto operands = getQubitOperands<qco::QubitType>(unitary);
    const auto indices = llvm::map_to_vector(
        operands, [&](Value qubit) { return wires.at(qubit); });
    auto powOp = qco::PowOp::create(
        rewriter, op.getLoc(), qubits, op.getExponent(),
        [&](ValueRange args) -> SmallVector<Value> {
          return cloneIntoBody(unitary, indices, args, rewriter);
        });
    qubits.assign(powOp.getResults().begin(), powOp.getResults().end());
    for (auto [result, index] :
         llvm::zip_equal(unitary->getResults(), indices)) {
      wires[result] = index;
    }
  }

  rewriter.replaceOp(op,
                     llvm::map_to_vector(body->getTerminator()->getOperands(),
                                         [&](Value yielded) {
                                           return qubits[wires.at(yielded)];
                                         }));
  return success();
}

namespace {

struct UnrollModifiers final : impl::UnrollModifiersBase<UnrollModifiers> {
protected:
  void runOnOperation() override {
    SmallVector<Operation*> modifiers;
    getOperation()->walk([&](Operation* op) {
      if (isa<qc::CtrlOp, qc::InvOp, qc::PowOp, qco::CtrlOp, qco::InvOp,
              qco::PowOp>(op)) {
        modifiers.push_back(op);
      }
    });

    // The walk visits nested modifiers before their parents, so unrolling the
    // collected modifiers in order reaches a fixpoint in a single sweep.
    IRRewriter rewriter(&getContext());
    for (auto* modifier : modifiers) {
      llvm::TypeSwitch<Operation*>(modifier)
          .Case<qc::CtrlOp, qc::InvOp, qc::PowOp, qco::CtrlOp, qco::InvOp,
                qco::PowOp>([&](auto op) {
            if (failed(unrollModifier(op, rewriter))) {
              LLVM_DEBUG(llvm::dbgs() << "Failed to unroll " << op->getName()
                                      << " at " << op.getLoc() << "\n");
            }
          });
    }
  }
};

} // namespace
} // namespace mlir::mqt
