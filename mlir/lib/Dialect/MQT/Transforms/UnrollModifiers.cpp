/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/MQT/Utils/Math.h"
#include "mlir/Dialect/MQT/Utils/Modifier.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

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
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

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
  return exponent && isIntegerExponent(*exponent);
}

//===----------------------------------------------------------------------===//
// QC
//===----------------------------------------------------------------------===//

/// Clone @p unitary into the body of a new `qc` modifier, replacing its qubits
/// with the block arguments @p args.
static void cloneIntoBody(qc::UnitaryOpInterface unitary, ValueRange args,
                          RewriterBase& rewriter) {
  IRMapping mapping;
  mapping.map(unitary.getQubits(), args);
  rewriter.clone(*unitary.getOperation(), mapping);
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
    const auto targets = llvm::map_to_vector(unitary.getQubits(), [&](Value q) {
      return getValueFromBlockArgument(q, op.getTargets());
    });
    qc::CtrlOp::create(
        rewriter, op.getLoc(), op.getControls(), targets,
        [&](ValueRange args) { cloneIntoBody(unitary, args, rewriter); });
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
    const auto qubits = llvm::map_to_vector(unitary.getQubits(), [&](Value q) {
      return getValueFromBlockArgument(q, op.getQubits());
    });
    qc::InvOp::create(rewriter, op.getLoc(), qubits, [&](ValueRange args) {
      cloneIntoBody(unitary, args, rewriter);
    });
  }
  rewriter.eraseOp(op);
  return success();
}

/// Check that the unitary operations in @p body act on disjoint qubits.
static bool hasDisjointBodyQubits(Block& body) {
  DenseSet<Value> qubits;
  for (auto unitary : body.getOps<qc::UnitaryOpInterface>()) {
    for (auto qubit : unitary.getQubits()) {
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
    const auto qubits = llvm::map_to_vector(unitary.getQubits(), [&](Value q) {
      return getValueFromBlockArgument(q, op.getQubits());
    });
    qc::PowOp::create(
        rewriter, op.getLoc(), op.getExponent(), qubits,
        [&](ValueRange args) { cloneIntoBody(unitary, args, rewriter); });
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// QCO
//===----------------------------------------------------------------------===//

/// Clone @p unitary into the body of a new `qco` modifier, replacing its input
/// qubits with the block arguments @p args, and return its output qubits.
static SmallVector<Value> cloneIntoBody(qco::UnitaryOpInterface unitary,
                                        ValueRange args,
                                        RewriterBase& rewriter) {
  IRMapping mapping;
  mapping.map(unitary.getInputQubits(), args);
  auto clone = cast<qco::UnitaryOpInterface>(
      rewriter.clone(*unitary.getOperation(), mapping));
  return llvm::map_to_vector(clone.getOutputQubits(),
                             [](OpResult result) -> Value { return result; });
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

  // Thread the body's linear qubit values through the new modifiers.
  IRMapping qubits;
  qubits.map(body->getArguments(), op.getTargetsIn());

  rewriter.setInsertionPoint(op);
  SmallVector<Value> controls(op.getControlsIn());
  for (auto unitary : body->getOps<qco::UnitaryOpInterface>()) {
    const auto targets = llvm::map_to_vector(
        unitary.getInputQubits(), [&](Value q) { return qubits.lookup(q); });
    auto ctrlOp =
        qco::CtrlOp::create(rewriter, op.getLoc(), controls, targets,
                            [&](ValueRange args) -> SmallVector<Value> {
                              return cloneIntoBody(unitary, args, rewriter);
                            });
    controls.assign(ctrlOp.getControlsOut().begin(),
                    ctrlOp.getControlsOut().end());
    qubits.map(unitary.getOutputQubits(), ctrlOp.getTargetsOut());
  }

  SmallVector<Value> results(controls);
  for (auto yielded : body->getTerminator()->getOperands()) {
    results.push_back(qubits.lookup(yielded));
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

  // Inverting the body reverses its data flow: the modifier inputs enter at the
  // yielded values and leave at the block arguments.
  IRMapping qubits;
  qubits.map(body->getTerminator()->getOperands(), op.getQubitsIn());

  rewriter.setInsertionPoint(op);
  auto unitaries = llvm::to_vector(body->getOps<qco::UnitaryOpInterface>());
  for (auto unitary : llvm::reverse(unitaries)) {
    const auto inputs = llvm::map_to_vector(
        unitary.getOutputQubits(), [&](Value q) { return qubits.lookup(q); });
    auto invOp =
        qco::InvOp::create(rewriter, op.getLoc(), inputs,
                           [&](ValueRange args) -> SmallVector<Value> {
                             return cloneIntoBody(unitary, args, rewriter);
                           });
    qubits.map(unitary.getInputQubits(), invOp.getResults());
  }

  rewriter.replaceOp(
      op, llvm::map_to_vector(body->getArguments(),
                              [&](Value arg) { return qubits.lookup(arg); }));
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
    for (auto [qubit, result] :
         llvm::zip_equal(unitary.getInputQubits(), unitary.getOutputQubits())) {
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

  IRMapping qubits;
  qubits.map(body->getArguments(), op.getQubitsIn());

  rewriter.setInsertionPoint(op);
  for (auto unitary : body->getOps<qco::UnitaryOpInterface>()) {
    const auto inputs = llvm::map_to_vector(
        unitary.getInputQubits(), [&](Value q) { return qubits.lookup(q); });
    auto powOp =
        qco::PowOp::create(rewriter, op.getLoc(), inputs, op.getExponent(),
                           [&](ValueRange args) -> SmallVector<Value> {
                             return cloneIntoBody(unitary, args, rewriter);
                           });
    qubits.map(unitary.getOutputQubits(), powOp.getResults());
  }

  rewriter.replaceOp(op,
                     llvm::map_to_vector(body->getTerminator()->getOperands(),
                                         [&](Value yielded) {
                                           return qubits.lookup(yielded);
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
