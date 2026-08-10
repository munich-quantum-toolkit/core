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
#include <llvm/ADT/DenseSet.h>
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

/// Clone @p unitary into the body of a new modifier, replacing its qubit
/// operands @p qubits with the block arguments @p args, and return its results.
static SmallVector<Value> cloneIntoBody(Operation* unitary, ValueRange qubits,
                                        ValueRange args,
                                        RewriterBase& rewriter) {
  IRMapping mapping;
  mapping.map(qubits, args);
  auto results = rewriter.clone(*unitary, mapping)->getResults();
  return {results.begin(), results.end()};
}

/// Check whether the exponent of @p op is a compile-time known integer.
template <typename PowOp> static bool hasIntegerExponent(PowOp op) {
  const auto exponent = op.getExponentValue();
  return exponent && utils::isIntegerExponent(*exponent);
}

//===----------------------------------------------------------------------===//
// QC
//===----------------------------------------------------------------------===//

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
    const auto qubits = getQubitOperands<qc::QubitType>(unitary);
    const auto targets = llvm::map_to_vector(qubits, [&](Value qubit) {
      return utils::getValueFromBlockArgument(qubit, op.getTargets());
    });
    qc::CtrlOp::create(rewriter, op.getLoc(), op.getControls(), targets,
                       [&](ValueRange args) {
                         cloneIntoBody(unitary, qubits, args, rewriter);
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
    const auto qubits = getQubitOperands<qc::QubitType>(unitary);
    const auto targets = llvm::map_to_vector(qubits, [&](Value qubit) {
      return utils::getValueFromBlockArgument(qubit, op.getQubits());
    });
    qc::InvOp::create(rewriter, op.getLoc(), targets, [&](ValueRange args) {
      cloneIntoBody(unitary, qubits, args, rewriter);
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
    const auto qubits = getQubitOperands<qc::QubitType>(unitary);
    const auto targets = llvm::map_to_vector(qubits, [&](Value qubit) {
      return utils::getValueFromBlockArgument(qubit, op.getQubits());
    });
    qc::PowOp::create(rewriter, op.getLoc(), op.getExponent(), targets,
                      [&](ValueRange args) {
                        cloneIntoBody(unitary, qubits, args, rewriter);
                      });
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// QCO
//===----------------------------------------------------------------------===//

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

  // Maps the qubits of the original body to the qubit values threaded through
  // the new modifiers. The inputs of the modifier enter the body at its block
  // arguments and leave it at the qubits it yields.
  IRMapping qubits;
  qubits.map(body->getArguments(), op.getTargetsIn());

  SmallVector<Value> controls(op.getControlsIn());

  rewriter.setInsertionPoint(op);
  for (auto unitary : body->getOps<qco::UnitaryOpInterface>()) {
    const auto operands = getQubitOperands<qco::QubitType>(unitary);
    const auto targets = llvm::map_to_vector(
        operands, [&](Value qubit) { return qubits.lookup(qubit); });
    auto ctrlOp = qco::CtrlOp::create(
        rewriter, op.getLoc(), controls, targets,
        [&](ValueRange args) -> SmallVector<Value> {
          return cloneIntoBody(unitary, operands, args, rewriter);
        });
    auto controlsOut = ctrlOp.getControlsOut();
    controls.assign(controlsOut.begin(), controlsOut.end());
    qubits.map(unitary->getResults(), ctrlOp.getTargetsOut());
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

  // Maps the qubits of the original body to the qubit values threaded through
  // the new modifiers. Inverting the body reverses its direction, so the inputs
  // of the modifier enter the body at the qubits it yields and leave it at its
  // block arguments.
  IRMapping qubits;
  qubits.map(body->getTerminator()->getOperands(), op.getQubitsIn());

  rewriter.setInsertionPoint(op);
  // (a b)^-1 = b^-1 a^-1, so the operations are inverted in reverse order.
  for (auto unitary : llvm::reverse(body->getOps<qco::UnitaryOpInterface>())) {
    const auto operands = getQubitOperands<qco::QubitType>(unitary);
    const auto targets =
        llvm::map_to_vector(unitary->getResults(),
                            [&](Value qubit) { return qubits.lookup(qubit); });
    auto invOp = qco::InvOp::create(rewriter, op.getLoc(), targets,
                                    [&](ValueRange args) -> SmallVector<Value> {
                                      return cloneIntoBody(unitary, operands,
                                                           args, rewriter);
                                    });
    qubits.map(operands, invOp.getResults());
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

  // Maps the qubits of the original body to the qubit values threaded through
  // the new modifiers.
  IRMapping qubits;
  qubits.map(body->getArguments(), op.getQubitsIn());

  rewriter.setInsertionPoint(op);
  for (auto unitary : body->getOps<qco::UnitaryOpInterface>()) {
    const auto operands = getQubitOperands<qco::QubitType>(unitary);
    const auto targets = llvm::map_to_vector(
        operands, [&](Value qubit) { return qubits.lookup(qubit); });
    auto powOp = qco::PowOp::create(
        rewriter, op.getLoc(), targets, op.getExponent(),
        [&](ValueRange args) -> SmallVector<Value> {
          return cloneIntoBody(unitary, operands, args, rewriter);
        });
    qubits.map(unitary->getResults(), powOp.getResults());
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
