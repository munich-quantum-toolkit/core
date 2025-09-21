/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#define DEBUG_TYPE "routing-verification"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGVERIFICATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

using namespace mlir;

/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

namespace {

/**
 * @brief The datatype for qubit indices. For now, 64bit.
 */
using QubitIndex = std::size_t;

/**
 * @brief Maps SSA values to hardware indices.
 */
using QubitIndexMap = llvm::DenseMap<Value, QubitIndex>;

/**
 * @brief Manages the stack of qubit index maps.
 *
 * Similarly to the routing pass we use a stack here to deal with
 * nested regions such as for-loops and if-statements.
 */
class VerificationStack {
public:
  /// @brief Put empty state on stack.
  void push() { stack_.emplace_back(); }
  /// @brief Put @p map on stack.
  void push(const QubitIndexMap& map) { stack_.push_back(map); }
  /// @brief Return top of stack.
  [[nodiscard]] QubitIndexMap& top() { return stack_.back(); }
  /// @brief Return the element below the top of stack.
  [[nodiscard]] QubitIndexMap& belowTop() { return stack_[stack_.size() - 2]; }
  /// @brief Push a copy of the top element onto the stack.
  void duplicateTop() { stack_.push_back(stack_.back()); }
  /// @brief Pop the top from the stack.
  void pop() { stack_.pop_back(); }
  /// @brief Returns the size of the stack.
  [[nodiscard]] std::size_t size() const { return stack_.size(); }
  /// @brief Reset the state.
  void reset() { stack_.clear(); }

private:
  llvm::SmallVector<QubitIndexMap> stack_;
};

/**
 * @brief Forward SSA values in map.
 */
void forwardOne(QubitIndexMap& map, const Value in, const Value out) {
  assert(in != out && "'in' must not equal 'out'");
  const auto it = map.find(in);
  assert(it != map.end() && "map must contain 'in'");
  map[out] = it->second;
  map.erase(in);
}

/**
 * @brief Handle `func::FuncOp`.
 *
 * Resets the state and pushes empty map. Skips non entry-point functions.
 */
WalkResult handleFunc(func::FuncOp op, VerificationStack& stack) {
  if (!op->hasAttr(ENTRY_POINT_ATTR)) {
    return WalkResult::skip();
  }
  stack.reset();
  stack.push();
  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region: Pop the top of the stack.
 */
WalkResult handleReturn(VerificationStack& stack) {
  stack.pop();
  return WalkResult::advance();
}

/**
 * @brief Handle `scf::ForOp`.
 *
 * Prepares state for nested regions: Pushes a copy of the top of the state on
 * the stack. Forwards all out-of-loop and in-loop SSA values for their
 * respective map in the stack.
 */
WalkResult handleFor(scf::ForOp op, VerificationStack& stack) {
  stack.duplicateTop();

  for (const auto [arg, res, iter] :
       llvm::zip(op.getInitArgs(), op.getResults(), op.getRegionIterArgs())) {
    if (isa<QubitType>(arg.getType())) {
      forwardOne(stack.belowTop(), arg, res);
      forwardOne(stack.top(), arg, iter);
    }
  }

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a nested region: Pop the top of the stack.
 */
WalkResult handleYield(scf::YieldOp op, VerificationStack& stack) {
  if (isa<scf::ForOp>(op->getParentOp())) {
    if (stack.size() < 2) {
      return op->emitOpError() << "expected at least two elements on stack.";
    }

    stack.pop();
  }
  return WalkResult::advance();
}

/**
 * @brief Adds hardware qubit to qubit index map.
 */
WalkResult handleQubit(QubitOp op, VerificationStack& stack,
                       const Architecture& arch) {
  if (stack.top().size() == arch.nqubits()) {
    return op->emitOpError()
           << "requires " << (arch.nqubits() + 1)
           << " qubits but target architecture '" << arch.name()
           << "' only supports " << arch.nqubits() << " qubits";
  }

  stack.top()[op.getQubit()] = op.getIndex();
  return WalkResult::advance();
}

/**
 * @brief Verifies if the unitary acts on either zero, one, or two qubits:
 * - Advances for zero qubit unitaries (Nothing to do)
 * - Forwards SSA values for one qubit.
 * - Verifies executability for two-qubit gates for the given architecture and
 *   forwards SSA values.
 */
WalkResult handleUnitary(UnitaryInterface op, VerificationStack& stack,
                         const Architecture& arch) {
  const std::size_t nacts = op.getAllInQubits().size();
  if (nacts == 0) {
    return WalkResult::advance();
  }

  if (nacts > 2) {
    return op->emitOpError() << "acts on more than two qubits";
  }

  const Value in0 = op.getAllInQubits()[0];
  const Value out0 = op.getAllOutQubits()[0];

  if (nacts == 1) {
    forwardOne(stack.top(), in0, out0);
    return WalkResult::advance();
  }

  const Value in1 = op.getAllInQubits()[1];
  const Value out1 = op.getAllOutQubits()[1];

  if (!arch.areAdjacent(stack.top().at(in0), stack.top().at(in1))) {
    return op->emitOpError()
           << "(" << stack.top()[in0] << "," << stack.top()[in1] << ")"
           << " is not executable on target architecture '" << arch.name()
           << "'";
  }

  forwardOne(stack.top(), in0, out0);
  forwardOne(stack.top(), in1, out1);

  return WalkResult::advance();
}

/**
 * @brief Forwards the SSA values.
 */
WalkResult handleReset(ResetOp op, VerificationStack& stack) {
  forwardOne(stack.top(), op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Forwards the SSA values.
 */
WalkResult handleMeasure(MeasureOp op, VerificationStack& stack) {
  forwardOne(stack.top(), op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief This pass verifies that the constraints of a target architecture are
 * met.
 */
struct RoutingVerificationPass final
    : impl::RoutingVerificationPassBase<RoutingVerificationPass> {
  void runOnOperation() override {
    VerificationStack stack;

    const auto arch = getArchitecture(ArchitectureName::MQTTest);
    const auto res =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
          return TypeSwitch<Operation*, WalkResult>(op)
              /// built-in Dialect
              .Case<ModuleOp>(
                  [&](ModuleOp /* op */) { return WalkResult::advance(); })
              /// func Dialect
              .Case<func::FuncOp>(
                  [&](func::FuncOp op) { return handleFunc(op, stack); })
              .Case<func::ReturnOp>(
                  [&](func::ReturnOp /* op */) { return handleReturn(stack); })
              /// scf Dialect
              .Case<scf::ForOp>(
                  [&](scf::ForOp op) { return handleFor(op, stack); })
              .Case<scf::YieldOp>(
                  [&](scf::YieldOp op) { return handleYield(op, stack); })
              /// mqtopt Dialect
              .Case<AllocQubitOp, DeallocQubitOp>([&](auto op) {
                return WalkResult(
                    op->emitOpError("not allowed for transpiled program"));
              })
              .Case<ResetOp>([&](ResetOp op) { return handleReset(op, stack); })
              .Case<MeasureOp>([&](MeasureOp measure) {
                return handleMeasure(measure, stack);
              })
              .Case<QubitOp>(
                  [&](QubitOp op) { return handleQubit(op, stack, *arch); })
              .Case<UnitaryInterface>([&](UnitaryInterface unitary) {
                return handleUnitary(unitary, stack, *arch);
              })
              /// Skip the rest.
              .Default([](auto) { return WalkResult::skip(); });
        });

    if (res.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mqt::ir::opt
