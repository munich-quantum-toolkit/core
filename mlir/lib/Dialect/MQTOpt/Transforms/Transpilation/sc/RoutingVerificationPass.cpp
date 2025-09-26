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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingStack.h"

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
#include <vector>

#define DEBUG_TYPE "routing-verification"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGVERIFICATIONSCPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

/**
 * @brief Attribute to forward function-level attributes to LLVM IR.
 */
constexpr llvm::StringLiteral PASSTHROUGH_ATTR{"passthrough"};

/**
 * @brief 'For' pushes once onto the stack, hence the parent is at depth one.
 */
constexpr std::size_t FOR_PARENT_DEPTH = 1UL;

/**
 * @brief 'If' pushes twice onto the stack, hence the parent is at depth two.
 */
constexpr std::size_t IF_PARENT_DEPTH = 2UL;

/**
 * @brief The datatype for qubit indices. For now, 64bit.
 */
using QubitIndex = std::size_t;

/**
 * @brief Maps SSA values to hardware indices.
 */
using QubitIndexMap = llvm::DenseMap<Value, QubitIndex>;

/**
 * @brief Replace an old SSA value with a new one.
 */
void remapQubitValue(QubitIndexMap& state, const Value in, const Value out) {
  assert(in != out && "'in' must not equal 'out'");
  const auto it = state.find(in);
  assert(it != state.end() && "map must contain 'in'");
  state[out] = it->second;
  state.erase(in);
}

/**
 * @brief Return true if the function contains "entry_point" in the passthrough
 * attribute.
 */
bool isEntryPoint(func::FuncOp op) {
  const auto passthroughAttr = op->getAttrOfType<ArrayAttr>(PASSTHROUGH_ATTR);
  if (!passthroughAttr) {
    return false;
  }

  return llvm::any_of(passthroughAttr, [](const Attribute attr) {
    return isa<StringAttr>(attr) && cast<StringAttr>(attr) == ENTRY_POINT_ATTR;
  });
}

/**
 * @brief Resets the state and pushes empty map. Skips non entry-point
 * functions.
 */
WalkResult handleFunc(func::FuncOp op, RoutingStack<QubitIndexMap>& stack) {
  if (!isEntryPoint(op)) {
    return WalkResult::skip();
  }
  stack.clear();
  stack.emplace();
  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region: Pop the top of the stack.
 */
WalkResult handleReturn(RoutingStack<QubitIndexMap>& stack) {
  stack.pop();
  return WalkResult::advance();
}

/**
 * @brief Prepares state for nested regions: Pushes a copy of the state on
 * the stack. Forwards all out-of-loop and in-loop SSA values for their
 * respective map in the stack.
 */
WalkResult handleFor(scf::ForOp op, RoutingStack<QubitIndexMap>& stack) {
  stack.duplicateTop();

  for (const auto [arg, res, iter] :
       llvm::zip(op.getInitArgs(), op.getResults(), op.getRegionIterArgs())) {
    if (isa<QubitType>(arg.getType())) {
      remapQubitValue(stack.getItemAtDepth(FOR_PARENT_DEPTH), arg, res);
      remapQubitValue(stack.top(), arg, iter);
    }
  }

  return WalkResult::advance();
}

/**
 * @brief Prepares state for nested regions: Pushes two copies of the state on
 * the stack. Forwards the results in the parent state.
 */
WalkResult handleIf(scf::IfOp op, RoutingStack<QubitIndexMap>& stack) {
  /// Collect hardware qubits.
  SmallVector<Value> qubits(op->getNumResults());
  for (const auto [q, i] : stack.top()) {
    qubits[i] = q;
  }

  /// Prepare stack.
  stack.duplicateTop(); // Else
  stack.duplicateTop(); // If

  /// Forward results for all hardware qubits.
  QubitIndexMap& stateBeforeIf = stack.getItemAtDepth(IF_PARENT_DEPTH);
  for (std::size_t i = 0; i < op.getNumResults(); ++i) {
    const Value in = qubits[i];
    const Value out = op->getResult(i);
    remapQubitValue(stateBeforeIf, in, out);
  }

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a nested region: Pop the top of the stack.
 */
WalkResult handleYield(scf::YieldOp op, RoutingStack<QubitIndexMap>& stack) {
  if (isa<scf::ForOp>(op->getParentOp()) || isa<scf::IfOp>(op->getParentOp())) {
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
WalkResult handleQubit(QubitOp op, RoutingStack<QubitIndexMap>& stack,
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
WalkResult handleUnitary(UnitaryInterface op,
                         RoutingStack<QubitIndexMap>& stack,
                         const Architecture& arch) {
  const std::vector<Value> inQubits = op.getAllInQubits();
  const std::vector<Value> outQubits = op.getAllOutQubits();
  const std::size_t nacts = inQubits.size();

  if (nacts == 0) {
    return WalkResult::advance();
  }

  if (nacts > 2) {
    return op->emitOpError() << "acts on more than two qubits";
  }

  const Value in0 = inQubits[0];
  const Value out0 = outQubits[0];

  QubitIndexMap& state = stack.top();

  if (nacts == 1) {
    remapQubitValue(state, in0, out0);
    return WalkResult::advance();
  }

  const Value in1 = inQubits[1];
  const Value out1 = outQubits[1];

  if (!arch.areAdjacent(state.at(in0), state.at(in1))) {
    return op->emitOpError() << "(" << state[in0] << "," << state[in1] << ")"
                             << " is not executable on target architecture '"
                             << arch.name() << "'";
  }

  remapQubitValue(state, in0, out0);
  remapQubitValue(state, in1, out1);

  return WalkResult::advance();
}

/**
 * @brief Forwards the SSA values.
 */
WalkResult handleReset(ResetOp op, RoutingStack<QubitIndexMap>& stack) {
  remapQubitValue(stack.top(), op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Forwards the SSA values.
 */
WalkResult handleMeasure(MeasureOp op, RoutingStack<QubitIndexMap>& stack) {
  remapQubitValue(stack.top(), op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief This pass verifies that the constraints of a target architecture are
 * met.
 */
struct RoutingVerificationPassSC final
    : impl::RoutingVerificationSCPassBase<RoutingVerificationPassSC> {
  void runOnOperation() override {
    RoutingStack<QubitIndexMap> stack;

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
              .Case<scf::IfOp>(
                  [&](scf::IfOp op) { return handleIf(op, stack); })
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
