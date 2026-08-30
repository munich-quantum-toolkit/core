/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagationAnalysis.hpp"

#include "UnionTable.hpp"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {

/// @brief Materializes a value range into an owned vector.
template <typename Range> static SmallVector<Value> toVec(const Range& range) {
  return {range.begin(), range.end()};
}

/// @brief Whether v is a qubit argument of the entry-point function's entry
/// block (so its initial state is |0>, per the pass contract). Arguments of any
/// other function have unknown provenance.
static bool isEntryPointQubitArgument(Value v) {
  const auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || !isa<QubitType>(arg.getType())) {
    return false;
  }
  Block* const block = arg.getOwner();
  return block->isEntryBlock() && block->getParentOp() != nullptr &&
         mqt::isEntryPoint(block->getParentOp());
}

/// @brief Ensures every qubit operand of op is tracked before use: an
/// entry-point argument starts in |0>, anything else of unknown provenance
/// collapses to top.
static void ensureSeeded(UnionTable& table, Operation* const op) {
  for (Value operand : op->getOperands()) {
    if (!isa<QubitType>(operand.getType()) || table.isTracked(operand)) {
      continue;
    }
    table.seedQubit(operand);
    if (!isEntryPointQubitArgument(operand)) {
      table.markQubitsTop(operand);
    }
  }
}

//===----------------------------------------------------------------------===//
// UnionTableLattice
//===----------------------------------------------------------------------===//

ChangeResult UnionTableLattice::join(const AbstractDenseLattice& other) {
  const auto& rhs = llvm::cast<UnionTableLattice>(other);
  if (!rhs.initialized) {
    return ChangeResult::NoChange;
  }
  return joinUnionTable(rhs.table);
}

ChangeResult UnionTableLattice::setUnionTable(UnionTable next) {
  if (initialized && next == table) {
    return ChangeResult::NoChange;
  }
  table = std::move(next);
  initialized = true;
  return ChangeResult::Change;
}

ChangeResult UnionTableLattice::joinUnionTable(const UnionTable& rhs) {
  if (!initialized) {
    return setUnionTable(rhs);
  }
  UnionTable joined = table;
  joined.join(rhs);
  return setUnionTable(std::move(joined));
}

void UnionTableLattice::print(raw_ostream& os) const {
  if (!initialized) {
    os << "<uninitialized>";
    return;
  }
  table.print(os);
}

//===----------------------------------------------------------------------===//
// ConstantPropagationAnalysis
//===----------------------------------------------------------------------===//

ConstantPropagationAnalysis::ConstantPropagationAnalysis(
    DataFlowSolver& solver, const size_t maxNonzeroAmplitudes,
    const size_t maxHybridStates)
    : DenseForwardDataFlowAnalysis(solver),
      maxNonzeroAmplitudes(maxNonzeroAmplitudes),
      maxHybridStates(maxHybridStates) {}

LogicalResult ConstantPropagationAnalysis::initialize(Operation* top) {
  // Does not reason across a call boundary: any call anywhere in the module
  // makes every program point top. Uncalled helper functions are fine - the
  // analysis just treats their (non-entry-point) qubit arguments as unknown.
  bool hasCall = false;
  top->walk([&](Operation* op) { hasCall |= isa<CallOpInterface>(op); });
  bailToTop = hasCall;
  return DenseForwardDataFlowAnalysis::initialize(top);
}

UnionTable ConstantPropagationAnalysis::freshTable() const {
  UnionTable table(maxNonzeroAmplitudes, maxHybridStates);
  if (bailToTop) {
    table.markAllTop();
  }
  return table;
}

void ConstantPropagationAnalysis::setToEntryState(UnionTableLattice* lattice) {
  // Entry qubits are seeded lazily on first use (see ensureSeeded); the entry
  // state is just an empty, budgeted table.
  propagateIfChanged(lattice, lattice->setUnionTable(freshTable()));
}

LogicalResult ConstantPropagationAnalysis::visitOperation(
    Operation* op, const UnionTableLattice& before, UnionTableLattice* after) {
  if (bailToTop) {
    propagateIfChanged(after, after->setUnionTable(freshTable()));
    return success();
  }

  // Region-branch ops (qco.if, qco.index_switch) are handled through
  // visitRegionBranchControlFlowTransfer; nothing to do for the op itself.
  if (isa<RegionBranchOpInterface>(op)) {
    return success();
  }

  // Bodies of qco.ctrl / qco.inv / qco.pow are interpreted by the enclosing
  // modifier's handler, so their nested ops just pass the state through.
  if (Operation* const parent = op->getParentOp();
      parent != nullptr && isa<CtrlOp, InvOp, PowOp>(parent)) {
    propagateIfChanged(after, after->setUnionTable(before.getUnionTable()));
    return success();
  }

  UnionTable table = before.getUnionTable();
  if (failed(applyOperation(table, op, /*quantumControls=*/{}))) {
    return op->emitError()
           << "constant propagation cannot interpret '" << op->getName()
           << "' (unsupported operation, or a propagation bug left the state "
              "inconsistent)";
  }
  propagateIfChanged(after, after->setUnionTable(std::move(table)));
  return success();
}

LogicalResult ConstantPropagationAnalysis::applyOperation(
    UnionTable& table, Operation* op, const ArrayRef<Value> quantumControls) {
  ensureSeeded(table, op);

  return TypeSwitch<Operation*, LogicalResult>(op)
      .Case<AllocOp>([&](AllocOp alloc) {
        table.seedQubit(alloc.getResult());
        return success();
      })
      .Case<StaticOp>([&](StaticOp stat) {
        table.seedQubit(stat.getQubit());
        return success();
      })
      .Case<qtensor::ExtractOp>([&](qtensor::ExtractOp extract) {
        table.seedQubit(extract.getResult());
        return success();
      })
      .Case<SinkOp, qtensor::InsertOp, qtensor::AllocOp,
            qtensor::FromElementsOp, qtensor::DeallocOp, YieldOp,
            func::ReturnOp, func::FuncOp, ModuleOp>(
          [](Operation*) { return success(); })
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constant) -> LogicalResult {
            const Attribute value = constant.getValue();
            if (!isa<IntegerAttr, FloatAttr>(value)) {
              return failure();
            }
            table.seedClassical(constant.getResult(), value);
            return success();
          })
      .Case<MeasureOp>([&](MeasureOp measure) {
        return table.measureQubit(measure.getQubitIn(), measure.getQubitOut(),
                                  measure.getResult());
      })
      .Case<ResetOp>([&](ResetOp reset) {
        return table.resetQubit(reset.getQubitIn(), reset.getQubitOut());
      })
      .Case<GPhaseOp>([&](GPhaseOp gphase) {
        return table.addGlobalPhase(gphase.getTheta(), quantumControls,
                                    quantumControls);
      })
      .Case<CtrlOp>([&](const CtrlOp ctrl) {
        return applyCtrl(table, ctrl, quantumControls);
      })
      .Case<IfOp, IndexSwitchOp>([](Operation*) {
        // Region-branch ops are routed by the framework
        // (visitRegionBranchControlFlowTransfer); reaching one here means it is
        // nested in a modifier body, which the QCO verifier forbids.
        return failure();
      })
      .Case<UnitaryOpInterface>([&](UnitaryOpInterface gate) {
        // Every remaining unitary: base gates, and qco.inv / qco.pow bodies
        // (which apply here when they expose a compile-time matrix, top out
        // otherwise).
        return applyUnitary(table, gate, quantumControls);
      })
      .Default([&](Operation* other) -> LogicalResult {
        // Not a QCO operation. Anything clear of qubits is a classical op to
        // fold; an unrecognized qubit-touching op is unsupported.
        const auto isQubit = [](const Type t) { return isa<QubitType>(t); };
        if (llvm::any_of(other->getOperandTypes(), isQubit) ||
            llvm::any_of(other->getResultTypes(), isQubit)) {
          return failure();
        }
        table.propagateClassical(other);
        return success();
      });
}

LogicalResult ConstantPropagationAnalysis::applyUnitary(
    UnionTable& table, UnitaryOpInterface gate,
    const ArrayRef<Value> quantumControls) {
  const auto targetsIn = toVec(gate.getInputTargets());
  const auto targetsOut = toVec(gate.getOutputTargets());

  Matrix2x2 matrix2;
  if (gate.getNumTargets() == 1 && gate.getUnitaryMatrix2x2(matrix2)) {
    return table.applyMatrix1Q(targetsIn[0], targetsOut[0], matrix2,
                               quantumControls, quantumControls);
  }
  Matrix4x4 matrix4;
  if (gate.getNumTargets() == 2 && gate.getUnitaryMatrix4x4(matrix4)) {
    return table.applyMatrix2Q(targetsIn[0], targetsIn[1], targetsOut[0],
                               targetsOut[1], matrix4, quantumControls,
                               quantumControls);
  }
  // Parametric-without-constant, >2-qubit, dynamic-matrix, or an unmodelled
  // qco.inv / qco.pow body: the targets become top.
  table.markQubitsTop(targetsIn);
  table.forwardValues(targetsIn, targetsOut);
  return success();
}

LogicalResult
ConstantPropagationAnalysis::applyCtrl(UnionTable& table, CtrlOp ctrl,
                                       const ArrayRef<Value> quantumControls) {
  Block& body = ctrl.getRegion().front();

  table.forwardValues(toVec(ctrl.getInputTargets()),
                      toVec(body.getArguments()));

  SmallVector<Value> innerControls(quantumControls);
  llvm::append_range(innerControls, ctrl.getInputControls());

  for (Operation& nested : body.without_terminator()) {
    if (failed(applyOperation(table, &nested, innerControls))) {
      return failure();
    }
  }

  auto yield = cast<YieldOp>(body.getTerminator());
  table.forwardValues(toVec(yield.getOperands()),
                      toVec(ctrl.getOutputTargets()));
  table.forwardValues(toVec(ctrl.getInputControls()),
                      toVec(ctrl.getOutputControls()));
  return success();
}

void ConstantPropagationAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, const std::optional<unsigned> regionFrom,
    const std::optional<unsigned> regionTo, const UnionTableLattice& before,
    UnionTableLattice* after) {
  // nullopt = the parent op; a value = the index of one of `branch`'s regions.
  auto ifOp = dyn_cast<IfOp>(branch.getOperation());
  if (bailToTop || !ifOp || !before.isInitialized()) {
    // bailToTop, qco.index_switch, or any not-yet-modelled region-branch op:
    // a plain join of `before` into `after`, without the operand->block-arg
    // (enter) / yield->result (leave) renaming that Case A / Case C do below.
    // The renamed value is thus untracked downstream, so the next
    // applyOperation that consumes it runs ensureSeeded, which - the value not
    // being an entry-point argument - marks its qubits top. (Under bailToTop
    // `before` is already all-top; an uninitialized `before` makes the join a
    // no-op.)
    DenseForwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, before, after);
    return;
  }

  UnionTable table = before.getUnionTable();

  if (!regionFrom.has_value() && regionTo.has_value()) {
    // Entering a branch: the op's linear operands become the region's block
    // arguments. Both then and else regions get the same incoming state; a
    // constant condition is exploited at rewrite time.
    Block& body = ifOp->getRegion(*regionTo).front();
    table.forwardValues(toVec(ifOp.getQubits()), toVec(body.getArguments()));
    propagateIfChanged(after, after->setUnionTable(std::move(table)));
    return;
  }

  if (!regionFrom.has_value() || regionTo.has_value()) {
    // Happens if loops produce region -> region calls (not supported) or a
    // region is empty (e.g. an else branch does not exist)
    return;
  }

  // Leaving a branch: the region's yield becomes the op's results. Classical
  // results precede the linear ones. Each region contributes one exit edge; the
  // lattice accumulates them via join.
  auto yield =
      cast<YieldOp>(ifOp->getRegion(*regionFrom).front().getTerminator());
  const size_t numClassical = ifOp.getClassicalResults().size();
  table.forwardValues(toVec(yield.getOperands().take_front(numClassical)),
                      toVec(ifOp.getClassicalResults()));
  table.forwardValues(toVec(yield.getOperands().drop_front(numClassical)),
                      toVec(ifOp.getLinearResults()));
  propagateIfChanged(after, after->joinUnionTable(table));
}

} // namespace mlir::qco
