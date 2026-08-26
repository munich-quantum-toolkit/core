/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagation/ConstantPropagationLattice.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

// Adjust these includes to your actual generated QCO interface/type headers.
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::qco {

static unsigned int maxTrackedAmplitudes = 8;
static unsigned int maxTrackedHybridStates = 4;

static bool isQubitType(const Type ty) { return isa<QubitType>(ty); }

static bool isClassicalType(const Type ty) { return ty.isIntOrIndexOrFloat(); }

class HybridStateLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  explicit HybridStateLattice(const Value anchor)
      : AbstractSparseLattice(anchor),
        value(HybridStateSet(maxTrackedAmplitudes, maxTrackedHybridStates)) {}

  explicit HybridStateLattice(const Value anchor, const HybridStateSet& state)
      : AbstractSparseLattice(anchor), value(state) {}

  const HybridStateSet& getValue() const { return value; }

  ChangeResult join(const AbstractSparseLattice& rhs) override {
    const auto rhsHS = llvm::cast<HybridStateLattice>(rhs);
    const HybridStateSet old = value;
    value.join(rhsHS.getValue());
    return old == value ? ChangeResult::NoChange : ChangeResult::Change;
  }

  ChangeResult meet(const AbstractSparseLattice& rhs) override {
    return join(rhs);
  }

  void print(raw_ostream& os) const override;

private:
  HybridStateSet value;
};

class HybridConstantPropagationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<HybridStateLattice> {
public:
  explicit HybridConstantPropagationAnalysis(DataFlowSolver& solver)
      : SparseForwardDataFlowAnalysis(solver) {}

  void setToEntryState(dataflow::AbstractSparseLattice* lattice) override {
    const auto value = lattice->getAnchor();
    auto* hybrid = llvm::cast<HybridStateLattice>(lattice);
    // TODO: Propagate the values to the HybridState
    // auto newLattice = HybridStateLattice();
    // propagateIfChanged(hybrid, hybrid->join(newLattice));
  }

  LogicalResult
  visitOperation(Operation* op,
                 const ArrayRef<const HybridStateLattice*> operands,
                 ArrayRef<HybridStateLattice*> results) override {
    HybridStateSet input = gatherInputState(operands);

    if (input.areStatesTop()) {
      // TODO: Forward Qubits to results
      // setAllResults(results, HybridStateSet::top());
      return success();
    }

    if (const auto measureOp = dyn_cast<MeasureOp>(op)) {
      visitMeasureOp(measureOp, input, results);
      return success();
    }

    if (const auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
      return visitUnitaryOp(unitary, input, results);
    }

    if (const auto ctrlOp = dyn_cast<CtrlOp>(op)) {
      visitCtrlOp(ctrlOp, input, results);
      return success();
    }

    if (llvm::all_of(op->getResultTypes(), isClassicalType)) {
      return visitClassicalOp(op, input, results);
    }

    visitFallback(op, input, results);
    return success();
  }

private:
  static HybridStateLattice* asHybrid(dataflow::AbstractSparseLattice* l) {
    return llvm::cast<HybridStateLattice>(l);
  }

  static const HybridStateLattice*
  asHybrid(const dataflow::AbstractSparseLattice* l) {
    return llvm::cast<HybridStateLattice>(l);
  }

  static HybridStateSet
  gatherInputState(const ArrayRef<const HybridStateLattice*> operands) {
    if (operands.size() == 1) {
      return operands[0]->getValue();
    }

    auto result = operands[0]->getValue().mergeStates(operands[1]->getValue());
    for (unsigned int i = 2; i < operands.size(); ++i) {
      result = result.mergeStates(operands[i]->getValue());
    }
    return result;
  }

  LogicalResult visitClassicalOp(Operation* op, HybridStateSet& input,
                                 ArrayRef<HybridStateLattice*> results) {
    if (input.applyClassicalOperation(op).failed()) {
      return failure();
    }
    for (auto [resLattice, resValue] : llvm::zip(results, op->getResults())) {
      const auto newLattice = HybridStateLattice(resValue, input);
      propagateIfChanged(resLattice, resLattice->join(newLattice));
    }
    return success();
  }

  LogicalResult visitUnitaryOp(UnitaryOpInterface unitary,
                               HybridStateSet& input,
                               ArrayRef<HybridStateLattice*> results) {
    if (input.applyUnitaryOperation(&unitary).failed()) {
      return failure();
    }
    for (auto [resLattice, resValue] :
         llvm::zip(results, unitary->getResults())) {
      const auto newLattice = HybridStateLattice(resValue, input);
      propagateIfChanged(resLattice, resLattice->join(newLattice));
    }
    return success();
  }

  void visitMeasureOp(MeasureOp op, const HybridStateSet& input,
                      ArrayRef<HybridStateLattice*> results) {
    // HybridStateSet output;
    // output.states.clear();
    //
    // Value inQubit = op.getOperand();
    // Value outQubit = op.getResult(0);
    // Value outClassical = op.getResult(1);
    //
    // for (const HybridState& state : input.states) {
    //   auto successors =
    //       state.quantumState.measure(inQubit, outQubit, op.getContext());
    //   if (successors.empty()) {
    //     HybridState next = state;
    //     next.quantumState.markTop(inQubit);
    //     output.addState(std::move(next));
    //     continue;
    //   }
    //
    //   const QuantumComponent* component =
    //       state.quantumState.getComponent(inQubit);
    //   double prob0 = 0.0;
    //   double prob1 = 0.0;
    //   if (component && !component->isTop) {
    //     auto idx = component->indexOf(inQubit);
    //     if (idx) {
    //       for (const auto& it : component->amplitudes) {
    //         double p = std::norm(it.second);
    //         if (((it.first >> *idx) & 1ULL) == 0ULL)
    //           prob0 += p;
    //         else
    //           prob1 += p;
    //       }
    //     }
    //   }
    //
    //   for (auto& succ : successors) {
    //     HybridState next = state;
    //     next.quantumState = std::move(succ.first);
    //     next.setClassical(outClassical, succ.second);
    //     if (isZeroAttribute(succ.second))
    //       next.probability *= prob0;
    //     else if (isTrueAttribute(succ.second))
    //       next.probability *= prob1;
    //     output.addState(std::move(next));
    //   }
    // }
    //
    // output.enforceMaxStates(maxTrackedStates);
    // setAllResults(results, output);
  }

  void visitCtrlOp(CtrlOp op, const HybridStateSet& input,
                   ArrayRef<HybridStateLattice*> results) {
    // Forward target inputs conservatively.
    // HybridStateSet output;
    // output.states = input.states;
    // output.isTop = input.isTop;
    //
    // unsigned numResults = op->getNumResults();
    // unsigned numOperands = op->getNumOperands();
    // unsigned numControls = numOperands - numResults;
    // (void)numControls;
    //
    // for (HybridState& state : output.states) {
    //   for (unsigned i = 0; i < numResults; ++i) {
    //     Value in = op->getOperand(numOperands - numResults + i);
    //     Value out = op->getResult(i);
    //     state.quantumState.forwardQubit(in, out);
    //   }
    // }
    //
    // output.enforceMaxStates(maxTrackedStates);
    // setAllResults(results, output);
  }

  void visitFallback(Operation* op, const HybridStateSet& input,
                     ArrayRef<HybridStateLattice*> results) {
    for (auto [resLattice, resValue] :
         llvm::zip(results, op->getResults())) {
      const auto newLattice = HybridStateLattice(resValue, input);
      propagateIfChanged(resLattice, resLattice->join(newLattice));
    }
  }
};

struct RemoveAlwaysZeroCtrlPattern : public OpRewritePattern<qco::CtrlOp> {
  RemoveAlwaysZeroCtrlPattern(MLIRContext* ctx, DataFlowSolver& solver)
      : OpRewritePattern<qco::CtrlOp>(ctx), solver(solver) {}

  LogicalResult matchAndRewrite(qco::CtrlOp op,
                                PatternRewriter& rewriter) const override {
    for (Value ctrl : op.getConditions()) {
      auto* state = solver.lookupState<HybridStateLattice>(ctrl);
      if (!state)
        return failure();
      if (!state->getValue().isAlwaysFalse(ctrl))
        continue;

      unsigned numResults = op->getNumResults();
      unsigned numOperands = op->getNumOperands();
      if (numOperands < numResults)
        return failure();

      SmallVector<Value> replacements;
      for (unsigned i = 0; i < numResults; ++i)
        replacements.push_back(op->getOperand(numOperands - numResults + i));

      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }

private:
  DataFlowSolver& solver;
};

struct ConstantPropagationPass
    : public mlir::mqt::impl::ConstantPropagationPassBase<
          ConstantPropagationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<HybridConstantPropagationAnalysis>(maxTrackedAmplitudes,
                                                   maxTrackedStates);

    if (failed(solver.initializeAndRun(module))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveAlwaysZeroCtrlPattern>(&getContext(), solver);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::qco

std::unique_ptr<Pass> mlir::mqt::createConstantPropagationPass() {
  return std::make_unique<ConstantPropagationPass>();
}

void mlir::mqt::registerConstantPropagationPass() {
  PassRegistration<ConstantPropagationPass>();
}