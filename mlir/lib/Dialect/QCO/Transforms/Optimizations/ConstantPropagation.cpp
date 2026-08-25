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

static bool isQubitType(const Type ty) { return isa<mlir::qco::QubitType>(ty); }

static bool isClassicalType(const Type ty) { return ty.isIntOrIndexOrFloat(); }

static std::optional<Attribute> foldWithState(Operation* op,
                                              const HybridState& state) {
  SmallVector<Attribute> operandAttrs;
  operandAttrs.reserve(op->getNumOperands());
  for (const Value operand : op->getOperands()) {
    auto attr = state.getClassical(operand);
    if (!attr) {
      return std::nullopt;
    }
    operandAttrs.push_back(*attr);
  }

  SmallVector<OpFoldResult> foldResults;
  if (succeeded(op->fold(operandAttrs, foldResults)) &&
      foldResults.size() == 1) {
    if (auto attr = llvm::dyn_cast<Attribute>(foldResults.front())) {
      return attr;
    }
  }
  return std::nullopt;
}

class HybridStateLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  explicit HybridStateLattice(const Value anchor)
      : AbstractSparseLattice(anchor),
        value(HybridStateSet(maxTrackedAmplitudes, maxTrackedHybridStates)) {}

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
                 const ArrayRef<HybridStateLattice*> results) override {
    const HybridStateSet input = gatherInputState(operands);

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
      visitUnitaryOp(op, unitary, input, results);
      return success();
    }

    if (const auto ctrlOp = dyn_cast<CtrlOp>(op)) {
      visitCtrlOp(ctrlOp, input, results);
      return success();
    }

    if (llvm::all_of(op->getResultTypes(), isClassicalType)) {
      visitClassicalOp(op, input, results);
      return success();
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

  void setAllResults(const ArrayRef<HybridStateLattice*> results,
                     const HybridStateSet& state) {
    for (auto* res : results) {
      auto* lat = asHybrid(res);
      propagateIfChanged(lat, lat->join(state));
    }
  }

  void visitClassicalOp(Operation* op, const HybridStateSet& input,
                        ArrayRef<HybridStateLattice*> results) {
    HybridStateSet output;
    output.states.clear();

    for (const HybridState& state : input.states) {
      HybridState next = state;
      auto attr = foldWithState(op, state);
      if (!attr) {
        output.addState(std::move(next));
        continue;
      }
      if (!op->getResults().empty())
        next.setClassical(op->getResult(0), *attr);
      output.addState(std::move(next));
    }

    output.enforceMaxStates(maxTrackedStates);
    setAllResults(results, output);
  }

  void visitUnitaryOp(Operation* op, qco::UnitaryOpInterface unitary,
                      const HybridStateSet& input,
                      ArrayRef<HybridStateLattice*> results) {
    HybridStateSet output;
    output.states.clear();

    SmallVector<Value> inputs(op->getOperands().begin(),
                              op->getOperands().end());
    SmallVector<Value> outputsV(op->getResults().begin(),
                                op->getResults().end());
    UnitaryMatrix matrix = unitary.getUnitaryMatrix();

    for (const HybridState& state : input.states) {
      HybridState next = state;
      if (failed(next.quantumState.applyUnitary(inputs, matrix, outputsV,
                                                maxTrackedAmplitudes))) {
        for (Value out : outputsV)
          next.quantumState.markTop(out);
      }
      output.addState(std::move(next));
    }

    output.enforceMaxStates(maxTrackedStates);
    setAllResults(results, output);
  }

  void visitMeasureOp(qco::MeasureOp op, const HybridStateSet& input,
                      ArrayRef<HybridStateLattice*> results) {
    HybridStateSet output;
    output.states.clear();

    Value inQubit = op.getOperand();
    Value outQubit = op.getResult(0);
    Value outClassical = op.getResult(1);

    for (const HybridState& state : input.states) {
      auto successors =
          state.quantumState.measure(inQubit, outQubit, op.getContext());
      if (successors.empty()) {
        HybridState next = state;
        next.quantumState.markTop(inQubit);
        output.addState(std::move(next));
        continue;
      }

      const QuantumComponent* component =
          state.quantumState.getComponent(inQubit);
      double prob0 = 0.0;
      double prob1 = 0.0;
      if (component && !component->isTop) {
        auto idx = component->indexOf(inQubit);
        if (idx) {
          for (const auto& it : component->amplitudes) {
            double p = std::norm(it.second);
            if (((it.first >> *idx) & 1ULL) == 0ULL)
              prob0 += p;
            else
              prob1 += p;
          }
        }
      }

      for (auto& succ : successors) {
        HybridState next = state;
        next.quantumState = std::move(succ.first);
        next.setClassical(outClassical, succ.second);
        if (isZeroAttribute(succ.second))
          next.probability *= prob0;
        else if (isTrueAttribute(succ.second))
          next.probability *= prob1;
        output.addState(std::move(next));
      }
    }

    output.enforceMaxStates(maxTrackedStates);
    setAllResults(results, output);
  }

  void visitCtrlOp(qco::CtrlOp op, const HybridStateSet& input,
                   ArrayRef<HybridStateLattice*> results) {
    // Forward target inputs conservatively.
    HybridStateSet output;
    output.states = input.states;
    output.isTop = input.isTop;

    unsigned numResults = op->getNumResults();
    unsigned numOperands = op->getNumOperands();
    unsigned numControls = numOperands - numResults;
    (void)numControls;

    for (HybridState& state : output.states) {
      for (unsigned i = 0; i < numResults; ++i) {
        Value in = op->getOperand(numOperands - numResults + i);
        Value out = op->getResult(i);
        state.quantumState.forwardQubit(in, out);
      }
    }

    output.enforceMaxStates(maxTrackedStates);
    setAllResults(results, output);
  }

  void visitFallback(Operation* op, const HybridStateSet& input,
                     ArrayRef<HybridStateLattice*> results) {
    HybridStateSet output = input;
    for (HybridState& state : output.states) {
      for (Value res : op->getResults()) {
        if (isQubitType(res.getType()))
          state.quantumState.initializeQubit(res),
              state.quantumState.markTop(res);
      }
    }
    output.enforceMaxStates(maxTrackedStates);
    setAllResults(results, output);
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