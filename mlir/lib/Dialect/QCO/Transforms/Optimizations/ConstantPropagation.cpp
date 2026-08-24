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
#include "mlir/Dialect/QCO/IR/QCOOpsTypes.h.inc"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::mqt::qco {

static bool isQubitType(Type ty) { return isa<QubitType>(ty); }

static bool isClassicalType(Type ty) { return ty.isIntOrIndexOrFloat(); }

static std::optional<Attribute> foldWithState(Operation* op,
                                              const HybridState& state) {
  SmallVector<Attribute> operandAttrs;
  operandAttrs.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
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
  explicit HybridStateLattice(Value anchor)
      : dataflow::AbstractSparseLattice(anchor) {}

  const HybridStateSet& getValue() const { return value; }

  ChangeResult join(const HybridStateSet& rhs) {
    HybridStateSet old = value;
    value.join(rhs);
    return old == value ? ChangeResult::NoChange : ChangeResult::Change;
  }

private:
  HybridStateSet value = HybridStateSet::singletonInitial();
};

class HybridConstantPropagationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<HybridStateLattice> {
public:
  explicit HybridConstantPropagationAnalysis(DataFlowSolver& solver,
                                             unsigned maxTrackedAmplitudes,
                                             unsigned maxTrackedStates)
      : dataflow::SparseForwardDataFlowAnalysis<HybridStateLattice>(solver),
        maxTrackedAmplitudes(maxTrackedAmplitudes),
        maxTrackedStates(maxTrackedStates) {}

  void setToEntryState(dataflow::AbstractSparseLattice* lattice) override {
    auto* hybrid = llvm::cast<HybridStateLattice>(lattice);
    propagateIfChanged(hybrid,
                       hybrid->join(HybridStateSet::singletonInitial()));
  }

  LogicalResult
  visitOperation(Operation* op,
                 ArrayRef<const dataflow::AbstractSparseLattice*> operands,
                 ArrayRef<dataflow::AbstractSparseLattice*> results) override {
    HybridStateSet input = gatherInputState(operands);

    if (input.isTop) {
      setAllResults(results, HybridStateSet::top());
      return success();
    }

    if (auto measureOp = dyn_cast<qco::MeasureOp>(op)) {
      visitMeasureOp(measureOp, input, results);
      return success();
    }

    if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(op)) {
      visitUnitaryOp(op, unitary, input, results);
      return success();
    }

    if (auto ctrlOp = dyn_cast<qco::CtrlOp>(op)) {
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
  unsigned maxTrackedAmplitudes;
  unsigned maxTrackedStates;

  static HybridStateLattice* asHybrid(dataflow::AbstractSparseLattice* l) {
    return llvm::cast<HybridStateLattice>(l);
  }

  static const HybridStateLattice*
  asHybrid(const dataflow::AbstractSparseLattice* l) {
    return llvm::cast<HybridStateLattice>(l);
  }

  HybridStateSet
  gatherInputState(ArrayRef<const dataflow::AbstractSparseLattice*> operands) {
    HybridStateSet input = HybridStateSet::singletonInitial();
    bool first = true;
    for (const auto* operand : operands) {
      const HybridStateSet& state = asHybrid(operand)->getValue();
      if (first) {
        input = state;
        first = false;
      } else {
        input.join(state);
      }
    }
    return input;
  }

  void setAllResults(ArrayRef<dataflow::AbstractSparseLattice*> results,
                     const HybridStateSet& state) {
    for (auto* res : results) {
      auto* lat = asHybrid(res);
      propagateIfChanged(lat, lat->join(state));
    }
  }

  void visitClassicalOp(Operation* op, const HybridStateSet& input,
                        ArrayRef<dataflow::AbstractSparseLattice*> results) {
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
                      ArrayRef<dataflow::AbstractSparseLattice*> results) {
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
                      ArrayRef<dataflow::AbstractSparseLattice*> results) {
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
        else if (isOneAttribute(succ.second))
          next.probability *= prob1;
        output.addState(std::move(next));
      }
    }

    output.enforceMaxStates(maxTrackedStates);
    setAllResults(results, output);
  }

  void visitCtrlOp(qco::CtrlOp op, const HybridStateSet& input,
                   ArrayRef<dataflow::AbstractSparseLattice*> results) {
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
                     ArrayRef<dataflow::AbstractSparseLattice*> results) {
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
      if (!state->getValue().isAlwaysZero(ctrl))
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

} // namespace mlir::mqt::qco

std::unique_ptr<Pass> mlir::mqt::createConstantPropagationPass() {
  return std::make_unique<ConstantPropagationPass>();
}

void mlir::mqt::registerConstantPropagationPass() {
  PassRegistration<ConstantPropagationPass>();
}