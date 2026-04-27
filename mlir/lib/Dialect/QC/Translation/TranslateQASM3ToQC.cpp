/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"

#include "ir/Definitions.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qasm3/Exception.hpp"
#include "qasm3/Gate.hpp"
#include "qasm3/InstVisitor.hpp"
#include "qasm3/NestedEnvironment.hpp"
#include "qasm3/Parser.hpp"
#include "qasm3/Statement.hpp"
#include "qasm3/StdGates.hpp"
#include "qasm3/Types.hpp"
#include "qasm3/passes/ConstEvalPass.hpp"
#include "qasm3/passes/TypeCheckPass.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallDenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <istream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir::qc {

namespace {

//===----------------------------------------------------------------------===//
// Gate dispatch table
//===----------------------------------------------------------------------===//

/// Signature: (builder, gate-operands, evaluated-parameters).
/// For gates with implicit controls (cx, ccx, ...) all qubits including
/// the controls are in the qubits array, matching QASM3 operand order.
using GateFn = std::function<void(QCProgramBuilder&, llvm::ArrayRef<Value>,
                                  llvm::ArrayRef<double>)>;

/**
 * Build the static gate-name → GateFn dispatch table.
 * Each entry maps a QASM3 gate identifier to a lambda that emits the
 * corresponding QC dialect op via QCProgramBuilder.
 */
static llvm::StringMap<GateFn> buildGateDispatch() {
  llvm::StringMap<GateFn> d;

  // 0-target, 1-param
  d["gphase"] = [](auto& b, auto /*q*/, auto p) { b.gphase(p[0]); };

  // 1-target, 0-param
  d["id"] = [](auto& b, auto q, auto) { b.id(q[0]); };
  d["x"] = [](auto& b, auto q, auto) { b.x(q[0]); };
  d["y"] = [](auto& b, auto q, auto) { b.y(q[0]); };
  d["z"] = [](auto& b, auto q, auto) { b.z(q[0]); };
  d["h"] = [](auto& b, auto q, auto) { b.h(q[0]); };
  d["s"] = [](auto& b, auto q, auto) { b.s(q[0]); };
  d["sdg"] = [](auto& b, auto q, auto) { b.sdg(q[0]); };
  d["t"] = [](auto& b, auto q, auto) { b.t(q[0]); };
  d["tdg"] = [](auto& b, auto q, auto) { b.tdg(q[0]); };
  d["sx"] = [](auto& b, auto q, auto) { b.sx(q[0]); };
  d["sxdg"] = [](auto& b, auto q, auto) { b.sxdg(q[0]); };

  // 1-target, 1-param
  d["rx"] = [](auto& b, auto q, auto p) { b.rx(p[0], q[0]); };
  d["ry"] = [](auto& b, auto q, auto p) { b.ry(p[0], q[0]); };
  d["rz"] = [](auto& b, auto q, auto p) { b.rz(p[0], q[0]); };
  d["p"] = [](auto& b, auto q, auto p) { b.p(p[0], q[0]); };
  d["u1"] = [](auto& b, auto q, auto p) { b.p(p[0], q[0]); };    // alias
  d["phase"] = [](auto& b, auto q, auto p) { b.p(p[0], q[0]); }; // alias

  // 1-target, 2-param
  d["r"] = [](auto& b, auto q, auto p) { b.r(p[0], p[1], q[0]); };
  d["u2"] = [](auto& b, auto q, auto p) { b.u2(p[0], p[1], q[0]); };

  // 1-target, 3-param
  d["U"] = [](auto& b, auto q, auto p) { b.u(p[0], p[1], p[2], q[0]); };
  d["u3"] = [](auto& b, auto q, auto p) {
    b.u(p[0], p[1], p[2], q[0]);
  }; // alias
  d["u"] = [](auto& b, auto q, auto p) {
    b.u(p[0], p[1], p[2], q[0]);
  }; // alias

  // 1-ctrl + 1-target, 0-param  (q[0]=ctrl, q[1]=target)
  d["cx"] = [](auto& b, auto q, auto) { b.cx(q[0], q[1]); };
  d["cnot"] = [](auto& b, auto q, auto) { b.cx(q[0], q[1]); }; // alias
  d["cy"] = [](auto& b, auto q, auto) { b.cy(q[0], q[1]); };
  d["cz"] = [](auto& b, auto q, auto) { b.cz(q[0], q[1]); };
  d["ch"] = [](auto& b, auto q, auto) { b.ch(q[0], q[1]); };
  d["csx"] = [](auto& b, auto q, auto) { b.csx(q[0], q[1]); };

  // 1-ctrl + 1-target, 1-param
  d["crx"] = [](auto& b, auto q, auto p) { b.crx(p[0], q[0], q[1]); };
  d["cry"] = [](auto& b, auto q, auto p) { b.cry(p[0], q[0], q[1]); };
  d["crz"] = [](auto& b, auto q, auto p) { b.crz(p[0], q[0], q[1]); };
  d["cp"] = [](auto& b, auto q, auto p) { b.cp(p[0], q[0], q[1]); };
  d["cphase"] = [](auto& b, auto q, auto p) {
    b.cp(p[0], q[0], q[1]);
  }; // alias

  // 2-ctrl + 1-target, 0-param  (q[0],q[1]=ctrl, q[2]=target)
  d["ccx"] = [](auto& b, auto q, auto) { b.mcx({q[0], q[1]}, q[2]); };
  d["toffoli"] = [](auto& b, auto q, auto) {
    b.mcx({q[0], q[1]}, q[2]);
  }; // alias
  d["ccz"] = [](auto& b, auto q, auto) { b.mcz({q[0], q[1]}, q[2]); };

  // 2-target, 0-param
  d["swap"] = [](auto& b, auto q, auto) { b.swap(q[0], q[1]); };
  d["iswap"] = [](auto& b, auto q, auto) { b.iswap(q[0], q[1]); };
  d["dcx"] = [](auto& b, auto q, auto) { b.dcx(q[0], q[1]); };
  d["ecr"] = [](auto& b, auto q, auto) { b.ecr(q[0], q[1]); };

  // 1-ctrl + 2-target, 0-param  (q[0]=ctrl, q[1],q[2]=targets)
  d["cswap"] = [](auto& b, auto q, auto) { b.cswap(q[0], q[1], q[2]); };
  d["fredkin"] = [](auto& b, auto q, auto) {
    b.cswap(q[0], q[1], q[2]);
  }; // alias

  // 2-target, 2-param
  d["xx_plus_yy"] = [](auto& b, auto q, auto p) {
    b.xx_plus_yy(p[0], p[1], q[0], q[1]);
  };
  d["xx_minus_yy"] = [](auto& b, auto q, auto p) {
    b.xx_minus_yy(p[0], p[1], q[0], q[1]);
  };

  // 2-target, 1-param
  d["rxx"] = [](auto& b, auto q, auto p) { b.rxx(p[0], q[0], q[1]); };
  d["ryy"] = [](auto& b, auto q, auto p) { b.ryy(p[0], q[0], q[1]); };
  d["rzx"] = [](auto& b, auto q, auto p) { b.rzx(p[0], q[0], q[1]); };
  d["rzz"] = [](auto& b, auto q, auto p) { b.rzz(p[0], q[0], q[1]); };

  // MCX variants: q[0..N-2] are controls, q[N-1] is the target.
  // These are not in stdgates.inc but are widely used (Qiskit-style).
  auto mcxFn = [](auto& b, auto q, auto) { b.mcx(q.drop_back(1), q.back()); };
  d["mcx"] = mcxFn;
  d["mcx_gray"] = mcxFn;
  d["mcphase"] = [](auto& b, auto q, auto p) {
    b.mcp(p[0], q.drop_back(1), q.back());
  };
  // vchain/recursive carry ancilla qubits; strip them using Qiskit's formula
  d["mcx_vchain"] = [](auto& b, auto q, auto) {
    const size_t n = q.size() - ((q.size() + 1) / 2) + 2;
    b.mcx(q.slice(0, n - 1), q[n - 1]);
  };
  d["mcx_recursive"] = [](auto& b, auto q, auto) {
    const size_t n = (q.size() > 5) ? q.size() - 1 : q.size();
    b.mcx(q.slice(0, n - 1), q[n - 1]);
  };

  return d;
}

/// Static gate dispatch table, built once at startup.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static const llvm::StringMap<GateFn> GATE_DISPATCH = buildGateDispatch();

//===----------------------------------------------------------------------===//
// MLIRQasmImporter
//===----------------------------------------------------------------------===//

/// Local qubit scope used during compound gate body expansion.
/// Maps argument name → vector of MLIR qubit Values.
using QubitScope = llvm::StringMap<llvm::SmallVector<Value>>;

/** AST visitor that translates a QASM3 program directly into the QC dialect.
 *
 * Implements qasm3::InstVisitor to walk the AST produced by qasm3::Parser and
 * emit QC dialect ops via QCProgramBuilder, bypassing qc::QuantumComputation.
 * Const-evaluation and type-checking passes run in lock-step with the walk.
 */
class MLIRQasmImporter final : public qasm3::InstVisitor {
public:
  explicit MLIRQasmImporter(MLIRContext* ctx)
      : builder(ctx), typeCheckPass(constEvalPass),
        gates(qasm3::STANDARD_GATES) {
    initBuiltins();
    builder.initialize();
  }

  void
  visitProgram(const std::vector<std::shared_ptr<qasm3::Statement>>& program) {
    for (const auto& stmt : program) {
      constEvalPass.processStatement(*stmt);
      typeCheckPass.processStatement(*stmt);
      stmt->accept(this);
    }
  }

  OwningOpRef<ModuleOp> finalize() { return builder.finalize(); }

private:
  QCProgramBuilder builder;
  qasm3::const_eval::ConstEvalPass constEvalPass;
  qasm3::type_checking::TypeCheckPass typeCheckPass;
  qasm3::NestedEnvironment<std::shared_ptr<qasm3::DeclarationStatement>>
      declarations;

  /// Top-level qubit registers: register name → SSA Values
  QubitScope qubitRegisters;

  /// Classical bit registers: register name → ClassicalRegister
  llvm::StringMap<QCProgramBuilder::ClassicalRegister> classicalRegisters;

  /// Measurement result tracking: register name → vector of i1 Values.
  /// Updated each time a measure is emitted. Used for if/else conditions.
  llvm::StringMap<llvm::SmallVector<Value>> bitValues;

  /// Gate library: standard gates + user-defined compound gates
  std::map<std::string, std::shared_ptr<qasm3::Gate>> gates;

  bool openQASM2CompatMode{false};

  //===--- Initialization -----------------------------------------------===//

  /// Seed the const-eval and type-check passes with QASM3 built-in constants
  /// (pi, euler, tau) and prime the type environment with their types.
  void initBuiltins() {
    using namespace qasm3::const_eval;
    using namespace qasm3::type_checking;
    auto floatTy = InferredType{std::dynamic_pointer_cast<qasm3::ResolvedType>(
        std::make_shared<qasm3::DesignatedType<uint64_t>>(qasm3::Float, 64))};

    auto addConstant = [&](const std::string& name, double val) {
      constEvalPass.addConst(name, ConstEvalValue(val));
      typeCheckPass.addBuiltin(name, floatTy);
    };

    addConstant("pi", ::qc::PI);
    addConstant("π", ::qc::PI);
    addConstant("tau", ::qc::TAU);
    addConstant("τ", ::qc::TAU);
    addConstant("euler", ::qc::E);
    addConstant("ℇ", ::qc::E);

    // MCX variants: not in OQ3 stdlib, variable qubit arity.
    // GateInfo fields are not arity-checked for StandardGate in this importer;
    // the GATE_DISPATCH lambdas handle the variable-arity logic at call time.
    const qasm3::GateInfo mcxInfo{.nControls = 0,
                                  .nTargets = 0,
                                  .nParameters = 0,
                                  .type = ::qc::OpType::X};
    const qasm3::GateInfo mcphaseInfo{.nControls = 0,
                                      .nTargets = 0,
                                      .nParameters = 1,
                                      .type = ::qc::OpType::P};
    gates["mcx"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcx_gray"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcx_vchain"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcx_recursive"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcphase"] = std::make_shared<qasm3::StandardGate>(mcphaseInfo);
  }

  //===--- InstVisitor overrides ----------------------------------------===//

  void visitVersionDeclaration(const std::shared_ptr<qasm3::VersionDeclaration>
                                   versionDeclaration) override {
    if (versionDeclaration->version < 3) {
      openQASM2CompatMode = true;
    }
  }

  void visitDeclarationStatement(
      std::shared_ptr<qasm3::DeclarationStatement> stmt) override {
    const auto& id = stmt->identifier;
    if (declarations.find(id).has_value()) {
      throw qasm3::CompilerError("Identifier '" + id + "' already declared.",
                                 stmt->debugInfo);
    }
    declarations.emplace(id, stmt);

    const auto ty = std::get<1>(stmt->type);
    const auto sizedTy =
        std::dynamic_pointer_cast<qasm3::DesignatedType<uint64_t>>(ty);
    if (!sizedTy) {
      throw qasm3::CompilerError("Only sized types are supported.",
                                 stmt->debugInfo);
    }
    const auto size = static_cast<int64_t>(sizedTy->getDesignator());

    switch (sizedTy->type) {
    case qasm3::Qubit: {
      auto reg = builder.allocQubitRegister(size);
      llvm::SmallVector<Value> qubits;
      qubits.reserve(static_cast<size_t>(size));
      for (int64_t i = 0; i < size; ++i) {
        qubits.push_back(reg[static_cast<size_t>(i)]);
      }
      qubitRegisters[id] = std::move(qubits);
      break;
    }
    case qasm3::Bit:
    case qasm3::Int:
    case qasm3::Uint: {
      classicalRegisters[id] = builder.allocClassicalBitRegister(size, id);
      break;
    }
    default:
      break;
    }

    // Handle initializer (measure only)
    if (stmt->expression) {
      const auto& expr = stmt->expression->expression;
      if (const auto measureExpr =
              std::dynamic_pointer_cast<qasm3::MeasureExpression>(expr)) {
        auto lhsId = std::make_shared<qasm3::IndexedIdentifier>(id);
        visitMeasureAssignment(lhsId, measureExpr, stmt->debugInfo);
        return;
      }
      if (stmt->isConst) {
        return; // nothing to emit
      }
      throw qasm3::CompilerError(
          "Only measure statements are supported as initializers.",
          stmt->debugInfo);
    }
  }

  void visitAssignmentStatement(
      std::shared_ptr<qasm3::AssignmentStatement> stmt) override {
    const auto& id = stmt->identifier->identifier;
    assert(declarations.find(id).has_value() && "Checked by type check pass");
    assert(!declarations.find(id)->get()->isConst &&
           "Checked by type check pass");

    const auto& expr = stmt->expression->expression;
    if (const auto measureExpr =
            std::dynamic_pointer_cast<qasm3::MeasureExpression>(expr)) {
      visitMeasureAssignment(stmt->identifier, measureExpr, stmt->debugInfo);
      return;
    }

    // TODO: In the future, handle classical computation.
    throw qasm3::CompilerError("Classical computation not yet supported.",
                               stmt->debugInfo);
  }

  void
  visitGateStatement(std::shared_ptr<qasm3::GateDeclaration> stmt) override {
    auto id = stmt->identifier;
    if (stmt->isOpaque) {
      if (gates.find(id) == gates.end()) {
        throw qasm3::CompilerError("Unsupported opaque gate '" + id + "'.",
                                   stmt->debugInfo);
      }
      return;
    }
    if (gates.count(id) != 0U) {
      if (std::dynamic_pointer_cast<qasm3::StandardGate>(gates[id])) {
        return; // ignore redeclaration of standard gate
      }
      throw qasm3::CompilerError("Gate '" + id + "' already declared.",
                                 stmt->debugInfo);
    }
    std::vector<std::string> paramNames;
    for (const auto& p : stmt->parameters->identifiers) {
      if (std::ranges::find(paramNames, p->identifier) != paramNames.end()) {
        throw qasm3::CompilerError("Parameter '" + p->identifier +
                                       "' already declared in gate '" + id +
                                       "'.",
                                   stmt->debugInfo);
      }
      paramNames.push_back(p->identifier);
    }
    std::vector<std::string> qubitNames;
    for (const auto& q : stmt->qubits->identifiers) {
      if (std::ranges::find(qubitNames, q->identifier) != qubitNames.end()) {
        throw qasm3::CompilerError("Qubit '" + q->identifier +
                                       "' already declared in gate '" + id +
                                       "'.",
                                   stmt->debugInfo);
      }
      qubitNames.push_back(q->identifier);
    }
    gates[id] = std::make_shared<qasm3::CompoundGate>(
        std::move(paramNames), std::move(qubitNames), stmt->statements);
  }

  void visitGateCallStatement(
      std::shared_ptr<qasm3::GateCallStatement> stmt) override {
    applyGateCallStatement(stmt, qubitRegisters);
  }

  /** Emit measure ops for \p target = measure \p measureExpr.
   * Handles both full-register and single-bit assignments; wires measure
   * results into the classical register's bit-value map for later use.
   */
  void visitMeasureAssignment(
      const std::shared_ptr<qasm3::IndexedIdentifier>& target,
      const std::shared_ptr<qasm3::MeasureExpression>& measureExpr,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    auto qubits = resolveGateOperand(measureExpr->gate, debugInfo);
    auto bits = resolveClassicalBits(target, debugInfo);
    if (qubits.size() != bits.size()) {
      throw qasm3::CompilerError(
          "Classical and quantum registers must have the same width in measure "
          "statement. Classical register '" +
              target->identifier + "' has " + std::to_string(bits.size()) +
              " bits, but quantum register '" + measureExpr->gate->getName() +
              "' has " + std::to_string(qubits.size()) + " qubits.",
          debugInfo);
    }
    for (size_t i = 0; i < qubits.size(); ++i) {
      // Use the MeasureOp directly to capture the i1 result for if/else
      auto measureOp = MeasureOp::create(
          builder, qubits[i], builder.getStringAttr(bits[i].registerName),
          builder.getI64IntegerAttr(bits[i].registerSize),
          builder.getI64IntegerAttr(bits[i].registerIndex));
      Value result = measureOp.getResult();

      // Track the result for use in if/else conditions
      const auto& regName = bits[i].registerName;
      auto& regBits = bitValues[regName];
      const auto idx = static_cast<size_t>(bits[i].registerIndex);
      if (regBits.size() <= idx) {
        regBits.resize(idx + 1);
      }
      regBits[idx] = result;
    }
  }

  void visitBarrierStatement(
      std::shared_ptr<qasm3::BarrierStatement> stmt) override {
    llvm::SmallVector<Value> qubits;
    for (const auto& gate : stmt->gates) {
      auto resolved = resolveGateOperand(gate, stmt->debugInfo);
      qubits.append(resolved.begin(), resolved.end());
    }
    builder.barrier(qubits);
  }

  void
  visitResetStatement(std::shared_ptr<qasm3::ResetStatement> stmt) override {
    for (auto q : resolveGateOperand(stmt->gate, stmt->debugInfo)) {
      builder.reset(q);
    }
  }

  void visitIfStatement(std::shared_ptr<qasm3::IfStatement> stmt) override {
    if (stmt->thenStatements.empty() && stmt->elseStatements.empty()) {
      return;
    }

    Value condition = translateCondition(stmt->condition, stmt->debugInfo);
    const bool hasElse = !stmt->elseStatements.empty();

    auto ifOp =
        scf::IfOp::create(builder, condition, /*withElseRegion=*/hasElse);

    // Then block
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    emitBlockStatements(stmt->thenStatements, stmt->debugInfo);

    // Else block
    if (hasElse) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      emitBlockStatements(stmt->elseStatements, stmt->debugInfo);
    }

    // Restore insertion point after the if op
    builder.setInsertionPointAfter(ifOp);
  }

  void
  visitInitialLayout(std::shared_ptr<qasm3::InitialLayout> layout) override {
    throw qasm3::CompilerError(
        "Initial layout pragmas are not supported in direct MLIR import.",
        layout->debugInfo);
  }

  void visitOutputPermutation(
      std::shared_ptr<qasm3::OutputPermutation> perm) override {
    throw qasm3::CompilerError(
        "Output permutation pragmas are not supported in direct MLIR import.",
        perm->debugInfo);
  }

  //===--- Core gate application ----------------------------------------===//

  /// Apply a gate call statement, resolving qubits from \p scope.
  /// For top-level calls pass \p qubitRegisters; for compound gate bodies
  /// pass the local argument scope.
  void
  applyGateCallStatement(const std::shared_ptr<qasm3::GateCallStatement>& stmt,
                         const QubitScope& scope) {
    const auto& id = stmt->identifier;

    auto it = gates.find(id);
    // `resolvedId` may differ from `id` when OQ2 compat strips 'c' prefixes.
    std::string resolvedId = id;
    size_t implicitCompatControls = 0;

    // OQ2 compat mode: strip leading 'c' prefixes and treat each as
    // an additional positive control.  E.g. "cmygate q0, q1" with compat on
    // becomes ctrl @ mygate q0, q1 when "mygate" is in the gate library.
    if (it == gates.end() && openQASM2CompatMode) {
      while (!resolvedId.empty() && resolvedId.front() == 'c') {
        resolvedId = resolvedId.substr(1);
        ++implicitCompatControls;
      }
      if (implicitCompatControls > 0) {
        it = gates.find(resolvedId);
      }
    }
    if (it == gates.end()) {
      throw qasm3::CompilerError("Unknown gate '" + id + "'.", stmt->debugInfo);
    }

    // Evaluate parameters to doubles
    std::vector<double> params;
    params.reserve(stmt->arguments.size());
    for (const auto& arg : stmt->arguments) {
      auto result = constEvalPass.visit(arg);
      if (!result.has_value()) {
        throw qasm3::CompilerError(
            "Gate parameter could not be const-evaluated.", stmt->debugInfo);
      }
      params.push_back(result->toExpr()->asFP());
    }

    // Parse modifiers: accumulate pos/neg controls and invert flag
    bool invert = false;
    size_t nModifierControls = 0;
    // (count, isPositive) per ctrl modifier, in order
    llvm::SmallVector<std::pair<size_t, bool>> ctrlSpec;
    for (const auto& mod : stmt->modifiers) {
      if (std::dynamic_pointer_cast<qasm3::InvGateModifier>(mod)) {
        invert = !invert;
      } else if (const auto* ctrlMod =
                     dynamic_cast<qasm3::CtrlGateModifier*>(mod.get())) {
        const size_t n =
            evaluatePositiveConstant(ctrlMod->expression, stmt->debugInfo, 1);
        ctrlSpec.emplace_back(n, ctrlMod->ctrlType);
        nModifierControls += n;
      } else {
        // TODO: add pow(n) support here once PowOp lands in main — detect
        // qasm3::PowGateModifier, evaluate n, wrap gate emission in PowOp.
        throw qasm3::CompilerError(
            "Only ctrl/negctrl/inv modifiers are supported.", stmt->debugInfo);
      }
    }

    // Expand each operand to its qubit Values
    std::vector<llvm::SmallVector<Value>> expandedOperands;
    expandedOperands.reserve(stmt->operands.size());
    for (const auto& operand : stmt->operands) {
      expandedOperands.push_back(
          resolveGateOperandInScope(operand, scope, stmt->debugInfo));
    }

    // First nModifierControls slots are modifier-derived controls
    llvm::SmallVector<Value> posControls;
    llvm::SmallVector<Value> negControls;
    size_t ctrlIdx = 0;
    for (const auto& [n, positive] : ctrlSpec) {
      for (size_t i = 0; i < n; ++i, ++ctrlIdx) {
        if (expandedOperands[ctrlIdx].size() != 1) {
          throw qasm3::CompilerError("Control operand must be a single qubit.",
                                     stmt->debugInfo);
        }
        if (positive) {
          posControls.push_back(expandedOperands[ctrlIdx][0]);
        } else {
          negControls.push_back(expandedOperands[ctrlIdx][0]);
        }
      }
    }

    // OQ2 compat implicit controls follow modifier controls
    for (size_t i = 0; i < implicitCompatControls; ++i, ++ctrlIdx) {
      if (ctrlIdx >= expandedOperands.size() ||
          expandedOperands[ctrlIdx].size() != 1) {
        throw qasm3::CompilerError(
            "Implicit OQ2 control operand must be a single qubit.",
            stmt->debugInfo);
      }
      posControls.push_back(expandedOperands[ctrlIdx][0]);
    }
    const size_t totalCtrlCount = nModifierControls + implicitCompatControls;

    // Remaining slots are the gate's own operands (may broadcast)
    std::vector<llvm::SmallVector<Value>> gateOperands(
        expandedOperands.begin() + static_cast<std::ptrdiff_t>(totalCtrlCount),
        expandedOperands.end());

    // Compound gate: inline expand
    if (const auto* compound =
            dynamic_cast<qasm3::CompoundGate*>(it->second.get())) {
      applyCompoundGate(*compound, gateOperands, posControls, negControls,
                        params, invert, stmt->debugInfo);
      return;
    }

    // Standard gate: validate param count then determine broadcast width
    if (it->second->getNParameters() != params.size()) {
      throw qasm3::CompilerError(
          "Gate '" + id + "' takes " +
              std::to_string(it->second->getNParameters()) +
              " parameters, but " + std::to_string(params.size()) +
              " were supplied.",
          stmt->debugInfo);
    }

    // Standard gate: determine broadcast width
    size_t broadcastWidth = 0;
    for (const auto& ops : gateOperands) {
      if (ops.size() > 1) {
        if (broadcastWidth == 0) {
          broadcastWidth = ops.size();
        } else if (broadcastWidth != ops.size()) {
          throw qasm3::CompilerError(
              "Broadcast operands must all have the same width.",
              stmt->debugInfo);
        }
      }
    }
    if (broadcastWidth == 0) {
      broadcastWidth = 1;
    }

    const auto dispIt = GATE_DISPATCH.find(resolvedId);
    if (dispIt == GATE_DISPATCH.end()) {
      throw qasm3::CompilerError("No MLIR mapping for gate '" + id + "'.",
                                 stmt->debugInfo);
    }

    for (size_t b = 0; b < broadcastWidth; ++b) {
      llvm::SmallVector<Value> iterQubits;
      iterQubits.reserve(gateOperands.size());
      for (const auto& ops : gateOperands) {
        iterQubits.push_back(ops.size() > 1 ? ops[b] : ops[0]);
      }

      // Check that no qubit appears twice across targets and controls.
      llvm::SmallDenseSet<Value> seen;
      for (auto q :
           llvm::concat<const Value>(iterQubits, posControls, negControls)) {
        if (!seen.insert(q).second) {
          throw qasm3::CompilerError("Duplicate qubit in gate '" + id +
                                         "' operands.",
                                     stmt->debugInfo);
        }
      }

      emitGate(dispIt->second, iterQubits, params, posControls, negControls,
               invert);
    }
  }

  /// Emit a single gate application, wrapping with ctrl/inv as needed.
  void emitGate(const GateFn& gateFn, llvm::ArrayRef<Value> qubits,
                llvm::ArrayRef<double> params,
                llvm::ArrayRef<Value> posControls,
                llvm::ArrayRef<Value> negControls, bool invert) {
    auto inner = [&] { gateFn(builder, qubits, params); };

    auto withInv = [&] {
      if (invert) {
        builder.inv(llvm::function_ref<void()>(inner));
      } else {
        inner();
      }
    };

    if (posControls.empty() && negControls.empty()) {
      withInv();
      return;
    }

    // Negative controls: X-bracket
    for (auto q : negControls) {
      builder.x(q);
    }
    llvm::SmallVector<Value> allControls(posControls.begin(),
                                         posControls.end());
    allControls.append(negControls.begin(), negControls.end());
    builder.ctrl(allControls, llvm::function_ref<void()>(withInv));
    for (auto q : negControls) {
      builder.x(q);
    }
  }

  /// Inline-expand a compound (user-defined) gate.
  void
  applyCompoundGate(const qasm3::CompoundGate& gate,
                    const std::vector<llvm::SmallVector<Value>>& gateOperands,
                    llvm::ArrayRef<Value> posControls,
                    llvm::ArrayRef<Value> negControls,
                    llvm::ArrayRef<double> params, bool invert,
                    const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    if (gate.targetNames.size() != gateOperands.size()) {
      throw qasm3::CompilerError("Compound gate operand count mismatch.",
                                 debugInfo);
    }
    if (gate.parameterNames.size() != params.size()) {
      throw qasm3::CompilerError("Compound gate parameter count mismatch.",
                                 debugInfo);
    }

    // Build local scope: argument name → Values
    QubitScope localScope;
    for (size_t i = 0; i < gate.targetNames.size(); ++i) {
      localScope[gate.targetNames[i]] = llvm::SmallVector<Value>(
          gateOperands[i].begin(), gateOperands[i].end());
    }

    // Bind parameters as constants
    constEvalPass.pushEnv();
    for (size_t i = 0; i < gate.parameterNames.size(); ++i) {
      constEvalPass.addConst(gate.parameterNames[i],
                             qasm3::const_eval::ConstEvalValue(params[i]));
    }

    auto bodyFn = [&] {
      for (const auto& bodyStmt : gate.body) {
        if (const auto gateCall =
                std::dynamic_pointer_cast<qasm3::GateCallStatement>(bodyStmt)) {
          applyGateCallStatement(gateCall, localScope);
        } else if (const auto barrier =
                       std::dynamic_pointer_cast<qasm3::BarrierStatement>(
                           bodyStmt)) {
          llvm::SmallVector<Value> qubits;
          for (const auto& g : barrier->gates) {
            auto resolved =
                resolveGateOperandInScope(g, localScope, barrier->debugInfo);
            qubits.append(resolved.begin(), resolved.end());
          }
          builder.barrier(qubits);
        } else if (const auto reset =
                       std::dynamic_pointer_cast<qasm3::ResetStatement>(
                           bodyStmt)) {
          for (auto q : resolveGateOperandInScope(reset->gate, localScope,
                                                  reset->debugInfo)) {
            builder.reset(q);
          }
        }
      }
    };

    auto withInv = [&] {
      if (invert) {
        builder.inv(llvm::function_ref<void()>(bodyFn));
      } else {
        bodyFn();
      }
    };

    if (posControls.empty() && negControls.empty()) {
      withInv();
    } else {
      for (auto q : negControls) {
        builder.x(q);
      }
      llvm::SmallVector<Value> allControls(posControls.begin(),
                                           posControls.end());
      allControls.append(negControls.begin(), negControls.end());
      builder.ctrl(allControls, llvm::function_ref<void()>(withInv));
      for (auto q : negControls) {
        builder.x(q);
      }
    }

    constEvalPass.popEnv();
  }

  //===--- If/else helpers ------------------------------------------------===//

  /// Emit quantum statements inside an if/else block.
  void emitBlockStatements(
      const std::vector<std::shared_ptr<qasm3::Statement>>& statements,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    for (const auto& statement : statements) {
      auto gateCall =
          std::dynamic_pointer_cast<qasm3::GateCallStatement>(statement);
      if (gateCall == nullptr) {
        throw qasm3::CompilerError(
            "Only quantum statements are supported in if/else blocks.",
            debugInfo);
      }
      applyGateCallStatement(gateCall, qubitRegisters);
    }
  }

  /// Translate a QASM3 condition expression to an i1 MLIR Value.
  /// Supports:
  ///   - Single bit: `c[0]` or `!c[0]` / `~c[0]`
  ///   - Register comparison: `creg == N`, `creg != N`, etc.
  [[nodiscard]] Value
  translateCondition(const std::shared_ptr<qasm3::Expression>& condition,
                     const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    // Case 1: Binary comparison (creg == N, creg != N, etc.)
    if (const auto binaryExpr =
            std::dynamic_pointer_cast<qasm3::BinaryExpression>(condition)) {
      return translateBinaryCondition(binaryExpr, debugInfo);
    }

    // Case 2: Unary negation (!c[0] or ~c[0])
    if (const auto unaryExpr =
            std::dynamic_pointer_cast<qasm3::UnaryExpression>(condition)) {
      assert(unaryExpr->op == qasm3::UnaryExpression::LogicalNot ||
             unaryExpr->op == qasm3::UnaryExpression::BitwiseNot);
      const auto idExpr = std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(
          unaryExpr->operand);
      Value bitVal = lookupBitValue(idExpr, debugInfo);
      // Negate: XOR with true
      Value trueVal = arith::ConstantOp::create(
          builder, builder.getIntegerAttr(builder.getI1Type(), 1));
      return arith::XOrIOp::create(builder, bitVal, trueVal);
    }

    // Case 3: Single bit (c[0] — truthy)
    if (const auto idExpr =
            std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(condition)) {
      return lookupBitValue(idExpr, debugInfo);
    }

    throw qasm3::CompilerError(
        "Unsupported condition expression in if statement.", debugInfo);
  }

  /// Translate a binary comparison condition (creg == N, etc.)
  [[nodiscard]] Value translateBinaryCondition(
      const std::shared_ptr<qasm3::BinaryExpression>& binaryExpr,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    const auto comparisonKind = qasm3::getComparisonKind(binaryExpr->op);
    if (!comparisonKind) {
      throw qasm3::CompilerError("Unsupported comparison operator.", debugInfo);
    }

    // Determine which side is the identifier and which is the constant
    auto lhsIsIdentifier =
        std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(binaryExpr->lhs);

    const auto& idExpr =
        lhsIsIdentifier ? std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(
                              binaryExpr->lhs)
                        : std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(
                              binaryExpr->rhs);
    const auto& constExpr =
        lhsIsIdentifier
            ? std::dynamic_pointer_cast<qasm3::Constant>(binaryExpr->rhs)
            : std::dynamic_pointer_cast<qasm3::Constant>(binaryExpr->lhs);

    if (!idExpr || !constExpr) {
      throw qasm3::CompilerError(
          "Only classical registers and constants are supported in conditions.",
          debugInfo);
    }

    const auto& regName = idExpr->identifier;
    const uint64_t expectedVal = constExpr->getUInt();

    // Look up classical register to get its size
    auto cregIt = classicalRegisters.find(regName);
    if (cregIt == classicalRegisters.end()) {
      throw qasm3::CompilerError("Unknown classical register '" + regName +
                                     "' in condition.",
                                 debugInfo);
    }
    const auto regSize = static_cast<size_t>(cregIt->second.size);

    auto bitIt = bitValues.find(regName);
    if (bitIt == bitValues.end()) {
      throw qasm3::CompilerError(
          "Classical register '" + regName +
              "' has no measurement results to use in condition.",
          debugInfo);
    }
    const auto& regBits = bitIt->second;

    const auto intWidth = std::max(regSize, static_cast<size_t>(64));
    auto intTy = builder.getIntegerType(static_cast<unsigned>(intWidth));

    // Indexed access (c[i] == N): compare the single bit directly
    if (!idExpr->indices.empty()) {
      const auto idx = evaluatePositiveConstant(
          idExpr->indices[0]->indexExpressions[0], debugInfo);
      if (idx >= regBits.size() || !regBits[idx]) {
        throw qasm3::CompilerError(
            "Bit " + std::to_string(idx) + " of register '" + regName +
                "' was not measured before use in condition.",
            debugInfo);
      }
      Value extended = arith::ExtUIOp::create(builder, intTy, regBits[idx]);
      Value expected = arith::ConstantOp::create(
          builder, builder.getIntegerAttr(intTy, expectedVal));
      auto pred = convertComparisonKind(*comparisonKind);
      return arith::CmpIOp::create(builder, pred, extended, expected);
    }

    // Full-register access (c == N): compose bits into an integer
    // result = b0 | (b1 << 1) | (b2 << 2) | ...
    Value composed =
        arith::ConstantOp::create(builder, builder.getIntegerAttr(intTy, 0));

    for (size_t i = 0; i < regSize; ++i) {
      if (i >= regBits.size() || !regBits[i]) {
        throw qasm3::CompilerError(
            "Bit " + std::to_string(i) + " of register '" + regName +
                "' was not measured before use in condition.",
            debugInfo);
      }
      // Extend i1 to integer type
      Value extended = arith::ExtUIOp::create(builder, intTy, regBits[i]);
      if (i > 0) {
        Value shiftAmt = arith::ConstantOp::create(
            builder, builder.getIntegerAttr(intTy, i));
        extended = arith::ShLIOp::create(builder, extended, shiftAmt);
      }
      composed = arith::OrIOp::create(builder, composed, extended);
    }

    // Compare
    Value expectedConst = arith::ConstantOp::create(
        builder, builder.getIntegerAttr(intTy, expectedVal));

    auto pred = convertComparisonKind(*comparisonKind);
    return arith::CmpIOp::create(builder, pred, composed, expectedConst);
  }

  /// Look up the most recent measurement result for a single classical bit.
  [[nodiscard]] Value
  lookupBitValue(const std::shared_ptr<qasm3::IndexedIdentifier>& idExpr,
                 const std::shared_ptr<qasm3::DebugInfo>& debugInfo) const {
    const auto& regName = idExpr->identifier;
    auto it = bitValues.find(regName);
    if (it == bitValues.end()) {
      throw qasm3::CompilerError(
          "Classical register '" + regName +
              "' has no measurement results to use in condition.",
          debugInfo);
    }
    const auto& regBits = it->second;

    // Single bit — must be indexed
    if (idExpr->indices.empty()) {
      if (regBits.size() != 1) {
        throw qasm3::CompilerError(
            "Condition on full register '" + regName +
                "' requires a comparison operator (e.g. creg == 0).",
            debugInfo);
      }
      if (!regBits[0]) {
        throw qasm3::CompilerError(
            "Bit 0 of register '" + regName +
                "' was not measured before use in condition.",
            debugInfo);
      }
      return regBits[0];
    }

    const auto idx = evaluatePositiveConstant(
        idExpr->indices[0]->indexExpressions[0], debugInfo);
    if (idx >= regBits.size() || !regBits[idx]) {
      throw qasm3::CompilerError(
          "Bit " + std::to_string(idx) + " of register '" + regName +
              "' was not measured before use in condition.",
          debugInfo);
    }
    return regBits[idx];
  }

  /// Convert qc::ComparisonKind to arith::CmpIPredicate.
  static arith::CmpIPredicate convertComparisonKind(::qc::ComparisonKind kind) {
    switch (kind) {
    case ::qc::ComparisonKind::Eq:
      return arith::CmpIPredicate::eq;
    case ::qc::ComparisonKind::Neq:
      return arith::CmpIPredicate::ne;
    case ::qc::ComparisonKind::Lt:
      return arith::CmpIPredicate::ult;
    case ::qc::ComparisonKind::Leq:
      return arith::CmpIPredicate::ule;
    case ::qc::ComparisonKind::Gt:
      return arith::CmpIPredicate::ugt;
    case ::qc::ComparisonKind::Geq:
      return arith::CmpIPredicate::uge;
    }
    llvm_unreachable("unknown ComparisonKind");
  }

  //===--- Operand resolution helpers ------------------------------------===//

  /// Resolve a gate operand against the top-level qubit register map.
  llvm::SmallVector<Value>
  resolveGateOperand(const std::shared_ptr<qasm3::GateOperand>& operand,
                     const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    return resolveGateOperandInScope(operand, qubitRegisters, debugInfo);
  }

  /** Resolve a gate operand against \p scope (top-level registers or a
   * compound-gate local argument scope). Returns the MLIR Values for the
   * qubit(s) named by \p operand — a full register or a single indexed qubit.
   */
  llvm::SmallVector<Value> resolveGateOperandInScope(
      const std::shared_ptr<qasm3::GateOperand>& operand,
      const QubitScope& scope,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    if (operand->isHardwareQubit()) {
      return {builder.staticQubit(operand->getHardwareQubit())};
    }

    const auto idExpr = operand->getIdentifier();
    const auto& name = idExpr->identifier;

    auto it = scope.find(name);
    if (it == scope.end()) {
      throw qasm3::CompilerError("Unknown qubit register '" + name + "'.",
                                 debugInfo);
    }
    const auto& qubits = it->second;

    if (idExpr->indices.empty()) {
      return qubits; // full register
    }

    if (idExpr->indices.size() > 1) {
      throw qasm3::CompilerError("Only single-index expressions are supported.",
                                 debugInfo);
    }
    const auto& indexExpression = idExpr->indices[0]->indexExpressions[0];
    const auto idx = evaluatePositiveConstant(indexExpression, debugInfo);
    if (idx >= qubits.size()) {
      throw qasm3::CompilerError("Qubit index out of bounds.", debugInfo);
    }
    return {qubits[idx]};
  }

  /** Resolve \p target to a list of classical bits in a known register.
   * Returns all bits for an unindexed identifier, or a single bit otherwise.
   */
  std::vector<QCProgramBuilder::Bit> resolveClassicalBits(
      const std::shared_ptr<qasm3::IndexedIdentifier>& target,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) const {
    const auto& name = target->identifier;
    auto it = classicalRegisters.find(name);
    if (it == classicalRegisters.end()) {
      throw qasm3::CompilerError("Unknown classical register '" + name + "'.",
                                 debugInfo);
    }
    const auto& creg = it->second;

    std::vector<QCProgramBuilder::Bit> bits;
    if (target->indices.empty()) {
      for (int64_t i = 0; i < creg.size; ++i) {
        bits.push_back(creg[i]);
      }
      return bits;
    }
    const auto& indexExpression = target->indices[0]->indexExpressions[0];
    const auto idx = evaluatePositiveConstant(indexExpression, debugInfo);
    bits.push_back(creg[idx]);
    return bits;
  }

  /// Evaluate \p expr as a non-negative integer constant.
  /// Returns \p defaultValue if \p expr is null; throws on non-constant input.
  static size_t
  evaluatePositiveConstant(const std::shared_ptr<qasm3::Expression>& expr,
                           const std::shared_ptr<qasm3::DebugInfo>& debugInfo,
                           size_t defaultValue = 0) {
    if (!expr) {
      return defaultValue;
    }
    const auto constVal = std::dynamic_pointer_cast<qasm3::Constant>(expr);
    if (!constVal) {
      throw qasm3::CompilerError("Expected a constant integer expression.",
                                 debugInfo);
    }
    return static_cast<size_t>(constVal->getUInt());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> translateQASM3ToQC(MLIRContext* context,
                                         std::istream& input) {
  try {
    qasm3::Parser parser(input);
    const auto program = parser.parseProgram();
    MLIRQasmImporter importer(context);
    importer.visitProgram(program);
    return importer.finalize();
  } catch (const qasm3::CompilerError& e) {
    llvm::errs() << "QASM3 import error: " << e.what() << "\n";
    return nullptr;
  } catch (const std::exception& e) {
    llvm::errs() << "QASM3 import error: " << e.what() << "\n";
    return nullptr;
  }
}

OwningOpRef<ModuleOp> translateQASM3ToQC(MLIRContext* context,
                                         const std::string& filename) {
  std::ifstream file(filename);
  if (!file.good()) {
    llvm::errs() << "Could not open file '" << filename << "'\n";
    return nullptr;
  }
  return translateQASM3ToQC(context, file);
}

} // namespace mlir::qc
