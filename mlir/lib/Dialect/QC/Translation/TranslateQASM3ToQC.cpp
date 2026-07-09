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
#include "ir/operations/OpType.hpp"
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::qc {

namespace {

/// Signature: (builder, gate operands, evaluated parameters).
/// For gates with implicit controls (cx, ccx, ...), all qubits including
/// the controls are part of the range, matching OpenQASM 3 operand order.
using GateFn =
    std::function<void(QCProgramBuilder&, ValueRange, ArrayRef<double>)>;

} // namespace

/**
 * @brief Build the table mapping OpenQASM 3 gate identifiers to
 * QCProgramBuilder emitters.
 *
 * @details
 * Each entry maps an OpenQASM 3 gate identifier to a lambda that emits the
 * corresponding QC operation via the QCProgramBuilder.
 */
static llvm::StringMap<GateFn> buildGateDispatch() {
  llvm::StringMap<GateFn> d;

  // ZeroTargetOneParameter
  d["gphase"] = [](auto& b, auto /*q*/, auto p) { b.gphase(p[0]); };

  // OneTargetZeroParameter
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

  // OneTargetOneParameter
  d["rx"] = [](auto& b, auto q, auto p) { b.rx(p[0], q[0]); };
  d["ry"] = [](auto& b, auto q, auto p) { b.ry(p[0], q[0]); };
  d["rz"] = [](auto& b, auto q, auto p) { b.rz(p[0], q[0]); };
  d["p"] = [](auto& b, auto q, auto p) { b.p(p[0], q[0]); };
  d["u1"] = [](auto& b, auto q, auto p) { b.p(p[0], q[0]); };    // alias
  d["phase"] = [](auto& b, auto q, auto p) { b.p(p[0], q[0]); }; // alias

  // OneTargetTwoParameter
  d["r"] = [](auto& b, auto q, auto p) { b.r(p[0], p[1], q[0]); };
  d["u2"] = [](auto& b, auto q, auto p) { b.u2(p[0], p[1], q[0]); };

  // OneTargetThreeParameter
  auto uFn = [](auto& b, auto q, auto p) { b.u(p[0], p[1], p[2], q[0]); };
  d["U"] = uFn;
  d["u3"] = uFn; // alias
  d["u"] = uFn;  // alias

  // TwoTargetZeroParameter
  d["swap"] = [](auto& b, auto q, auto) { b.swap(q[0], q[1]); };
  d["iswap"] = [](auto& b, auto q, auto) { b.iswap(q[0], q[1]); };
  d["dcx"] = [](auto& b, auto q, auto) { b.dcx(q[0], q[1]); };
  d["ecr"] = [](auto& b, auto q, auto) { b.ecr(q[0], q[1]); };

  // ThreeTargetZeroParameter
  d["rccx"] = [](auto& b, auto q, auto) { b.rccx(q[0], q[1], q[2]); };

  // TwoTargetOneParameter
  d["rxx"] = [](auto& b, auto q, auto p) { b.rxx(p[0], q[0], q[1]); };
  d["ryy"] = [](auto& b, auto q, auto p) { b.ryy(p[0], q[0], q[1]); };
  d["rzx"] = [](auto& b, auto q, auto p) { b.rzx(p[0], q[0], q[1]); };
  d["rzz"] = [](auto& b, auto q, auto p) { b.rzz(p[0], q[0], q[1]); };

  // TwoTargetTwoParameter
  d["xx_plus_yy"] = [](auto& b, auto q, auto p) {
    b.xx_plus_yy(p[0], p[1], q[0], q[1]);
  };
  d["xx_minus_yy"] = [](auto& b, auto q, auto p) {
    b.xx_minus_yy(p[0], p[1], q[0], q[1]);
  };

  // Controlled OneTargetZeroParameter
  d["cx"] = [](auto& b, auto q, auto) { b.cx(q[0], q[1]); };
  d["cnot"] = [](auto& b, auto q, auto) { b.cx(q[0], q[1]); }; // alias
  d["cy"] = [](auto& b, auto q, auto) { b.cy(q[0], q[1]); };
  d["cz"] = [](auto& b, auto q, auto) { b.cz(q[0], q[1]); };
  d["ch"] = [](auto& b, auto q, auto) { b.ch(q[0], q[1]); };
  d["csx"] = [](auto& b, auto q, auto) { b.csx(q[0], q[1]); };

  // Controlled OneTargetOneParameter
  d["crx"] = [](auto& b, auto q, auto p) { b.crx(p[0], q[0], q[1]); };
  d["cry"] = [](auto& b, auto q, auto p) { b.cry(p[0], q[0], q[1]); };
  d["crz"] = [](auto& b, auto q, auto p) { b.crz(p[0], q[0], q[1]); };
  d["cp"] = [](auto& b, auto q, auto p) { b.cp(p[0], q[0], q[1]); };
  d["cphase"] = [](auto& b, auto q, auto p) {
    b.cp(p[0], q[0], q[1]);
  }; // alias

  // Controlled TwoTargetZeroParameter
  d["cswap"] = [](auto& b, auto q, auto) { b.cswap(q[0], q[1], q[2]); };
  d["fredkin"] = [](auto& b, auto q, auto) {
    b.cswap(q[0], q[1], q[2]);
  }; // alias

  // Multi-controlled gates
  auto mcxFn = [](auto& b, auto q, auto) { b.mcx(q.drop_back(1), q.back()); };
  d["mcx"] = mcxFn;
  d["mcx_gray"] = mcxFn;

  d["mcx_vchain"] = [](auto& b, auto q, auto) {
    const size_t n = q.size() - ((q.size() + 1) / 2) + 2;
    b.mcx(q.slice(0, n - 1), q[n - 1]);
  };

  d["mcx_recursive"] = [](auto& b, auto q, auto) {
    const size_t n = (q.size() > 5) ? q.size() - 1 : q.size();
    b.mcx(q.slice(0, n - 1), q[n - 1]);
  };

  d["mcphase"] = [](auto& b, auto q, auto p) {
    b.mcp(p[0], q.drop_back(1), q.back());
  };

  return d;
}

static llvm::StringMap<std::shared_ptr<qasm3::Gate>> convertToStringMap(
    const std::map<std::string, std::shared_ptr<qasm3::Gate>>& sourceMap) {
  llvm::StringMap<std::shared_ptr<qasm3::Gate>> targetMap;
  for (const auto& [key, value] : sourceMap) {
    targetMap.insert(std::make_pair(key, value));
  }
  return targetMap;
}

namespace {

/// Map from OpenQASM 3 gate identifier to QCProgramBuilder emitter.
const llvm::StringMap<GateFn> GATE_DISPATCH = buildGateDispatch();

/// Map of qubits in the current scope.
using QubitScope = llvm::StringMap<SmallVector<Value>>;

/**
 * @brief AST visitor that translates an OpenQASM 3 program to a QC program.
 *
 * @details
 * Implements qasm3::InstVisitor to walk the AST produced by qasm3::Parser and
 * emit QC operations via the QCProgramBuilder.
 */
class MLIRQasmImporter final : public qasm3::InstVisitor {
public:
  explicit MLIRQasmImporter(MLIRContext* ctx)
      : builder(ctx), typeCheckPass(constEvalPass),
        gates(convertToStringMap(qasm3::STANDARD_GATES)) {
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

  /// Map from qubit-register name to allocated qubit values.
  QubitScope qubitRegisters;

  /// Map from classical-register name to ClassicalRegister.
  llvm::StringMap<QCProgramBuilder::ClassicalRegister> classicalRegisters;

  /// Map from classical-register name to measurement results.
  llvm::StringMap<SmallVector<Value>> bitValues;

  /// Map from gate identifier to OpenQASM 3 definition.
  llvm::StringMap<std::shared_ptr<qasm3::Gate>> gates;

  bool openQASM2CompatMode{false};

  //===--- Initialization -----------------------------------------------===//

  void initBuiltins() {
    using namespace qasm3::const_eval;
    using namespace qasm3::type_checking;

    auto floatType =
        InferredType{std::dynamic_pointer_cast<qasm3::ResolvedType>(
            std::make_shared<qasm3::DesignatedType<uint64_t>>(qasm3::Float,
                                                              64))};

    auto addConstant = [&](const std::string& name, double value) {
      constEvalPass.addConst(name, ConstEvalValue(value));
      typeCheckPass.addBuiltin(name, floatType);
    };

    addConstant("pi", ::qc::PI);
    addConstant("π", ::qc::PI);
    addConstant("tau", ::qc::TAU);
    addConstant("τ", ::qc::TAU);
    addConstant("euler", ::qc::E);
    addConstant("ℇ", ::qc::E);

    const qasm3::GateInfo mcxInfo{.nControls = 0,
                                  .nTargets = 0,
                                  .nParameters = 0,
                                  .type = ::qc::OpType::X};
    gates["mcx"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcx_gray"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcx_vchain"] = std::make_shared<qasm3::StandardGate>(mcxInfo);
    gates["mcx_recursive"] = std::make_shared<qasm3::StandardGate>(mcxInfo);

    const qasm3::GateInfo mcphaseInfo{.nControls = 0,
                                      .nTargets = 0,
                                      .nParameters = 1,
                                      .type = ::qc::OpType::P};
    gates["mcphase"] = std::make_shared<qasm3::StandardGate>(mcphaseInfo);
  }

public:
  //===--- InstVisitor overrides ----------------------------------------===//

  void
  visitGateStatement(std::shared_ptr<qasm3::GateDeclaration> stmt) override {
    const auto& id = stmt->identifier;
    if (stmt->isOpaque) {
      if (!gates.contains(id)) {
        throw qasm3::CompilerError("Unsupported opaque gate '" + id + "'.",
                                   stmt->debugInfo);
      }
      return;
    }
    if (gates.contains(id)) {
      throw qasm3::CompilerError("Gate '" + id + "' already declared.",
                                 stmt->debugInfo);
    }
    std::vector<std::string> params;
    for (const auto& p : stmt->parameters->identifiers) {
      const auto& param = p->identifier;
      if (std::ranges::find(params, param) != params.end()) {
        throw qasm3::CompilerError(
            "Parameter is already declared in compound gate.", stmt->debugInfo);
      }
      params.push_back(param);
    }
    std::vector<std::string> targets;
    for (const auto& t : stmt->qubits->identifiers) {
      const auto& target = t->identifier;
      if (std::ranges::find(targets, target) != targets.end()) {
        throw qasm3::CompilerError(
            "Target is already declared in compound gate.", stmt->debugInfo);
      }
      targets.push_back(target);
    }
    gates[id] = std::make_shared<qasm3::CompoundGate>(
        std::move(params), std::move(targets), stmt->statements);
  }

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

    if (stmt->isConst) {
      // Nothing to emit
      return;
    }

    const auto sizedType =
        std::dynamic_pointer_cast<qasm3::DesignatedType<uint64_t>>(
            std::get<1>(stmt->type));
    if (!sizedType) {
      throw qasm3::CompilerError("Only sized types are supported.",
                                 stmt->debugInfo);
    }
    const auto size = static_cast<int64_t>(sizedType->getDesignator());

    switch (sizedType->type) {
    case qasm3::Qubit: {
      const auto& reg = builder.allocQubitRegister(size);
      qubitRegisters[id] = reg.qubits;
      break;
    }
    case qasm3::Bit:
    case qasm3::Int:
    case qasm3::Uint: {
      classicalRegisters[id] = builder.allocClassicalBitRegister(size, id);
      break;
    }
    default:
      throw qasm3::CompilerError("Unsupported declaration type.",
                                 stmt->debugInfo);
    }

    // Handle declarations through measure expressions
    if (stmt->expression) {
      const auto& innerExpr = stmt->expression->expression;
      if (const auto measureExpr =
              std::dynamic_pointer_cast<qasm3::MeasureExpression>(innerExpr)) {
        auto target = std::make_shared<qasm3::IndexedIdentifier>(id);
        visitMeasureAssignment(target, measureExpr, stmt->debugInfo);
        return;
      }
      throw qasm3::CompilerError(
          "Only measure expressions can declare variables.", stmt->debugInfo);
    }
  }

  void visitInitialLayout(
      std::shared_ptr<qasm3::InitialLayout> /*initialLayout*/) override {}

  void visitOutputPermutation(
      std::shared_ptr<qasm3::OutputPermutation> /*outputPermutation*/)
      override {}

  void visitGateCallStatement(
      std::shared_ptr<qasm3::GateCallStatement> stmt) override {
    applyGateCallStatement(stmt, qubitRegisters);
  }

  void visitAssignmentStatement(
      std::shared_ptr<qasm3::AssignmentStatement> stmt) override {
    const auto& innerId = stmt->identifier->identifier;
    assert(declarations.find(innerId).has_value());
    assert(!declarations.find(innerId)->get()->isConst);

    const auto& innerExpr = stmt->expression->expression;
    if (const auto measureExpr =
            std::dynamic_pointer_cast<qasm3::MeasureExpression>(innerExpr)) {
      visitMeasureAssignment(stmt->identifier, measureExpr, stmt->debugInfo);
      return;
    }

    throw qasm3::CompilerError("Classical computations are not supported.",
                               stmt->debugInfo);
  }

  void visitMeasureAssignment(
      const std::shared_ptr<qasm3::IndexedIdentifier>& target,
      const std::shared_ptr<qasm3::MeasureExpression>& measureExpr,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    const auto& bits = resolveClassicalBits(target, debugInfo);
    const auto& operand = resolveGateOperand(measureExpr->gate, debugInfo);
    SmallVector<Value> qubits;
    if (std::holds_alternative<Value>(operand)) {
      qubits.push_back(std::get<Value>(operand));
    } else {
      qubits = std::get<SmallVector<Value>>(operand);
    }
    if (bits.size() != qubits.size()) {
      throw qasm3::CompilerError("The classical register and the quantum "
                                 "register must have the same width.",
                                 debugInfo);
    }
    for (const auto& [bit, qubit] : llvm::zip_equal(bits, qubits)) {
      auto result = MeasureOp::create(
                        builder, qubit, builder.getStringAttr(bit.registerName),
                        builder.getI64IntegerAttr(bit.registerSize),
                        builder.getI64IntegerAttr(bit.registerIndex))
                        .getResult();
      auto& regBits = bitValues[bit.registerName];
      const auto index = static_cast<size_t>(bit.registerIndex);
      if (regBits.size() <= index) {
        regBits.resize(index + 1);
      }
      regBits[index] = result;
    }
  }

  void visitBarrierStatement(
      std::shared_ptr<qasm3::BarrierStatement> stmt) override {
    SmallVector<Value> qubits;
    for (const auto& gate : stmt->gates) {
      const auto& operand = resolveGateOperand(gate, stmt->debugInfo);
      if (std::holds_alternative<Value>(operand)) {
        qubits.push_back(std::get<Value>(operand));
      } else {
        llvm::append_range(qubits, std::get<SmallVector<Value>>(operand));
      }
    }
    builder.barrier(qubits);
  }

  void
  visitResetStatement(std::shared_ptr<qasm3::ResetStatement> stmt) override {
    const auto& operand = resolveGateOperand(stmt->gate, stmt->debugInfo);
    if (std::holds_alternative<Value>(operand)) {
      builder.reset(std::get<Value>(operand));
    } else {
      for (auto qubit : std::get<SmallVector<Value>>(operand)) {
        builder.reset(qubit);
      }
    }
  }

  void visitIfStatement(std::shared_ptr<qasm3::IfStatement> stmt) override {
    if (stmt->thenStatements.empty() && stmt->elseStatements.empty()) {
      throw qasm3::CompilerError(
          "If statements with empty then and else blocks are not supported.",
          stmt->debugInfo);
    }

    auto condition = translateCondition(stmt->condition, stmt->debugInfo);
    auto hasElse = !stmt->elseStatements.empty();

    std::vector<std::shared_ptr<qasm3::Statement>> thenStatements;
    if (stmt->thenStatements.empty()) {
      thenStatements = stmt->elseStatements;
      hasElse = false;
      auto trueValue = builder.boolConstant(true);
      condition =
          arith::XOrIOp::create(builder, condition, trueValue).getResult();
    } else {
      thenStatements = stmt->thenStatements;
    }

    auto ifOp =
        scf::IfOp::create(builder, condition, /*withElseRegion=*/hasElse);

    // Save current insertion point
    OpBuilder::InsertionGuard guard(builder);

    // Then block
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    emitBlockStatements(thenStatements, stmt->debugInfo);

    // Else block
    if (hasElse) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      emitBlockStatements(stmt->elseStatements, stmt->debugInfo);
    }
  }

  //===--- Core gate application ----------------------------------------===//

  /**
   * @brief Apply a GateCallStatement by emitting the corresponding QC
   * operations.
   *
   * @param stmt The GateCallStatement to apply.
   * @param scope The current qubit scope for resolving operands. If called from
   * the main visitor, this is the top-level qubitRegisters map. If called
   * recursively for a compound gate, this is the local scope of the
   * CompoundGate.
   */
  void
  applyGateCallStatement(const std::shared_ptr<qasm3::GateCallStatement>& stmt,
                         const QubitScope& scope) {
    const auto& id = stmt->identifier;
    auto it = gates.find(id);

    // OpenQASM 2 compatibility:
    // Strip leading c characters and treat them as implicit control modifiers
    auto resolvedId = id;
    size_t numCompatControls = 0;
    if (openQASM2CompatMode && it == gates.end()) {
      while (!resolvedId.empty() && resolvedId.front() == 'c') {
        resolvedId = resolvedId.substr(1);
        ++numCompatControls;
      }
      if (numCompatControls > 0) {
        it = gates.find(resolvedId);
      }
    }

    if (it == gates.end()) {
      throw qasm3::CompilerError("No OpenQASM definition found for gate '" +
                                     id + "'.",
                                 stmt->debugInfo);
    }

    // Evaluate parameters to doubles
    SmallVector<double> params;
    params.reserve(stmt->arguments.size());
    for (const auto& arg : stmt->arguments) {
      auto result = constEvalPass.visit(arg);
      if (!result.has_value()) {
        throw qasm3::CompilerError("Gate parameter could not be evaluated.",
                                   stmt->debugInfo);
      }
      params.push_back(result->toExpr()->asFP());
    }

    // Expand operands to MLIR values
    SmallVector<Value> operands;
    SmallVector<SmallVector<Value>> operandsBroadcasting;
    auto broadcasting = false;
    for (const auto& operand : stmt->operands) {
      const auto& resolvedOperand =
          resolveGateOperandInScope(operand, scope, stmt->debugInfo);
      if (const auto* operand = std::get_if<Value>(&resolvedOperand)) {
        operands.push_back(*operand);
      } else if (const auto* operand =
                     std::get_if<SmallVector<Value>>(&resolvedOperand)) {
        operandsBroadcasting.push_back(*operand);
        broadcasting = true;
      }
    }

    if (broadcasting && !operands.empty()) {
      throw qasm3::CompilerError("Gate operands must be single qubits or "
                                 "quantum registers and not a mix of both.",
                                 stmt->debugInfo);
    }

    if (broadcasting && numCompatControls != 0) {
      throw qasm3::CompilerError("OpenQASM 2 gates cannot be broadcasted.",
                                 stmt->debugInfo);
    }

    size_t broadcastWidth = 0;
    if (broadcasting) {
      for (const auto& operand : operandsBroadcasting) {
        if (broadcastWidth == 0) {
          broadcastWidth = operand.size();
        } else if (broadcastWidth != operand.size()) {
          throw qasm3::CompilerError(
              "All broadcasting operands must have the same width.",
              stmt->debugInfo);
        }
      }
    }

    auto invert = false;
    size_t numControls = 0;
    SmallVector<Value> posControls;
    SmallVector<Value> negControls;
    SmallVector<SmallVector<Value>> posControlsBroadcasting;
    SmallVector<SmallVector<Value>> negControlsBroadcasting;

    // Parse modifiers
    for (const auto& mod : stmt->modifiers) {
      if (std::dynamic_pointer_cast<qasm3::InvGateModifier>(mod)) {
        invert = !invert;
      } else if (const auto* ctrlMod =
                     dynamic_cast<qasm3::CtrlGateModifier*>(mod.get())) {
        const auto n =
            evaluatePositiveConstant(ctrlMod->expression, stmt->debugInfo, 1);
        for (size_t i = 0; i < n; ++i, ++numControls) {
          const auto positive = ctrlMod->ctrlType;
          if (!broadcasting) {
            if (numControls >= operands.size()) {
              throw qasm3::CompilerError("Control index out of bounds.",
                                         stmt->debugInfo);
            }
            auto operand = operands[numControls];
            if (positive) {
              posControls.push_back(operand);
            } else {
              negControls.push_back(operand);
            }
          } else {
            if (numControls >= operandsBroadcasting.size()) {
              throw qasm3::CompilerError("Control index out of bounds.",
                                         stmt->debugInfo);
            }
            const auto& operand = operandsBroadcasting[numControls];
            if (positive) {
              posControlsBroadcasting.push_back(operand);
            } else {
              negControlsBroadcasting.push_back(operand);
            }
          }
        }
      } else {
        throw qasm3::CompilerError(
            "Only ctrl, negctrl, and inv modifiers are supported.",
            stmt->debugInfo);
      }
    }

    // OpenQASM 2 compatibility:
    // Append implicit control qubits
    for (size_t i = 0; i < numCompatControls; ++i, ++numControls) {
      if (numControls >= operands.size()) {
        throw qasm3::CompilerError("Control index out of bounds.",
                                   stmt->debugInfo);
      }
      posControls.push_back(operands[numControls]);
    }

    // Remaining operands are target qubits
    SmallVector<Value> targets;
    SmallVector<SmallVector<Value>> targetsBroadcasting;
    if (!broadcasting) {
      targets = llvm::to_vector(llvm::drop_begin(operands, numControls));
    } else {
      targetsBroadcasting =
          llvm::to_vector(llvm::drop_begin(operandsBroadcasting, numControls));
    }

    // Inline compound gate
    if (const auto* compound =
            dynamic_cast<qasm3::CompoundGate*>(it->second.get())) {
      if (broadcasting) {
        throw qasm3::CompilerError(
            "Broadcasted compound gates are not supported.", stmt->debugInfo);
      }
      applyCompoundGate(*compound, params, targets, posControls, negControls,
                        invert, stmt->debugInfo);
      return;
    }

    // Emit standard gate
    const auto dispIt = GATE_DISPATCH.find(resolvedId);
    if (dispIt == GATE_DISPATCH.end()) {
      throw qasm3::CompilerError(
          "No MLIR definition found for gate '" + id + "'.", stmt->debugInfo);
    }

    if (it->second->getNParameters() != params.size()) {
      throw qasm3::CompilerError("Invalid number of parameters for gate '" +
                                     id + "'.",
                                 stmt->debugInfo);
    }

    if (!broadcasting) {
      emitGate(dispIt->second, params, targets, posControls, negControls,
               invert);
    } else {
      for (size_t b = 0; b < broadcastWidth; ++b) {
        SmallVector<Value> bTargets;
        bTargets.reserve(targetsBroadcasting.size());
        for (const auto& target : targetsBroadcasting) {
          bTargets.push_back(target[b]);
        }
        SmallVector<Value> bPosControls;
        bPosControls.reserve(posControlsBroadcasting.size());
        for (const auto& ctrl : posControlsBroadcasting) {
          bPosControls.push_back(ctrl[b]);
        }
        SmallVector<Value> bNegControls;
        bNegControls.reserve(negControlsBroadcasting.size());
        for (const auto& ctrl : negControlsBroadcasting) {
          bNegControls.push_back(ctrl[b]);
        }
        emitGate(dispIt->second, params, bTargets, bPosControls, bNegControls,
                 invert);
      }
    }
  }

  /// Helper function to build a gate with potential modifiers.
  void buildModifiedGate(function_ref<void(ValueRange)> bodyFn,
                         ValueRange targets, ValueRange posControls,
                         ValueRange negControls, bool invert) {
    auto wrappedBodyFn = [&](ValueRange qubits) {
      if (invert) {
        builder.inv(qubits, function_ref<void(ValueRange)>(bodyFn));
      } else {
        bodyFn(qubits);
      }
    };

    if (posControls.empty() && negControls.empty()) {
      wrappedBodyFn(targets);
      return;
    }

    SmallVector<Value> controls;
    controls.append(posControls.begin(), posControls.end());
    controls.append(negControls.begin(), negControls.end());

    for (auto control : negControls) {
      builder.x(control);
    }
    builder.ctrl(controls, targets,
                 function_ref<void(ValueRange)>(wrappedBodyFn));
    for (auto control : negControls) {
      builder.x(control);
    }
  }

  /// Emit a standard gate.
  void emitGate(const GateFn& gateFn, const SmallVector<double>& params,
                ValueRange targets, ValueRange posControls,
                ValueRange negControls, bool invert) {
    auto bodyFn = [&](ValueRange qubits) { gateFn(builder, qubits, params); };
    buildModifiedGate(bodyFn, targets, posControls, negControls, invert);
  }

  /// Inline a compound gate.
  void applyCompoundGate(const qasm3::CompoundGate& gate,
                         const SmallVector<double>& params, ValueRange targets,
                         ValueRange posControls, ValueRange negControls,
                         bool invert,
                         const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    assert(gate.parameterNames.size() == params.size());
    assert(gate.targetNames.size() == targets.size());

    // Map from internal target name to index in targets list. This map is
    // needed because the qubits may be aliased if the CompoundGate is inlined
    // within a modifier region.
    llvm::StringMap<SmallVector<size_t>> targetsMap;

    for (const auto& [targetName, target] :
         llvm::zip_equal(gate.targetNames, targets)) {
      auto it = llvm::find(targets, target);
      if (it == targets.end()) {
        throw qasm3::CompilerError(
            "Target '" + targetName + "' not found in operands.", debugInfo);
      }
      const auto index =
          static_cast<size_t>(std::distance(targets.begin(), it));
      targetsMap[targetName].push_back(index);
    }

    // Bind parameters as constants
    constEvalPass.pushEnv();
    for (size_t i = 0; i < gate.parameterNames.size(); ++i) {
      constEvalPass.addConst(gate.parameterNames[i],
                             qasm3::const_eval::ConstEvalValue(params[i]));
    }

    auto bodyFn = [&](ValueRange qubits) {
      QubitScope localScope;
      for (const auto& [name, indices] : targetsMap) {
        SmallVector<Value> args;
        for (auto index : indices) {
          args.push_back(qubits[index]);
        }
        localScope[name] = std::move(args);
      }
      for (const auto& stmt : gate.body) {
        if (const auto gateCall =
                std::dynamic_pointer_cast<qasm3::GateCallStatement>(stmt)) {
          applyGateCallStatement(gateCall, localScope);
          continue;
        }
        throw qasm3::CompilerError("Compound operations with non-quantum "
                                   "statements are not supported.",
                                   debugInfo);
      }
    };

    buildModifiedGate(bodyFn, targets, posControls, negControls, invert);

    constEvalPass.popEnv();
  }

  //===--- IfStatement helpers ------------------------------------------===//

  /// Helper function to emit quantum statements within an IfOp's then/else
  /// regions.
  void emitBlockStatements(
      const std::vector<std::shared_ptr<qasm3::Statement>>& statements,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    for (const auto& stmt : statements) {
      if (const auto gateCall =
              std::dynamic_pointer_cast<qasm3::GateCallStatement>(stmt)) {
        applyGateCallStatement(gateCall, qubitRegisters);
        continue;
      }
      throw qasm3::CompilerError(
          "If statements with non-quantum statements are not supported.",
          debugInfo);
    }
  }

  /// Translate an OpenQASM 3 condition to MLIR.
  [[nodiscard]] Value
  translateCondition(const std::shared_ptr<qasm3::Expression>& condition,
                     const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    // Single bit (c[0])
    if (const auto& id =
            std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(condition)) {
      return lookupBitValue(id, debugInfo);
    }

    // Unary negation (!c[0] or ~c[0])
    if (const auto unaryExpr =
            std::dynamic_pointer_cast<qasm3::UnaryExpression>(condition)) {
      if (unaryExpr->op != qasm3::UnaryExpression::LogicalNot &&
          unaryExpr->op != qasm3::UnaryExpression::BitwiseNot) {
        throw qasm3::CompilerError(
            "Only ! and ~ are supported in if statements.", debugInfo);
      }
      const auto& id = std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(
          unaryExpr->operand);
      if (!id) {
        throw qasm3::CompilerError("Unary expression has unsupported operand.",
                                   debugInfo);
      }
      auto value = lookupBitValue(id, debugInfo);
      auto trueValue = builder.boolConstant(true);
      return arith::XOrIOp::create(builder, value, trueValue).getResult();
    }

    // Register comparison (creg == N, creg != N, etc.)
    if (const auto binaryExpr =
            std::dynamic_pointer_cast<qasm3::BinaryExpression>(condition)) {
      throw qasm3::CompilerError("Register comparisons are not supported.",
                                 debugInfo);
    }

    throw qasm3::CompilerError(
        "Unsupported condition expression in if statement.", debugInfo);
  }

  /// Look up the most recent measurement result for a classical bit.
  [[nodiscard]] Value
  lookupBitValue(const std::shared_ptr<qasm3::IndexedIdentifier>& id,
                 const std::shared_ptr<qasm3::DebugInfo>& debugInfo) const {
    const auto& regName = id->identifier;
    auto it = bitValues.find(regName);
    if (it == bitValues.end()) {
      throw qasm3::CompilerError("No classical bit of register '" + regName +
                                     "' has been measured yet.",
                                 debugInfo);
    }
    const auto& regBits = it->second;

    if (id->indices.empty()) {
      assert(regBits.size() == 1);
      return regBits[0];
    }

    if (id->indices.size() != 1 ||
        id->indices[0]->indexExpressions.size() != 1) {
      throw qasm3::CompilerError("Only single-index expressions are supported.",
                                 debugInfo);
    }
    const auto& indexExpression = id->indices[0]->indexExpressions[0];
    const auto index = evaluatePositiveConstant(indexExpression, debugInfo);
    if (index >= regBits.size() || !regBits[index]) {
      throw qasm3::CompilerError("Bit " + std::to_string(index) +
                                     " of register '" + regName +
                                     "' has been not measured yet.",
                                 debugInfo);
    }
    return regBits[index];
  }

  //===--- Operand resolution helpers ------------------------------------===//

  /**
   * @brief Resolve a qubit operand against the top-level qubitRegisters map.
   *
   * @return A variant containing
   * - a `Value` if the operand is, e.g., `q[0]`,
   * - a `Value` if the operand `q` is a single-qubit register, or
   * - a `SmallVector<Value>` if the operand `q` is a multi-qubit register.
   */
  [[nodiscard]] std::variant<Value, SmallVector<Value>>
  resolveGateOperand(const std::shared_ptr<qasm3::GateOperand>& operand,
                     const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    return resolveGateOperandInScope(operand, qubitRegisters, debugInfo);
  }

  /**
   * @brief Resolve a qubit operand against @p scope.
   *
   * @return A variant containing
   * - a `Value` if the operand is, e.g., `q[0]`,
   * - a `Value` if the operand `q` is a single-qubit register, or
   * - a `SmallVector<Value>` if the operand `q` is a multi-qubit register.
   */
  [[nodiscard]] std::variant<Value, SmallVector<Value>>
  resolveGateOperandInScope(
      const std::shared_ptr<qasm3::GateOperand>& operand,
      const QubitScope& scope,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    if (operand->isHardwareQubit()) {
      return builder.staticQubit(operand->getHardwareQubit());
    }

    const auto& id = operand->getIdentifier();
    const auto& name = id->identifier;
    auto it = scope.find(name);
    if (it == scope.end()) {
      throw qasm3::CompilerError("Unknown qubit register '" + name + "'.",
                                 debugInfo);
    }

    const auto& qubits = it->second;

    if (id->indices.empty()) {
      if (qubits.size() == 1) {
        return qubits[0];
      }
      // Return full register
      return qubits;
    }

    if (id->indices.size() != 1 ||
        id->indices[0]->indexExpressions.size() != 1) {
      throw qasm3::CompilerError("Only single-index expressions are supported.",
                                 debugInfo);
    }
    const auto& indexExpression = id->indices[0]->indexExpressions[0];
    const auto index = evaluatePositiveConstant(indexExpression, debugInfo);
    if (index >= qubits.size()) {
      throw qasm3::CompilerError("Qubit index out of bounds.", debugInfo);
    }
    return qubits[index];
  }

  /// Resolve a classical bit operand.
  [[nodiscard]] SmallVector<QCProgramBuilder::Bit> resolveClassicalBits(
      const std::shared_ptr<qasm3::IndexedIdentifier>& operand,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) const {
    const auto& name = operand->identifier;
    auto it = classicalRegisters.find(name);
    if (it == classicalRegisters.end()) {
      throw qasm3::CompilerError("Unknown classical register '" + name + "'.",
                                 debugInfo);
    }

    const auto& creg = it->second;
    SmallVector<QCProgramBuilder::Bit> bits;

    if (operand->indices.empty()) {
      for (int64_t i = 0; i < creg.size; ++i) {
        bits.push_back(creg[i]);
      }
      return bits;
    }

    if (operand->indices.size() != 1 ||
        operand->indices[0]->indexExpressions.size() != 1) {
      throw qasm3::CompilerError("Only single-index expressions are supported.",
                                 debugInfo);
    }
    const auto& indexExpression = operand->indices[0]->indexExpressions[0];
    const auto index = evaluatePositiveConstant(indexExpression, debugInfo);
    if (std::cmp_greater_equal(index, creg.size)) {
      throw qasm3::CompilerError("Classical bit index out of bounds.",
                                 debugInfo);
    }
    bits.push_back(creg[static_cast<int64_t>(index)]);
    return bits;
  }

  /// Evaluate a constant expression to a positive integer.
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

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  try {
    auto buffer =
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer();
    std::string_view view(buffer.data(), buffer.size());
    std::istringstream input((std::string(view)));

    qasm3::Parser parser(input);
    const auto program = parser.parseProgram();

    MLIRQasmImporter importer(context);
    importer.visitProgram(program);
    return importer.finalize();
  } catch (const qasm3::CompilerError& e) {
    llvm::errs() << "Import error: " << e.what() << "\n";
    return nullptr;
  } catch (const std::exception& e) {
    llvm::errs() << "Import error: " << e.what() << "\n";
    return nullptr;
  }
}

OwningOpRef<ModuleOp> translateQASM3ToQC(StringRef source,
                                         MLIRContext* context) {
  llvm::SourceMgr sourceMgr;
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(source);
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());
  return translateQASM3ToQC(sourceMgr, context);
}

} // namespace mlir::qc
