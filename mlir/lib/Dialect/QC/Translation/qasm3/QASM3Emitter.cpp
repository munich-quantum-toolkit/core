/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/qasm3/QASM3Emitter.h"

#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/qasm3/QASM3Lexer.h"
#include "mlir/Dialect/QC/Translation/qasm3/QASM3Parser.h"
#include "qasm3/Gate.hpp"
#include "qasm3/StdGates.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qc {

namespace {

using qasm3::GateInfo;

using detail::Condition;
using detail::Expr;
using detail::GateCall;
using detail::Lexer;
using detail::Modifier;
using detail::Operand;
using detail::Parser;
using detail::Reference;

/// Signature: (builder, gate operands, gate parameters).
using GateFn = std::function<void(QCProgramBuilder&, ValueRange, ValueRange)>;

/// A stored compound-gate definition. Array members are bump-allocated by the
/// parser and remain valid for the duration of the import.
struct StoredGate {
  llvm::ArrayRef<llvm::StringRef> parameters;
  llvm::ArrayRef<llvm::StringRef> targets;
  llvm::ArrayRef<GateCall> body;
};

/**
 * @brief A named qubit binding in scope.
 *
 * @details
 * For top-level registers, `memref` holds the backing register and `qubits`
 * holds the eagerly extracted values. For compound gates, `memref` is null and
 * `qubits` holds the aliased values.
 */
struct QubitBinding {
  Value memref;
  SmallVector<Value> qubits;
};

using QubitScope = llvm::StringMap<QubitBinding>;

/// Look up a built-in numeric constant and emit it as an `f64`-typed value.
std::optional<Value> lookupBuiltinConstant(llvm::StringRef name,
                                           QCProgramBuilder& builder) {
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, builder.getF64FloatAttr(value))
        .getResult();
  };
  if (name == "pi" || name == "π") {
    return constant(::qc::PI);
  }
  if (name == "tau" || name == "τ") {
    return constant(::qc::TAU);
  }
  if (name == "euler" || name == "ℇ") {
    return constant(::qc::E);
  }
  return std::nullopt;
}

/// Build the table mapping each gate identifier to a `QCProgramBuilder`
/// emitter.
llvm::StringMap<GateFn> buildGateDispatch() {
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

/// Map from gate identifier to `QCProgramBuilder` emitter.
const llvm::StringMap<GateFn> GATE_DISPATCH = buildGateDispatch();

/// Build the table mapping a gate identifier to its metadata.
llvm::StringMap<std::variant<GateInfo, StoredGate>> buildGateTable() {
  llvm::StringMap<std::variant<GateInfo, StoredGate>> t;
  for (const auto& [name, gate] : qasm3::STANDARD_GATES) {
    const auto* standard = dynamic_cast<qasm3::StandardGate*>(gate.get());
    assert(standard != nullptr && "STANDARD_GATES entry is not a StandardGate");
    t.insert({name, standard->info});
  }

  const GateInfo mcxInfo{
      .nControls = 0, .nTargets = 0, .nParameters = 0, .type = ::qc::OpType::X};
  t["mcx"] = mcxInfo;
  t["mcx_gray"] = mcxInfo;
  t["mcx_vchain"] = mcxInfo;
  t["mcx_recursive"] = mcxInfo;

  t["mcphase"] = GateInfo{
      .nControls = 0, .nTargets = 0, .nParameters = 1, .type = ::qc::OpType::P};

  return t;
}

//===----------------------------------------------------------------------===//
// QCEmitter
//===----------------------------------------------------------------------===//

/**
 * @brief Lowers OpenQASM 3 parse events to QC-dialect operations.
 *
 * @details
 * Models the sink concept consumed by `Parser`. All events emit eagerly via the
 * `QCProgramBuilder` and report errors as diagnostics through `LogicalResult`;
 * name resolution, semantic validation, and target-specific restrictions are
 * handled here.
 */
class QCEmitter {
public:
  QCEmitter(MLIRContext* ctx, llvm::SourceMgr& sourceMgr)
      : builder(ctx), sourceMgr(sourceMgr), gates(buildGateTable()) {}

  void initialize() { builder.initialize(); }
  OwningOpRef<ModuleOp> finalize() { return builder.finalize(); }

  //===--- Diagnostics --------------------------------------------------===//

  LogicalResult error(llvm::SMLoc loc, const Twine& message) {
    return emitError(translate(loc), message);
  }

  //===--- Program events -----------------------------------------------===//

  LogicalResult version(double /*version*/) {
    // The version declaration has no effect on translation.
    return success();
  }

  LogicalResult include(llvm::SMLoc loc, llvm::StringRef filename) {
    if (filename != "stdgates.inc" && filename != "qelib1.inc") {
      return error(loc, "unsupported include '" + filename +
                            "'; only 'stdgates.inc' and 'qelib1.inc' are "
                            "supported");
    }
    return success();
  }

  LogicalResult scalarDecl(llvm::SMLoc loc, detail::ScalarType type,
                           llvm::StringRef id, const Expr* initializer) {
    if (failed(declare(loc, id))) {
      return failure();
    }
    switch (type) {
    case detail::ScalarType::Float:
      variableKinds[id] = VariableKind::Float;
      break;
    case detail::ScalarType::Int:
      variableKinds[id] = VariableKind::Int;
      break;
    }
    if (initializer != nullptr) {
      return assignScalar(id, *initializer);
    }
    return success();
  }

  LogicalResult boolDecl(llvm::SMLoc loc, llvm::StringRef id,
                         const Condition* initializer) {
    if (failed(declare(loc, id))) {
      return failure();
    }
    variableKinds[id] = VariableKind::Bool;
    if (initializer != nullptr) {
      return assignBool(id, *initializer);
    }
    return success();
  }

  LogicalResult scalarAssign(llvm::SMLoc /*loc*/, llvm::StringRef id,
                             const Expr& value) {
    return assignScalar(id, value);
  }

  LogicalResult boolAssign(llvm::SMLoc /*loc*/, llvm::StringRef id,
                           const Condition& value) {
    return assignBool(id, value);
  }

  [[nodiscard]] std::optional<detail::ClassicalKind>
  classicalKind(llvm::StringRef id) const {
    auto it = variableKinds.find(id);
    if (it == variableKinds.end()) {
      return std::nullopt;
    }
    return it->second == VariableKind::Bool ? detail::ClassicalKind::Boolean
                                            : detail::ClassicalKind::Numeric;
  }

  /// Bind (or rebind) a numeric scalar `id` to the value of @p value. Shared by
  /// `scalarDecl` and `scalarAssign`.
  LogicalResult assignScalar(llvm::StringRef id, const Expr& value) {
    if (variableKinds.lookup(id) == VariableKind::Float) {
      auto emitted = emitAngle(value);
      if (failed(emitted)) {
        return failure();
      }
      parameterConstants.insert(id, *emitted);
      return success();
    }
    auto emitted = evaluateConstant(value);
    if (failed(emitted)) {
      return failure();
    }
    integerConstants.insert(id, *emitted);
    return success();
  }

  /// Bind (or rebind) a boolean scalar `id`. Shared by `boolDecl` and
  /// `boolAssign`.
  LogicalResult assignBool(llvm::StringRef id, const Condition& value) {
    auto emitted = emitCondition(value);
    if (failed(emitted)) {
      return failure();
    }
    booleanConstants.insert(id, *emitted);
    return success();
  }

  LogicalResult qubitRegister(llvm::SMLoc loc, llvm::StringRef id,
                              const Expr* size) {
    if (failed(declare(loc, id))) {
      return failure();
    }
    if (size != nullptr) {
      auto count = evaluateConstant(*size);
      if (failed(count)) {
        return failure();
      }
      const auto reg = builder.allocQubitRegister(*count);
      qubitRegisters[id] = {.memref = reg.value, .qubits = reg.qubits};
    } else {
      qubitRegisters[id] = {.memref = nullptr,
                            .qubits = {builder.allocQubit()}};
    }
    return success();
  }

  LogicalResult classicalRegister(llvm::SMLoc loc, llvm::StringRef id,
                                  const Expr* size) {
    if (failed(declare(loc, id))) {
      return failure();
    }
    int64_t count = 1;
    if (size != nullptr) {
      auto evaluated = evaluateConstant(*size);
      if (failed(evaluated)) {
        return failure();
      }
      count = *evaluated;
    }
    classicalRegisters[id] = builder.allocClassicalBitRegister(count, id.str());
    return success();
  }

  //===--- Measurement, reset, barrier ----------------------------------===//

  LogicalResult measure(llvm::SMLoc loc, const Reference& target,
                        const Operand& operand) {
    auto bits = resolveClassicalBits(target);
    if (failed(bits)) {
      return failure();
    }
    auto resolved = resolveOperand(operand, qubitRegisters);
    if (failed(resolved)) {
      return failure();
    }
    SmallVector<Value> qubits;
    if (std::holds_alternative<Value>(*resolved)) {
      qubits.push_back(std::get<Value>(*resolved));
    } else {
      qubits = std::get<SmallVector<Value>>(*resolved);
    }
    if (bits->size() != qubits.size()) {
      return error(loc, "the classical register and the quantum register must "
                        "have the same width");
    }
    for (const auto& [bit, qubit] : llvm::zip_equal(*bits, qubits)) {
      auto result = MeasureOp::create(
                        builder, qubit, builder.getStringAttr(bit.registerName),
                        builder.getI64IntegerAttr(bit.registerSize),
                        builder.getI64IntegerAttr(bit.registerIndex))
                        .getResult();
      auto& registerBits = bitValues[bit.registerName];
      const auto index = static_cast<size_t>(bit.registerIndex);
      if (registerBits.size() <= index) {
        registerBits.resize(index + 1);
      }
      registerBits[index] = result;
    }
    return success();
  }

  LogicalResult reset(llvm::SMLoc /*loc*/, const Operand& operand) {
    auto resolved = resolveOperand(operand, qubitRegisters);
    if (failed(resolved)) {
      return failure();
    }
    if (std::holds_alternative<Value>(*resolved)) {
      builder.reset(std::get<Value>(*resolved));
    } else {
      for (auto qubit : std::get<SmallVector<Value>>(*resolved)) {
        builder.reset(qubit);
      }
    }
    return success();
  }

  LogicalResult barrier(llvm::SMLoc /*loc*/, llvm::ArrayRef<Operand> operands) {
    SmallVector<Value> qubits;
    for (const auto& operand : operands) {
      auto resolved = resolveOperand(operand, qubitRegisters);
      if (failed(resolved)) {
        return failure();
      }
      if (std::holds_alternative<Value>(*resolved)) {
        qubits.push_back(std::get<Value>(*resolved));
      } else {
        llvm::append_range(qubits, std::get<SmallVector<Value>>(*resolved));
      }
    }
    builder.barrier(qubits);
    return success();
  }

  //===--- Gate definitions and calls -----------------------------------===//

  LogicalResult gateDefinition(llvm::SMLoc loc, llvm::StringRef id,
                               llvm::ArrayRef<llvm::StringRef> parameters,
                               llvm::ArrayRef<llvm::StringRef> targets,
                               llvm::ArrayRef<GateCall> body) {
    if (gates.contains(id)) {
      return error(loc, "gate '" + id + "' already declared");
    }
    for (size_t i = 0; i < parameters.size(); ++i) {
      for (size_t j = i + 1; j < parameters.size(); ++j) {
        if (parameters[i] == parameters[j]) {
          return error(loc, "parameter is already declared in compound gate");
        }
      }
    }
    for (size_t i = 0; i < targets.size(); ++i) {
      for (size_t j = i + 1; j < targets.size(); ++j) {
        if (targets[i] == targets[j]) {
          return error(loc, "target is already declared in compound gate");
        }
      }
    }
    gates[id] =
        StoredGate{.parameters = parameters, .targets = targets, .body = body};
    return success();
  }

  LogicalResult gateCall(const GateCall& call) {
    return emitGateCall(call, qubitRegisters);
  }

  //===--- Control flow -------------------------------------------------===//

  LogicalResult conditionOnly(llvm::SMLoc /*loc*/, const Condition& condition) {
    return LogicalResult{emitCondition(condition)};
  }

  struct IfScope {
    scf::IfOp op;
    OpBuilder::InsertPoint savedInsertionPoint;
  };

  FailureOr<IfScope> ifBegin(llvm::SMLoc /*loc*/, const Condition& condition,
                             bool invert) {
    auto value = emitCondition(condition);
    if (failed(value)) {
      return failure();
    }
    Value condValue = *value;
    if (invert) {
      auto trueValue = builder.boolConstant(true);
      condValue =
          arith::XOrIOp::create(builder, condValue, trueValue).getResult();
    }
    auto ifOp = scf::IfOp::create(builder, condValue, /*withElseRegion=*/false);
    IfScope scope{.op = ifOp,
                  .savedInsertionPoint = builder.saveInsertionPoint()};
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    return scope;
  }

  LogicalResult ifElse(IfScope& scope) {
    auto* elseBlock = builder.createBlock(&scope.op.getElseRegion());
    builder.setInsertionPointToStart(elseBlock);
    return success();
  }

  LogicalResult ifEnd(IfScope& scope, bool hadElse) {
    if (hadElse) {
      scf::YieldOp::create(builder);
    }
    builder.restoreInsertionPoint(scope.savedInsertionPoint);
    return success();
  }

  LogicalResult forStmt(llvm::SMLoc loc, llvm::StringRef variable,
                        const Expr& start, const Expr& step, const Expr& stop,
                        function_ref<LogicalResult()> body) {
    auto stepConstant = evaluateConstant(step);
    if (failed(stepConstant)) {
      return failure();
    }
    if (*stepConstant < 1) {
      return error(loc, "for loops with a non-positive step are not supported");
    }

    auto startValue = emitIndex(start);
    auto stepValue = emitIndex(step);
    auto stopValue = emitIndex(stop);
    if (failed(startValue) || failed(stepValue) || failed(stopValue)) {
      return failure();
    }

    // OpenQASM 3's range is inclusive of the stop value.
    auto one =
        arith::ConstantOp::create(builder, builder.getIndexAttr(1)).getResult();
    Value inclusiveStop =
        arith::AddIOp::create(builder, *stopValue, one).getResult();

    ValueScope loopScope(loopVariables);
    ValueScope loadScope(dynamicallyLoadedQubits);

    LogicalResult status = success();
    builder.scfFor(*startValue, inclusiveStop, *stepValue, [&](Value iv) {
      loopVariables.insert(variable, iv);
      if (succeeded(status)) {
        status = body();
      }
    });
    return status;
  }

  LogicalResult whileStmt(llvm::SMLoc /*loc*/, const Condition& condition,
                          function_ref<LogicalResult()> body) {
    LogicalResult status = success();
    builder.scfWhile(
        [&] {
          ValueScope loadScope(dynamicallyLoadedQubits);
          auto value = emitCondition(condition);
          if (failed(value)) {
            status = failure();
            builder.scfCondition(builder.boolConstant(false));
          } else {
            builder.scfCondition(*value);
          }
        },
        [&] {
          ValueScope loadScope(dynamicallyLoadedQubits);
          if (succeeded(status)) {
            status = body();
          }
        });
    return status;
  }

private:
  using ScopedValueTable = llvm::ScopedHashTable<llvm::StringRef, Value>;
  using ValueScope = llvm::ScopedHashTableScope<llvm::StringRef, Value>;
  using IntegerTable = llvm::ScopedHashTable<llvm::StringRef, int64_t>;
  using IntegerScope = llvm::ScopedHashTableScope<llvm::StringRef, int64_t>;

  /// The declared type of a classical scalar variable, tracked so that
  /// assignments know how to lower their right-hand side.
  enum class VariableKind : uint8_t { Float, Int, Bool };

  //===--- Locations ----------------------------------------------------===//

  Location translate(llvm::SMLoc loc) {
    if (!loc.isValid()) {
      return UnknownLoc::get(builder.getContext());
    }
    const auto [line, col] = sourceMgr.getLineAndColumn(loc);
    return FileLineColLoc::get(builder.getStringAttr("<qasm3>"), line, col);
  }

  //===--- Symbol declarations ------------------------------------------===//

  LogicalResult declare(llvm::SMLoc loc, llvm::StringRef id) {
    if (!declaredNames.insert(id).second) {
      return error(loc, "identifier '" + id + "' already declared");
    }
    return success();
  }

  //===--- Conditions ---------------------------------------------------===//

  FailureOr<Value> emitCondition(const Condition& condition) {
    switch (condition.kind) {
    case Condition::Kind::Measurement: {
      auto resolved = resolveOperand(condition.operand, qubitRegisters);
      if (failed(resolved)) {
        return failure();
      }
      if (!std::holds_alternative<Value>(*resolved)) {
        return error(condition.loc,
                     "measurement condition must be a single qubit");
      }
      return builder.measure(std::get<Value>(*resolved));
    }
    case Condition::Kind::Bit:
      // A bare identifier may name a declared `bool`; otherwise it is a
      // classical bit reference.
      if (condition.bit.index == nullptr) {
        if (Value value = booleanConstants.lookup(condition.bit.identifier)) {
          return value;
        }
      }
      return lookupBitValue(condition.bit);
    case Condition::Kind::Literal:
      return builder.boolConstant(condition.literalValue);
    case Condition::Kind::Not: {
      auto value = emitCondition(*condition.lhs);
      if (failed(value)) {
        return failure();
      }
      auto trueValue = builder.boolConstant(true);
      return arith::XOrIOp::create(builder, *value, trueValue).getResult();
    }
    case Condition::Kind::And:
    case Condition::Kind::Or: {
      auto lhs = emitCondition(*condition.lhs);
      auto rhs = emitCondition(*condition.rhs);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      if (condition.kind == Condition::Kind::And) {
        return arith::AndIOp::create(builder, *lhs, *rhs).getResult();
      }
      return arith::OrIOp::create(builder, *lhs, *rhs).getResult();
    }
    }
    llvm_unreachable("unknown condition kind");
  }

  FailureOr<Value> lookupBitValue(const Reference& bit) {
    auto it = bitValues.find(bit.identifier);
    if (it == bitValues.end()) {
      return error(bit.loc, "no classical bit of register '" + bit.identifier +
                                "' has been measured yet");
    }
    const auto& registerBits = it->second;

    if (bit.index == nullptr) {
      return registerBits[0];
    }
    auto index = evaluateConstant(*bit.index);
    if (failed(index)) {
      return failure();
    }
    const auto i = static_cast<size_t>(*index);
    if (i >= registerBits.size() || !registerBits[i]) {
      return error(bit.loc, "bit " + Twine(*index) + " of register '" +
                                bit.identifier + "' has not been measured yet");
    }
    return registerBits[i];
  }

  //===--- Gate calls ---------------------------------------------------===//

  template <typename T> struct ControlsAndTargets {
    bool invert = false;
    SmallVector<T> posControls;
    SmallVector<T> negControls;
    SmallVector<T> targets;
  };

  FailureOr<size_t> controlCount(const Expr* argument) {
    if (argument == nullptr) {
      return static_cast<size_t>(1);
    }
    auto value = evaluateConstant(*argument);
    if (failed(value)) {
      return failure();
    }
    return static_cast<size_t>(*value);
  }

  template <typename T>
  FailureOr<ControlsAndTargets<T>>
  splitControlsAndTargets(llvm::ArrayRef<Modifier> modifiers,
                          const SmallVector<T>& operands, llvm::SMLoc loc) {
    ControlsAndTargets<T> result;
    size_t numControls = 0;
    for (const auto& modifier : modifiers) {
      switch (modifier.kind) {
      case Modifier::Kind::Inv:
        result.invert = !result.invert;
        break;
      case Modifier::Kind::Pow:
        return error(loc, "power modifiers are not supported yet");
      case Modifier::Kind::Ctrl:
      case Modifier::Kind::NegCtrl: {
        auto count = controlCount(modifier.argument);
        if (failed(count)) {
          return failure();
        }
        auto& controls = modifier.kind == Modifier::Kind::Ctrl
                             ? result.posControls
                             : result.negControls;
        for (size_t i = 0; i < *count; ++i, ++numControls) {
          if (numControls >= operands.size()) {
            return error(loc, "control index out of bounds");
          }
          controls.push_back(operands[numControls]);
        }
        break;
      }
      }
    }
    result.targets = llvm::to_vector(llvm::drop_begin(operands, numControls));
    return result;
  }

  FailureOr<const GateFn*> resolveStandardGate(const GateInfo& gate,
                                               llvm::StringRef id,
                                               ValueRange parameters,
                                               llvm::SMLoc loc) {
    const auto it = GATE_DISPATCH.find(id);
    if (it == GATE_DISPATCH.end()) {
      return error(loc, "no MLIR definition found for gate '" + id + "'");
    }
    if (gate.nParameters != parameters.size()) {
      return error(loc, "invalid number of parameters for gate '" + id + "'");
    }
    return &it->second;
  }

  void emitModifiedGate(function_ref<void(ValueRange)> bodyFn,
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

  void emitStandardGate(const GateFn& gateFn, ValueRange parameters,
                        ValueRange targets, ValueRange posControls,
                        ValueRange negControls, bool invert) {
    auto bodyFn = [&](ValueRange qubits) {
      gateFn(builder, qubits, parameters);
    };
    emitModifiedGate(bodyFn, targets, posControls, negControls, invert);
  }

  LogicalResult emitCompoundGate(const StoredGate& gate, ValueRange parameters,
                                 ValueRange targets, ValueRange posControls,
                                 ValueRange negControls, bool invert,
                                 llvm::SMLoc loc) {
    if (gate.parameters.size() != parameters.size() ||
        gate.targets.size() != targets.size()) {
      return error(loc, "invalid number of arguments for compound gate");
    }

    // Map each internal target name to its position(s) in the operand list.
    // Positions may repeat if the qubits are aliased within a modifier region.
    llvm::StringMap<SmallVector<size_t>> targetsMap;
    for (const auto& [targetName, target] :
         llvm::zip_equal(gate.targets, targets)) {
      auto iter = llvm::find(targets, target);
      if (iter == targets.end()) {
        return error(loc, "target '" + targetName + "' not found in operands");
      }
      targetsMap[targetName].push_back(
          static_cast<size_t>(std::distance(targets.begin(), iter)));
    }

    ValueScope parameterScope(parameterConstants);
    for (const auto& [name, value] :
         llvm::zip_equal(gate.parameters, parameters)) {
      parameterConstants.insert(name, value);
    }

    LogicalResult status = success();
    auto bodyFn = [&](ValueRange qubits) {
      QubitScope localScope;
      for (const auto& [name, indices] : targetsMap) {
        SmallVector<Value> args;
        for (auto index : indices) {
          args.push_back(qubits[index]);
        }
        localScope[name] = {.memref = nullptr, .qubits = std::move(args)};
      }
      for (const auto& bodyCall : gate.body) {
        if (succeeded(status)) {
          status = emitGateCall(bodyCall, localScope);
        }
      }
    };
    emitModifiedGate(bodyFn, targets, posControls, negControls, invert);
    return status;
  }

  LogicalResult emitBroadcastedGateCall(
      const GateInfo& gate, llvm::StringRef id, ValueRange parameters,
      llvm::ArrayRef<Modifier> modifiers,
      const SmallVector<SmallVector<Value>>& operands, llvm::SMLoc loc) {
    const auto broadcastWidth = operands.front().size();
    for (const auto& operand : operands) {
      if (operand.size() != broadcastWidth) {
        return error(loc, "all broadcasting operands must have the same width");
      }
    }

    auto split =
        splitControlsAndTargets<SmallVector<Value>>(modifiers, operands, loc);
    if (failed(split)) {
      return failure();
    }
    auto gateFn = resolveStandardGate(gate, id, parameters, loc);
    if (failed(gateFn)) {
      return failure();
    }

    for (size_t b = 0; b < broadcastWidth; ++b) {
      const auto slice = [&](const SmallVector<SmallVector<Value>>& lists) {
        SmallVector<Value> qubits;
        qubits.reserve(lists.size());
        for (const auto& list : lists) {
          qubits.push_back(list[b]);
        }
        return qubits;
      };
      emitStandardGate(**gateFn, parameters, slice(split->targets),
                       slice(split->posControls), slice(split->negControls),
                       split->invert);
    }
    return success();
  }

  LogicalResult emitGateCall(const GateCall& call, const QubitScope& scope) {
    auto it = gates.find(call.identifier);
    if (it == gates.end()) {
      return error(call.loc, "no OpenQASM definition found for gate '" +
                                 call.identifier + "'");
    }

    SmallVector<Value> parameters;
    parameters.reserve(call.parameters.size());
    for (const auto* argument : call.parameters) {
      auto value = emitAngle(*argument);
      if (failed(value)) {
        return failure();
      }
      parameters.push_back(*value);
    }

    SmallVector<Value> operands;
    SmallVector<SmallVector<Value>> broadcasting;
    for (const auto& operand : call.operands) {
      auto resolved = resolveOperand(operand, scope);
      if (failed(resolved)) {
        return failure();
      }
      if (const auto* value = std::get_if<Value>(&*resolved)) {
        operands.push_back(*value);
      } else {
        broadcasting.push_back(std::get<SmallVector<Value>>(*resolved));
      }
    }

    if (!broadcasting.empty() && !operands.empty()) {
      return error(call.loc, "gate operands must be single qubits or quantum "
                             "registers and not a mix of both");
    }

    if (!broadcasting.empty()) {
      if (std::holds_alternative<StoredGate>(it->second)) {
        return error(call.loc,
                     "broadcasted compound gates are not supported yet");
      }
      return emitBroadcastedGateCall(std::get<GateInfo>(it->second),
                                     call.identifier, parameters,
                                     call.modifiers, broadcasting, call.loc);
    }

    auto split =
        splitControlsAndTargets<Value>(call.modifiers, operands, call.loc);
    if (failed(split)) {
      return failure();
    }

    if (const auto* compound = std::get_if<StoredGate>(&it->second)) {
      return emitCompoundGate(*compound, parameters, split->targets,
                              split->posControls, split->negControls,
                              split->invert, call.loc);
    }

    const auto& gate = std::get<GateInfo>(it->second);
    auto gateFn =
        resolveStandardGate(gate, call.identifier, parameters, call.loc);
    if (failed(gateFn)) {
      return failure();
    }
    emitStandardGate(**gateFn, parameters, split->targets, split->posControls,
                     split->negControls, split->invert);
    return success();
  }

  //===--- Operand resolution -------------------------------------------===//

  FailureOr<std::variant<Value, SmallVector<Value>>>
  resolveOperand(const Operand& operand, const QubitScope& scope) {
    if (operand.hardwareQubit) {
      return std::variant<Value, SmallVector<Value>>{
          builder.staticQubit(*operand.hardwareQubit)};
    }

    auto it = scope.find(operand.identifier);
    if (it == scope.end()) {
      return error(operand.loc,
                   "unknown qubit register '" + operand.identifier + "'");
    }
    const auto& binding = it->second;

    if (operand.index == nullptr) {
      if (binding.qubits.size() == 1) {
        return std::variant<Value, SmallVector<Value>>{binding.qubits[0]};
      }
      return std::variant<Value, SmallVector<Value>>{binding.qubits};
    }

    const auto& indexExpr = *operand.index;
    if (isConstantIndex(indexExpr)) {
      auto index = evaluateConstant(indexExpr);
      if (failed(index)) {
        return failure();
      }
      const auto i = static_cast<size_t>(*index);
      if (i >= binding.qubits.size()) {
        return error(operand.loc, "qubit index out of bounds");
      }
      return std::variant<Value, SmallVector<Value>>{binding.qubits[i]};
    }

    if (!binding.memref) {
      return error(operand.loc,
                   "dynamic qubit indexing requires a qubit register");
    }
    auto loaded =
        loadDynamicElement(operand.identifier, binding.memref, indexExpr);
    if (failed(loaded)) {
      return failure();
    }
    return std::variant<Value, SmallVector<Value>>{*loaded};
  }

  bool isConstantIndex(const Expr& expr) const {
    switch (expr.kind) {
    case Expr::Kind::Int:
    case Expr::Kind::Float:
      return true;
    case Expr::Kind::Ident:
      // A declared integer constant folds; a loop variable does not.
      return integerConstants.count(expr.ident) != 0;
    case Expr::Kind::Neg:
      return isConstantIndex(*expr.lhs);
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div:
      return isConstantIndex(*expr.lhs) && isConstantIndex(*expr.rhs);
    }
    llvm_unreachable("unknown expression kind");
  }

  static std::string indexKey(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return std::to_string(expr.intValue);
    case Expr::Kind::Float:
      return std::to_string(expr.floatValue);
    case Expr::Kind::Ident:
      return expr.ident.str();
    case Expr::Kind::Neg:
      return "(-" + indexKey(*expr.lhs) + ")";
    case Expr::Kind::Add:
      return "(" + indexKey(*expr.lhs) + "+" + indexKey(*expr.rhs) + ")";
    case Expr::Kind::Sub:
      return "(" + indexKey(*expr.lhs) + "-" + indexKey(*expr.rhs) + ")";
    case Expr::Kind::Mul:
      return "(" + indexKey(*expr.lhs) + "*" + indexKey(*expr.rhs) + ")";
    case Expr::Kind::Div:
      return "(" + indexKey(*expr.lhs) + "/" + indexKey(*expr.rhs) + ")";
    }
    llvm_unreachable("unknown expression kind");
  }

  FailureOr<Value> loadDynamicElement(llvm::StringRef name, Value memref,
                                      const Expr& indexExpr) {
    const std::string key = name.str() + "[" + indexKey(indexExpr) + "]";
    if (Value cached = dynamicallyLoadedQubits.lookup(key)) {
      return cached;
    }
    auto index = emitIndex(indexExpr);
    if (failed(index)) {
      return failure();
    }
    Value loaded = builder.memrefLoad(memref, *index);
    dynamicallyLoadedQubits.insert(keySaver.save(key), loaded);
    return loaded;
  }

  FailureOr<SmallVector<QCProgramBuilder::Bit>>
  resolveClassicalBits(const Reference& reference) {
    auto it = classicalRegisters.find(reference.identifier);
    if (it == classicalRegisters.end()) {
      return error(reference.loc,
                   "unknown classical register '" + reference.identifier + "'");
    }
    const auto& creg = it->second;

    SmallVector<QCProgramBuilder::Bit> bits;
    if (reference.index == nullptr) {
      for (int64_t i = 0; i < creg.size; ++i) {
        bits.push_back(creg[i]);
      }
      return bits;
    }
    auto index = evaluateConstant(*reference.index);
    if (failed(index)) {
      return failure();
    }
    if (*index < 0 || *index >= creg.size) {
      return error(reference.loc, "classical bit index out of bounds");
    }
    bits.push_back(creg[*index]);
    return bits;
  }

  //===--- Expression lowering ------------------------------------------===//

  FailureOr<Value> emitAngle(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return arith::ConstantOp::create(
                 builder,
                 builder.getF64FloatAttr(static_cast<double>(expr.intValue)))
          .getResult();
    case Expr::Kind::Float:
      return arith::ConstantOp::create(builder,
                                       builder.getF64FloatAttr(expr.floatValue))
          .getResult();
    case Expr::Kind::Ident:
      return resolveParameter(expr.ident, expr.loc);
    case Expr::Kind::Neg: {
      auto operand = emitAngle(*expr.lhs);
      if (failed(operand)) {
        return failure();
      }
      return arith::NegFOp::create(builder, *operand).getResult();
    }
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div: {
      auto lhs = emitAngle(*expr.lhs);
      auto rhs = emitAngle(*expr.rhs);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      switch (expr.kind) {
      case Expr::Kind::Add:
        return arith::AddFOp::create(builder, *lhs, *rhs).getResult();
      case Expr::Kind::Sub:
        return arith::SubFOp::create(builder, *lhs, *rhs).getResult();
      case Expr::Kind::Mul:
        return arith::MulFOp::create(builder, *lhs, *rhs).getResult();
      default:
        return arith::DivFOp::create(builder, *lhs, *rhs).getResult();
      }
    }
    }
    llvm_unreachable("unknown expression kind");
  }

  FailureOr<Value> resolveParameter(llvm::StringRef name, llvm::SMLoc loc) {
    if (Value value = parameterConstants.lookup(name)) {
      return value;
    }
    // An integer constant used as an angle is materialized as an `f64`.
    if (integerConstants.count(name) != 0) {
      return arith::ConstantOp::create(
                 builder, builder.getF64FloatAttr(static_cast<double>(
                              integerConstants.lookup(name))))
          .getResult();
    }
    if (auto value = lookupBuiltinConstant(name, builder)) {
      return *value;
    }
    return error(loc, "unknown identifier '" + name + "'");
  }

  FailureOr<Value> emitIndex(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return arith::ConstantOp::create(builder,
                                       builder.getIndexAttr(expr.intValue))
          .getResult();
    case Expr::Kind::Neg: {
      auto operand = emitIndex(*expr.lhs);
      if (failed(operand)) {
        return failure();
      }
      auto zero = arith::ConstantOp::create(builder, builder.getIndexAttr(0))
                      .getResult();
      return arith::SubIOp::create(builder, zero, *operand).getResult();
    }
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div: {
      auto lhs = emitIndex(*expr.lhs);
      auto rhs = emitIndex(*expr.rhs);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      switch (expr.kind) {
      case Expr::Kind::Add:
        return arith::AddIOp::create(builder, *lhs, *rhs).getResult();
      case Expr::Kind::Sub:
        return arith::SubIOp::create(builder, *lhs, *rhs).getResult();
      case Expr::Kind::Mul:
        return arith::MulIOp::create(builder, *lhs, *rhs).getResult();
      default:
        return arith::DivSIOp::create(builder, *lhs, *rhs).getResult();
      }
    }
    case Expr::Kind::Ident:
      if (Value value = loopVariables.lookup(expr.ident)) {
        return value;
      }
      if (integerConstants.count(expr.ident) != 0) {
        return arith::ConstantOp::create(
                   builder,
                   builder.getIndexAttr(integerConstants.lookup(expr.ident)))
            .getResult();
      }
      return error(expr.loc, "expected an integer expression");
    case Expr::Kind::Float:
      return error(expr.loc, "expected an integer expression");
    }
    llvm_unreachable("unknown expression kind");
  }

  FailureOr<int64_t> evaluateConstant(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return expr.intValue;
    case Expr::Kind::Neg: {
      auto operand = evaluateConstant(*expr.lhs);
      if (failed(operand)) {
        return failure();
      }
      return -*operand;
    }
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div: {
      auto lhs = evaluateConstant(*expr.lhs);
      auto rhs = evaluateConstant(*expr.rhs);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      switch (expr.kind) {
      case Expr::Kind::Add:
        return *lhs + *rhs;
      case Expr::Kind::Sub:
        return *lhs - *rhs;
      case Expr::Kind::Mul:
        return *lhs * *rhs;
      default:
        if (*rhs == 0) {
          return error(expr.loc, "division by zero in constant expression");
        }
        return *lhs / *rhs;
      }
    }
    case Expr::Kind::Ident:
      if (integerConstants.count(expr.ident) != 0) {
        return integerConstants.lookup(expr.ident);
      }
      return error(expr.loc, "expected a constant integer expression");
    case Expr::Kind::Float:
      return error(expr.loc, "expected a constant integer expression");
    }
    llvm_unreachable("unknown expression kind");
  }

  //===--- State --------------------------------------------------------===//

  QCProgramBuilder builder;
  llvm::SourceMgr& sourceMgr;

  llvm::StringSet<> declaredNames;

  ScopedValueTable parameterConstants;
  ValueScope parameterConstantsScope{parameterConstants};
  IntegerTable integerConstants;
  IntegerScope integerConstantsScope{integerConstants};
  ScopedValueTable booleanConstants;
  ValueScope booleanConstantsScope{booleanConstants};
  ScopedValueTable loopVariables;
  ValueScope loopVariablesScope{loopVariables};
  ScopedValueTable dynamicallyLoadedQubits;
  ValueScope dynamicallyLoadedQubitsScope{dynamicallyLoadedQubits};

  llvm::BumpPtrAllocator keyStorage;
  llvm::StringSaver keySaver{keyStorage};

  llvm::StringMap<VariableKind> variableKinds;

  QubitScope qubitRegisters;
  llvm::StringMap<QCProgramBuilder::ClassicalRegister> classicalRegisters;
  llvm::StringMap<SmallVector<Value>> bitValues;
  llvm::StringMap<std::variant<GateInfo, StoredGate>> gates;
};

} // namespace

namespace detail {

OwningOpRef<ModuleOp> importQASM3(llvm::SourceMgr& sourceMgr,
                                  MLIRContext* context) {
  const auto bufferId = sourceMgr.getMainFileID();
  const llvm::StringRef buffer =
      sourceMgr.getMemoryBuffer(bufferId)->getBuffer();

  Lexer lexer(buffer);
  QCEmitter emitter(context, sourceMgr);
  llvm::BumpPtrAllocator allocator;
  Parser<QCEmitter> parser(lexer, emitter, allocator);

  emitter.initialize();
  const auto status = parser.parseProgram();
  auto module = emitter.finalize();
  if (failed(status)) {
    return nullptr;
  }
  return module;
}

} // namespace detail

} // namespace mlir::qc
