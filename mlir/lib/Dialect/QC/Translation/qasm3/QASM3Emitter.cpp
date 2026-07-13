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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qc {

using qasm3::GateInfo;

using detail::BitReference;
using detail::Condition;
using detail::Expr;
using detail::GateCall;
using detail::Lexer;
using detail::Modifier;
using detail::Operand;
using detail::Parser;

namespace {

/**
 * @brief A stored compound-gate definition.
 *
 * @details
 * Array members are bump-allocated by the parser and remain valid for the
 * duration of the import.
 */
struct StoredGate {
  ArrayRef<StringRef> parameters;
  ArrayRef<StringRef> targets;
  ArrayRef<GateCall> body;
};

/**
 * @brief A named qubit binding in scope.
 *
 * @details
 * For top-level registers, `memref` holds the backing register and `qubits`
 * holds the eagerly extracted values. For compound gates, `memref` is @c null
 * and `qubits` holds the aliased values.
 */
struct QubitBinding {
  Value memref;
  SmallVector<Value> qubits;
};

using QubitScope = llvm::StringMap<QubitBinding>;

/**
 * @brief A declared classical register.
 *
 * @details
 * Registers are held in a vector in declaration order, so that the order in
 * which they are returned from the generated function is deterministic.
 */
struct ClassicalRegisterInfo {
  /// The location of the declaration, used to anchor diagnostics.
  SMLoc loc;
  QCProgramBuilder::ClassicalRegister reg;
  /// The measured value of each bit. Empty until the register is measured.
  SmallVector<Value> bits;
  /// Whether the register was declared with the `output` directive.
  bool isOutput = false;
};

/// Signature: (builder, gate operands, gate parameters). Owns the callable, as
/// `GATE_DISPATCH` outlives the lambdas that build it.
using GateFn = std::function<void(QCProgramBuilder&, ValueRange, ValueRange)>;

} // namespace

/// Build the table mapping each gate identifier to a `QCProgramBuilder`
/// emitter.
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

  return d;
}

/// Build the table mapping a gate identifier to its metadata.
static llvm::StringMap<std::variant<GateInfo, StoredGate>> buildGateTable() {
  llvm::StringMap<std::variant<GateInfo, StoredGate>> table;
  for (const auto& [name, gate] : qasm3::STANDARD_GATES) {
    const auto* standard = dynamic_cast<qasm3::StandardGate*>(gate.get());
    assert(standard != nullptr && "STANDARD_GATES entry is not a StandardGate");
    table.insert({name, standard->info});
  }
  return table;
}

/// Look up a built-in numeric constant and emit it as an `f64`-typed value.
static std::optional<Value> lookupBuiltinConstant(StringRef name,
                                                  QCProgramBuilder& builder) {
  if (name == "pi" || name == "π") {
    return builder.floatConstant(std::numbers::pi);
  }
  if (name == "tau" || name == "τ") {
    return builder.floatConstant(2 * std::numbers::pi);
  }
  if (name == "euler" || name == "ℇ") {
    return builder.floatConstant(std::numbers::e);
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// QCEmitter
//===----------------------------------------------------------------------===//

namespace {

/// Map from gate identifier to `QCProgramBuilder` emitter.
const llvm::StringMap<GateFn> GATE_DISPATCH = buildGateDispatch();

/**
 * @brief Lowers OpenQASM 3 parse events to QC operations.
 *
 * @details
 * Models the sink concept consumed by `Parser`. All events emit eagerly via the
 * `QCProgramBuilder` and report errors as diagnostics through `LogicalResult`.
 * Handles name resolution, semantic validation, and target-specific
 * restrictions.
 */
class QCEmitter {
public:
  QCEmitter(MLIRContext* ctx, llvm::SourceMgr& sourceMgr)
      : builder(ctx), sourceMgr(sourceMgr), gates(buildGateTable()) {}

  void initialize() { builder.initialize(); }

  FailureOr<OwningOpRef<ModuleOp>> finalize() {
    if (classicalRegisters.empty()) {
      return builder.finalize();
    }

    // Without an output directive, every classical register is returned
    const bool hasOutputs =
        any_of(classicalRegisters,
               [](const ClassicalRegisterInfo& info) { return info.isOutput; });

    SmallVector<Value> returnValues;
    for (const auto& info : classicalRegisters) {
      if (hasOutputs && !info.isOutput) {
        continue;
      }
      if (info.bits.empty()) {
        return error(info.loc, "output register '" + StringRef(info.reg.name) +
                                   "' is never measured");
      }
      if (info.bits.size() < static_cast<size_t>(info.reg.size)) {
        return error(info.loc, "not all bits of output register '" +
                                   StringRef(info.reg.name) + "' are measured");
      }
      append_range(returnValues, info.bits);
    }

    builder.retype(ValueRange(returnValues).getTypes());
    return builder.finalize(returnValues);
  }

  //===--- Diagnostics --------------------------------------------------===//

  LogicalResult error(SMLoc loc, const Twine& message) {
    Location location = UnknownLoc::get(builder.getContext());
    if (loc.isValid()) {
      const auto [line, col] = sourceMgr.getLineAndColumn(loc);
      location =
          FileLineColLoc::get(builder.getStringAttr("<qasm3>"), line, col);
    }
    return emitError(location, message);
  }

  //===--- Include ------------------------------------------------------===//

  LogicalResult include(SMLoc loc, StringRef filename) {
    if (filename != "stdgates.inc" && filename != "qelib1.inc") {
      return error(loc, "unsupported include '" + filename +
                            "'; only 'stdgates.inc' and 'qelib1.inc' are "
                            "supported");
    }
    return success();
  }

  //===--- Declarations and assignments ---------------------------------===//

  LogicalResult intDecl(SMLoc /*loc*/, StringRef id, const Expr* initializer,
                        bool isConst) {
    if (initializer != nullptr) {
      return assignInt(id, *initializer, isConst);
    }
    return success();
  }

  LogicalResult floatDecl(SMLoc /*loc*/, StringRef id,
                          const Expr* initializer) {
    if (initializer != nullptr) {
      return assignFloat(id, *initializer);
    }
    return success();
  }

  LogicalResult boolDecl(SMLoc /*loc*/, StringRef id,
                         const Condition* initializer) {
    if (initializer != nullptr) {
      return assignBool(id, *initializer);
    }
    return success();
  }

  LogicalResult intAssign(SMLoc /*loc*/, StringRef id, const Expr& value) {
    // The parser rejects assignments to a `const`
    return assignInt(id, value, /*isConst=*/false);
  }

  LogicalResult floatAssign(SMLoc /*loc*/, StringRef id, const Expr& value) {
    return assignFloat(id, value);
  }

  LogicalResult boolAssign(SMLoc /*loc*/, StringRef id,
                           const Condition& value) {
    return assignBool(id, value);
  }

  /**
   * @brief Bind an integer variable @p id to the value of @p value.
   *
   * @details
   * If @p isConst, the value is evaluated at compile time and stored in
   * `constantIntegers`. Otherwise, the value is emitted as a runtime value and
   * stored in `dynamicIntegers`.
   */
  LogicalResult assignInt(StringRef id, const Expr& value, bool isConst) {
    if (isConst) {
      auto folded = evaluateConstant(value);
      if (failed(folded)) {
        return failure();
      }
      constantIntegers.insert(id, *folded);
      return success();
    }
    auto emitted = emitIndex(value);
    if (failed(emitted)) {
      return failure();
    }
    dynamicIntegers.insert(id, *emitted);
    return success();
  }

  /// Bind a floating-point variable @p id to the value of @p value.
  LogicalResult assignFloat(StringRef id, const Expr& value) {
    auto emitted = emitFloat(value);
    if (failed(emitted)) {
      return failure();
    }
    floatValues.insert(id, *emitted);
    return success();
  }

  /// Bind a boolean variable @p id to the value of @p value.
  LogicalResult assignBool(StringRef id, const Condition& value) {
    auto emitted = emitCondition(value);
    if (failed(emitted)) {
      return failure();
    }
    booleanValues.insert(id, *emitted);
    return success();
  }

  LogicalResult qubitRegister(SMLoc /*loc*/, StringRef id, const Expr* size) {
    if (size != nullptr) {
      auto sizeConstant = evaluateConstant(*size);
      if (failed(sizeConstant)) {
        return failure();
      }
      const auto [value, qubits] = builder.allocQubitRegister(*sizeConstant);
      qubitRegisters[id] = {.memref = value, .qubits = qubits};
    } else {
      qubitRegisters[id] = {.memref = nullptr,
                            .qubits = {builder.allocQubit()}};
    }
    return success();
  }

  LogicalResult classicalRegister(SMLoc loc, StringRef id, const Expr* size,
                                  bool isOutput) {
    if (size != nullptr) {
      auto sizeConstant = evaluateConstant(*size);
      if (failed(sizeConstant)) {
        return failure();
      }
      classicalRegisters.push_back(
          {.loc = loc,
           .reg = builder.allocClassicalBitRegister(*sizeConstant, id.str()),
           .isOutput = isOutput});
    } else {
      classicalRegisters.push_back(
          {.loc = loc,
           .reg = builder.allocClassicalBitRegister(1, id.str()),
           .isOutput = isOutput});
    }
    return success();
  }

  //===--- Measure ------------------------------------------------------===//

  LogicalResult measure(SMLoc loc, const BitReference& target,
                        const Operand& operand) {
    auto cregOrFailure = findClassicalRegister(target.loc, target.identifier);
    if (failed(cregOrFailure)) {
      return failure();
    }
    auto* creg = *cregOrFailure;

    auto bits = resolveBitReference(*creg, target);
    if (failed(bits)) {
      return failure();
    }

    auto resolved = resolveOperand(operand, qubitRegisters);
    if (failed(resolved)) {
      return failure();
    }

    SmallVector<Value> qubits;
    if (auto* qubit = std::get_if<Value>(&*resolved)) {
      qubits.push_back(*qubit);
    } else {
      qubits = std::get<SmallVector<Value>>(*resolved);
    }

    if (bits->size() != qubits.size()) {
      return error(loc, "the classical register and the quantum register must "
                        "have the same width");
    }

    for (const auto& [bit, qubit] : zip_equal(*bits, qubits)) {
      auto result = MeasureOp::create(
                        builder, qubit, builder.getStringAttr(bit.registerName),
                        builder.getI64IntegerAttr(bit.registerSize),
                        builder.getI64IntegerAttr(bit.registerIndex))
                        .getResult();
      const auto index = static_cast<size_t>(bit.registerIndex);
      if (creg->bits.size() <= index) {
        creg->bits.resize(index + 1);
      }
      creg->bits[index] = result;
    }

    return success();
  }

  //===--- Reset --------------------------------------------------------===//

  LogicalResult reset(SMLoc /*loc*/, const Operand& operand) {
    auto resolved = resolveOperand(operand, qubitRegisters);
    if (failed(resolved)) {
      return failure();
    }
    if (auto* qubit = std::get_if<Value>(&*resolved)) {
      builder.reset(*qubit);
    } else {
      for (auto qubit : std::get<SmallVector<Value>>(*resolved)) {
        builder.reset(qubit);
      }
    }
    return success();
  }

  //===--- Barrier ------------------------------------------------------===//

  LogicalResult barrier(SMLoc /*loc*/, ArrayRef<Operand> operands) {
    SmallVector<Value> qubits;
    for (const auto& operand : operands) {
      auto resolved = resolveOperand(operand, qubitRegisters);
      if (failed(resolved)) {
        return failure();
      }
      if (auto* qubit = std::get_if<Value>(&*resolved)) {
        qubits.push_back(*qubit);
      } else {
        append_range(qubits, std::get<SmallVector<Value>>(*resolved));
      }
    }
    builder.barrier(qubits);
    return success();
  }

  //===--- Gate definitions and calls -----------------------------------===//

  LogicalResult gateDefinition(SMLoc loc, StringRef id,
                               ArrayRef<StringRef> parameters,
                               ArrayRef<StringRef> targets,
                               ArrayRef<GateCall> body) {
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

  LogicalResult ifConditionOnly(SMLoc /*loc*/, const Condition& condition) {
    return LogicalResult{emitCondition(condition)};
  }

  struct IfScope {
    scf::IfOp op;
    OpBuilder::InsertPoint savedInsertionPoint;
  };

  FailureOr<IfScope> ifBegin(SMLoc /*loc*/, const Condition& condition,
                             bool invert) {
    auto condOrFailure = emitCondition(condition);
    if (failed(condOrFailure)) {
      return failure();
    }
    auto cond = *condOrFailure;
    if (invert) {
      auto trueValue = builder.boolConstant(true);
      cond = arith::XOrIOp::create(builder, cond, trueValue).getResult();
    }
    auto ifOp = scf::IfOp::create(builder, cond, /*withElseRegion=*/false);
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

  LogicalResult forStmt(SMLoc loc, StringRef variable, const Expr& start,
                        const Expr& step, const Expr& stop,
                        function_ref<LogicalResult()> body) {
    auto startValue = emitIndex(start);
    auto stopValue = emitIndex(stop);
    if (failed(startValue) || failed(stopValue)) {
      return failure();
    }

    auto foldedStep = tryEvaluateConstant(step);
    if (failed(foldedStep)) {
      return failure();
    }

    // The magnitude of a statically negative step, or 0 if the step counts up
    // or is only known dynamically.
    int64_t stride = 0;
    if (const std::optional<int64_t> stepConstant = *foldedStep) {
      if (*stepConstant == 0) {
        return error(loc, "for loops with a zero step are not supported");
      }
      if (*stepConstant < 0) {
        stride = -*stepConstant;
      }
    }

    Value resolvedStart;
    Value resolvedStep;
    Value resolvedStop;
    Value strideValue;
    if (stride != 0) {
      // A negative step counts down, which `scf.for` cannot express. Count the
      // iterations up instead and recompute the loop variable from the
      // induction variable below, which preserves both the values the loop
      // visits and the order in which it visits them. The loop runs
      // `max(0, (start - stop) / stride + 1)` times.
      auto zero = builder.indexConstant(0);
      auto one = builder.indexConstant(1);
      strideValue = builder.indexConstant(stride);
      auto span =
          arith::SubIOp::create(builder, *startValue, *stopValue).getResult();
      auto steps =
          arith::DivSIOp::create(builder, span, strideValue).getResult();
      auto count = arith::AddIOp::create(builder, steps, one).getResult();
      auto isPositive =
          arith::CmpIOp::create(builder, arith::CmpIPredicate::sgt, count, zero)
              .getResult();
      resolvedStart = zero;
      resolvedStop =
          arith::SelectOp::create(builder, isPositive, count, zero).getResult();
      resolvedStep = one;
    } else {
      auto stepValue = emitIndex(step);
      if (failed(stepValue)) {
        return failure();
      }
      resolvedStart = *startValue;
      resolvedStep = *stepValue;
      // Add 1 to make the stop value inclusive, as OpenQASM 3's for loops are
      // inclusive.
      resolvedStop =
          arith::AddIOp::create(builder, *stopValue, builder.indexConstant(1))
              .getResult();
    }

    ValueScope ivScope(dynamicIntegers);
    ValueScope loadScope(dynamicallyLoadedQubits);

    auto status = success();
    builder.scfFor(resolvedStart, resolvedStop, resolvedStep, [&](Value iv) {
      Value resolvedIv = iv;
      if (stride != 0) {
        auto offset =
            arith::MulIOp::create(builder, iv, strideValue).getResult();
        resolvedIv =
            arith::SubIOp::create(builder, *startValue, offset).getResult();
      }
      dynamicIntegers.insert(variable, resolvedIv);
      status = body();
    });
    return status;
  }

  LogicalResult whileStmt(SMLoc /*loc*/, const Condition& condition,
                          function_ref<LogicalResult()> body) {
    auto status = success();
    builder.scfWhile(
        [&] {
          ValueScope loadScope(dynamicallyLoadedQubits);
          auto cond = emitCondition(condition);
          if (failed(cond)) {
            status = failure();
            builder.scfCondition(builder.boolConstant(false));
          } else {
            builder.scfCondition(*cond);
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
  using ValueTable = llvm::ScopedHashTable<StringRef, Value>;
  using ValueScope = llvm::ScopedHashTableScope<StringRef, Value>;
  using IntegerTable = llvm::ScopedHashTable<StringRef, int64_t>;
  using IntegerScope = llvm::ScopedHashTableScope<StringRef, int64_t>;

  //===--- Gate calls ---------------------------------------------------===//

  LogicalResult emitGateCall(const GateCall& call, const QubitScope& scope) {
    auto it = gates.find(call.identifier);
    if (it == gates.end()) {
      return error(call.loc, "no OpenQASM definition found for gate '" +
                                 call.identifier + "'");
    }

    // Resolve parameters
    SmallVector<Value> parameters;
    parameters.reserve(call.parameters.size());
    for (const auto* argument : call.parameters) {
      auto value = emitFloat(*argument);
      if (failed(value)) {
        return failure();
      }
      parameters.push_back(*value);
    }

    // Resolve operands. If any operand is a register, the call is broadcast
    // over the qubits of the register(s). All registers must have the same
    // length, and any single-qubit operands are repeated in each iteration.
    SmallVector<std::variant<Value, SmallVector<Value>>> resolvedOperands;
    resolvedOperands.reserve(call.operands.size());
    std::optional<size_t> broadcastWidth;
    for (const auto& operand : call.operands) {
      auto resolved = resolveOperand(operand, scope);
      if (failed(resolved)) {
        return failure();
      }
      if (const auto* reg = std::get_if<SmallVector<Value>>(&*resolved)) {
        if (broadcastWidth && *broadcastWidth != reg->size()) {
          return error(call.loc,
                       "all broadcasting operands must have the same width");
        }
        broadcastWidth = reg->size();
      }
      resolvedOperands.push_back(std::move(*resolved));
    }

    // No broadcasting
    if (!broadcastWidth) {
      SmallVector<Value> operands;
      operands.reserve(resolvedOperands.size());
      for (const auto& resolved : resolvedOperands) {
        operands.push_back(std::get<Value>(resolved));
      }
      return emitGateInvocation(it->second, call.identifier, parameters,
                                call.modifiers, operands, call.loc);
    }

    for (size_t b = 0; b < *broadcastWidth; ++b) {
      SmallVector<Value> operands;
      operands.reserve(resolvedOperands.size());
      for (const auto& resolved : resolvedOperands) {
        if (const auto* value = std::get_if<Value>(&resolved)) {
          operands.push_back(*value);
        } else {
          operands.push_back(std::get<SmallVector<Value>>(resolved)[b]);
        }
      }
      if (failed(emitGateInvocation(it->second, call.identifier, parameters,
                                    call.modifiers, operands, call.loc))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult
  emitGateInvocation(const std::variant<GateInfo, StoredGate>& definition,
                     StringRef id, ValueRange parameters,
                     ArrayRef<Modifier> modifiers,
                     const SmallVector<Value>& operands, SMLoc loc) {
    auto split = splitControlsAndTargets<Value>(modifiers, operands, loc);
    if (failed(split)) {
      return failure();
    }
    if (const auto* compound = std::get_if<StoredGate>(&definition)) {
      return emitCompoundGate(*compound, parameters, split->targets,
                              split->posControls, split->negControls,
                              split->invert, loc);
    }
    const auto& gate = std::get<GateInfo>(definition);
    auto gateFn = resolveStandardGate(gate, id, parameters, loc);
    if (failed(gateFn)) {
      return failure();
    }
    emitStandardGate(**gateFn, parameters, split->targets, split->posControls,
                     split->negControls, split->invert);
    return success();
  }

  template <typename T> struct ControlsAndTargets {
    bool invert = false;
    SmallVector<T> posControls;
    SmallVector<T> negControls;
    SmallVector<T> targets;
  };

  FailureOr<size_t> getControlCount(const Expr* argument) {
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
  splitControlsAndTargets(ArrayRef<Modifier> modifiers,
                          const SmallVector<T>& operands, SMLoc loc) {
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
        auto count = getControlCount(modifier.argument);
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
    result.targets = to_vector(drop_begin(operands, numControls));
    return result;
  }

  LogicalResult emitCompoundGate(const StoredGate& gate, ValueRange parameters,
                                 ValueRange targets, ValueRange posControls,
                                 ValueRange negControls, bool invert,
                                 SMLoc loc) {
    if (gate.parameters.size() != parameters.size()) {
      return error(loc, "invalid number of parameters for compound gate");
    }
    if (gate.targets.size() != targets.size()) {
      return error(loc, "invalid number of target qubits for compound gate");
    }

    // Map each internal target name to its position(s) in the operand list.
    // Positions may repeat if the qubits are aliased within a modifier region.
    llvm::StringMap<SmallVector<size_t>> targetsMap;
    for (const auto& [targetName, target] : zip_equal(gate.targets, targets)) {
      auto it = llvm::find(targets, target);
      if (it == targets.end()) {
        return error(loc, "target '" + targetName + "' not found in operands");
      }
      targetsMap[targetName].push_back(
          static_cast<size_t>(std::distance(targets.begin(), it)));
    }

    ValueScope parameterScope(floatValues);
    for (const auto& [name, value] : zip_equal(gate.parameters, parameters)) {
      floatValues.insert(name, value);
    }

    auto status = success();
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

  FailureOr<const GateFn*> resolveStandardGate(const GateInfo& gate,
                                               StringRef id,
                                               ValueRange parameters,
                                               SMLoc loc) {
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

  //===--- Bit reference resolution -------------------------------------===//

  /// Look up the classical register named @p name.
  FailureOr<ClassicalRegisterInfo*> findClassicalRegister(SMLoc loc,
                                                          StringRef name) {
    auto* it =
        find_if(classicalRegisters, [&](const ClassicalRegisterInfo& info) {
          return StringRef(info.reg.name) == name;
        });
    if (it == classicalRegisters.end()) {
      return error(loc, "unknown classical register '" + name + "'");
    }
    return it;
  }

  FailureOr<Value> lookupBitValue(const BitReference& bit) {
    auto creg = findClassicalRegister(bit.loc, bit.identifier);
    if (failed(creg)) {
      return failure();
    }
    const auto& registerBits = (*creg)->bits;
    if (registerBits.empty()) {
      return error(bit.loc, "no classical bit of register '" + bit.identifier +
                                "' has been measured yet");
    }
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
    const auto& [memref, qubits] = it->second;

    if (operand.index == nullptr) {
      if (qubits.size() == 1) {
        return std::variant<Value, SmallVector<Value>>{qubits[0]};
      }
      return std::variant<Value, SmallVector<Value>>{qubits};
    }

    const auto& index = *operand.index;

    auto foldedIndex = tryEvaluateConstant(index);
    if (failed(foldedIndex)) {
      return failure();
    }
    if (const std::optional<int64_t> indexConstant = *foldedIndex) {
      const auto i = static_cast<size_t>(*indexConstant);
      if (i >= qubits.size()) {
        return error(operand.loc, "qubit index out of bounds");
      }
      return std::variant<Value, SmallVector<Value>>{qubits[i]};
    }

    if (!memref) {
      return error(operand.loc,
                   "dynamic qubit indexing requires a qubit register");
    }
    auto loaded = loadDynamicElement(operand.identifier, memref, index);
    if (failed(loaded)) {
      return failure();
    }
    return std::variant<Value, SmallVector<Value>>{*loaded};
  }

  static std::string indexKey(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return std::to_string(expr.intValue);
    case Expr::Kind::Float:
      return std::to_string(expr.floatValue);
    case Expr::Kind::Identifier:
      return expr.identifier.str();
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

  FailureOr<Value> loadDynamicElement(StringRef name, Value memref,
                                      const Expr& expr) {
    const std::string key = name.str() + "[" + indexKey(expr) + "]";
    if (Value cached = dynamicallyLoadedQubits.lookup(key)) {
      return cached;
    }
    auto index = emitIndex(expr);
    if (failed(index)) {
      return failure();
    }
    auto loaded = builder.memrefLoad(memref, *index);
    dynamicallyLoadedQubits.insert(keySaver.save(key), loaded);
    return loaded;
  }

  FailureOr<SmallVector<QCProgramBuilder::Bit>>
  resolveBitReference(const ClassicalRegisterInfo& info,
                      const BitReference& reference) {
    const auto& creg = info.reg;
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

  //===--- Conditions ---------------------------------------------------===//

  FailureOr<Value> emitCondition(const Condition& condition) {
    switch (condition.kind) {
    case Condition::Kind::Measurement: {
      auto resolved = resolveOperand(condition.operand, qubitRegisters);
      if (failed(resolved)) {
        return failure();
      }
      auto* qubit = std::get_if<Value>(&*resolved);
      if (qubit == nullptr) {
        return error(condition.loc,
                     "measurement condition must be a single qubit");
      }
      return builder.measure(*qubit);
    }
    case Condition::Kind::Bit:
      // A bare identifier may name a declared `bool`; otherwise it is a
      // classical bit reference.
      if (condition.bit.index == nullptr) {
        if (Value value = booleanValues.lookup(condition.bit.identifier)) {
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

  //===--- Expressions --------------------------------------------------===//

  FailureOr<Value> emitFloat(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return builder.floatConstant(static_cast<double>(expr.intValue));
    case Expr::Kind::Float:
      return builder.floatConstant(expr.floatValue);
    case Expr::Kind::Identifier:
      return resolveParameter(expr.identifier, expr.loc);
    case Expr::Kind::Neg: {
      auto operand = emitFloat(*expr.lhs);
      if (failed(operand)) {
        return failure();
      }
      return arith::NegFOp::create(builder, *operand).getResult();
    }
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div: {
      auto lhs = emitFloat(*expr.lhs);
      auto rhs = emitFloat(*expr.rhs);
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

  FailureOr<Value> resolveParameter(StringRef name, SMLoc loc) {
    if (auto value = floatValues.lookup(name)) {
      return value;
    }
    if (auto value = constantIntegers.lookup(name)) {
      return builder.floatConstant(static_cast<double>(value));
    }
    if (auto value = dynamicIntegers.lookup(name)) {
      auto integer =
          arith::IndexCastOp::create(builder, builder.getI64Type(), value)
              .getResult();
      return arith::SIToFPOp::create(builder, builder.getF64Type(), integer)
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
      return builder.indexConstant(expr.intValue);
    case Expr::Kind::Neg: {
      auto operand = emitIndex(*expr.lhs);
      if (failed(operand)) {
        return failure();
      }
      auto zero = builder.indexConstant(0);
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
    case Expr::Kind::Identifier:
      if (auto value = constantIntegers.lookup(expr.identifier)) {
        return builder.indexConstant(value);
      }
      if (auto value = dynamicIntegers.lookup(expr.identifier)) {
        return value;
      }
      return error(expr.loc, "expected an integer expression");
    case Expr::Kind::Float:
      return error(expr.loc, "expected an integer expression");
    }
    llvm_unreachable("unknown expression kind");
  }

  /**
   * @brief Fold @p expr to a compile-time integer constant.
   *
   * @details
   * A `std::nullopt` result means @p expr is not a compile-time constant, which
   * is not necessarily an error: the caller decides whether to require one or
   * to fall back to emitting the expression. A constant expression that is
   * malformed is diagnosed and returns failure.
   */
  FailureOr<std::optional<int64_t>> tryEvaluateConstant(const Expr& expr) {
    switch (expr.kind) {
    case Expr::Kind::Int:
      return std::optional{expr.intValue};
    case Expr::Kind::Float:
      return std::optional<int64_t>{};
    case Expr::Kind::Identifier:
      if (auto value = constantIntegers.lookup(expr.identifier)) {
        return std::optional{value};
      }
      return std::optional<int64_t>{};
    case Expr::Kind::Neg: {
      auto operand = tryEvaluateConstant(*expr.lhs);
      if (failed(operand) || !*operand) {
        return operand;
      }
      return std::optional{-**operand};
    }
    case Expr::Kind::Add:
    case Expr::Kind::Sub:
    case Expr::Kind::Mul:
    case Expr::Kind::Div: {
      auto lhs = tryEvaluateConstant(*expr.lhs);
      auto rhs = tryEvaluateConstant(*expr.rhs);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      if (!*lhs || !*rhs) {
        return std::optional<int64_t>{};
      }
      switch (expr.kind) {
      case Expr::Kind::Add:
        return std::optional{**lhs + **rhs};
      case Expr::Kind::Sub:
        return std::optional{**lhs - **rhs};
      case Expr::Kind::Mul:
        return std::optional{**lhs * **rhs};
      default:
        if (**rhs == 0) {
          return error(expr.loc, "division by zero in constant expression");
        }
        return std::optional{**lhs / **rhs};
      }
    }
    }
    llvm_unreachable("unknown expression kind");
  }

  /// Fold @p expr to a compile-time integer constant, requiring that it is one.
  FailureOr<int64_t> evaluateConstant(const Expr& expr) {
    auto folded = tryEvaluateConstant(expr);
    if (failed(folded)) {
      return failure();
    }
    if (!*folded) {
      return error(expr.loc, "expected a constant integer expression");
    }
    return **folded;
  }

  //===--- State --------------------------------------------------------===//

  QCProgramBuilder builder;
  llvm::SourceMgr& sourceMgr;

  ValueTable floatValues;
  ValueScope floatValuesScope{floatValues};

  IntegerTable constantIntegers;
  IntegerScope constantIntegersScope{constantIntegers};
  ValueTable dynamicIntegers;
  ValueScope dynamicIntegersScope{dynamicIntegers};

  ValueTable booleanValues;
  ValueScope booleanValuesScope{booleanValues};

  ValueTable dynamicallyLoadedQubits;
  ValueScope dynamicallyLoadedQubitsScope{dynamicallyLoadedQubits};

  llvm::BumpPtrAllocator keyStorage;
  llvm::StringSaver keySaver{keyStorage};

  QubitScope qubitRegisters;
  SmallVector<ClassicalRegisterInfo> classicalRegisters;
  llvm::StringMap<std::variant<GateInfo, StoredGate>> gates;
};

} // namespace

namespace detail {

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  const auto bufferId = sourceMgr.getMainFileID();
  const StringRef buffer = sourceMgr.getMemoryBuffer(bufferId)->getBuffer();

  Lexer lexer(buffer);
  QCEmitter emitter(context, sourceMgr);
  llvm::BumpPtrAllocator allocator;
  Parser<QCEmitter> parser(lexer, emitter, allocator);

  emitter.initialize();
  const auto status = parser.parseProgram();
  auto mod = emitter.finalize();
  if (failed(status) || failed(mod)) {
    return nullptr;
  }
  return std::move(*mod);
}

} // namespace detail

} // namespace mlir::qc
