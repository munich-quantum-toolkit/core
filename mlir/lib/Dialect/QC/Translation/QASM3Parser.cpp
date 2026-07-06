/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QASM3Parser.h"

#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qasm3/DebugInfo.hpp"
#include "qasm3/Exception.hpp"
#include "qasm3/Gate.hpp"
#include "qasm3/NestedEnvironment.hpp"
#include "qasm3/Scanner.hpp"
#include "qasm3/StdGates.hpp"
#include "qasm3/Token.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::qc {

namespace {

using qasm3::CompilerError;
using qasm3::DebugInfo;
using qasm3::GateInfo;
using qasm3::Token;

/// Signature: (builder, gate operands, gate parameters).
using GateFn = std::function<void(QCProgramBuilder&, ValueRange, ValueRange)>;

/// Build the table mapping each gate identifier to a lambda that emits the
/// corresponding QC operation via the `QCProgramBuilder`.
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

/**
 * @brief Represents a named qubit binding in scope.
 *
 * @details
 * For top-level registers, `memref` holds the backing register and `qubits`
 * holds eagerly extracted values. For compound gates, `memref` is null and
 * `qubits` holds the alaiased values.
 */
struct QubitBinding {
  Value memref;
  SmallVector<Value> qubits;
};

/// Map of qubits in the current scope.
using QubitScope = llvm::StringMap<QubitBinding>;

/// Look up a built-in numeric constant and emit it as an `f64`-typed MLIR
/// value.
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

//===----------------------------------------------------------------------===//
// Parsed representations
//===----------------------------------------------------------------------===//

/// A parsed expression.
struct ParsedExpr {
  enum class Kind : uint8_t { IntLiteral, FloatLiteral, Ident, Unary, Binary };

  Kind kind{Kind::IntLiteral};

  // Literal
  int64_t intValue{};
  double floatValue{};

  // Ident
  std::string name;

  // Unary or Binary
  Token::Kind op{};
  std::vector<ParsedExpr> children;
};

/// A parsed gate modifier.
struct ParsedModifier {
  Token::Kind kind{Token::Kind::Inv};
  std::optional<ParsedExpr> expression;
};

/// A parsed gate operand. Either a (possibly indexed) named qubit, or a
/// hardware qubit.
struct ParsedOperand {
  std::string name;
  std::optional<ParsedExpr> index;
  std::optional<uint64_t> hardwareQubit;
};

/// A parsed gate call.
struct ParsedGateCall {
  std::string identifier;
  std::vector<ParsedModifier> modifiers;
  std::vector<ParsedExpr> parameters;
  std::vector<ParsedOperand> operands;
  std::shared_ptr<DebugInfo> debugInfo;
};

/// A parsed compound gate.
struct ParsedCompoundGate {
  std::vector<std::string> parameterNames;
  std::vector<std::string> targetNames;
  std::vector<ParsedGateCall> body;
};

/// A parsed classical bit reference.
struct ParsedBitRef {
  std::string name;
  std::optional<ParsedExpr> index;
};

/// Build the table mapping a gate identifier to its metadata.
llvm::StringMap<std::variant<GateInfo, ParsedCompoundGate>> buildGateTable() {
  llvm::StringMap<std::variant<GateInfo, ParsedCompoundGate>> t;
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
// QASM3Parser
//===----------------------------------------------------------------------===//

/**
 * @brief OpenQASM 3 parser that builds a QC program.
 *
 * @details
 * Consumes `qasm3::Scanner` tokens directly and emits QC operations via the
 * `QCProgramBuilder` as it parses. Each `parse` function parses all tokens of a
 * given statement. Validation helpers raise `CompilerError`s wherever needed.
 */
class QASM3Parser final {
public:
  explicit QASM3Parser(MLIRContext* ctx)
      : builder(ctx), gates(buildGateTable()) {}

  OwningOpRef<ModuleOp> parse(std::string_view source) {
    input = std::make_unique<std::istringstream>(std::string(source));
    scanner = std::make_unique<qasm3::Scanner>(input.get());

    // Initialize the token windows
    currentToken = scanner->next();
    nextToken = scanner->next();

    builder.initialize();
    parseProgram();
    return builder.finalize();
  }

private:
  //===--- State --------------------------------------------------------===//

  QCProgramBuilder builder;

  std::unique_ptr<std::istream> input;
  std::unique_ptr<qasm3::Scanner> scanner;
  Token currentToken{0, 0};
  Token nextToken{0, 0};

  /// Set of declared identifiers.
  llvm::StringSet<> declaredNames;

  /// Map from a parameter constant to its `f64`-typed MLIR value.
  qasm3::NestedEnvironment<Value> parameterConstants;

  /// Map from an induction variable to its MLIR value.
  qasm3::NestedEnvironment<Value> loopVariables;

  /// Cache of dynamically loaded qubits.
  qasm3::NestedEnvironment<Value> dynamicallyLoadedQubits;

  /// Map from qubit-register name to `QubitScope`.
  QubitScope qubitRegisters;

  /// Map from classical-register name to `ClassicalRegister`.
  llvm::StringMap<QCProgramBuilder::ClassicalRegister> classicalRegisters;

  /// Map from classical-register name to measurement results.
  llvm::StringMap<SmallVector<Value>> bitValues;

  /// Map from gate identifier to its metadata.
  llvm::StringMap<std::variant<GateInfo, ParsedCompoundGate>> gates;

  bool openQASM2CompatMode{false};

  //===--- Token scaffolding --------------------------------------------===//

  /**
   * @brief Advance the token cursor by one.
   *
   * @details
   * `current()` moves to what `peek()` returned, and a fresh token is pulled
   * from the scanner into the lookahead.
   */
  void advance() {
    currentToken = nextToken;
    nextToken = scanner->next();
  }

  /// The token the parser is currently positioned on.
  [[nodiscard]] const Token& current() const { return currentToken; }

  /// The next token, without consuming it.
  [[nodiscard]] const Token& peek() const { return nextToken; }

  /// Whether the entire input has been consumed.
  [[nodiscard]] bool isAtEnd() const {
    return currentToken.kind == Token::Kind::Eof;
  }

  /// Create a `DebugInfo` object from @p token.
  [[nodiscard]] std::shared_ptr<DebugInfo>
  makeDebugInfo(const Token& token) const {
    return std::make_shared<DebugInfo>(token.line, token.col, "<input>");
  }

  /// Throw a `CompilerError` at the position of @p token.
  void error(const Token& token, const std::string& msg) const {
    throw CompilerError(msg, makeDebugInfo(token));
  }

  /// Throw a `CompilerError` using an existing @p debugInfo.
  void error(const DebugInfo& debugInfo, const std::string& msg) const {
    throw CompilerError(msg, std::make_shared<DebugInfo>(debugInfo));
  }

  /// Assert that the current token matches an expected kind, then advance.
  Token expect(const Token::Kind expected) {
    auto token = current();
    if (token.kind != expected) {
      error(token, "Expected '" + Token::kindToString(expected) + "', got '" +
                       Token::kindToString(token.kind) + "'.");
    }
    advance();
    return token;
  }

  //===--- Program and statement dispatch -------------------------------===//

  void parseProgram() {
    while (!isAtEnd()) {
      parseStatement();
    }
  }

  void parseStatement() {
    switch (current().kind) {
    case Token::Kind::OpenQasm:
      parseVersionDeclaration();
      return;
    case Token::Kind::Include:
      parseInclude();
      return;
    case Token::Kind::Const:
      parseConstantDeclaration();
      return;
    case Token::Kind::Qubit:
      parseQubitDeclaration();
      return;
    case Token::Kind::Qreg:
      parseQregDeclaration();
      return;
    case Token::Kind::Bit:
      parseBitDeclaration();
      return;
    case Token::Kind::CReg:
      parseCregDeclaration();
      return;
    case Token::Kind::Int:
    case Token::Kind::Uint:
    case Token::Kind::Bool:
    case Token::Kind::Float:
    case Token::Kind::Angle:
    case Token::Kind::Duration:
      error(current(), "Declaration type is not supported yet.");
      return;
    case Token::Kind::Gate:
      parseGateDeclaration();
      return;
    case Token::Kind::Opaque:
      error(current(), "Opaque gate declarations are not supported.");
      return;
    case Token::Kind::InitialLayout:
    case Token::Kind::OutputPermutation:
      // No counterparts in QC
      advance();
      return;
    case Token::Kind::Barrier:
      parseBarrier();
      return;
    case Token::Kind::Reset:
      parseReset();
      return;
    case Token::Kind::Measure:
      parseMeasure();
      return;
    case Token::Kind::If:
      parseIf();
      return;
    case Token::Kind::For:
      parseFor();
      return;
    case Token::Kind::While:
      parseWhile();
      return;
    case Token::Kind::Inv:
    case Token::Kind::Pow:
    case Token::Kind::Ctrl:
    case Token::Kind::NegCtrl:
    case Token::Kind::Gphase:
      emitGateCall(parseGateCall(), qubitRegisters);
      return;
    case Token::Kind::Identifier:
      switch (peek().kind) {
      case Token::Kind::LBracket:
      case Token::Kind::Equals:
      case Token::Kind::PlusEquals:
      case Token::Kind::MinusEquals:
      case Token::Kind::AsteriskEquals:
      case Token::Kind::SlashEquals:
      case Token::Kind::AmpersandEquals:
      case Token::Kind::PipeEquals:
      case Token::Kind::TildeEquals:
      case Token::Kind::CaretEquals:
      case Token::Kind::LeftShitEquals:
      case Token::Kind::RightShiftEquals:
      case Token::Kind::PercentEquals:
      case Token::Kind::DoubleAsteriskEquals:
        parseAssignment();
        return;
      default:
        emitGateCall(parseGateCall(), qubitRegisters);
        return;
      }
    default:
      error(current(), "Unexpected token '" + current().toString() + "'.");
    }
  }

  /// Parse a `{ ... }` block or a single statement.
  void parseBlockOrStatement() {
    if (current().kind == Token::Kind::LBrace) {
      advance();
      while (!isAtEnd() && current().kind != Token::Kind::RBrace) {
        parseStatement();
      }
      expect(Token::Kind::RBrace);
    } else {
      parseStatement();
    }
  }

  //===--- Version ------------------------------------------------------===//

  void parseVersionDeclaration() {
    expect(Token::Kind::OpenQasm);
    double version = 0.0;
    if (current().kind == Token::Kind::FloatLiteral) {
      version = current().valReal;
      advance();
    } else if (current().kind == Token::Kind::IntegerLiteral) {
      version = static_cast<double>(current().val);
      advance();
    } else {
      error(current(),
            "Version declaration must be a float or integer literal.");
    }
    expect(Token::Kind::Semicolon);
    if (version < 3) {
      openQASM2CompatMode = true;
    }
  }

  //===--- Include ------------------------------------------------------===//

  void parseInclude() {
    const auto beginToken = expect(Token::Kind::Include);
    const auto filename = expect(Token::Kind::StringLiteral).str;
    expect(Token::Kind::Semicolon);
    if (filename != "stdgates.inc" && filename != "qelib1.inc") {
      error(beginToken, "Unsupported include '" + filename +
                            "'. Only 'stdgates.inc' and 'qelib1.inc' are "
                            "supported.");
    }
  }

  //===--- Declarations -------------------------------------------------===//

  /// Parse `const float <id> = <expression>;`.
  void parseConstantDeclaration() {
    const auto constToken = expect(Token::Kind::Const);
    const auto debugInfo = makeDebugInfo(constToken);

    if (current().kind != Token::Kind::Float ||
        peek().kind != Token::Kind::Identifier) {
      error(*debugInfo, "Only `const float <id> = <expression>;` declarations "
                        "are supported for now.");
    }
    expect(Token::Kind::Float);
    const auto id = expect(Token::Kind::Identifier).str;

    expect(Token::Kind::Equals);
    auto initExpr = parseExpression();
    expect(Token::Kind::Semicolon);

    registerDeclaredName(constToken, id);
    parameterConstants.emplace(id, emitFloatExpression(initExpr, debugInfo));
  }

  /// Parse `qubit[<n>] <id>;`.
  void parseQubitDeclaration() {
    const auto qubit = expect(Token::Kind::Qubit);
    const auto debugInfo = makeDebugInfo(qubit);
    std::optional<int64_t> size;
    if (current().kind == Token::Kind::LBracket) {
      size = parseDesignator(debugInfo);
    }
    const auto id = expect(Token::Kind::Identifier).str;
    expect(Token::Kind::Semicolon);
    registerDeclaredName(qubit, id);
    if (size) {
      const auto reg = builder.allocQubitRegister(*size);
      qubitRegisters[id] = {reg.value, reg.qubits};
    } else {
      qubitRegisters[id] = {nullptr, {builder.allocQubit()}};
    }
  }

  /// Parse `qreg <id>[<n>];`.
  void parseQregDeclaration() {
    const auto qreg = expect(Token::Kind::Qreg);
    const auto debugInfo = makeDebugInfo(qreg);
    const auto id = expect(Token::Kind::Identifier).str;
    std::optional<int64_t> size;
    if (current().kind == Token::Kind::LBracket) {
      size = parseDesignator(debugInfo);
    }
    expect(Token::Kind::Semicolon);
    registerDeclaredName(qreg, id);
    if (size) {
      const auto reg = builder.allocQubitRegister(*size);
      qubitRegisters[id] = {reg.value, reg.qubits};
    } else {
      qubitRegisters[id] = {nullptr, {builder.allocQubit()}};
    }
  }

  /// Parse `bit[<n>] <id> (= <measurement>);`.
  void parseBitDeclaration() {
    const auto bit = expect(Token::Kind::Bit);
    const auto debugInfo = makeDebugInfo(bit);

    std::optional<int64_t> size;
    if (current().kind == Token::Kind::LBracket) {
      size = parseDesignator(debugInfo);
    }

    const auto id = expect(Token::Kind::Identifier).str;

    std::optional<ParsedOperand> operand;
    if (current().kind == Token::Kind::Equals) {
      advance();
      expect(Token::Kind::Measure);
      operand = parseGateOperand();
    }

    expect(Token::Kind::Semicolon);

    registerDeclaredName(bit, id);
    classicalRegisters[id] =
        builder.allocClassicalBitRegister(size.value_or(1), id);

    if (operand) {
      emitMeasureAssignment(ParsedBitRef{id, std::nullopt}, *operand,
                            debugInfo);
    }
  }

  /// Parse `creg <id>[<n>];`.
  void parseCregDeclaration() {
    const auto creg = expect(Token::Kind::CReg);
    const auto debugInfo = makeDebugInfo(creg);
    const auto id = expect(Token::Kind::Identifier).str;
    std::optional<int64_t> size;
    if (current().kind == Token::Kind::LBracket) {
      size = parseDesignator(debugInfo);
    }
    expect(Token::Kind::Semicolon);
    registerDeclaredName(creg, id);
    classicalRegisters[id] =
        builder.allocClassicalBitRegister(size.value_or(1), id);
  }

  /// Parse a `[<expression>]` designator.
  int64_t parseDesignator(const std::shared_ptr<DebugInfo>& debugInfo) {
    expect(Token::Kind::LBracket);
    const auto expr = parseExpression();
    expect(Token::Kind::RBracket);
    return evaluateIntegerConstant(expr, debugInfo);
  }

  void registerDeclaredName(const Token& token, const std::string& id) {
    if (!declaredNames.insert(id).second) {
      error(token, "Identifier '" + id + "' already declared.");
    }
  }

  //===--- Assignments --------------------------------------------------===//

  /// Parse `c = measure q;`.
  void parseAssignment() {
    const auto debugInfo = makeDebugInfo(current());
    const auto target = parseBitRef();

    if (current().kind != Token::Kind::Equals) {
      error(current(), "Classical computations are not supported yet.");
    }
    advance();

    if (current().kind != Token::Kind::Measure) {
      error(current(), "Classical computations are not supported yet.");
    }
    advance();

    const auto operand = parseGateOperand();
    expect(Token::Kind::Semicolon);

    emitMeasureAssignment(target, operand, debugInfo);
  }

  //===--- Measurement --------------------------------------------------===//

  /// Parse `measure q -> c;`.
  void parseMeasure() {
    const auto measure = expect(Token::Kind::Measure);
    const auto debugInfo = makeDebugInfo(measure);
    const auto operand = parseGateOperand();
    expect(Token::Kind::Arrow);
    const auto target = parseBitRef();
    expect(Token::Kind::Semicolon);
    emitMeasureAssignment(target, operand, debugInfo);
  }

  void emitMeasureAssignment(const ParsedBitRef& target,
                             const ParsedOperand& operand,
                             const std::shared_ptr<DebugInfo>& debugInfo) {
    const auto bits = resolveClassicalBits(target, debugInfo);
    const auto resolved = resolveGateOperand(operand, debugInfo);
    SmallVector<Value> qubits;
    if (std::holds_alternative<Value>(resolved)) {
      qubits.push_back(std::get<Value>(resolved));
    } else {
      qubits = std::get<SmallVector<Value>>(resolved);
    }
    if (bits.size() != qubits.size()) {
      error(*debugInfo, "The classical register and the quantum register must "
                        "have the same width.");
    }
    for (const auto& [bit, qubit] : llvm::zip_equal(bits, qubits)) {
      const auto& registerName = bit.registerName;
      const auto registerSize = bit.registerSize;
      const auto registerIndex = bit.registerIndex;
      auto result =
          MeasureOp::create(builder, qubit, builder.getStringAttr(registerName),
                            builder.getI64IntegerAttr(registerSize),
                            builder.getI64IntegerAttr(registerIndex))
              .getResult();
      auto& registerBits = bitValues[registerName];
      const auto index = static_cast<size_t>(registerIndex);
      if (registerBits.size() <= index) {
        registerBits.resize(index + 1);
      }
      registerBits[index] = result;
    }
  }

  //===--- Barrier ------------------------------------------------------===//

  void parseBarrier() {
    const auto barrier = expect(Token::Kind::Barrier);
    const auto debugInfo = makeDebugInfo(barrier);
    SmallVector<Value> qubits;
    while (current().kind != Token::Kind::Semicolon) {
      const auto operand = parseGateOperand();
      const auto resolved = resolveGateOperand(operand, debugInfo);
      if (std::holds_alternative<Value>(resolved)) {
        qubits.push_back(std::get<Value>(resolved));
      } else {
        llvm::append_range(qubits, std::get<SmallVector<Value>>(resolved));
      }
      if (current().kind != Token::Kind::Semicolon) {
        expect(Token::Kind::Comma);
      }
    }
    expect(Token::Kind::Semicolon);
    builder.barrier(qubits);
  }

  //===--- Reset --------------------------------------------------------===//

  void parseReset() {
    const auto reset = expect(Token::Kind::Reset);
    const auto debugInfo = makeDebugInfo(reset);
    const auto operand = parseGateOperand();
    const auto resolved = resolveGateOperand(operand, debugInfo);
    if (std::holds_alternative<Value>(resolved)) {
      builder.reset(std::get<Value>(resolved));
    } else {
      for (auto qubit : std::get<SmallVector<Value>>(resolved)) {
        builder.reset(qubit);
      }
    }
    expect(Token::Kind::Semicolon);
  }

  //===--- SCF ----------------------------------------------------------===//

  void parseIf() {
    expect(Token::Kind::If);
    expect(Token::Kind::LParen);
    auto condition = parseCondition();
    expect(Token::Kind::RParen);

    // Empty then block
    if (current().kind == Token::Kind::LBrace &&
        peek().kind == Token::Kind::RBrace) {
      expect(Token::Kind::LBrace);
      expect(Token::Kind::RBrace);
      if (current().kind != Token::Kind::Else) {
        return;
      }
      expect(Token::Kind::Else);
      auto trueValue = builder.boolConstant(true);
      condition =
          arith::XOrIOp::create(builder, condition, trueValue).getResult();
      auto ifOp = scf::IfOp::create(builder, condition,
                                    /*withElseRegion=*/false);
      OpBuilder::InsertionGuard guard(builder);
      auto* thenBlock = &ifOp.getThenRegion().front();
      builder.setInsertionPointToStart(thenBlock);
      parseBlockOrStatement();
      return;
    }

    auto ifOp = scf::IfOp::create(builder, condition, /*withElseRegion=*/false);

    OpBuilder::InsertionGuard guard(builder);

    auto* thenBlock = &ifOp.getThenRegion().front();
    builder.setInsertionPointToStart(thenBlock);
    parseBlockOrStatement();

    if (current().kind == Token::Kind::Else) {
      expect(Token::Kind::Else);
      if (current().kind == Token::Kind::LBrace &&
          peek().kind == Token::Kind::RBrace) {
        expect(Token::Kind::LBrace);
        expect(Token::Kind::RBrace);
        return;
      }

      auto* elseBlock = builder.createBlock(&ifOp.getElseRegion());
      builder.setInsertionPointToStart(elseBlock);
      parseBlockOrStatement();
      scf::YieldOp::create(builder);
    }
  }

  void parseFor() {
    const auto forToken = expect(Token::Kind::For);
    const auto debugInfo = makeDebugInfo(forToken);

    if (current().kind != Token::Kind::Int &&
        current().kind != Token::Kind::Uint) {
      error(current(), "Expected 'int' or 'uint' after 'for'.");
    }
    advance();

    const auto loopVariable = expect(Token::Kind::Identifier).str;

    expect(Token::Kind::In);
    expect(Token::Kind::LBracket);

    const auto start = parseExpression();
    expect(Token::Kind::Colon);
    const auto second = parseExpression();

    ParsedExpr step = {.kind = ParsedExpr::Kind::IntLiteral, .intValue = 1};
    ParsedExpr stop;
    if (current().kind == Token::Kind::Colon) {
      advance();
      step = second;
      stop = parseExpression();
    } else {
      stop = second;
    }

    expect(Token::Kind::RBracket);

    auto startVal = emitIntegerExpression(start, debugInfo);
    auto stepVal = emitIntegerExpression(step, debugInfo);
    auto stopVal = emitIntegerExpression(stop, debugInfo);

    // OpenQASM 3's range is inclusive of the stop value
    auto one =
        arith::ConstantOp::create(builder, builder.getIndexAttr(1)).getResult();
    stopVal = arith::AddIOp::create(builder, stopVal, one).getResult();

    loopVariables.push();
    dynamicallyLoadedQubits.push();

    builder.scfFor(startVal, stopVal, stepVal, [&](Value iv) {
      loopVariables.emplace(loopVariable, iv);
      parseBlockOrStatement();
    });

    dynamicallyLoadedQubits.pop();
    loopVariables.pop();
  }

  void parseWhile() {
    expect(Token::Kind::While);
    expect(Token::Kind::LParen);

    builder.scfWhile(
        [&] {
          dynamicallyLoadedQubits.push();
          const auto condition = parseCondition();
          expect(Token::Kind::RParen);
          builder.scfCondition(condition);
          dynamicallyLoadedQubits.pop();
        },
        [&] {
          dynamicallyLoadedQubits.push();
          parseBlockOrStatement();
          dynamicallyLoadedQubits.pop();
        });
  }

  /// Translate a condition to an `i1`-typed MLIR value.
  [[nodiscard]] Value parseCondition() {
    const auto debugInfo = makeDebugInfo(current());

    // Measurement (e.g., measure q)
    if (current().kind == Token::Kind::Measure) {
      advance();
      const auto operand = parseGateOperand();
      const auto resolved = resolveGateOperand(operand, debugInfo);
      if (!std::holds_alternative<Value>(resolved)) {
        error(*debugInfo, "Measurement condition must be a single qubit.");
      }
      return builder.measure(std::get<Value>(resolved));
    }

    // Unary negation (!c[0] or ~c[0])
    if (current().kind == Token::Kind::ExclamationPoint ||
        current().kind == Token::Kind::Tilde) {
      advance();
      if (current().kind != Token::Kind::Identifier) {
        error(current(), "Unary expression has unsupported operand.");
      }
      const auto bit = parseBitRef();
      const auto value = lookupBitValue(bit, debugInfo);
      auto trueValue = builder.boolConstant(true);
      return arith::XOrIOp::create(builder, value, trueValue).getResult();
    }

    // Single bit (c or c[0]), or an unsupported register comparison
    if (current().kind == Token::Kind::Identifier) {
      const auto bit = parseBitRef();
      switch (current().kind) {
      case Token::Kind::DoubleEquals:
      case Token::Kind::NotEquals:
      case Token::Kind::LessThan:
      case Token::Kind::GreaterThan:
      case Token::Kind::LessThanEquals:
      case Token::Kind::GreaterThanEquals:
        error(*debugInfo, "Register comparisons are not supported.");
      default:
        break;
      }
      return lookupBitValue(bit, debugInfo);
    }

    error(*debugInfo, "Unsupported condition expression in if statement.");
  }

  /// Look up the most recent measurement result for a classical bit.
  [[nodiscard]] Value
  lookupBitValue(const ParsedBitRef& bit,
                 const std::shared_ptr<DebugInfo>& debugInfo) const {
    const auto& registerName = bit.name;
    auto it = bitValues.find(registerName);
    if (it == bitValues.end()) {
      error(*debugInfo, "No classical bit of register '" + registerName +
                            "' has been measured yet.");
    }
    const auto& registerBits = it->second;

    if (!bit.index) {
      assert(registerBits.size() == 1);
      return registerBits[0];
    }

    const auto index = evaluateNonNegativeConstant(*bit.index, debugInfo);
    if (index >= registerBits.size() || !registerBits[index]) {
      error(*debugInfo, "Bit " + std::to_string(index) + " of register '" +
                            registerName + "' has been not measured yet.");
    }
    return registerBits[index];
  }

  //===--- Gate declarations --------------------------------------------===//

  void parseGateDeclaration() {
    const auto gate = expect(Token::Kind::Gate);

    const auto id = expect(Token::Kind::Identifier).str;
    if (gates.contains(id)) {
      error(gate, "Gate '" + id + "' already declared.");
    }

    // Parse parameters
    std::vector<std::string> parameters;
    if (current().kind == Token::Kind::LParen) {
      advance();
      parameters = parseIdentifierList();
      expect(Token::Kind::RParen);
    }

    // Parse targets
    const auto targets = parseIdentifierList();

    // Parse body
    expect(Token::Kind::LBrace);
    std::vector<ParsedGateCall> body;
    while (current().kind != Token::Kind::RBrace) {
      body.push_back(parseGateCall());
    }
    expect(Token::Kind::RBrace);

    // Verify parameters
    for (size_t i = 0; i < parameters.size(); ++i) {
      for (size_t j = i + 1; j < parameters.size(); ++j) {
        if (parameters[i] == parameters[j]) {
          error(gate, "Parameter is already declared in compound gate.");
        }
      }
    }

    // Verify targets
    for (size_t i = 0; i < targets.size(); ++i) {
      for (size_t j = i + 1; j < targets.size(); ++j) {
        if (targets[i] == targets[j]) {
          error(gate, "Target is already declared in compound gate.");
        }
      }
    }

    gates[id] = ParsedCompoundGate{std::move(parameters), std::move(targets),
                                   std::move(body)};
  }

  std::vector<std::string> parseIdentifierList() {
    std::vector<std::string> identifiers;
    identifiers.push_back(expect(Token::Kind::Identifier).str);
    while (current().kind == Token::Kind::Comma) {
      advance();
      identifiers.push_back(expect(Token::Kind::Identifier).str);
    }
    return identifiers;
  }

  //===--- Gate calls ---------------------------------------------------===//

  ParsedGateCall parseGateCall() {
    ParsedGateCall call;
    call.debugInfo = makeDebugInfo(current());

    // Parse modifiers
    while (current().kind == Token::Kind::Inv ||
           current().kind == Token::Kind::Pow ||
           current().kind == Token::Kind::Ctrl ||
           current().kind == Token::Kind::NegCtrl) {
      call.modifiers.push_back(parseGateModifier());
      expect(Token::Kind::At);
    }

    // Parse identifier
    if (current().kind == Token::Kind::Gphase) {
      advance();
      call.identifier = "gphase";
    } else {
      call.identifier = expect(Token::Kind::Identifier).str;
    }

    // Parse parameters
    if (current().kind == Token::Kind::LParen) {
      advance();
      while (current().kind != Token::Kind::RParen) {
        call.parameters.push_back(parseExpression());
        if (current().kind != Token::Kind::RParen) {
          expect(Token::Kind::Comma);
        }
      }
      expect(Token::Kind::RParen);
    }

    if (current().kind == Token::Kind::LBracket) {
      error(current(), "Gate calls with designators are not supported yet.");
    }

    // Parse operands
    while (current().kind != Token::Kind::Semicolon) {
      call.operands.push_back(parseGateOperand());
      if (current().kind != Token::Kind::Semicolon) {
        expect(Token::Kind::Comma);
      }
    }

    expect(Token::Kind::Semicolon);
    return call;
  }

  ParsedModifier parseGateModifier() {
    ParsedModifier mod;
    mod.kind = current().kind;
    switch (current().kind) {
    case Token::Kind::Inv:
      advance();
      return mod;
    case Token::Kind::Pow:
      advance();
      expect(Token::Kind::LParen);
      mod.expression = parseExpression();
      expect(Token::Kind::RParen);
      return mod;
    case Token::Kind::Ctrl:
    case Token::Kind::NegCtrl:
      advance();
      if (current().kind == Token::Kind::LParen) {
        advance();
        mod.expression = parseExpression();
        expect(Token::Kind::RParen);
      }
      return mod;
    default:
      llvm_unreachable("Unknown gate modifier");
    }
  }

  ParsedOperand parseGateOperand() {
    ParsedOperand operand;
    if (current().kind == Token::Kind::HardwareQubit) {
      operand.hardwareQubit = static_cast<uint64_t>(current().val);
      advance();
      return operand;
    }
    operand.name = expect(Token::Kind::Identifier).str;
    if (current().kind == Token::Kind::LBracket) {
      advance();
      operand.index = parseExpression();
      expect(Token::Kind::RBracket);
    }
    return operand;
  }

  ParsedBitRef parseBitRef() {
    ParsedBitRef ref;
    ref.name = expect(Token::Kind::Identifier).str;
    if (current().kind == Token::Kind::LBracket) {
      advance();
      ref.index = parseExpression();
      expect(Token::Kind::RBracket);
    }
    return ref;
  }

  /// Resolve and emit a @p gate against @p scope.
  void emitGateCall(const ParsedGateCall& call, const QubitScope& scope) {
    const auto& id = call.identifier;
    const auto& debugInfo = call.debugInfo;

    // Resolve identifier
    auto it = gates.find(id);

    // OpenQASM 2 compatibility: strip leading `c` characters and treat them as
    // implicit control modifiers
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
      error(*debugInfo, "No OpenQASM definition found for gate '" + id + "'.");
    }

    // Resolve parameters
    SmallVector<Value> params;
    params.reserve(call.parameters.size());
    for (const auto& arg : call.parameters) {
      params.push_back(emitFloatExpression(arg, debugInfo));
    }

    // Resolve operands
    SmallVector<Value> operands;
    SmallVector<SmallVector<Value>> operandsBroadcasting;
    for (const auto& operand : call.operands) {
      const auto resolved =
          resolveGateOperandInScope(operand, scope, debugInfo);
      if (const auto* value = std::get_if<Value>(&resolved)) {
        operands.push_back(*value);
      } else {
        operandsBroadcasting.push_back(std::get<SmallVector<Value>>(resolved));
      }
    }

    if (!operandsBroadcasting.empty() && !operands.empty()) {
      error(*debugInfo, "Gate operands must be single qubits or quantum "
                        "registers and not a mix of both.");
    }

    // Handle broadcasted calls
    if (!operandsBroadcasting.empty()) {
      if (numCompatControls != 0) {
        error(*debugInfo, "OpenQASM 2 gates cannot be broadcasted.");
      }
      if (std::holds_alternative<ParsedCompoundGate>(it->second)) {
        error(*debugInfo, "Broadcasted compound gates are not supported yet.");
      }
      emitBroadcastedGateCall(std::get<GateInfo>(it->second), id, resolvedId,
                              params, call.modifiers, operandsBroadcasting,
                              debugInfo);
      return;
    }

    // Handle non-broadcasted calls
    const auto split = splitControlsAndTargets<Value>(
        call.modifiers, numCompatControls, operands, debugInfo);

    // Inline compound gate
    if (const auto* compound = std::get_if<ParsedCompoundGate>(&it->second)) {
      emitCompoundGate(*compound, params, split.targets, split.posControls,
                       split.negControls, split.invert, debugInfo);
      return;
    }

    // Emit standard gate
    const auto& gate = std::get<GateInfo>(it->second);
    const auto& gateFn =
        resolveStandardGate(gate, id, resolvedId, params, debugInfo);
    emitStandardGate(gateFn, params, split.targets, split.posControls,
                     split.negControls, split.invert);
  }

  /// Emit a broadcasted gate call.
  void emitBroadcastedGateCall(const GateInfo& gate, const std::string& fullId,
                               llvm::StringRef resolvedId, ValueRange params,
                               const std::vector<ParsedModifier>& modifiers,
                               const SmallVector<SmallVector<Value>>& operands,
                               const std::shared_ptr<DebugInfo>& debugInfo) {
    const auto broadcastWidth = operands.front().size();
    for (const auto& operand : operands) {
      if (operand.size() != broadcastWidth) {
        error(*debugInfo,
              "All broadcasting operands must have the same width.");
      }
    }

    // OpenQASM 2 gates cannot be broadcasted
    const auto split = splitControlsAndTargets<SmallVector<Value>>(
        modifiers, /*numCompatControls=*/0, operands, debugInfo);

    const auto& gateFn =
        resolveStandardGate(gate, fullId, resolvedId, params, debugInfo);

    for (size_t b = 0; b < broadcastWidth; ++b) {
      const auto slice = [&](const SmallVector<SmallVector<Value>>& lists) {
        SmallVector<Value> qubits;
        qubits.reserve(lists.size());
        for (const auto& list : lists) {
          qubits.push_back(list[b]);
        }
        return qubits;
      };
      emitStandardGate(gateFn, params, slice(split.targets),
                       slice(split.posControls), slice(split.negControls),
                       split.invert);
    }
  }

  template <typename T> struct ControlsAndTargets {
    bool invert = false;
    SmallVector<T> posControls;
    SmallVector<T> negControls;
    SmallVector<T> targets;
  };

  /// Partition @p operands into controls and targets.
  template <typename T>
  ControlsAndTargets<T>
  splitControlsAndTargets(const std::vector<ParsedModifier>& modifiers,
                          size_t numCompatControls,
                          const SmallVector<T>& operands,
                          const std::shared_ptr<DebugInfo>& debugInfo) const {
    ControlsAndTargets<T> result;
    size_t numControls = 0;
    for (const auto& mod : modifiers) {
      switch (mod.kind) {
      case Token::Kind::Inv:
        result.invert = !result.invert;
        break;
      case Token::Kind::Pow:
        error(*debugInfo, "Power modifiers are not supported yet.");
      case Token::Kind::Ctrl: {
        const auto n =
            evaluateNonNegativeConstant(mod.expression, debugInfo, 1);
        for (size_t i = 0; i < n; ++i, ++numControls) {
          if (numControls >= operands.size()) {
            error(*debugInfo, "Control index out of bounds.");
          }
          result.posControls.push_back(operands[numControls]);
        }
        break;
      }
      case Token::Kind::NegCtrl: {
        const auto n =
            evaluateNonNegativeConstant(mod.expression, debugInfo, 1);
        for (size_t i = 0; i < n; ++i, ++numControls) {
          if (numControls >= operands.size()) {
            error(*debugInfo, "Control index out of bounds.");
          }
          result.negControls.push_back(operands[numControls]);
        }
        break;
      }
      default:
        llvm_unreachable("Unknown gate modifier");
      }
    }

    // OpenQASM 2 compatibility: append implicit control qubits
    for (size_t i = 0; i < numCompatControls; ++i, ++numControls) {
      if (numControls >= operands.size()) {
        error(*debugInfo, "Control index out of bounds.");
      }
      result.posControls.push_back(operands[numControls]);
    }

    result.targets = llvm::to_vector(llvm::drop_begin(operands, numControls));
    return result;
  }

  /// Look up the `QCProgramBuilder` emitter of a @p gate.
  const GateFn&
  resolveStandardGate(const GateInfo& gate, const std::string& fullId,
                      llvm::StringRef resolvedId, ValueRange params,
                      const std::shared_ptr<DebugInfo>& debugInfo) const {
    const auto dispIt = GATE_DISPATCH.find(resolvedId);
    if (dispIt == GATE_DISPATCH.end()) {
      error(*debugInfo, "No MLIR definition found for gate '" + fullId + "'.");
    }
    if (gate.nParameters != params.size()) {
      error(*debugInfo,
            "Invalid number of parameters for gate '" + fullId + "'.");
    }
    return dispIt->second;
  }

  /// Emit a gate body wrapped in its modifiers.
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

  /// Emit a standard gate.
  void emitStandardGate(const GateFn& gateFn, ValueRange params,
                        ValueRange targets, ValueRange posControls,
                        ValueRange negControls, bool invert) {
    auto bodyFn = [&](ValueRange qubits) { gateFn(builder, qubits, params); };
    emitModifiedGate(bodyFn, targets, posControls, negControls, invert);
  }

  /// Inline a compound gate.
  void emitCompoundGate(const ParsedCompoundGate& gate, ValueRange params,
                        ValueRange targets, ValueRange posControls,
                        ValueRange negControls, bool invert,
                        const std::shared_ptr<DebugInfo>& debugInfo) {
    assert(gate.parameterNames.size() == params.size());
    assert(gate.targetNames.size() == targets.size());

    // Map from internal target name to index in targets list. This map is
    // needed because the qubits may be aliased if the compound gate is inlined
    // within a modifier region.
    llvm::StringMap<SmallVector<size_t>> targetsMap;

    for (const auto& [targetName, target] :
         llvm::zip_equal(gate.targetNames, targets)) {
      auto iter = llvm::find(targets, target);
      if (iter == targets.end()) {
        error(*debugInfo, "Target '" + targetName + "' not found in operands.");
      }
      const auto index =
          static_cast<size_t>(std::distance(targets.begin(), iter));
      targetsMap[targetName].push_back(index);
    }

    // Bind parameters so they can be referenced by name inside the gate body.
    parameterConstants.push();
    for (size_t i = 0; i < gate.parameterNames.size(); ++i) {
      parameterConstants.emplace(gate.parameterNames[i], params[i]);
    }

    auto bodyFn = [&](ValueRange qubits) {
      QubitScope localScope;
      for (const auto& [name, indices] : targetsMap) {
        SmallVector<Value> args;
        for (auto index : indices) {
          args.push_back(qubits[index]);
        }
        localScope[name] = {nullptr, std::move(args)};
      }
      for (const auto& bodyCall : gate.body) {
        emitGateCall(bodyCall, localScope);
      }
    };

    emitModifiedGate(bodyFn, targets, posControls, negControls, invert);

    parameterConstants.pop();
  }

  //===--- Operand resolution -------------------------------------------===//

  [[nodiscard]] std::variant<Value, SmallVector<Value>>
  resolveGateOperand(const ParsedOperand& operand,
                     const std::shared_ptr<DebugInfo>& debugInfo) {
    return resolveGateOperandInScope(operand, qubitRegisters, debugInfo);
  }

  [[nodiscard]] std::variant<Value, SmallVector<Value>>
  resolveGateOperandInScope(const ParsedOperand& operand,
                            const QubitScope& scope,
                            const std::shared_ptr<DebugInfo>& debugInfo) {
    if (operand.hardwareQubit) {
      return builder.staticQubit(*operand.hardwareQubit);
    }

    const auto& name = operand.name;
    auto it = scope.find(name);
    if (it == scope.end()) {
      error(*debugInfo, "Unknown qubit register '" + name + "'.");
    }

    const auto& binding = it->second;

    if (!operand.index) {
      if (binding.qubits.size() == 1) {
        return binding.qubits[0];
      }
      // Return full register
      return binding.qubits;
    }

    const auto& indexExpr = *operand.index;
    if (isConstantIndex(indexExpr)) {
      const auto index = evaluateNonNegativeConstant(indexExpr, debugInfo);
      if (index >= binding.qubits.size()) {
        error(*debugInfo, "Qubit index out of bounds.");
      }
      return binding.qubits[index];
    }

    if (!binding.memref) {
      error(*debugInfo, "Dynamic qubit indexing requires a qubit register.");
    }
    return loadDynamicElement(name, binding.memref, indexExpr, debugInfo);
  }

  /// Whether @p expr can be folded to a constant at translation time.
  static bool isConstantIndex(const ParsedExpr& expr) {
    switch (expr.kind) {
    case ParsedExpr::Kind::IntLiteral:
    case ParsedExpr::Kind::FloatLiteral:
      return true;
    case ParsedExpr::Kind::Ident:
      return false;
    case ParsedExpr::Kind::Unary:
      return isConstantIndex(expr.children[0]);
    case ParsedExpr::Kind::Binary:
      return isConstantIndex(expr.children[0]) &&
             isConstantIndex(expr.children[1]);
    }
    return false;
  }

  /// Get a stable string representation of an index expression.
  static std::string getIndexKey(const ParsedExpr& expr) {
    switch (expr.kind) {
    case ParsedExpr::Kind::IntLiteral:
      return std::to_string(expr.intValue);
    case ParsedExpr::Kind::FloatLiteral:
      return std::to_string(expr.floatValue);
    case ParsedExpr::Kind::Ident:
      return expr.name;
    case ParsedExpr::Kind::Unary:
      return "(" + Token::kindToString(expr.op) +
             getIndexKey(expr.children[0]) + ")";
    case ParsedExpr::Kind::Binary:
      return "(" + getIndexKey(expr.children[0]) +
             Token::kindToString(expr.op) + getIndexKey(expr.children[1]) + ")";
    }
    return "";
  }

  /**
   * @brief Load a qubit from @p memref at a runtime index
   *
   * @details
   * The result within the current region is cached so that repeated references
   * to the same element reuse a single load.
   */
  [[nodiscard]] Value
  loadDynamicElement(const std::string& name, Value memref,
                     const ParsedExpr& indexExpr,
                     const std::shared_ptr<DebugInfo>& debugInfo) {
    const auto key = name + "[" + getIndexKey(indexExpr) + "]";
    if (const auto cached = dynamicallyLoadedQubits.find(key)) {
      return *cached;
    }
    const auto index = emitIntegerExpression(indexExpr, debugInfo);
    const auto loaded = builder.memrefLoad(memref, index);
    dynamicallyLoadedQubits.emplace(key, loaded);
    return loaded;
  }

  //===--- Bit resolution -----------------------------------------------===//

  [[nodiscard]] SmallVector<QCProgramBuilder::Bit>
  resolveClassicalBits(const ParsedBitRef& operand,
                       const std::shared_ptr<DebugInfo>& debugInfo) const {
    const auto& name = operand.name;
    auto it = classicalRegisters.find(name);
    if (it == classicalRegisters.end()) {
      error(*debugInfo, "Unknown classical register '" + name + "'.");
    }

    const auto& creg = it->second;
    SmallVector<QCProgramBuilder::Bit> bits;

    if (!operand.index) {
      for (int64_t i = 0; i < creg.size; ++i) {
        bits.push_back(creg[i]);
      }
      return bits;
    }

    const auto index = evaluateNonNegativeConstant(*operand.index, debugInfo);
    if (std::cmp_greater_equal(index, creg.size)) {
      error(*debugInfo, "Classical bit index out of bounds.");
    }
    bits.push_back(creg[static_cast<int64_t>(index)]);
    return bits;
  }

  //===--- Expression parsing -------------------------------------------===//

  /// expr := term (('+' | '-') term)*
  ParsedExpr parseExpression() {
    auto x = parseTerm();
    while (current().kind == Token::Kind::Plus ||
           current().kind == Token::Kind::Minus) {
      const auto op = current().kind;
      advance();
      auto y = parseTerm();
      ParsedExpr binary;
      binary.kind = ParsedExpr::Kind::Binary;
      binary.op = op;
      binary.children.push_back(std::move(x));
      binary.children.push_back(std::move(y));
      x = std::move(binary);
    }
    return x;
  }

  /// term := unary (('*' | '/') unary)*
  ParsedExpr parseTerm() {
    auto x = parseUnary();
    while (current().kind == Token::Kind::Asterisk ||
           current().kind == Token::Kind::Slash) {
      const auto op = current().kind;
      advance();
      auto y = parseUnary();
      ParsedExpr binary;
      binary.kind = ParsedExpr::Kind::Binary;
      binary.op = op;
      binary.children.push_back(std::move(x));
      binary.children.push_back(std::move(y));
      x = std::move(binary);
    }
    return x;
  }

  /// unary := '-' unary | primary
  ParsedExpr parseUnary() {
    if (current().kind == Token::Kind::Minus) {
      advance();
      ParsedExpr unary;
      unary.kind = ParsedExpr::Kind::Unary;
      unary.op = Token::Kind::Minus;
      unary.children.push_back(parseUnary());
      return unary;
    }
    return parsePrimary();
  }

  /// primary := literal | ident | '(' expr ')'
  ParsedExpr parsePrimary() {
    ParsedExpr e;
    switch (current().kind) {
    case Token::Kind::FloatLiteral: {
      const auto value = expect(Token::Kind::FloatLiteral);
      e.kind = ParsedExpr::Kind::FloatLiteral;
      e.floatValue = value.valReal;
      return e;
    }
    case Token::Kind::IntegerLiteral: {
      const auto value = expect(Token::Kind::IntegerLiteral);
      e.kind = ParsedExpr::Kind::IntLiteral;
      e.intValue = value.val;
      return e;
    }
    case Token::Kind::Identifier: {
      const auto value = expect(Token::Kind::Identifier);
      e.kind = ParsedExpr::Kind::Ident;
      e.name = value.str;
      return e;
    }
    case Token::Kind::LParen: {
      expect(Token::Kind::LParen);
      auto inner = parseExpression();
      expect(Token::Kind::RParen);
      return inner;
    }
    default:
      error(current(),
            "Expected expression, got '" + current().toString() + "'.");
    }
  }

  //===--- Expression evaluation ----------------------------------------===//

  /// Translate a `ParsedExpr` to an `f64`-typed MLIR value.
  [[nodiscard]] Value
  emitFloatExpression(const ParsedExpr& expr,
                      const std::shared_ptr<DebugInfo>& debugInfo) {
    switch (expr.kind) {
    case ParsedExpr::Kind::IntLiteral:
      return arith::ConstantOp::create(builder,
                                       builder.getF64FloatAttr(expr.intValue))
          .getResult();
    case ParsedExpr::Kind::FloatLiteral:
      return arith::ConstantOp::create(builder,
                                       builder.getF64FloatAttr(expr.floatValue))
          .getResult();
    case ParsedExpr::Kind::Ident:
      return resolveParameterIdentifier(expr.name, debugInfo);
    case ParsedExpr::Kind::Unary: {
      const auto operand = emitFloatExpression(expr.children[0], debugInfo);
      if (expr.op == Token::Kind::Minus) {
        return arith::NegFOp::create(builder, operand).getResult();
      }
      error(*debugInfo,
            "Unsupported unary operator in gate parameter expression.");
    }
    case ParsedExpr::Kind::Binary: {
      const auto lhs = emitFloatExpression(expr.children[0], debugInfo);
      const auto rhs = emitFloatExpression(expr.children[1], debugInfo);
      switch (expr.op) {
      case Token::Kind::Plus:
        return arith::AddFOp::create(builder, lhs, rhs).getResult();
      case Token::Kind::Minus:
        return arith::SubFOp::create(builder, lhs, rhs).getResult();
      case Token::Kind::Asterisk:
        return arith::MulFOp::create(builder, lhs, rhs).getResult();
      case Token::Kind::Slash:
        return arith::DivFOp::create(builder, lhs, rhs).getResult();
      default:
        error(*debugInfo,
              "Unsupported binary operator in gate parameter expression.");
      }
    }
    }
    error(*debugInfo, "Unsupported gate parameter expression.");
  }

  [[nodiscard]] Value
  resolveParameterIdentifier(const std::string& name,
                             const std::shared_ptr<DebugInfo>& debugInfo) {
    if (const auto value = parameterConstants.find(name)) {
      return *value;
    }
    if (const auto value = lookupBuiltinConstant(name, builder)) {
      return *value;
    }
    error(*debugInfo, "Unknown identifier '" + name + "'.");
  }

  /// Translate a `ParsedExpr` to an `index`-typed MLIR value.
  [[nodiscard]] Value
  emitIntegerExpression(const ParsedExpr& expr,
                        const std::shared_ptr<DebugInfo>& debugInfo) {
    switch (expr.kind) {
    case ParsedExpr::Kind::IntLiteral:
      return arith::ConstantOp::create(builder,
                                       builder.getIndexAttr(expr.intValue))
          .getResult();
    case ParsedExpr::Kind::Unary: {
      if (expr.op == Token::Kind::Minus) {
        const auto operand = emitIntegerExpression(expr.children[0], debugInfo);
        const auto zero =
            arith::ConstantOp::create(builder, builder.getIndexAttr(0))
                .getResult();
        return arith::SubIOp::create(builder, zero, operand).getResult();
      }
      error(*debugInfo, "Unsupported unary operator in integer expression.");
    }
    case ParsedExpr::Kind::Binary: {
      const auto lhs = emitIntegerExpression(expr.children[0], debugInfo);
      const auto rhs = emitIntegerExpression(expr.children[1], debugInfo);
      switch (expr.op) {
      case Token::Kind::Plus:
        return arith::AddIOp::create(builder, lhs, rhs).getResult();
      case Token::Kind::Minus:
        return arith::SubIOp::create(builder, lhs, rhs).getResult();
      case Token::Kind::Asterisk:
        return arith::MulIOp::create(builder, lhs, rhs).getResult();
      case Token::Kind::Slash:
        return arith::DivSIOp::create(builder, lhs, rhs).getResult();
      default:
        error(*debugInfo, "Unsupported binary operator in integer expression.");
      }
    }
    case ParsedExpr::Kind::Ident:
      if (const auto iv = loopVariables.find(expr.name)) {
        return *iv;
      }
      error(*debugInfo, "Expected an integer expression.");
    case ParsedExpr::Kind::FloatLiteral:
    default:
      error(*debugInfo, "Expected an integer expression.");
    }
  }

  /// Statically evaluate `ParsedExpr` to an `int64_t`.
  [[nodiscard]] int64_t
  evaluateIntegerConstant(const ParsedExpr& expr,
                          const std::shared_ptr<DebugInfo>& debugInfo) const {
    switch (expr.kind) {
    case ParsedExpr::Kind::IntLiteral:
      return expr.intValue;
    case ParsedExpr::Kind::Unary:
      if (expr.op == Token::Kind::Minus) {
        return -evaluateIntegerConstant(expr.children[0], debugInfo);
      }
      error(*debugInfo, "Expected a constant integer expression.");
    case ParsedExpr::Kind::Binary: {
      const auto lhs = evaluateIntegerConstant(expr.children[0], debugInfo);
      const auto rhs = evaluateIntegerConstant(expr.children[1], debugInfo);
      switch (expr.op) {
      case Token::Kind::Plus:
        return lhs + rhs;
      case Token::Kind::Minus:
        return lhs - rhs;
      case Token::Kind::Asterisk:
        return lhs * rhs;
      case Token::Kind::Slash:
        if (rhs == 0) {
          error(*debugInfo, "Division by zero in constant expression.");
        }
        return lhs / rhs;
      default:
        error(*debugInfo, "Expected a constant integer expression.");
      }
    }
    case ParsedExpr::Kind::FloatLiteral:
    case ParsedExpr::Kind::Ident:
    default:
      error(*debugInfo, "Expected a constant integer expression.");
    }
  }

  /// Statically evaluate `ParsedExpr` to an `size_t`.
  [[nodiscard]] size_t evaluateNonNegativeConstant(
      const ParsedExpr& expr,
      const std::shared_ptr<DebugInfo>& debugInfo) const {
    return static_cast<size_t>(evaluateIntegerConstant(expr, debugInfo));
  }

  /// Statically evaluate `ParsedExpr` to an `size_t`, using @p defaultValue
  /// when the expression is absent.
  [[nodiscard]] size_t
  evaluateNonNegativeConstant(const std::optional<ParsedExpr>& expr,
                              const std::shared_ptr<DebugInfo>& debugInfo,
                              size_t defaultValue) const {
    if (!expr) {
      return defaultValue;
    }
    return evaluateNonNegativeConstant(*expr, debugInfo);
  }
};

} // namespace

namespace detail {

OwningOpRef<ModuleOp> parseQASM3(std::string_view source,
                                 MLIRContext* context) {
  QASM3Parser parser(context);
  return parser.parse(source);
}

} // namespace detail

} // namespace mlir::qc
