/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "OpenQASMLexer.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <utility>

namespace mlir::oq3::frontend::detail {

/// An exact OpenQASM version, preserving the decimal minor component.
struct Version {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
};

enum class ScalarKind : std::uint8_t { Bool, Int, Uint, Float };

/**
 * @defgroup ParseVocabulary Transient parse vocabulary
 * @brief The vocabulary the parser hands to a sink.
 *
 * @details
 * These types are cheap and trivially destructible. Expressions are allocated
 * in a bump allocator; other values borrow parser-local storage for the
 * duration of a sink call. `SyntaxBuilder` copies each completed construct into
 * the persistent, owning syntax program.
 */

/**
 * @ingroup ParseVocabulary
 * @brief A (sub-)expression.
 *
 * @details
 * Bump-allocated; children are borrowed pointers.
 */
struct Expr {
  enum class Kind : uint8_t {
    Int,
    Float,
    Bool,
    Identifier,
    Index,
    Measurement,
    Neg,
    Not,
    BitNot,
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    // Built-in math functions
    ArcCos,
    ArcSin,
    ArcTan,
    Cos,
    Exp,
    Log,
    Mod,
    BuiltinMod,
    BuiltinPow,
    Pow,
    Sin,
    Sqrt,
    Tan,
  };

  SMLoc loc;
  Kind kind = Kind::Int;
  uint64_t intValue = 0;
  double floatValue = 0.0;
  bool boolValue = false;
  StringRef identifier;
  std::optional<uint64_t> hardwareQubit;
  const Expr* lhs = nullptr;
  const Expr* rhs = nullptr;
};

/// Get the kind of the built-in math function @p name.
[[nodiscard]] inline std::optional<Expr::Kind>
getMathFunctionKind(StringRef name) {
  return llvm::StringSwitch<std::optional<Expr::Kind>>(name)
      .Case("arccos", Expr::Kind::ArcCos)
      .Case("arcsin", Expr::Kind::ArcSin)
      .Case("arctan", Expr::Kind::ArcTan)
      .Case("cos", Expr::Kind::Cos)
      .Case("exp", Expr::Kind::Exp)
      .Case("log", Expr::Kind::Log)
      .Case("ln", Expr::Kind::Log)
      .Case("mod", Expr::Kind::BuiltinMod)
      .Case("pow", Expr::Kind::BuiltinPow)
      .Case("sin", Expr::Kind::Sin)
      .Case("sqrt", Expr::Kind::Sqrt)
      .Case("tan", Expr::Kind::Tan)
      .Default(std::nullopt);
}

/**
 * @ingroup ParseVocabulary
 * @brief A gate modifier: `inv @`, `pow(e) @`, `ctrl(e) @`, or `negctrl(e) @`.
 */
struct Modifier {
  enum class Kind : uint8_t { Inv, Pow, Ctrl, NegCtrl };
  Kind kind = Kind::Inv;
  const Expr* argument = nullptr;
};

/**
 * @ingroup ParseVocabulary
 * @brief A gate operand: a (possibly indexed) identifier, or a hardware qubit.
 */
struct Operand {
  SMLoc loc;
  StringRef identifier;
  const Expr* index = nullptr;
  std::optional<uint64_t> hardwareQubit;
};

/// A (possibly indexed) classical reference (e.g., `c` or `c[0]`).
struct BitReference {
  SMLoc loc;
  StringRef identifier;
  const Expr* index = nullptr;
};

/**
 * @ingroup ParseVocabulary
 * @brief A parsed gate call.
 *
 * @details
 * Array members are borrowed for the duration of the sink call.
 */
struct GateCall {
  SMLoc loc;
  StringRef identifier;
  ArrayRef<Modifier> modifiers;
  ArrayRef<const Expr*> parameters;
  ArrayRef<Operand> operands;
};

//===----------------------------------------------------------------------===//
// Sink concept
//===----------------------------------------------------------------------===//

/**
 * @brief The interface a `Parser` drives to materialize parsed constructs.
 *
 * @details
 * A sink consumes the events produced by `Parser`. The production sink copies
 * them into target-independent persistent syntax; the parser is templated so
 * dispatch remains static. Diagnostics are routed through `error`.
 *
 * Control flow uses continuations so the persistent syntax builder can select
 * the destination body while the parser recursively consumes a source block.
 */
template <class S>
concept QASMSink =
    requires(S s, SMLoc loc, StringRef str, const Expr& expr,
             const Operand& operand, const BitReference& reference,
             const GateCall& call, ArrayRef<Operand> operands,
             ArrayRef<StringRef> names, function_ref<LogicalResult()> cont,
             Version version, bool flag) {
      s.error(loc, str);
      s.version(loc, version);
      s.include(loc, str);
      s.scalarDecl(loc, ScalarKind::Int, str, &expr, flag);
      s.assignment(loc, reference, expr);
      s.qubitRegister(loc, str, &expr);
      s.classicalRegister(loc, str, &expr, &expr, flag);
      s.measure(loc, &reference, operand);
      s.reset(loc, operand);
      s.barrier(loc, operands);
      s.gateCall(call);
      s.gateDefinition(loc, str, names, names, cont);
      s.ifStmt(loc, expr, cont, cont);
      s.forStmt(loc, str, flag, expr, expr, expr, cont);
      s.whileStmt(loc, expr, cont);
    };

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/**
 * @brief A single-pass recursive-descent parser for OpenQASM 3.
 *
 * @details
 * The parser is target-independent. Its builder materializes a persistent
 * syntax program; expressions and temporary gate-definition vocabulary are
 * bump-allocated only for the duration of parsing.
 */
template <class Sink>
  requires QASMSink<Sink>
class Parser {
public:
  Parser(Lexer& lexer, Sink& sink, llvm::BumpPtrAllocator& allocator)
      : lexer(lexer), sink(sink), allocator(allocator) {
    currentToken = lexer.next();
    nextToken = lexer.next();
  }

  [[nodiscard]] LogicalResult parseProgram() {
    bool hadError = false;
    while (!atEnd()) {
      if (failed(parseStatement())) {
        hadError = true;
        synchronizeAfterError();
      }
    }
    return success(!hadError);
  }

private:
  //===--- Token scaffolding --------------------------------------------===//

  void advance() {
    currentToken = nextToken;
    nextToken = lexer.next();
  }

  void synchronizeAfterError() {
    while (!atEnd() && current().kind != TokenKind::Semicolon &&
           current().kind != TokenKind::RBrace) {
      advance();
    }
    if (!atEnd()) {
      advance();
    }
  }

  [[nodiscard]] const Token& current() const { return currentToken; }
  [[nodiscard]] const Token& peek() const { return nextToken; }
  [[nodiscard]] bool atEnd() const {
    return currentToken.kind == TokenKind::Eof;
  }

  [[nodiscard]] LogicalResult expect(const TokenKind kind) {
    if (current().kind != kind) {
      if (current().kind == TokenKind::UnsupportedKeyword) {
        return unsupportedKeyword();
      }
      return sink.error(current().loc, Twine("expected ") + describe(kind) +
                                           ", got " + describe(current().kind));
    }
    advance();
    return success();
  }

  [[nodiscard]] LogicalResult unsupportedKeyword() {
    return sink.error(current().loc, Twine("reserved keyword '") +
                                         current().spelling +
                                         "' is not supported");
  }

  [[nodiscard]] LogicalResult expectedIdentifier(const StringRef message) {
    if (current().kind == TokenKind::UnsupportedKeyword) {
      return unsupportedKeyword();
    }
    return sink.error(current().loc, message);
  }

  //===--- Allocation helpers -------------------------------------------===//

  [[nodiscard]] Expr* makeExpr() {
    return new (allocator.Allocate<Expr>()) Expr();
  }

  //===--- Program and statements ---------------------------------------===//

  [[nodiscard]] LogicalResult parseStatement() {
    switch (current().kind) {
    case TokenKind::OpenQASM:
      return parseVersion();
    case TokenKind::Include:
      return parseInclude();
    case TokenKind::Const:
    case TokenKind::Bool:
    case TokenKind::Int:
    case TokenKind::Uint:
    case TokenKind::Float:
      return parseScalarDeclaration();
    case TokenKind::Angle:
    case TokenKind::Duration:
      return sink.error(current().loc,
                        "'angle' and 'duration' declarations are not supported "
                        "yet");
    case TokenKind::Qubit:
      return parseQuantumDecl();
    case TokenKind::Qreg:
      return parseQregDecl();
    case TokenKind::Bit:
      return parseClassicalDecl(/*isOutput=*/false);
    case TokenKind::CReg:
      return parseCregDecl();
    case TokenKind::Output:
      return parseOutputDecl();
    case TokenKind::Gate:
      return parseGateStatement();
    case TokenKind::Opaque:
      return sink.error(current().loc,
                        "opaque gate declarations are not supported");
    case TokenKind::Barrier:
      return parseBarrier();
    case TokenKind::Reset:
      return parseReset();
    case TokenKind::Measure:
      return parseMeasure();
    case TokenKind::If:
      return parseIf();
    case TokenKind::For:
      return parseFor();
    case TokenKind::While:
      return parseWhile();
    case TokenKind::Inv:
    case TokenKind::Pow:
    case TokenKind::Ctrl:
    case TokenKind::NegCtrl:
    case TokenKind::Gphase:
      return parseGateCallStatement();
    case TokenKind::Identifier:
      switch (peek().kind) {
      case TokenKind::LBracket:
      case TokenKind::Equals:
      case TokenKind::CompoundAssign:
        return parseAssignment();
      default:
        return parseGateCallStatement();
      }
    case TokenKind::UnterminatedComment:
      return sink.error(current().loc, "unterminated block comment");
    case TokenKind::UnsupportedKeyword:
      return unsupportedKeyword();
    default:
      return sink.error(current().loc, "unexpected token");
    }
  }

  /**
   * @brief Parse a `{ ... }` block or a single statement.
   *
   * @details
   * The block is parsed in a new scope.
   */
  [[nodiscard]] LogicalResult parseBlock() {
    ++blockDepth;
    const auto result = parseBlockInScope();
    --blockDepth;
    return result;
  }

  /**
   * @brief Parse a `{ ... }` block or a single statement.
   *
   * @details
   * The block is parsed into the current scope.
   */
  [[nodiscard]] LogicalResult parseBlockInScope() {
    if (current().kind == TokenKind::LBrace) {
      advance();
      while (!atEnd() && current().kind != TokenKind::RBrace) {
        if (failed(parseStatement())) {
          return failure();
        }
      }
      return expect(TokenKind::RBrace);
    }
    return parseStatement();
  }

  //===--- Helpers ------------------------------------------------------===//

  [[nodiscard]] FailureOr<const Expr*> parseDesignator() {
    if (failed(expect(TokenKind::LBracket))) {
      return failure();
    }
    auto expr = parseExpression();
    if (failed(expr)) {
      return failure();
    }
    if (failed(expect(TokenKind::RBracket))) {
      return failure();
    }
    return *expr;
  }

  //===--- Version ------------------------------------------------------===//

  [[nodiscard]] LogicalResult parseVersion() {
    const auto loc = current().loc;
    advance(); // OPENQASM
    if (current().kind != TokenKind::FloatLiteral &&
        current().kind != TokenKind::IntegerLiteral) {
      return sink.error(current().loc,
                        "version must be a float or integer literal");
    }
    auto majorText = current().spelling;
    StringRef minorText;
    bool hasMinor = false;
    if (const auto separator = majorText.find('.');
        separator != StringRef::npos) {
      hasMinor = true;
      minorText = majorText.drop_front(separator + 1);
      majorText = majorText.take_front(separator);
    }
    const auto decimalDigits = [](const StringRef text) {
      return !text.empty() && llvm::all_of(text, [](const char value) {
        return value >= '0' && value <= '9';
      });
    };
    std::uint64_t major = 0;
    std::uint64_t minor = 0;
    if (!decimalDigits(majorText) || (hasMinor && !decimalDigits(minorText)) ||
        majorText.getAsInteger(10, major) ||
        (!minorText.empty() && minorText.getAsInteger(10, minor)) ||
        major > std::numeric_limits<std::uint32_t>::max() ||
        minor > std::numeric_limits<std::uint32_t>::max()) {
      return sink.error(current().loc, "invalid OpenQASM version string");
    }
    const Version version{.major = static_cast<std::uint32_t>(major),
                          .minor = static_cast<std::uint32_t>(minor)};
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.version(loc, version);
  }

  //===--- Include ------------------------------------------------------===//

  [[nodiscard]] LogicalResult parseInclude() {
    const auto loc = current().loc;
    if (blockDepth != 0) {
      return sink.error(loc, "include directives are only allowed globally");
    }
    advance(); // include
    if (current().kind != TokenKind::StringLiteral) {
      return sink.error(current().loc, "expected a string literal");
    }
    const auto filename = current().stringValue;
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.include(loc, filename);
  }

  //===--- Declarations -------------------------------------------------===//

  /// Parse `[const] (int|uint|float|bool) <id> = <initializer>;`.
  [[nodiscard]] LogicalResult parseScalarDeclaration() {
    const auto loc = current().loc;

    bool isConst = false;
    if (current().kind == TokenKind::Const) {
      isConst = true;
      advance(); // const
    }

    const auto kind = current().kind;
    switch (kind) {
    case TokenKind::Bool:
      break;
    case TokenKind::Int:
      break;
    case TokenKind::Uint:
      break;
    case TokenKind::Float:
      break;
    case TokenKind::Angle:
    case TokenKind::Duration:
      return sink.error(current().loc,
                        "'angle' and 'duration' declarations are not supported "
                        "yet");
    case TokenKind::UnsupportedKeyword:
      return unsupportedKeyword();
    default:
      return sink.error(current().loc, "expected a scalar type");
    }
    advance(); // type

    if ((kind == TokenKind::Int || kind == TokenKind::Uint) &&
        current().kind == TokenKind::LBracket) {
      return sink.error(current().loc,
                        "Integer declarations currently require the default "
                        "64-bit width");
    }
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected identifier");
    }
    const auto id = current().identifier;
    advance();

    const bool hasInitializer = current().kind == TokenKind::Equals;
    if (hasInitializer) {
      advance();
    }
    if (isConst && !hasInitializer) {
      return sink.error(loc, Twine("'const' declaration of '") + id +
                                 "' requires an initializer");
    }

    const Expr* initializer = nullptr;
    if (hasInitializer) {
      auto value = parseExpression();
      if (failed(value)) {
        return failure();
      }
      initializer = *value;
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    const auto scalarKind = kind == TokenKind::Bool   ? ScalarKind::Bool
                            : kind == TokenKind::Int  ? ScalarKind::Int
                            : kind == TokenKind::Uint ? ScalarKind::Uint
                                                      : ScalarKind::Float;
    return sink.scalarDecl(loc, scalarKind, id, initializer, isConst);
  }

  /// Parse `qubit[<n>] <id>;`.
  [[nodiscard]] LogicalResult parseQuantumDecl() {
    const auto loc = current().loc;
    advance(); // qubit
    const Expr* size = nullptr;
    if (current().kind == TokenKind::LBracket) {
      auto designator = parseDesignator();
      if (failed(designator)) {
        return failure();
      }
      size = *designator;
    }
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected identifier");
    }
    const auto id = current().identifier;
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.qubitRegister(loc, id, size);
  }

  /// Parse `qreg <id>[<n>];`.
  [[nodiscard]] LogicalResult parseQregDecl() {
    const auto loc = current().loc;
    advance(); // qreg
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected identifier");
    }
    const auto id = current().identifier;
    advance();
    const Expr* size = nullptr;
    if (current().kind == TokenKind::LBracket) {
      auto designator = parseDesignator();
      if (failed(designator)) {
        return failure();
      }
      size = *designator;
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.qubitRegister(loc, id, size);
  }

  /// Parse `output bit[<n>] <id> (= <measurement>);`.
  [[nodiscard]] LogicalResult parseOutputDecl() {
    const auto loc = current().loc;
    advance(); // output
    if (current().kind == TokenKind::UnsupportedKeyword) {
      return unsupportedKeyword();
    }
    if (current().kind != TokenKind::Bit) {
      return sink.error(current().loc,
                        "only 'bit' registers can be declared as outputs");
    }
    return parseClassicalDecl(/*isOutput=*/true);
  }

  /// Parse `bit[<n>] <id> (= <measurement>);`.
  [[nodiscard]] LogicalResult parseClassicalDecl(bool isOutput) {
    const auto loc = current().loc;
    advance(); // bit
    const Expr* size = nullptr;
    if (current().kind == TokenKind::LBracket) {
      auto designator = parseDesignator();
      if (failed(designator)) {
        return failure();
      }
      size = *designator;
    }
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected identifier");
    }
    const auto id = current().identifier;
    advance();

    const Expr* initializer = nullptr;
    std::optional<Operand> measureSource;
    if (current().kind == TokenKind::Equals) {
      advance();
      if (current().kind == TokenKind::Measure) {
        advance();
        auto operand = parseGateOperand();
        if (failed(operand)) {
          return failure();
        }
        measureSource = *operand;
      } else {
        auto expression = parseExpression();
        if (failed(expression)) {
          return failure();
        }
        initializer = *expression;
      }
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }

    if (failed(sink.classicalRegister(loc, id, size, initializer, isOutput))) {
      return failure();
    }
    if (measureSource) {
      const BitReference target{.loc = loc, .identifier = id, .index = nullptr};
      return sink.measure(loc, &target, *measureSource);
    }
    return success();
  }

  /// Parse `creg <id>[<n>];`.
  [[nodiscard]] LogicalResult parseCregDecl() {
    const auto loc = current().loc;
    advance(); // creg
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected identifier");
    }
    const auto id = current().identifier;
    advance();
    const Expr* size = nullptr;
    if (current().kind == TokenKind::LBracket) {
      auto designator = parseDesignator();
      if (failed(designator)) {
        return failure();
      }
      size = *designator;
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.classicalRegister(loc, id, size, /*initializer=*/nullptr,
                                  /*output=*/false);
  }

  //===--- Assignment ---------------------------------------------------===//

  /// Parse an assignment.
  [[nodiscard]] LogicalResult parseAssignment() {
    const auto loc = current().loc;
    auto target = parseBitReference();
    if (failed(target)) {
      return failure();
    }

    const bool compound = current().kind == TokenKind::CompoundAssign;
    const auto compoundLocation = current().loc;
    const auto compoundSpelling = current().spelling;
    if (compound) {
      if (target->index != nullptr) {
        return sink.error(current().loc,
                          "indexed compound assignments are not supported");
      }
      advance();
    } else if (failed(expect(TokenKind::Equals))) {
      return failure();
    }

    if (!compound && current().kind == TokenKind::Measure) {
      advance();
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      if (failed(expect(TokenKind::Semicolon))) {
        return failure();
      }
      return sink.measure(loc, &*target, *operand);
    }

    auto value = parseExpression();
    if (failed(value)) {
      return failure();
    }
    const Expr* assignedValue = *value;
    if (compound) {
      const auto kind = llvm::StringSwitch<std::optional<Expr::Kind>>(
                            compoundSpelling.drop_back())
                            .Case("+", Expr::Kind::Add)
                            .Case("-", Expr::Kind::Sub)
                            .Case("*", Expr::Kind::Mul)
                            .Case("/", Expr::Kind::Div)
                            .Case("%", Expr::Kind::Mod)
                            .Case("**", Expr::Kind::Pow)
                            .Case("&", Expr::Kind::BitAnd)
                            .Case("|", Expr::Kind::BitOr)
                            .Case("^", Expr::Kind::BitXor)
                            .Case("<<", Expr::Kind::ShiftLeft)
                            .Case(">>", Expr::Kind::ShiftRight)
                            .Default(std::nullopt);
      if (!kind) {
        return sink.error(compoundLocation,
                          "unsupported compound assignment operator");
      }
      auto* previous = makeExpr();
      previous->loc = loc;
      previous->kind = Expr::Kind::Identifier;
      previous->identifier = target->identifier;
      assignedValue =
          makeBinary(*kind, previous, assignedValue, compoundLocation);
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.assignment(loc, *target, *assignedValue);
  }

  //===--- Measure ------------------------------------------------------===//

  [[nodiscard]] LogicalResult parseMeasure() {
    const auto loc = current().loc;
    advance(); // measure
    auto operand = parseGateOperand();
    if (failed(operand)) {
      return failure();
    }
    if (current().kind == TokenKind::Semicolon) {
      advance();
      return sink.measure(loc, /*target=*/nullptr, *operand);
    }
    if (failed(expect(TokenKind::Arrow))) {
      return failure();
    }
    auto target = parseBitReference();
    if (failed(target)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.measure(loc, &*target, *operand);
  }

  //===--- Reset --------------------------------------------------------===//

  [[nodiscard]] LogicalResult parseReset() {
    const auto loc = current().loc;
    advance(); // reset
    auto operand = parseGateOperand();
    if (failed(operand)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.reset(loc, *operand);
  }

  //===--- Barrier ------------------------------------------------------===//

  [[nodiscard]] LogicalResult parseBarrier() {
    const auto loc = current().loc;
    advance(); // barrier
    SmallVector<Operand> operands;
    while (current().kind != TokenKind::Semicolon) {
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      operands.push_back(*operand);
      if (current().kind != TokenKind::Semicolon &&
          failed(expect(TokenKind::Comma))) {
        return failure();
      }
    }
    advance(); // ;
    return sink.barrier(loc, operands);
  }

  //===--- Gate definitions and calls -----------------------------------===//

  [[nodiscard]] LogicalResult parseGateStatement() {
    const auto loc = current().loc;
    advance(); // gate
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected gate name");
    }
    const auto id = current().identifier;
    advance();

    // Parse parameters
    SmallVector<StringRef> parameters;
    if (current().kind == TokenKind::LParen) {
      advance();
      if (current().kind != TokenKind::RParen &&
          failed(parseIdentifierList(parameters))) {
        return failure();
      }
      if (failed(expect(TokenKind::RParen))) {
        return failure();
      }
    }

    // Parse target qubits
    SmallVector<StringRef> targets;
    if (failed(parseIdentifierList(targets))) {
      return failure();
    }
    if (current().kind != TokenKind::LBrace) {
      return sink.error(current().loc,
                        "expected '{' to begin gate definition body");
    }

    return sink.gateDefinition(loc, id, parameters, targets,
                               [this] { return parseBlock(); });
  }

  [[nodiscard]] LogicalResult parseGateCallStatement() {
    // The scratch buffers must outlive the sink call, so they live here rather
    // than inside `parseGateCall`.
    SmallVector<Modifier> modifiers;
    SmallVector<const Expr*> parameters;
    SmallVector<Operand> operands;
    auto call = parseGateCall(modifiers, parameters, operands);
    if (failed(call)) {
      return failure();
    }
    return sink.gateCall(*call);
  }

  [[nodiscard]] FailureOr<GateCall>
  parseGateCall(SmallVectorImpl<Modifier>& modifiers,
                SmallVectorImpl<const Expr*>& parameters,
                SmallVectorImpl<Operand>& operands) {
    GateCall call;
    call.loc = current().loc;

    while (current().kind == TokenKind::Inv ||
           current().kind == TokenKind::Pow ||
           current().kind == TokenKind::Ctrl ||
           current().kind == TokenKind::NegCtrl) {
      auto modifier = parseModifier();
      if (failed(modifier)) {
        return failure();
      }
      modifiers.push_back(*modifier);
      if (failed(expect(TokenKind::At))) {
        return failure();
      }
    }

    switch (current().kind) {
    case TokenKind::Gphase: {
      call.identifier = "gphase";
      advance();
      break;
    }
    case TokenKind::Identifier: {
      call.identifier = current().identifier;
      advance();
      break;
    }
    default:
      if (current().kind == TokenKind::UnsupportedKeyword) {
        return unsupportedKeyword();
      }
      return sink.error(current().loc, "expected gate name");
    }

    // Parse parameters
    if (current().kind == TokenKind::LParen) {
      advance();
      while (current().kind != TokenKind::RParen) {
        auto parameter = parseExpression();
        if (failed(parameter)) {
          return failure();
        }
        parameters.push_back(*parameter);
        if (current().kind != TokenKind::RParen &&
            failed(expect(TokenKind::Comma))) {
          return failure();
        }
      }
      advance(); // )
    }

    if (current().kind == TokenKind::LBracket) {
      return sink.error(current().loc,
                        "gate-call designators are outside the supported "
                        "OpenQASM subset");
    }

    // Parse target qubits
    while (current().kind != TokenKind::Semicolon) {
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      operands.push_back(*operand);
      if (current().kind != TokenKind::Semicolon &&
          failed(expect(TokenKind::Comma))) {
        return failure();
      }
    }
    advance(); // ;

    call.modifiers = modifiers;
    call.parameters = parameters;
    call.operands = operands;
    return call;
  }

  [[nodiscard]] FailureOr<Modifier> parseModifier() {
    Modifier modifier;
    switch (current().kind) {
    case TokenKind::Inv:
      modifier.kind = Modifier::Kind::Inv;
      advance();
      return modifier;
    case TokenKind::Pow:
      modifier.kind = Modifier::Kind::Pow;
      advance();
      if (failed(expect(TokenKind::LParen))) {
        return failure();
      }
      {
        auto argument = parseExpression();
        if (failed(argument)) {
          return failure();
        }
        modifier.argument = *argument;
      }
      if (failed(expect(TokenKind::RParen))) {
        return failure();
      }
      return modifier;
    case TokenKind::Ctrl:
    case TokenKind::NegCtrl:
      modifier.kind = current().kind == TokenKind::Ctrl
                          ? Modifier::Kind::Ctrl
                          : Modifier::Kind::NegCtrl;
      advance();
      if (current().kind == TokenKind::LParen) {
        advance();
        auto argument = parseExpression();
        if (failed(argument)) {
          return failure();
        }
        modifier.argument = *argument;
        if (failed(expect(TokenKind::RParen))) {
          return failure();
        }
      }
      return modifier;
    default:
      return sink.error(current().loc, "expected a gate modifier");
    }
  }

  [[nodiscard]] FailureOr<Operand> parseGateOperand() {
    Operand operand;
    operand.loc = current().loc;
    if (current().kind == TokenKind::HardwareQubit) {
      operand.hardwareQubit = static_cast<uint64_t>(current().intValue);
      advance();
      return operand;
    }
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected a gate operand");
    }
    operand.identifier = current().identifier;
    advance();
    const Expr* index = nullptr;
    if (current().kind == TokenKind::LBracket) {
      auto designator = parseDesignator();
      if (failed(designator)) {
        return failure();
      }
      index = *designator;
    }
    operand.index = index;
    return operand;
  }

  [[nodiscard]] FailureOr<BitReference> parseBitReference() {
    BitReference reference;
    reference.loc = current().loc;
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected an identifier");
    }
    reference.identifier = current().identifier;
    advance();
    const Expr* index = nullptr;
    if (current().kind == TokenKind::LBracket) {
      auto designator = parseDesignator();
      if (failed(designator)) {
        return failure();
      }
      index = *designator;
    }
    reference.index = index;
    return reference;
  }

  [[nodiscard]] LogicalResult
  parseIdentifierList(SmallVectorImpl<StringRef>& identifiers) {
    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected an identifier");
    }
    identifiers.push_back(current().identifier);
    advance();
    while (current().kind == TokenKind::Comma) {
      advance();
      if (current().kind != TokenKind::Identifier) {
        break;
      }
      identifiers.push_back(current().identifier);
      advance();
    }
    return success();
  }

  //===--- Control flow -------------------------------------------------===//

  [[nodiscard]] LogicalResult parseIf() {
    const auto loc = current().loc;
    advance(); // if
    if (failed(expect(TokenKind::LParen))) {
      return failure();
    }
    auto conditionOrFailure = parseExpression();
    if (failed(conditionOrFailure)) {
      return failure();
    }
    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }
    return sink.ifStmt(
        loc, **conditionOrFailure, [this] { return parseBlock(); },
        [this] { return parseElse(); });
  }

  [[nodiscard]] LogicalResult parseElse() {
    if (current().kind != TokenKind::Else) {
      return success();
    }
    advance(); // else
    return parseBlock();
  }

  [[nodiscard]] LogicalResult parseFor() {
    const auto loc = current().loc;
    advance(); // for

    if (current().kind == TokenKind::UnsupportedKeyword) {
      return unsupportedKeyword();
    }
    if (current().kind != TokenKind::Int && current().kind != TokenKind::Uint) {
      return sink.error(current().loc, "expected 'int' or 'uint' after 'for'");
    }
    const bool isUnsigned = current().kind == TokenKind::Uint;
    advance(); // type

    if (current().kind != TokenKind::Identifier) {
      return expectedIdentifier("expected loop variable");
    }
    const auto iv = current();
    advance(); // identifier

    if (failed(expect(TokenKind::In)) || failed(expect(TokenKind::LBracket))) {
      return failure();
    }
    auto start = parseExpression();
    if (failed(start)) {
      return failure();
    }
    if (failed(expect(TokenKind::Colon))) {
      return failure();
    }
    auto second = parseExpression();
    if (failed(second)) {
      return failure();
    }

    const Expr* step = nullptr;
    const Expr* stop = nullptr;
    if (current().kind == TokenKind::Colon) {
      advance();
      auto third = parseExpression();
      if (failed(third)) {
        return failure();
      }
      step = *second;
      stop = *third;
    } else {
      auto* one = makeExpr();
      one->loc = loc;
      one->kind = Expr::Kind::Int;
      one->intValue = 1;
      step = one;
      stop = *second;
    }
    if (failed(expect(TokenKind::RBracket))) {
      return failure();
    }

    return sink.forStmt(loc, iv.identifier, isUnsigned, **start, *step, *stop,
                        [this] { return parseBlock(); });
  }

  [[nodiscard]] LogicalResult parseWhile() {
    const auto loc = current().loc;
    advance(); // while
    if (failed(expect(TokenKind::LParen))) {
      return failure();
    }
    auto conditionOrFailure = parseExpression();
    if (failed(conditionOrFailure)) {
      return failure();
    }
    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }
    return sink.whileStmt(loc, **conditionOrFailure,
                          [this] { return parseBlock(); });
  }

  //===--- Expressions --------------------------------------------------===//

  /// Parse an expression using OpenQASM's precedence hierarchy.
  [[nodiscard]] FailureOr<const Expr*> parseExpression() {
    auto lhs = parseLogicalAnd();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::PipePipe) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseLogicalAnd();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(Expr::Kind::Or, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseLogicalAnd() {
    auto lhs = parseBitwiseOr();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::AmpAmp) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseBitwiseOr();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(Expr::Kind::And, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseBitwiseOr() {
    auto lhs = parseBitwiseXor();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::Pipe) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseBitwiseXor();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(Expr::Kind::BitOr, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseBitwiseXor() {
    auto lhs = parseBitwiseAnd();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::Caret) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseBitwiseAnd();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(Expr::Kind::BitXor, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseBitwiseAnd() {
    auto lhs = parseEquality();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::Amp) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseEquality();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(Expr::Kind::BitAnd, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseEquality() {
    auto lhs = parseRelational();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::EqualsEquals ||
           current().kind == TokenKind::NotEquals) {
      const auto kind = current().kind == TokenKind::EqualsEquals
                            ? Expr::Kind::Equal
                            : Expr::Kind::NotEqual;
      const auto loc = current().loc;
      advance();
      auto rhs = parseRelational();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(kind, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseRelational() {
    auto lhs = parseShift();
    if (failed(lhs)) {
      return failure();
    }
    const auto kind = [&]() -> std::optional<Expr::Kind> {
      switch (current().kind) {
      case TokenKind::Less:
        return Expr::Kind::Less;
      case TokenKind::LessEquals:
        return Expr::Kind::LessEqual;
      case TokenKind::Greater:
        return Expr::Kind::Greater;
      case TokenKind::GreaterEquals:
        return Expr::Kind::GreaterEqual;
      default:
        return std::nullopt;
      }
    }();
    if (!kind) {
      return lhs;
    }
    const auto loc = current().loc;
    advance();
    auto rhs = parseShift();
    if (failed(rhs)) {
      return failure();
    }
    return makeBinary(*kind, *lhs, *rhs, loc);
  }

  [[nodiscard]] FailureOr<const Expr*> parseShift() {
    auto lhs = parseAdditive();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::ShiftLeft ||
           current().kind == TokenKind::ShiftRight) {
      const auto kind = current().kind == TokenKind::ShiftLeft
                            ? Expr::Kind::ShiftLeft
                            : Expr::Kind::ShiftRight;
      const auto loc = current().loc;
      advance();
      auto rhs = parseAdditive();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(kind, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseAdditive() {
    auto lhs = parseTerm();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::Plus ||
           current().kind == TokenKind::Minus) {
      const auto kind =
          current().kind == TokenKind::Plus ? Expr::Kind::Add : Expr::Kind::Sub;
      const auto loc = current().loc;
      advance();
      auto rhs = parseTerm();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(kind, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseTerm() {
    auto lhs = parseUnary();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::Asterisk ||
           current().kind == TokenKind::Slash ||
           current().kind == TokenKind::Percent) {
      const auto kind = current().kind == TokenKind::Asterisk ? Expr::Kind::Mul
                        : current().kind == TokenKind::Slash  ? Expr::Kind::Div
                                                              : Expr::Kind::Mod;
      const auto loc = current().loc;
      advance();
      auto rhs = parseUnary();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinary(kind, result, *rhs, loc);
    }
    return result;
  }

  [[nodiscard]] FailureOr<const Expr*> parseUnary() {
    if (current().kind == TokenKind::Minus ||
        current().kind == TokenKind::ExclamationPoint ||
        current().kind == TokenKind::Tilde) {
      const auto loc = current().loc;
      const auto kind = current().kind == TokenKind::Minus ? Expr::Kind::Neg
                        : current().kind == TokenKind::Tilde
                            ? Expr::Kind::BitNot
                            : Expr::Kind::Not;
      advance();
      auto operand = parseUnary();
      if (failed(operand)) {
        return failure();
      }
      auto* expr = makeExpr();
      expr->loc = loc;
      expr->kind = kind;
      expr->lhs = *operand;
      return expr;
    }
    return parsePower();
  }

  [[nodiscard]] FailureOr<const Expr*> parsePower() {
    auto lhs = parsePrimary();
    if (failed(lhs) || current().kind != TokenKind::DoubleAsterisk) {
      return lhs;
    }
    const auto loc = current().loc;
    advance();
    auto rhs = parseUnary();
    if (failed(rhs)) {
      return failure();
    }
    return makeBinary(Expr::Kind::Pow, *lhs, *rhs, loc);
  }

  [[nodiscard]] FailureOr<const Expr*> parsePrimary() {
    auto* expr = makeExpr();
    expr->loc = current().loc;
    switch (current().kind) {
    case TokenKind::True:
    case TokenKind::False:
      expr->kind = Expr::Kind::Bool;
      expr->boolValue = current().kind == TokenKind::True;
      advance();
      return expr;
    case TokenKind::FloatLiteral:
      expr->kind = Expr::Kind::Float;
      expr->floatValue = current().floatValue;
      advance();
      return expr;
    case TokenKind::IntegerLiteral:
      expr->kind = Expr::Kind::Int;
      expr->intValue = current().intValue;
      advance();
      return expr;
    case TokenKind::Identifier: {
      if (peek().kind == TokenKind::LParen) {
        const auto kind = getMathFunctionKind(current().identifier);
        if (!kind) {
          return sink.error(current().loc, Twine("unknown function '") +
                                               current().identifier + "'");
        }
        return parseMathCall(*kind, expr);
      }
      expr->kind = Expr::Kind::Identifier;
      expr->identifier = current().identifier;
      advance();
      if (current().kind == TokenKind::LBracket) {
        auto designator = parseDesignator();
        if (failed(designator)) {
          return failure();
        }
        expr->kind = Expr::Kind::Index;
        expr->lhs = *designator;
      }
      return expr;
    }
    case TokenKind::Measure: {
      advance();
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      expr->kind = Expr::Kind::Measurement;
      expr->identifier = operand->identifier;
      expr->hardwareQubit = operand->hardwareQubit;
      expr->lhs = operand->index;
      return expr;
    }
    // `pow` is also a gate modifier, so it has a dedicated token.
    case TokenKind::Pow:
      return parseMathCall(Expr::Kind::BuiltinPow, expr);
    case TokenKind::LParen: {
      advance();
      auto inner = parseExpression();
      if (failed(inner)) {
        return failure();
      }
      if (failed(expect(TokenKind::RParen))) {
        return failure();
      }
      return *inner;
    }
    case TokenKind::UnsupportedKeyword:
      return unsupportedKeyword();
    default:
      return sink.error(current().loc, "expected expression");
    }
  }

  /// Parse the argument list of a call to the built-in math function @p kind.
  [[nodiscard]] FailureOr<const Expr*> parseMathCall(Expr::Kind kind,
                                                     Expr* expr) {
    expr->kind = kind;
    advance(); // function name

    if (failed(expect(TokenKind::LParen))) {
      return failure();
    }

    auto lhs = parseExpression();
    if (failed(lhs)) {
      return failure();
    }
    expr->lhs = *lhs;

    if (kind == Expr::Kind::BuiltinMod || kind == Expr::Kind::BuiltinPow) {
      if (failed(expect(TokenKind::Comma))) {
        return failure();
      }
      auto rhs = parseExpression();
      if (failed(rhs)) {
        return failure();
      }
      expr->rhs = *rhs;
    }

    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }
    return expr;
  }

  [[nodiscard]] Expr* makeBinary(const Expr::Kind kind, const Expr* lhs,
                                 const Expr* rhs, const SMLoc loc) {
    auto* expr = makeExpr();
    expr->loc = loc;
    expr->kind = kind;
    expr->lhs = lhs;
    expr->rhs = rhs;
    return expr;
  }

  Lexer& lexer;
  Sink& sink;
  llvm::BumpPtrAllocator& allocator;
  Token currentToken;
  Token nextToken;
  std::size_t blockDepth = 0;
};

} // namespace mlir::oq3::frontend::detail
