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

#include "mlir/Dialect/QC/Translation/qasm3/QASM3Lexer.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace mlir::qc::detail {

//===----------------------------------------------------------------------===//
// Transient parse vocabulary
//
// These types are the vocabulary the parser hands to a sink. They are cheap,
// trivially destructible, and (where they need to outlive a single statement,
// i.e. gate-definition bodies) allocated in a bump allocator. There is no
// persistent whole-program syntax tree: flat statements stream straight to the
// sink as they are recognized.
//===----------------------------------------------------------------------===//

/// A (sub-)expression. Bump-allocated; children are borrowed pointers.
struct Expr {
  enum class Kind : uint8_t { Int, Float, Ident, Neg, Add, Sub, Mul, Div };

  Kind kind = Kind::Int;
  int64_t intValue = 0;
  double floatValue = 0.0;
  llvm::StringRef ident;
  const Expr* lhs = nullptr;
  const Expr* rhs = nullptr;
  llvm::SMLoc loc;
};

/// A gate modifier: `inv @`, `pow(e) @`, `ctrl(e) @`, or `negctrl(e) @`.
struct Modifier {
  enum class Kind : uint8_t { Inv, Pow, Ctrl, NegCtrl };
  Kind kind = Kind::Inv;
  const Expr* argument = nullptr; ///< `pow`/`ctrl`/`negctrl` argument, or null.
};

/// A gate operand: a (possibly indexed) identifier, or a hardware qubit.
struct Operand {
  llvm::StringRef identifier;
  const Expr* index = nullptr;
  std::optional<uint64_t> hardwareQubit;
  llvm::SMLoc loc;
};

/// A (possibly indexed) classical reference, e.g. `c` or `c[0]`.
struct Reference {
  llvm::StringRef identifier;
  const Expr* index = nullptr;
  llvm::SMLoc loc;
};

/// A resolved gate call. Array members are borrowed for the duration of the
/// sink call (top-level) or bump-allocated (gate-definition bodies).
struct GateCall {
  llvm::StringRef identifier;
  llvm::ArrayRef<Modifier> modifiers;
  llvm::ArrayRef<const Expr*> parameters;
  llvm::ArrayRef<Operand> operands;
  llvm::SMLoc loc;
};

/// A branch condition: a boolean-valued expression. Bump-allocated; children
/// are borrowed pointers. Register comparisons (e.g. `c == 5`) are not yet
/// supported.
struct Condition {
  enum class Kind : uint8_t { Measurement, Bit, Literal, Not, And, Or };
  Kind kind = Kind::Bit;
  Operand operand;                ///< For `Measurement`.
  Reference bit;                  ///< For `Bit`.
  bool literalValue = false;      ///< For `Literal`.
  const Condition* lhs = nullptr; ///< For `Not`, `And`, and `Or`.
  const Condition* rhs = nullptr; ///< For `And` and `Or`.
  llvm::SMLoc loc;
};

/// A supported arithmetic scalar declaration type (`bool` is handled
/// separately, as its initializer is a boolean condition). `const` and
/// non-`const` declarations are lowered identically, so constness is not
/// tracked here.
enum class ScalarType : uint8_t { Float, Int };

//===----------------------------------------------------------------------===//
// Sink concept
//===----------------------------------------------------------------------===//

/**
 * @brief The interface a `Parser` drives to lower parse events to a target.
 *
 * @details
 * A sink consumes the events produced by `Parser` and lowers them to a target
 * representation. The parser is templated over the sink, so dispatch is fully
 * static. Diagnostics are routed through `error`; every event returns a
 * `LogicalResult` (or, for `if`, an opaque scope handle).
 *
 * Control flow uses continuations: the sink opens the target region and calls
 * back into the parser to emit the body into it, so no block is ever
 * materialized.
 */
template <class S>
concept QASMSink = requires(
    S s, llvm::SMLoc loc, llvm::StringRef str, const Expr& expr,
    const Operand& operand, const Reference& reference,
    const Condition& condition, const GateCall& call,
    llvm::ArrayRef<Operand> operands, llvm::ArrayRef<llvm::StringRef> names,
    llvm::ArrayRef<GateCall> body, llvm::function_ref<LogicalResult()> cont,
    double d, bool flag, ScalarType scalarType) {
  s.error(loc, str);
  s.version(d);
  s.include(loc, str);
  s.scalarDecl(loc, scalarType, str, expr);
  s.boolDecl(loc, str, condition);
  s.qubitRegister(loc, str, &expr);
  s.classicalRegister(loc, str, &expr);
  s.measure(loc, reference, operand);
  s.reset(loc, operand);
  s.barrier(loc, operands);
  s.gateCall(call);
  s.gateDefinition(loc, str, names, names, body);
  s.conditionOnly(loc, condition);
  s.ifBegin(loc, condition, flag);
  s.forStmt(loc, str, expr, expr, expr, cont);
  s.whileStmt(loc, condition, cont);
  // The sink must additionally provide `ifElse(scope)` and `ifEnd(scope,
  // bool)` taking the opaque scope returned by `ifBegin`; those are not
  // expressible here without naming the scope type.
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/**
 * @brief A single-pass recursive-descent parser for OpenQASM 3.
 *
 * @details
 * The parser owns no persistent syntax tree. As it recognizes each construct it
 * calls the corresponding sink event; control-flow bodies are emitted by having
 * the sink call back through the continuations the parser supplies. Expressions
 * and gate-definition bodies are bump-allocated.
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
    while (!isAtEnd()) {
      if (failed(parseStatement())) {
        return failure();
      }
    }
    return success();
  }

private:
  //===--- Token scaffolding --------------------------------------------===//

  void advance() {
    currentToken = nextToken;
    nextToken = lexer.next();
  }

  [[nodiscard]] const Token& current() const { return currentToken; }
  [[nodiscard]] const Token& peek() const { return nextToken; }
  [[nodiscard]] bool isAtEnd() const {
    return currentToken.kind == TokenKind::Eof;
  }

  [[nodiscard]] LogicalResult expect(const TokenKind kind) {
    if (current().kind != kind) {
      return sink.error(current().loc, "expected " + describe(kind) + ", got " +
                                           describe(current().kind));
    }
    advance();
    return success();
  }

  //===--- Allocation helpers -------------------------------------------===//

  [[nodiscard]] Expr* makeExpr() {
    return new (allocator.Allocate<Expr>()) Expr();
  }

  [[nodiscard]] Condition* makeCondition() {
    return new (allocator.Allocate<Condition>()) Condition();
  }

  template <class T>
  [[nodiscard]] llvm::ArrayRef<T> copyToArena(llvm::ArrayRef<T> values) {
    if (values.empty()) {
      return {};
    }
    auto* storage = allocator.Allocate<T>(values.size());
    std::uninitialized_copy(values.begin(), values.end(), storage);
    return {storage, values.size()};
  }

  //===--- Program and statements ---------------------------------------===//

  [[nodiscard]] LogicalResult parseStatement() {
    switch (current().kind) {
    case TokenKind::OpenQASM:
      return parseVersion();
    case TokenKind::Include:
      return parseInclude();
    case TokenKind::Const:
    case TokenKind::Int:
    case TokenKind::Uint:
    case TokenKind::Bool:
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
      return parseOldStyleDecl(/*classical=*/false);
    case TokenKind::CReg:
      return parseOldStyleDecl(/*classical=*/true);
    case TokenKind::Bit:
      return parseClassicalDecl();
    case TokenKind::Gate:
      return parseGateDefinition();
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
    default:
      return sink.error(current().loc, "unexpected token");
    }
  }

  /// Parse a `{ ... }` block or a single statement, emitting into whatever
  /// region the sink has made current.
  [[nodiscard]] LogicalResult parseBlock() {
    if (current().kind == TokenKind::LBrace) {
      advance();
      while (!isAtEnd() && current().kind != TokenKind::RBrace) {
        if (failed(parseStatement())) {
          return failure();
        }
      }
      return expect(TokenKind::RBrace);
    }
    return parseStatement();
  }

  //===--- Version and include ------------------------------------------===//

  [[nodiscard]] LogicalResult parseVersion() {
    advance(); // OPENQASM
    double version = 0.0;
    if (current().kind == TokenKind::FloatLiteral) {
      version = current().floatValue;
    } else if (current().kind == TokenKind::IntegerLiteral) {
      version = static_cast<double>(current().intValue);
    } else {
      return sink.error(current().loc,
                        "version must be a float or integer literal");
    }
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.version(version);
  }

  [[nodiscard]] LogicalResult parseInclude() {
    const auto loc = current().loc;
    advance(); // include
    if (current().kind != TokenKind::StringLiteral) {
      return sink.error(current().loc, "expected a string literal");
    }
    const auto filename = current().spelling;
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.include(loc, filename);
  }

  //===--- Declarations -------------------------------------------------===//

  /// Parse `[const] (int|uint|float|bool) <id> = <initializer>;`. `const` and
  /// non-`const` declarations are parsed and lowered identically. `bool`
  /// initializers are boolean conditions; all others are arithmetic.
  [[nodiscard]] LogicalResult parseScalarDeclaration() {
    const auto loc = current().loc;
    if (current().kind == TokenKind::Const) {
      advance(); // const
    }

    ScalarType scalarType = ScalarType::Float;
    bool isBool = false;
    switch (current().kind) {
    case TokenKind::Float:
      scalarType = ScalarType::Float;
      break;
    case TokenKind::Int:
    case TokenKind::Uint:
      scalarType = ScalarType::Int;
      break;
    case TokenKind::Bool:
      isBool = true;
      break;
    case TokenKind::Angle:
    case TokenKind::Duration:
      return sink.error(current().loc,
                        "'angle' and 'duration' declarations are not supported "
                        "yet");
    default:
      return sink.error(current().loc, "expected a scalar type");
    }
    advance(); // type

    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected identifier");
    }
    const auto id = current().spelling;
    advance();

    // Only initialized declarations are supported, since there is no classical
    // assignment to give an uninitialized variable a value later.
    if (failed(expect(TokenKind::Equals))) {
      return failure();
    }

    if (isBool) {
      auto condition = parseCondition();
      if (failed(condition)) {
        return failure();
      }
      if (failed(expect(TokenKind::Semicolon))) {
        return failure();
      }
      return sink.boolDecl(loc, id, **condition);
    }

    auto value = parseExpression();
    if (failed(value)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.scalarDecl(loc, scalarType, id, **value);
  }

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
      return sink.error(current().loc, "expected identifier");
    }
    const auto id = current().spelling;
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.qubitRegister(loc, id, size);
  }

  [[nodiscard]] LogicalResult parseOldStyleDecl(const bool classical) {
    const auto loc = current().loc;
    advance(); // qreg / creg
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected identifier");
    }
    const auto id = current().spelling;
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
    return classical ? sink.classicalRegister(loc, id, size)
                     : sink.qubitRegister(loc, id, size);
  }

  [[nodiscard]] LogicalResult parseClassicalDecl() {
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
      return sink.error(current().loc, "expected identifier");
    }
    const auto id = current().spelling;
    advance();

    std::optional<Operand> measureSource;
    if (current().kind == TokenKind::Equals) {
      advance();
      if (failed(expect(TokenKind::Measure))) {
        return failure();
      }
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      measureSource = *operand;
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }

    if (failed(sink.classicalRegister(loc, id, size))) {
      return failure();
    }
    if (measureSource) {
      const Reference target{.identifier = id, .index = nullptr, .loc = loc};
      return sink.measure(loc, target, *measureSource);
    }
    return success();
  }

  [[nodiscard]] mlir::FailureOr<const Expr*> parseDesignator() {
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

  //===--- Assignment and measurement -----------------------------------===//

  [[nodiscard]] LogicalResult parseAssignment() {
    const auto loc = current().loc;
    auto target = parseReference();
    if (failed(target)) {
      return failure();
    }
    if (current().kind != TokenKind::Equals) {
      return sink.error(current().loc,
                        "classical computations are not supported yet");
    }
    advance();
    if (current().kind != TokenKind::Measure) {
      return sink.error(current().loc,
                        "classical computations are not supported yet");
    }
    advance();
    auto operand = parseGateOperand();
    if (failed(operand)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.measure(loc, *target, *operand);
  }

  [[nodiscard]] LogicalResult parseMeasure() {
    const auto loc = current().loc;
    advance(); // measure
    auto operand = parseGateOperand();
    if (failed(operand)) {
      return failure();
    }
    if (failed(expect(TokenKind::Arrow))) {
      return failure();
    }
    auto target = parseReference();
    if (failed(target)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.measure(loc, *target, *operand);
  }

  //===--- Barrier and reset --------------------------------------------===//

  [[nodiscard]] LogicalResult parseBarrier() {
    const auto loc = current().loc;
    advance(); // barrier
    llvm::SmallVector<Operand> operands;
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

  //===--- Control flow -------------------------------------------------===//

  [[nodiscard]] LogicalResult parseIf() {
    const auto loc = current().loc;
    advance(); // if
    if (failed(expect(TokenKind::LParen))) {
      return failure();
    }
    auto condition = parseCondition();
    if (failed(condition)) {
      return failure();
    }
    const Condition& cond = **condition;
    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }

    const bool thenEmpty =
        current().kind == TokenKind::LBrace && peek().kind == TokenKind::RBrace;
    if (thenEmpty) {
      advance(); // {
      advance(); // }
      if (current().kind != TokenKind::Else) {
        return sink.conditionOnly(loc, cond);
      }
      advance(); // else
      auto scope = sink.ifBegin(loc, cond, /*invert=*/true);
      if (failed(scope)) {
        return failure();
      }
      if (failed(parseBlock())) {
        return failure();
      }
      return sink.ifEnd(*scope, /*hadElse=*/false);
    }

    auto scope = sink.ifBegin(loc, cond, /*invert=*/false);
    if (failed(scope)) {
      return failure();
    }
    if (failed(parseBlock())) {
      return failure();
    }

    if (current().kind == TokenKind::Else) {
      advance(); // else
      const bool elseEmpty = current().kind == TokenKind::LBrace &&
                             peek().kind == TokenKind::RBrace;
      if (elseEmpty) {
        advance(); // {
        advance(); // }
        return sink.ifEnd(*scope, /*hadElse=*/false);
      }
      if (failed(sink.ifElse(*scope))) {
        return failure();
      }
      if (failed(parseBlock())) {
        return failure();
      }
      return sink.ifEnd(*scope, /*hadElse=*/true);
    }
    return sink.ifEnd(*scope, /*hadElse=*/false);
  }

  [[nodiscard]] LogicalResult parseFor() {
    const auto loc = current().loc;
    advance(); // for
    if (current().kind != TokenKind::Int && current().kind != TokenKind::Uint) {
      return sink.error(current().loc, "expected 'int' or 'uint' after 'for'");
    }
    advance();
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected loop variable");
    }
    const auto variable = current().spelling;
    advance();
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
      one->kind = Expr::Kind::Int;
      one->intValue = 1;
      one->loc = loc;
      step = one;
      stop = *second;
    }
    if (failed(expect(TokenKind::RBracket))) {
      return failure();
    }

    return sink.forStmt(loc, variable, **start, *step, *stop,
                        [this] { return parseBlock(); });
  }

  [[nodiscard]] LogicalResult parseWhile() {
    const auto loc = current().loc;
    advance(); // while
    if (failed(expect(TokenKind::LParen))) {
      return failure();
    }
    auto condition = parseCondition();
    if (failed(condition)) {
      return failure();
    }
    const Condition& cond = **condition;
    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }
    return sink.whileStmt(loc, cond, [this] { return parseBlock(); });
  }

  /// cond := andCond ('||' andCond)*
  [[nodiscard]] mlir::FailureOr<const Condition*> parseCondition() {
    auto lhs = parseAndCondition();
    if (failed(lhs)) {
      return failure();
    }
    const Condition* result = *lhs;
    while (current().kind == TokenKind::PipePipe) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseAndCondition();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinaryCondition(Condition::Kind::Or, result, *rhs, loc);
    }
    return result;
  }

  /// andCond := unaryCond ('&&' unaryCond)*
  [[nodiscard]] mlir::FailureOr<const Condition*> parseAndCondition() {
    auto lhs = parseUnaryCondition();
    if (failed(lhs)) {
      return failure();
    }
    const Condition* result = *lhs;
    while (current().kind == TokenKind::AmpAmp) {
      const auto loc = current().loc;
      advance();
      auto rhs = parseUnaryCondition();
      if (failed(rhs)) {
        return failure();
      }
      result = makeBinaryCondition(Condition::Kind::And, result, *rhs, loc);
    }
    return result;
  }

  /// unaryCond := ('!' | '~') unaryCond | primaryCond
  [[nodiscard]] mlir::FailureOr<const Condition*> parseUnaryCondition() {
    if (current().kind == TokenKind::ExclamationPoint ||
        current().kind == TokenKind::Tilde) {
      const auto loc = current().loc;
      advance();
      auto operand = parseUnaryCondition();
      if (failed(operand)) {
        return failure();
      }
      auto* condition = makeCondition();
      condition->kind = Condition::Kind::Not;
      condition->loc = loc;
      condition->lhs = *operand;
      return condition;
    }
    return parsePrimaryCondition();
  }

  /// primaryCond := 'measure' gateOperand | '(' cond ')' | reference
  [[nodiscard]] mlir::FailureOr<const Condition*> parsePrimaryCondition() {
    const auto loc = current().loc;

    if (current().kind == TokenKind::True ||
        current().kind == TokenKind::False) {
      auto* condition = makeCondition();
      condition->kind = Condition::Kind::Literal;
      condition->literalValue = current().kind == TokenKind::True;
      condition->loc = loc;
      advance();
      return finishPrimaryCondition(condition);
    }

    if (current().kind == TokenKind::Measure) {
      advance();
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      auto* condition = makeCondition();
      condition->kind = Condition::Kind::Measurement;
      condition->operand = *operand;
      condition->loc = loc;
      return finishPrimaryCondition(condition);
    }

    if (current().kind == TokenKind::LParen) {
      advance();
      auto inner = parseCondition();
      if (failed(inner)) {
        return failure();
      }
      if (failed(expect(TokenKind::RParen))) {
        return failure();
      }
      return finishPrimaryCondition(*inner);
    }

    if (current().kind == TokenKind::Identifier) {
      auto bit = parseReference();
      if (failed(bit)) {
        return failure();
      }
      auto* condition = makeCondition();
      condition->kind = Condition::Kind::Bit;
      condition->bit = *bit;
      condition->loc = loc;
      return finishPrimaryCondition(condition);
    }

    return sink.error(loc, "unsupported condition expression");
  }

  /// Reject a register comparison (e.g. `c == 5`) trailing a primary.
  [[nodiscard]] mlir::FailureOr<const Condition*>
  finishPrimaryCondition(const Condition* condition) {
    switch (current().kind) {
    case TokenKind::EqualsEquals:
    case TokenKind::NotEquals:
    case TokenKind::Less:
    case TokenKind::LessEquals:
    case TokenKind::Greater:
    case TokenKind::GreaterEquals:
      return sink.error(current().loc,
                        "register comparisons are not supported");
    default:
      return condition;
    }
  }

  [[nodiscard]] Condition* makeBinaryCondition(const Condition::Kind kind,
                                               const Condition* lhs,
                                               const Condition* rhs,
                                               const llvm::SMLoc loc) {
    auto* condition = makeCondition();
    condition->kind = kind;
    condition->loc = loc;
    condition->lhs = lhs;
    condition->rhs = rhs;
    return condition;
  }

  //===--- Gate definitions and calls -----------------------------------===//

  [[nodiscard]] LogicalResult parseGateDefinition() {
    const auto loc = current().loc;
    advance(); // gate
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected gate name");
    }
    const auto id = current().spelling;
    advance();

    llvm::SmallVector<llvm::StringRef> parameters;
    if (current().kind == TokenKind::LParen) {
      advance();
      if (failed(parseIdentifierList(parameters))) {
        return failure();
      }
      if (failed(expect(TokenKind::RParen))) {
        return failure();
      }
    }

    llvm::SmallVector<llvm::StringRef> targets;
    if (failed(parseIdentifierList(targets))) {
      return failure();
    }

    if (failed(expect(TokenKind::LBrace))) {
      return failure();
    }
    llvm::SmallVector<GateCall> body;
    while (current().kind != TokenKind::RBrace) {
      // Bodies outlive this frame, so the call's arrays are copied into the
      // arena; the scratch buffers below are discarded.
      llvm::SmallVector<Modifier> callModifiers;
      llvm::SmallVector<const Expr*> callParameters;
      llvm::SmallVector<Operand> callOperands;
      auto call = parseGateCall(callModifiers, callParameters, callOperands,
                                /*persist=*/true);
      if (failed(call)) {
        return failure();
      }
      body.push_back(*call);
    }
    if (failed(expect(TokenKind::RBrace))) {
      return failure();
    }

    return sink.gateDefinition(loc, id, copyToArena(llvm::ArrayRef(parameters)),
                               copyToArena(llvm::ArrayRef(targets)),
                               copyToArena(llvm::ArrayRef(body)));
  }

  [[nodiscard]] LogicalResult parseGateCallStatement() {
    // The scratch buffers must outlive the sink call, so they live here rather
    // than inside `parseGateCall`.
    llvm::SmallVector<Modifier> modifiers;
    llvm::SmallVector<const Expr*> parameters;
    llvm::SmallVector<Operand> operands;
    auto call = parseGateCall(modifiers, parameters, operands,
                              /*persist=*/false);
    if (failed(call)) {
      return failure();
    }
    return sink.gateCall(*call);
  }

  [[nodiscard]] mlir::FailureOr<GateCall>
  parseGateCall(llvm::SmallVectorImpl<Modifier>& modifiers,
                llvm::SmallVectorImpl<const Expr*>& parameters,
                llvm::SmallVectorImpl<Operand>& operands, const bool persist) {
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

    if (current().kind == TokenKind::Gphase) {
      call.identifier = "gphase";
      advance();
    } else if (current().kind == TokenKind::Identifier) {
      call.identifier = current().spelling;
      advance();
    } else {
      return sink.error(current().loc, "expected gate name");
    }

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
                        "gate calls with designators are not supported yet");
    }

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

    // Top-level calls are consumed by the sink while the caller's buffers are
    // still alive, so borrowing them is safe. Gate-definition bodies outlive
    // this frame, so their arrays are copied into the arena.
    if (persist) {
      call.modifiers = copyToArena(llvm::ArrayRef(modifiers));
      call.parameters = copyToArena(llvm::ArrayRef(parameters));
      call.operands = copyToArena(llvm::ArrayRef(operands));
      return call;
    }
    call.modifiers = modifiers;
    call.parameters = parameters;
    call.operands = operands;
    return call;
  }

  [[nodiscard]] mlir::FailureOr<Modifier> parseModifier() {
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

  [[nodiscard]] mlir::FailureOr<Operand> parseGateOperand() {
    Operand operand;
    operand.loc = current().loc;
    if (current().kind == TokenKind::HardwareQubit) {
      operand.hardwareQubit = static_cast<uint64_t>(current().intValue);
      advance();
      return operand;
    }
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected a gate operand");
    }
    operand.identifier = current().spelling;
    advance();
    if (current().kind == TokenKind::LBracket) {
      advance();
      auto index = parseExpression();
      if (failed(index)) {
        return failure();
      }
      operand.index = *index;
      if (failed(expect(TokenKind::RBracket))) {
        return failure();
      }
    }
    return operand;
  }

  [[nodiscard]] mlir::FailureOr<Reference> parseReference() {
    Reference reference;
    reference.loc = current().loc;
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected an identifier");
    }
    reference.identifier = current().spelling;
    advance();
    if (current().kind == TokenKind::LBracket) {
      advance();
      auto index = parseExpression();
      if (failed(index)) {
        return failure();
      }
      reference.index = *index;
      if (failed(expect(TokenKind::RBracket))) {
        return failure();
      }
    }
    return reference;
  }

  [[nodiscard]] LogicalResult
  parseIdentifierList(llvm::SmallVectorImpl<llvm::StringRef>& identifiers) {
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected an identifier");
    }
    identifiers.push_back(current().spelling);
    advance();
    while (current().kind == TokenKind::Comma) {
      advance();
      if (current().kind != TokenKind::Identifier) {
        return sink.error(current().loc, "expected an identifier");
      }
      identifiers.push_back(current().spelling);
      advance();
    }
    return success();
  }

  //===--- Expressions --------------------------------------------------===//

  [[nodiscard]] mlir::FailureOr<const Expr*> parseExpression() {
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

  [[nodiscard]] mlir::FailureOr<const Expr*> parseTerm() {
    auto lhs = parseUnary();
    if (failed(lhs)) {
      return failure();
    }
    const Expr* result = *lhs;
    while (current().kind == TokenKind::Asterisk ||
           current().kind == TokenKind::Slash) {
      const auto kind = current().kind == TokenKind::Asterisk ? Expr::Kind::Mul
                                                              : Expr::Kind::Div;
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

  [[nodiscard]] mlir::FailureOr<const Expr*> parseUnary() {
    if (current().kind == TokenKind::Minus) {
      const auto loc = current().loc;
      advance();
      auto operand = parseUnary();
      if (failed(operand)) {
        return failure();
      }
      auto* expr = makeExpr();
      expr->kind = Expr::Kind::Neg;
      expr->loc = loc;
      expr->lhs = *operand;
      return expr;
    }
    return parsePrimary();
  }

  [[nodiscard]] mlir::FailureOr<const Expr*> parsePrimary() {
    auto* expr = makeExpr();
    expr->loc = current().loc;
    switch (current().kind) {
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
    case TokenKind::Identifier:
      expr->kind = Expr::Kind::Ident;
      expr->ident = current().spelling;
      advance();
      return expr;
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
    default:
      return sink.error(current().loc, "expected expression");
    }
  }

  [[nodiscard]] Expr* makeBinary(const Expr::Kind kind, const Expr* lhs,
                                 const Expr* rhs, const llvm::SMLoc loc) {
    auto* expr = makeExpr();
    expr->kind = kind;
    expr->loc = loc;
    expr->lhs = lhs;
    expr->rhs = rhs;
    return expr;
  }

  Lexer& lexer;
  Sink& sink;
  llvm::BumpPtrAllocator& allocator;
  Token currentToken;
  Token nextToken;
};

} // namespace mlir::qc::detail
