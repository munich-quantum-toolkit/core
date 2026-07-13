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
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace mlir::qc::detail {

/**
 * @defgroup ParseVocabulary Transient parse vocabulary
 * @brief The vocabulary the parser hands to a sink.
 *
 * @details
 * These types are cheap, trivially destructible, and (where they need to
 * outlive a single statement, i.e. gate-definition bodies) allocated in a bump
 * allocator. There is no persistent whole-program syntax tree: flat statements
 * stream straight to the sink as they are recognized.
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
    Identifier,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    // Built-in math functions
    ArcCos,
    ArcSin,
    ArcTan,
    Cos,
    Exp,
    Log,
    Mod,
    Pow,
    Sin,
    Sqrt,
    Tan,
  };

  SMLoc loc;
  Kind kind = Kind::Int;
  int64_t intValue = 0;
  double floatValue = 0.0;
  StringRef identifier;
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
      .Case("mod", Expr::Kind::Mod)
      .Case("pow", Expr::Kind::Pow)
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
 * @brief A resolved gate call.
 *
 * @details
 * Array members are borrowed for the duration of the sink call (top-level) or
 * bump-allocated (gate-definition bodies).
 */
struct GateCall {
  SMLoc loc;
  StringRef identifier;
  ArrayRef<Modifier> modifiers;
  ArrayRef<const Expr*> parameters;
  ArrayRef<Operand> operands;
};

/**
 * @ingroup ParseVocabulary
 * @brief A branch condition: a boolean-valued expression.
 *
 * @details
 * Bump-allocated; children are borrowed pointers. Register comparisons (e.g.,
 * `c == 5`) are not supported yet.
 */
struct Condition {
  enum class Kind : uint8_t { Measurement, Bit, Literal, Not, And, Or };

  SMLoc loc;
  Kind kind = Kind::Bit;
  Operand operand;                ///< For `Measurement`.
  BitReference bit;               ///< For `Bit`.
  bool literalValue = false;      ///< For `Literal`.
  const Condition* lhs = nullptr; ///< For `Not`, `And`, and `Or`.
  const Condition* rhs = nullptr; ///< For `And` and `Or`.
};

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
concept QASMSink =
    requires(S s, SMLoc loc, StringRef str, const Expr& expr,
             const Operand& operand, const BitReference& reference,
             const Condition& condition, const GateCall& call,
             ArrayRef<Operand> operands, ArrayRef<StringRef> names,
             ArrayRef<GateCall> body, function_ref<LogicalResult()> cont,
             double d, bool flag) {
      s.error(loc, str);
      s.include(loc, str);
      s.boolDecl(loc, str, &condition);
      s.intDecl(loc, str, &expr, flag);
      s.floatDecl(loc, str, &expr);
      s.boolAssign(loc, str, condition);
      s.intAssign(loc, str, expr);
      s.floatAssign(loc, str, expr);
      s.qubitRegister(loc, str, &expr);
      s.classicalRegister(loc, str, &expr, flag);
      s.measure(loc, reference, operand);
      s.reset(loc, operand);
      s.barrier(loc, operands);
      s.gateCall(call);
      s.gateDefinition(loc, str, names, names, body);
      s.ifConditionOnly(loc, condition);
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
    // Initialize the outermost scope
    scopes.emplace_back();
  }

  [[nodiscard]] LogicalResult parseProgram() {
    while (!atEnd()) {
      if (failed(parseStatement())) {
        return failure();
      }
    }
    return success();
  }

private:
  /**
   * @brief A declared name.
   *
   * @details
   * The parser owns the names a program declares: it rejects redeclarations,
   * chooses the grammar of an assignment from what its target names, and
   * rejects an assignment to a `const`. Each declaration and assignment tells
   * the sink the type it needs, so the sink keeps no symbol table of its own.
   */
  struct Symbol {
    enum class Kind : uint8_t {
      Bool,
      Int,
      Float,
      QubitRegister,
      ClassicalRegister
    };

    Kind kind = Kind::Int;
    bool isConst = false; ///< Relevant if a scalar.
  };

  /**
   * @brief A lexical scope.
   *
   * @details
   * Names declared while it is alive go out of scope when it is destroyed.
   */
  class SymbolScope {
  public:
    explicit SymbolScope(Parser& parser) : parser(parser) {
      parser.scopes.emplace_back();
    }
    ~SymbolScope() { parser.scopes.pop_back(); }

    SymbolScope(const SymbolScope&) = delete;
    SymbolScope& operator=(const SymbolScope&) = delete;
    SymbolScope(SymbolScope&&) = delete;
    SymbolScope& operator=(SymbolScope&&) = delete;

  private:
    Parser& parser;
  };

  /// Lookup the symbol @p id refers to.
  [[nodiscard]] const Symbol* lookup(StringRef id) const {
    for (const auto& scope : reverse(scopes)) {
      const auto it = scope.find(id);
      if (it != scope.end()) {
        return &it->second;
      }
    }
    return nullptr;
  }

  /**
   * @brief Declare @p id in the innermost scope.
   *
   * @details
   * A name may be redeclared in an inner scope, where it shadows the outer one
   * until that scope ends, but not twice in the same scope.
   */
  [[nodiscard]] LogicalResult declare(SMLoc loc, StringRef id, Symbol symbol) {
    if (!scopes.back().insert({id, symbol}).second) {
      return sink.error(loc, "identifier '" + id + "' already declared");
    }
    return success();
  }

  //===--- Token scaffolding --------------------------------------------===//

  void advance() {
    currentToken = nextToken;
    nextToken = lexer.next();
  }

  [[nodiscard]] const Token& current() const { return currentToken; }
  [[nodiscard]] const Token& peek() const { return nextToken; }
  [[nodiscard]] bool atEnd() const {
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

  template <class T> [[nodiscard]] ArrayRef<T> copyToArena(ArrayRef<T> values) {
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
    default:
      return sink.error(current().loc, "unexpected token");
    }
  }

  /// Parse a `{ ... }` block or a single statement, emitting into whatever
  /// region the sink has made current.
  [[nodiscard]] LogicalResult parseBlock() {
    const SymbolScope scope(*this);
    return parseBlockInScope();
  }

  /// Parse a `{ ... }` block or a single statement into the current scope,
  /// which the caller has already opened.
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
    return success();
  }

  //===--- Include ------------------------------------------------------===//

  [[nodiscard]] LogicalResult parseInclude() {
    const auto loc = current().loc;
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

    typename Symbol::Kind kind{};
    switch (current().kind) {
    case TokenKind::Bool:
      kind = Symbol::Kind::Bool;
      break;
    case TokenKind::Int:
    case TokenKind::Uint:
      kind = Symbol::Kind::Int;
      break;
    case TokenKind::Float:
      kind = Symbol::Kind::Float;
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
    const auto id = current().identifier;
    advance();

    const bool hasInitializer = current().kind == TokenKind::Equals;
    if (hasInitializer) {
      advance();
    }
    if (isConst && !hasInitializer) {
      return sink.error(loc, "'const' declaration of '" + id +
                                 "' requires an initializer");
    }

    if (failed(declare(loc, id, {.kind = kind, .isConst = isConst}))) {
      return failure();
    }

    if (kind == Symbol::Kind::Bool) {
      const Condition* initializer = nullptr;
      if (hasInitializer) {
        auto condition = parseCondition();
        if (failed(condition)) {
          return failure();
        }
        initializer = *condition;
      }
      if (failed(expect(TokenKind::Semicolon))) {
        return failure();
      }
      return sink.boolDecl(loc, id, initializer);
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
    if (kind == Symbol::Kind::Int) {
      return sink.intDecl(loc, id, initializer, isConst);
    }
    return sink.floatDecl(loc, id, initializer);
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
      return sink.error(current().loc, "expected identifier");
    }
    const auto id = current().identifier;
    advance();
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    if (failed(declare(loc, id, {.kind = Symbol::Kind::QubitRegister}))) {
      return failure();
    }
    return sink.qubitRegister(loc, id, size);
  }

  /// Parse `qreg <id>[<n>];`.
  [[nodiscard]] LogicalResult parseQregDecl() {
    const auto loc = current().loc;
    advance(); // qreg
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected identifier");
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
    if (failed(declare(loc, id, {.kind = Symbol::Kind::QubitRegister}))) {
      return failure();
    }
    return sink.qubitRegister(loc, id, size);
  }

  /// Parse `output bit[<n>] <id> (= <measurement>);`.
  [[nodiscard]] LogicalResult parseOutputDecl() {
    advance(); // output
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
      return sink.error(current().loc, "expected identifier");
    }
    const auto id = current().identifier;
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

    if (failed(declare(loc, id, {.kind = Symbol::Kind::ClassicalRegister}))) {
      return failure();
    }
    if (failed(sink.classicalRegister(loc, id, size, isOutput))) {
      return failure();
    }
    if (measureSource) {
      const BitReference target{.loc = loc, .identifier = id, .index = nullptr};
      return sink.measure(loc, target, *measureSource);
    }
    return success();
  }

  /// Parse `creg <id>[<n>];`.
  [[nodiscard]] LogicalResult parseCregDecl() {
    const auto loc = current().loc;
    advance(); // creg
    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected identifier");
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
    if (failed(declare(loc, id, {.kind = Symbol::Kind::ClassicalRegister}))) {
      return failure();
    }
    return sink.classicalRegister(loc, id, size, /*isOutput=*/false);
  }

  //===--- Assignment ---------------------------------------------------===//

  /// Parse an assignment.
  [[nodiscard]] LogicalResult parseAssignment() {
    const auto loc = current().loc;
    const auto id = current().identifier;
    const auto* symbol = lookup(id);
    if (symbol == nullptr) {
      return sink.error(loc, "unknown identifier '" + id + "'");
    }
    switch (symbol->kind) {
    case Symbol::Kind::Bool:
    case Symbol::Kind::Int:
    case Symbol::Kind::Float:
      return parseScalarAssignment(*symbol);
    case Symbol::Kind::ClassicalRegister:
      return parseMeasureAssignment();
    case Symbol::Kind::QubitRegister:
      return sink.error(loc, "cannot assign to qubit register '" + id + "'");
    }
    llvm_unreachable("unknown symbol kind");
  }

  /// Parse `<id> = <value>;`.
  [[nodiscard]] LogicalResult parseScalarAssignment(Symbol symbol) {
    const auto loc = current().loc;
    const auto id = current().identifier;
    advance(); // identifier

    if (current().kind == TokenKind::LBracket) {
      return sink.error(current().loc,
                        "cannot assign to an element of a scalar");
    }
    if (symbol.isConst) {
      return sink.error(loc,
                        "cannot assign to the 'const'-declared '" + id + "'");
    }
    if (!scopes.back().contains(id)) {
      return sink.error(loc, "cannot assign to '" + id +
                                 "' from inside a block; it is declared in an "
                                 "enclosing scope");
    }
    if (current().kind == TokenKind::CompoundAssign) {
      return sink.error(current().loc,
                        "compound assignments are not supported yet");
    }
    if (failed(expect(TokenKind::Equals))) {
      return failure();
    }

    if (symbol.kind == Symbol::Kind::Bool) {
      auto condition = parseCondition();
      if (failed(condition)) {
        return failure();
      }
      if (failed(expect(TokenKind::Semicolon))) {
        return failure();
      }
      return sink.boolAssign(loc, id, **condition);
    }

    auto value = parseExpression();
    if (failed(value)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    if (symbol.kind == Symbol::Kind::Int) {
      return sink.intAssign(loc, id, **value);
    }
    return sink.floatAssign(loc, id, **value);
  }

  /// Parse `<bit-reference> = measure <operand>;`.
  [[nodiscard]] LogicalResult parseMeasureAssignment() {
    const auto loc = current().loc;

    auto target = parseBitReference();
    if (failed(target)) {
      return failure();
    }
    const auto& reference = *target;

    if (current().kind == TokenKind::CompoundAssign) {
      return sink.error(current().loc,
                        "compound assignments are not supported yet");
    }
    if (failed(expect(TokenKind::Equals))) {
      return failure();
    }
    if (failed(expect(TokenKind::Measure))) {
      return failure();
    }

    auto operand = parseGateOperand();
    if (failed(operand)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.measure(loc, reference, *operand);
  }

  //===--- Measure ------------------------------------------------------===//

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
    auto target = parseBitReference();
    if (failed(target)) {
      return failure();
    }
    if (failed(expect(TokenKind::Semicolon))) {
      return failure();
    }
    return sink.measure(loc, *target, *operand);
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
      return sink.error(current().loc, "expected gate name");
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

    // Parse body
    if (failed(expect(TokenKind::LBrace))) {
      return failure();
    }
    SmallVector<GateCall> body;
    while (current().kind != TokenKind::RBrace) {
      // Bodies outlive this frame, so the call's arrays are copied into the
      // arena; the scratch buffers below are discarded.
      SmallVector<Modifier> callModifiers;
      SmallVector<const Expr*> callParameters;
      SmallVector<Operand> callOperands;
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

    return sink.gateDefinition(loc, id, copyToArena(ArrayRef(parameters)),
                               copyToArena(ArrayRef(targets)),
                               copyToArena(ArrayRef(body)));
  }

  [[nodiscard]] LogicalResult parseGateCallStatement() {
    // The scratch buffers must outlive the sink call, so they live here rather
    // than inside `parseGateCall`.
    SmallVector<Modifier> modifiers;
    SmallVector<const Expr*> parameters;
    SmallVector<Operand> operands;
    auto call = parseGateCall(modifiers, parameters, operands,
                              /*persist=*/false);
    if (failed(call)) {
      return failure();
    }
    return sink.gateCall(*call);
  }

  [[nodiscard]] FailureOr<GateCall>
  parseGateCall(SmallVectorImpl<Modifier>& modifiers,
                SmallVectorImpl<const Expr*>& parameters,
                SmallVectorImpl<Operand>& operands, const bool persist) {
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
                        "gate calls with designators are not supported yet");
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

    // Top-level calls are consumed by the sink while the caller's buffers are
    // still alive, so borrowing them is safe. Gate definitions outlive this
    // frame, so their arrays are copied into the arena.
    if (persist) {
      call.modifiers = copyToArena(ArrayRef(modifiers));
      call.parameters = copyToArena(ArrayRef(parameters));
      call.operands = copyToArena(ArrayRef(operands));
      return call;
    }
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
      return sink.error(current().loc, "expected a gate operand");
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
      return sink.error(current().loc, "expected an identifier");
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
      return sink.error(current().loc, "expected an identifier");
    }
    identifiers.push_back(current().identifier);
    advance();
    while (current().kind == TokenKind::Comma) {
      advance();
      if (current().kind != TokenKind::Identifier) {
        return sink.error(current().loc, "expected an identifier");
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
    auto conditionOrFailure = parseCondition();
    if (failed(conditionOrFailure)) {
      return failure();
    }
    const auto& condition = **conditionOrFailure;
    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }

    const bool thenEmpty =
        current().kind == TokenKind::LBrace && peek().kind == TokenKind::RBrace;
    if (thenEmpty) {
      advance(); // {
      advance(); // }
      if (current().kind != TokenKind::Else) {
        return sink.ifConditionOnly(loc, condition);
      }
      advance(); // else
      auto scope = sink.ifBegin(loc, condition, /*invert=*/true);
      if (failed(scope)) {
        return failure();
      }
      if (failed(parseBlock())) {
        return failure();
      }
      return sink.ifEnd(*scope, /*hadElse=*/false);
    }

    auto scope = sink.ifBegin(loc, condition, /*invert=*/false);
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
    advance(); // type

    if (current().kind != TokenKind::Identifier) {
      return sink.error(current().loc, "expected loop variable");
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

    // The loop variable lives in the body's scope, as if it were declared as
    // its first statement. It is therefore not visible to the range expressions
    // above, and it cannot be redeclared directly in the body.
    const SymbolScope scope(*this);
    if (failed(declare(iv.loc, iv.identifier, {.kind = Symbol::Kind::Int}))) {
      return failure();
    }

    return sink.forStmt(loc, iv.identifier, **start, *step, *stop,
                        [this] { return parseBlockInScope(); });
  }

  [[nodiscard]] LogicalResult parseWhile() {
    const auto loc = current().loc;
    advance(); // while
    if (failed(expect(TokenKind::LParen))) {
      return failure();
    }
    auto conditionOrFailure = parseCondition();
    if (failed(conditionOrFailure)) {
      return failure();
    }
    const auto& condition = **conditionOrFailure;
    if (failed(expect(TokenKind::RParen))) {
      return failure();
    }
    return sink.whileStmt(loc, condition, [this] { return parseBlock(); });
  }

  //===--- Conditions --------------------------------------------------===//

  /// cond := andCond ('||' andCond)*
  [[nodiscard]] FailureOr<const Condition*> parseCondition() {
    auto lhs = parseAndCondition();
    if (failed(lhs)) {
      return failure();
    }
    const auto* result = *lhs;
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
  [[nodiscard]] FailureOr<const Condition*> parseAndCondition() {
    auto lhs = parseUnaryCondition();
    if (failed(lhs)) {
      return failure();
    }
    const auto* result = *lhs;
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
  [[nodiscard]] FailureOr<const Condition*> parseUnaryCondition() {
    if (current().kind == TokenKind::ExclamationPoint ||
        current().kind == TokenKind::Tilde) {
      const auto loc = current().loc;
      advance();
      auto operand = parseUnaryCondition();
      if (failed(operand)) {
        return failure();
      }
      auto* condition = makeCondition();
      condition->loc = loc;
      condition->kind = Condition::Kind::Not;
      condition->lhs = *operand;
      return condition;
    }
    return parsePrimaryCondition();
  }

  /// primaryCond := 'measure' gateOperand | '(' cond ')' | reference
  [[nodiscard]] FailureOr<const Condition*> parsePrimaryCondition() {
    const auto loc = current().loc;
    switch (current().kind) {
    case TokenKind::True:
    case TokenKind::False: {
      auto* condition = makeCondition();
      condition->loc = loc;
      condition->kind = Condition::Kind::Literal;
      condition->literalValue = current().kind == TokenKind::True;
      advance();
      return finishPrimaryCondition(condition);
    }
    case TokenKind::Measure: {
      advance();
      auto operand = parseGateOperand();
      if (failed(operand)) {
        return failure();
      }
      auto* condition = makeCondition();
      condition->loc = loc;
      condition->kind = Condition::Kind::Measurement;
      condition->operand = *operand;
      return finishPrimaryCondition(condition);
    }
    case TokenKind::LParen: {
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
    case TokenKind::Identifier: {
      auto bit = parseBitReference();
      if (failed(bit)) {
        return failure();
      }
      auto* condition = makeCondition();
      condition->loc = loc;
      condition->kind = Condition::Kind::Bit;
      condition->bit = *bit;
      return finishPrimaryCondition(condition);
    }
    default:
      return sink.error(loc, "unsupported condition expression");
    }
  }

  /// Reject a register comparison (e.g. `c == 5`) trailing a primary.
  [[nodiscard]] FailureOr<const Condition*>
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
                                               const SMLoc loc) {
    auto* condition = makeCondition();
    condition->loc = loc;
    condition->kind = kind;
    condition->lhs = lhs;
    condition->rhs = rhs;
    return condition;
  }

  //===--- Expressions --------------------------------------------------===//

  [[nodiscard]] FailureOr<const Expr*> parseExpression() {
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

  [[nodiscard]] FailureOr<const Expr*> parseUnary() {
    if (current().kind == TokenKind::Minus) {
      const auto loc = current().loc;
      advance();
      auto operand = parseUnary();
      if (failed(operand)) {
        return failure();
      }
      auto* expr = makeExpr();
      expr->loc = loc;
      expr->kind = Expr::Kind::Neg;
      expr->lhs = *operand;
      return expr;
    }
    return parsePrimary();
  }

  [[nodiscard]] FailureOr<const Expr*> parsePrimary() {
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
    case TokenKind::Identifier: {
      if (peek().kind == TokenKind::LParen) {
        const auto kind = getMathFunctionKind(current().identifier);
        if (!kind) {
          return sink.error(current().loc,
                            "unknown function '" + current().identifier + "'");
        }
        return parseMathCall(*kind, expr);
      }
      expr->kind = Expr::Kind::Identifier;
      expr->identifier = current().identifier;
      advance();
      return expr;
    }
    // `pow` is also a gate modifier, so it has a dedicated token
    case TokenKind::Pow:
      return parseMathCall(Expr::Kind::Pow, expr);
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

    if (kind == Expr::Kind::Mod || kind == Expr::Kind::Pow) {
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
  SmallVector<llvm::StringMap<Symbol>> scopes;
};

} // namespace mlir::qc::detail
