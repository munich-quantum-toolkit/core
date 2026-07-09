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

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/SMLoc.h>

#include <cstdint>

namespace mlir::qc::detail {

/// The kind of a lexical token.
enum class TokenKind : uint8_t {
  Eof,
  Error,

  // Keywords
  OpenQASM,
  Include,
  Const,
  Qubit,
  Qreg,
  Bit,
  CReg,
  Gate,
  Opaque,
  Barrier,
  Reset,
  Measure,
  If,
  Else,
  For,
  While,
  In,
  Gphase,
  Inv,
  Pow,
  Ctrl,
  NegCtrl,
  // Type keywords (only a subset is supported; the rest produce diagnostics)
  Int,
  Uint,
  Bool,
  Float,
  Angle,
  Duration,
  True,
  False,

  // Identifiers and literals
  Identifier,
  HardwareQubit,
  StringLiteral,
  IntegerLiteral,
  FloatLiteral,

  // Punctuation
  LParen,
  RParen,
  LBracket,
  RBracket,
  LBrace,
  RBrace,
  Comma,
  Semicolon,
  Colon,
  Arrow,
  At,

  // Operators
  Equals,
  Plus,
  Minus,
  Asterisk,
  Slash,
  Tilde,
  ExclamationPoint,
  AmpAmp,         // `&&`
  PipePipe,       // `||`
  CompoundAssign, // any `<op>=` such as `+=`, `-=`, ...

  // Comparisons
  EqualsEquals,
  NotEquals,
  Less,
  LessEquals,
  Greater,
  GreaterEquals,
};

/// A human-readable name for @p kind, used in diagnostics.
[[nodiscard]] llvm::StringRef describe(TokenKind kind);

/// A single lexical token. Spellings are zero-copy views into the source
/// buffer; literals are pre-parsed.
struct Token {
  TokenKind kind = TokenKind::Eof;
  llvm::StringRef spelling; ///< Identifier text or string-literal contents.
  llvm::SMLoc loc;
  int64_t intValue = 0;
  double floatValue = 0.0;
};

/**
 * @brief A zero-copy lexer over an OpenQASM 3 source buffer.
 *
 * @details
 * The lexer holds pointers into the buffer and produces tokens on demand
 * without allocating. Token locations are `SMLoc`s into the buffer, so they can
 * be resolved to line/column via the owning `llvm::SourceMgr`.
 */
class Lexer {
public:
  explicit Lexer(llvm::StringRef buffer)
      : cur(buffer.begin()), end(buffer.end()) {}

  /// Produce the next token, or an `Eof`/`Error` token at the end/on failure.
  [[nodiscard]] Token next();

private:
  [[nodiscard]] bool atEnd() const { return cur == end; }
  void skipTrivia();
  [[nodiscard]] Token lexIdentifierOrKeyword(const char* start);
  [[nodiscard]] Token lexNumber(const char* start);
  [[nodiscard]] Token lexString(const char* start);
  [[nodiscard]] Token lexHardwareQubit(const char* start);

  const char* cur;
  const char* end;
};

} // namespace mlir::qc::detail
