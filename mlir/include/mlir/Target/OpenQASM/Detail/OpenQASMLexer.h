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
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mlir::oq3::frontend::detail {

/// The kind of a lexical token.
enum class TokenKind : uint8_t {
  Eof,
  Error,
  UnterminatedComment,
  UnsupportedKeyword,

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
  Output,
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

  // Types
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
  DoubleAsterisk,
  Slash,
  Percent,
  Tilde,
  Amp,
  Pipe,
  Caret,
  ShiftLeft,
  ShiftRight,
  ExclamationPoint,
  AmpAmp,         // `&&`
  PipePipe,       // `||`
  CompoundAssign, // `+=`, `-=`, ...

  // Comparisons
  EqualsEquals,
  NotEquals,
  Less,
  LessEquals,
  Greater,
  GreaterEquals,
};

/// A human-readable name for @p kind, used in diagnostics.
[[nodiscard]] StringRef describe(TokenKind kind);

/**
 * @brief A single lexical token.
 *
 * @details
 * Spellings are zero-copy views into the source buffer. Literals are
 * pre-parsed.
 */
struct Token {
  TokenKind kind = TokenKind::Eof;
  SMLoc loc;
  StringRef identifier;  ///< For `Identifier` tokens.
  StringRef stringValue; ///< For `StringLiteral` tokens.
  StringRef spelling;    ///< Zero-copy source spelling.
  uint64_t intValue = 0;
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
  explicit Lexer(StringRef buffer) : cur(buffer.begin()), end(buffer.end()) {}

  /// Produce the next token, or an `Eof`/`Error` token at the end/on failure.
  [[nodiscard]] Token next();

private:
  [[nodiscard]] bool atEnd() const { return cur == end; }

  /// The character after the cursor, or `'\0'` at the end of the buffer.
  [[nodiscard]] char peek() const { return (cur + 1) != end ? cur[1] : '\0'; }

  /**
   * @brief Skip whitespace and comments.
   * @return The start of an unterminated block comment, or `nullptr` if the
   * trivia is well-formed.
   */
  [[nodiscard]] const char* skipTrivia();

  [[nodiscard]] Token lexIdentifierOrKeyword(const char* start);
  [[nodiscard]] Token lexNumber(const char* start);
  [[nodiscard]] Token lexString(const char* start, char quote);
  [[nodiscard]] Token lexHardwareQubit(const char* start);

  const char* cur;
  const char* end;
};

} // namespace mlir::oq3::frontend::detail
