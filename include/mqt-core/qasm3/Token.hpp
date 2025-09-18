/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>

namespace qasm3 {

struct Token {
  enum class Kind : uint8_t {
    None,

    OpenQasm,
    Include,
    DefCalGrammar,
    Def,
    Cal,
    DefCal,
    Gate,
    Opaque,
    Extern,
    Box,
    Let,

    Break,
    Continue,
    If,
    Else,
    End,
    Return,
    For,
    While,
    In,

    Pragma,

    // types
    Input,
    Output,
    Const,
    ReadOnly,
    Mutable,

    Qreg,
    Qubit,

    CReg,
    Bool,
    Bit,
    Int,
    Uint,
    Float,
    Angle,
    Complex,
    Array,
    Void,

    Duration,
    Stretch,

    // builtin identifiers
    Gphase,
    Inv,
    Pow,
    Ctrl,
    NegCtrl,

    Dim,

    DurationOf,

    Delay,
    Reset,
    Measure,
    Barrier,

    True,
    False,

    LBracket,
    RBracket,
    LBrace,
    RBrace,
    LParen,
    RParen,

    Colon,
    Semicolon,
    Eof,

    Dot,
    Comma,

    Equals,
    Arrow,
    Plus,
    DoublePlus,
    Minus,
    Asterisk,
    DoubleAsterisk,
    Slash,
    Percent,
    Pipe,
    DoublePipe,
    Ampersand,
    DoubleAmpersand,
    Caret,
    At,
    Tilde,
    ExclamationPoint,

    DoubleEquals,
    NotEquals,
    PlusEquals,
    MinusEquals,
    AsteriskEquals,
    SlashEquals,
    AmpersandEquals,
    PipeEquals,
    TildeEquals,
    CaretEquals,
    LeftShitEquals,
    RightShiftEquals,
    PercentEquals,
    DoubleAsteriskEquals,

    LessThan,
    LessThanEquals,
    GreaterThan,
    GreaterThanEquals,
    LeftShift,
    RightShift,

    Imag,

    Underscore,

    DoubleQuote,
    SingleQuote,
    BackSlash,

    Identifier,

    HardwareQubit,
    StringLiteral,
    IntegerLiteral,
    FloatLiteral,
    TimingLiteral,

    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,

    InitialLayout,
    OutputPermutation,
  };

  Kind kind = Kind::None;
  size_t line = 0;
  size_t col = 0;
  size_t endLine = 0;
  size_t endCol = 0;
  int64_t val{};
  bool isSigned{false};
  double valReal{};
  std::string str;

  Token(const size_t l, const size_t c)
      : line(l), col(c), endLine(l), endCol(c) {}
  Token(const Kind k, const size_t l, const size_t c)
      : kind(k), line(l), col(c), endLine(l), endCol(c) {}
  Token(const Kind k, const size_t l, const size_t c, std::string s)
      : kind(k), line(l), col(c), endLine(l), endCol(c), str(std::move(s)) {}

  static std::string kindToString(Kind kind);

  [[nodiscard]] std::string toString() const;

  friend std::ostream& operator<<(std::ostream& os, const Kind& k);

  friend std::ostream& operator<<(std::ostream& os, const Token& t);
};
} // namespace qasm3
