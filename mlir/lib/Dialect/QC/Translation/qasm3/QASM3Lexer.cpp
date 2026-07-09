/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/qasm3/QASM3Lexer.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/SMLoc.h>

#include <cstdint>

namespace mlir::qc::detail {

namespace {

[[nodiscard]] bool isIdentifierStart(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         static_cast<unsigned char>(c) >= 0x80; // allow UTF-8 (e.g. π, τ, ℇ)
}

[[nodiscard]] bool isIdentifierContinue(char c) {
  return isIdentifierStart(c) || (c >= '0' && c <= '9');
}

[[nodiscard]] bool isDigit(char c) { return c >= '0' && c <= '9'; }

[[nodiscard]] TokenKind keywordKind(llvm::StringRef text) {
  return llvm::StringSwitch<TokenKind>(text)
      .Case("OPENQASM", TokenKind::OpenQASM)
      .Case("include", TokenKind::Include)
      .Case("const", TokenKind::Const)
      .Case("qubit", TokenKind::Qubit)
      .Case("qreg", TokenKind::Qreg)
      .Case("bit", TokenKind::Bit)
      .Case("creg", TokenKind::CReg)
      .Case("gate", TokenKind::Gate)
      .Case("opaque", TokenKind::Opaque)
      .Case("barrier", TokenKind::Barrier)
      .Case("reset", TokenKind::Reset)
      .Case("measure", TokenKind::Measure)
      .Case("if", TokenKind::If)
      .Case("else", TokenKind::Else)
      .Case("for", TokenKind::For)
      .Case("while", TokenKind::While)
      .Case("in", TokenKind::In)
      .Case("gphase", TokenKind::Gphase)
      .Case("inv", TokenKind::Inv)
      .Case("pow", TokenKind::Pow)
      .Case("ctrl", TokenKind::Ctrl)
      .Case("negctrl", TokenKind::NegCtrl)
      .Case("int", TokenKind::Int)
      .Case("uint", TokenKind::Uint)
      .Case("bool", TokenKind::Bool)
      .Case("float", TokenKind::Float)
      .Case("angle", TokenKind::Angle)
      .Case("duration", TokenKind::Duration)
      .Default(TokenKind::Identifier);
}

} // namespace

llvm::StringRef describe(const TokenKind kind) {
  switch (kind) {
  case TokenKind::Eof:
    return "end of input";
  case TokenKind::Error:
    return "invalid token";
  case TokenKind::Identifier:
    return "identifier";
  case TokenKind::HardwareQubit:
    return "hardware qubit";
  case TokenKind::StringLiteral:
    return "string literal";
  case TokenKind::IntegerLiteral:
    return "integer literal";
  case TokenKind::FloatLiteral:
    return "float literal";
  case TokenKind::LParen:
    return "'('";
  case TokenKind::RParen:
    return "')'";
  case TokenKind::LBracket:
    return "'['";
  case TokenKind::RBracket:
    return "']'";
  case TokenKind::LBrace:
    return "'{'";
  case TokenKind::RBrace:
    return "'}'";
  case TokenKind::Comma:
    return "','";
  case TokenKind::Semicolon:
    return "';'";
  case TokenKind::Colon:
    return "':'";
  case TokenKind::Arrow:
    return "'->'";
  case TokenKind::At:
    return "'@'";
  case TokenKind::Equals:
    return "'='";
  default:
    return "token";
  }
}

void Lexer::skipTrivia() {
  while (!atEnd()) {
    const char c = *cur;
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      ++cur;
      continue;
    }
    if (c == '/' && (cur + 1) != end && cur[1] == '/') {
      // Line comment
      cur += 2;
      while (!atEnd() && *cur != '\n') {
        ++cur;
      }
      continue;
    }
    if (c == '/' && (cur + 1) != end && cur[1] == '*') {
      cur += 2;
      while (!atEnd() && !(*cur == '*' && (cur + 1) != end && cur[1] == '/')) {
        ++cur;
      }
      if (!atEnd()) {
        cur += 2; // consume the closing */
      }
      continue;
    }
    break;
  }
}

Token Lexer::lexIdentifierOrKeyword(const char* start) {
  while (!atEnd() && isIdentifierContinue(*cur)) {
    ++cur;
  }
  const llvm::StringRef text(start, static_cast<size_t>(cur - start));
  Token token;
  token.kind = keywordKind(text);
  token.spelling = text;
  token.loc = llvm::SMLoc::getFromPointer(start);
  return token;
}

Token Lexer::lexNumber(const char* start) {
  bool isFloat = false;
  while (!atEnd() && isDigit(*cur)) {
    ++cur;
  }
  if (!atEnd() && *cur == '.') {
    isFloat = true;
    ++cur;
    while (!atEnd() && isDigit(*cur)) {
      ++cur;
    }
  }
  if (!atEnd() && (*cur == 'e' || *cur == 'E')) {
    isFloat = true;
    ++cur;
    if (!atEnd() && (*cur == '+' || *cur == '-')) {
      ++cur;
    }
    while (!atEnd() && isDigit(*cur)) {
      ++cur;
    }
  }
  const llvm::StringRef text(start, static_cast<size_t>(cur - start));
  Token token;
  token.loc = llvm::SMLoc::getFromPointer(start);
  token.spelling = text;
  if (isFloat) {
    token.kind = TokenKind::FloatLiteral;
    if (text.getAsDouble(token.floatValue)) {
      token.kind = TokenKind::Error;
    }
  } else {
    token.kind = TokenKind::IntegerLiteral;
    if (text.getAsInteger(10, token.intValue)) {
      token.kind = TokenKind::Error;
    }
  }
  return token;
}

Token Lexer::lexString(const char* start) {
  ++cur; // consume opening quote
  const char* contentStart = cur;
  while (!atEnd() && *cur != '"') {
    ++cur;
  }
  Token token;
  token.loc = llvm::SMLoc::getFromPointer(start);
  if (atEnd()) {
    token.kind = TokenKind::Error;
    return token;
  }
  token.kind = TokenKind::StringLiteral;
  token.spelling =
      llvm::StringRef(contentStart, static_cast<size_t>(cur - contentStart));
  ++cur; // consume closing quote
  return token;
}

Token Lexer::lexHardwareQubit(const char* start) {
  ++cur; // consume '$'
  const char* digitsStart = cur;
  while (!atEnd() && isDigit(*cur)) {
    ++cur;
  }
  Token token;
  token.loc = llvm::SMLoc::getFromPointer(start);
  const llvm::StringRef digits(digitsStart,
                               static_cast<size_t>(cur - digitsStart));
  if (digits.empty() || digits.getAsInteger(10, token.intValue)) {
    token.kind = TokenKind::Error;
    return token;
  }
  token.kind = TokenKind::HardwareQubit;
  return token;
}

Token Lexer::next() {
  skipTrivia();

  Token token;
  token.loc = llvm::SMLoc::getFromPointer(cur);
  if (atEnd()) {
    token.kind = TokenKind::Eof;
    return token;
  }

  const char* start = cur;
  const char c = *cur;

  if (isIdentifierStart(c)) {
    return lexIdentifierOrKeyword(start);
  }
  if (isDigit(c)) {
    return lexNumber(start);
  }
  if (c == '"') {
    return lexString(start);
  }
  if (c == '$') {
    return lexHardwareQubit(start);
  }

  // Punctuation and operators.
  const auto peek = [&](const char expected) {
    return (cur + 1) != end && cur[1] == expected;
  };
  const auto single = [&](const TokenKind kind) {
    ++cur;
    token.kind = kind;
    return token;
  };
  const auto twoChar = [&](const TokenKind kind) {
    cur += 2;
    token.kind = kind;
    return token;
  };

  switch (c) {
  case '(':
    return single(TokenKind::LParen);
  case ')':
    return single(TokenKind::RParen);
  case '[':
    return single(TokenKind::LBracket);
  case ']':
    return single(TokenKind::RBracket);
  case '{':
    return single(TokenKind::LBrace);
  case '}':
    return single(TokenKind::RBrace);
  case ',':
    return single(TokenKind::Comma);
  case ';':
    return single(TokenKind::Semicolon);
  case ':':
    return single(TokenKind::Colon);
  case '@':
    return single(TokenKind::At);
  case '~':
    return peek('=') ? twoChar(TokenKind::CompoundAssign)
                     : single(TokenKind::Tilde);
  case '=':
    return peek('=') ? twoChar(TokenKind::EqualsEquals)
                     : single(TokenKind::Equals);
  case '!':
    return peek('=') ? twoChar(TokenKind::NotEquals)
                     : single(TokenKind::ExclamationPoint);
  case '<':
    return peek('=') ? twoChar(TokenKind::LessEquals) : single(TokenKind::Less);
  case '>':
    return peek('=') ? twoChar(TokenKind::GreaterEquals)
                     : single(TokenKind::Greater);
  case '+':
    return peek('=') ? twoChar(TokenKind::CompoundAssign)
                     : single(TokenKind::Plus);
  case '-':
    if (peek('>')) {
      return twoChar(TokenKind::Arrow);
    }
    return peek('=') ? twoChar(TokenKind::CompoundAssign)
                     : single(TokenKind::Minus);
  case '*':
    return peek('=') ? twoChar(TokenKind::CompoundAssign)
                     : single(TokenKind::Asterisk);
  case '/':
    return peek('=') ? twoChar(TokenKind::CompoundAssign)
                     : single(TokenKind::Slash);
  case '%':
  case '&':
  case '|':
  case '^':
    if (peek('=')) {
      return twoChar(TokenKind::CompoundAssign);
    }
    break;
  default:
    break;
  }

  return single(TokenKind::Error);
}

} // namespace mlir::qc::detail
