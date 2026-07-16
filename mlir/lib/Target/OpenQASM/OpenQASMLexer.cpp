/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "OpenQASMLexer.h"

#include "OpenQASMUnicode.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ConvertUTF.h>
#include <mlir/Support/LLVM.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <optional>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
namespace mlir::oq3::frontend::detail {

[[nodiscard]] static bool canStartIdentifier(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

[[nodiscard]] static bool canContinueIdentifier(char c) {
  return canStartIdentifier(c) || (c >= '0' && c <= '9');
}

[[nodiscard]] static bool isDigit(char c) { return c >= '0' && c <= '9'; }

[[nodiscard]] static bool isDigitForRadix(const char c, const unsigned radix) {
  if (c >= '0' && c <= '9') {
    return static_cast<unsigned>(c - '0') < radix;
  }
  if (c >= 'a' && c <= 'f') {
    return static_cast<unsigned>(c - 'a' + 10) < radix;
  }
  if (c >= 'A' && c <= 'F') {
    return static_cast<unsigned>(c - 'A' + 10) < radix;
  }
  return false;
}

template <class IsDigit>
[[nodiscard]] static bool hasValidSeparators(const StringRef text,
                                             IsDigit&& isValidDigit) {
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] != '_') {
      continue;
    }
    if (index == 0 || index + 1 == text.size() ||
        !isValidDigit(text[index - 1]) || !isValidDigit(text[index + 1])) {
      return false;
    }
  }
  return true;
}

struct DecodedCodePoint {
  std::uint32_t value = 0;
  std::size_t width = 0;
};

[[nodiscard]] static std::optional<DecodedCodePoint>
decodeCodePoint(const char* position, const char* end) {
  auto* source = reinterpret_cast<const llvm::UTF8*>(position);
  const auto* const begin = source;
  const auto* const sourceEnd = reinterpret_cast<const llvm::UTF8*>(end);
  llvm::UTF32 codePoint = 0;
  if (llvm::convertUTF8Sequence(&source, sourceEnd, &codePoint,
                                llvm::strictConversion) != llvm::conversionOK) {
    return std::nullopt;
  }
  return DecodedCodePoint{.value = codePoint,
                          .width = static_cast<std::size_t>(source - begin)};
}

[[nodiscard]] static TokenKind keywordKind(StringRef text) {
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
      .Case("output", TokenKind::Output)
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
      .Case("true", TokenKind::True)
      .Case("false", TokenKind::False)
      .Cases("defcalgrammar", "def", "cal", "defcal",
             TokenKind::UnsupportedKeyword)
      .Cases("extern", "box", "let", "break", "continue",
             TokenKind::UnsupportedKeyword)
      .Cases("end", "return", "switch", "case", "default",
             TokenKind::UnsupportedKeyword)
      .Cases("pragma", "input", "readonly", "mutable",
             TokenKind::UnsupportedKeyword)
      .Cases("complex", "array", "void", "stretch",
             TokenKind::UnsupportedKeyword)
      .Cases("durationof", "delay", "im", TokenKind::UnsupportedKeyword)
      .Default(TokenKind::Identifier);
}

StringRef describe(const TokenKind kind) {
  switch (kind) {
  case TokenKind::Eof:
    return "end of input";
  case TokenKind::Error:
    return "invalid token";
  case TokenKind::UnterminatedComment:
    return "unterminated block comment";
  case TokenKind::UnsupportedKeyword:
    return "unsupported reserved keyword";
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

const char* Lexer::skipTrivia() {
  while (!atEnd()) {
    const char c = *cur;
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      ++cur;
      continue;
    }
    if (c == '/' && peek() == '/') {
      // Line comment
      cur += 2;
      while (!atEnd() && *cur != '\n') {
        ++cur;
      }
      continue;
    }
    if (c == '/' && peek() == '*') {
      // Block comment
      const char* start = cur;
      cur += 2;
      while (!atEnd() && (*cur != '*' || peek() != '/')) {
        ++cur;
      }
      if (atEnd()) {
        // Report unterminated block comment
        return start;
      }
      // Consume the closing */
      cur += 2;
      continue;
    }
    break;
  }
  return nullptr;
}

Token Lexer::lexIdentifierOrKeyword(const char* start) {
  while (!atEnd()) {
    if (canContinueIdentifier(*cur)) {
      ++cur;
      continue;
    }
    if (static_cast<unsigned char>(*cur) < 0x80) {
      break;
    }
    const auto codePoint = decodeCodePoint(cur, end);
    if (!codePoint || !isOpenQASMIdentifierCodePoint(codePoint->value)) {
      break;
    }
    cur += codePoint->width;
  }
  const StringRef text(start, static_cast<size_t>(cur - start));
  Token token;
  token.loc = SMLoc::getFromPointer(start);
  token.spelling = text;
  token.kind = keywordKind(text);
  if (token.kind == TokenKind::Identifier) {
    token.identifier = text;
  }
  return token;
}

Token Lexer::lexNumber(const char* start) {
  const auto isSeparator = [](const char value) { return value == '_'; };
  if (*start == '0' && (peek() == 'b' || peek() == 'B' || peek() == 'o' ||
                        peek() == 'O' || peek() == 'x' || peek() == 'X')) {
    const char prefix = peek();
    cur += 2;
    const unsigned radix = prefix == 'b' || prefix == 'B'   ? 2
                           : prefix == 'o' || prefix == 'O' ? 8
                                                            : 16;
    const char* digits = cur;
    while (!atEnd() && (std::isalnum(static_cast<unsigned char>(*cur)) ||
                        isSeparator(*cur))) {
      ++cur;
    }
    const StringRef text(start, static_cast<size_t>(cur - start));
    const StringRef digitText(digits, static_cast<size_t>(cur - digits));
    Token token{.kind = TokenKind::IntegerLiteral,
                .loc = SMLoc::getFromPointer(start),
                .spelling = text};
    llvm::SmallString<32> normalized;
    for (const char value : digitText) {
      if (!isSeparator(value)) {
        normalized.push_back(value);
      }
    }
    if (digitText.empty() ||
        !hasValidSeparators(digitText,
                            [radix](const char value) {
                              return isDigitForRadix(value, radix);
                            }) ||
        StringRef(normalized).getAsInteger(radix, token.intValue)) {
      token.kind = TokenKind::Error;
    }
    return token;
  }

  bool isFloat = false;
  while (!atEnd() && (isDigit(*cur) || isSeparator(*cur))) {
    ++cur;
  }
  if (!atEnd() && *cur == '.') {
    isFloat = true;
    ++cur;
    while (!atEnd() && (isDigit(*cur) || isSeparator(*cur))) {
      ++cur;
    }
  }
  if (!atEnd() && (*cur == 'e' || *cur == 'E')) {
    isFloat = true;
    ++cur;
    if (!atEnd() && (*cur == '+' || *cur == '-')) {
      ++cur;
    }
    while (!atEnd() && (isDigit(*cur) || isSeparator(*cur))) {
      ++cur;
    }
  }
  const StringRef text(start, static_cast<size_t>(cur - start));
  Token token;
  token.loc = SMLoc::getFromPointer(start);
  token.spelling = text;
  llvm::SmallString<32> normalized;
  for (const char value : text) {
    if (!isSeparator(value)) {
      normalized.push_back(value);
    }
  }
  const bool invalidSeparators = !hasValidSeparators(
      text, [](const char value) { return isDigit(value); });
  if (isFloat) {
    token.kind = TokenKind::FloatLiteral;
    if (invalidSeparators ||
        StringRef(normalized).getAsDouble(token.floatValue) ||
        !std::isfinite(token.floatValue)) {
      token.kind = TokenKind::Error;
    }
  } else {
    token.kind = TokenKind::IntegerLiteral;
    if (invalidSeparators ||
        StringRef(normalized).getAsInteger(10, token.intValue)) {
      token.kind = TokenKind::Error;
    }
  }
  return token;
}

Token Lexer::lexString(const char* start, const char quote) {
  ++cur; // consume opening quote
  const char* contentStart = cur;
  while (!atEnd() && *cur != quote) {
    ++cur;
  }
  Token token;
  token.loc = SMLoc::getFromPointer(start);
  if (atEnd()) {
    token.kind = TokenKind::Error;
    return token;
  }
  token.kind = TokenKind::StringLiteral;
  token.stringValue =
      StringRef(contentStart, static_cast<size_t>(cur - contentStart));
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
  token.loc = SMLoc::getFromPointer(start);
  const StringRef digits(digitsStart, static_cast<size_t>(cur - digitsStart));
  if (digits.empty() || digits.getAsInteger(10, token.intValue)) {
    token.kind = TokenKind::Error;
    return token;
  }
  token.kind = TokenKind::HardwareQubit;
  return token;
}

Token Lexer::next() {
  const char* unterminatedComment = skipTrivia();

  Token token;
  if (unterminatedComment != nullptr) {
    token.loc = SMLoc::getFromPointer(unterminatedComment);
    token.kind = TokenKind::UnterminatedComment;
    return token;
  }

  token.loc = SMLoc::getFromPointer(cur);
  if (atEnd()) {
    token.kind = TokenKind::Eof;
    return token;
  }

  const char* start = cur;
  const char c = *cur;

  if (canStartIdentifier(c)) {
    return lexIdentifierOrKeyword(start);
  }
  const auto remaining = static_cast<std::size_t>(end - cur);
  const std::size_t unsupportedHashKeywordWidth =
      c == '#' && remaining >= 4 && StringRef(cur, 4) == "#dim" &&
              (remaining == 4 || !canContinueIdentifier(cur[4]))
          ? 4
      : c == '#' && remaining >= 7 && StringRef(cur, 7) == "#pragma" &&
              (remaining == 7 || !canContinueIdentifier(cur[7]))
          ? 7
          : 0;
  if (unsupportedHashKeywordWidth != 0) {
    cur += unsupportedHashKeywordWidth;
    token.kind = TokenKind::UnsupportedKeyword;
    token.spelling = StringRef(start, unsupportedHashKeywordWidth);
    return token;
  }
  if (static_cast<unsigned char>(c) >= 0x80) {
    const auto codePoint = decodeCodePoint(cur, end);
    if (codePoint && isOpenQASMIdentifierCodePoint(codePoint->value)) {
      return lexIdentifierOrKeyword(start);
    }
    cur += codePoint ? codePoint->width : 1;
    token.kind = TokenKind::Error;
    token.spelling = StringRef(start, static_cast<std::size_t>(cur - start));
    return token;
  }
  // A float literal may lead with a dot
  if (isDigit(c) || (c == '.' && isDigit(peek()))) {
    return lexNumber(start);
  }
  if (c == '"' || c == '\'') {
    return lexString(start, c);
  }
  if (c == '$') {
    return lexHardwareQubit(start);
  }

  // Punctuation and operators
  const auto single = [&](const TokenKind kind) {
    ++cur;
    token.kind = kind;
    token.spelling = StringRef(start, 1);
    return token;
  };
  const auto twoChar = [&](const TokenKind kind) {
    cur += 2;
    token.kind = kind;
    token.spelling = StringRef(start, 2);
    return token;
  };
  const auto threeChar = [&](const TokenKind kind) {
    cur += 3;
    token.kind = kind;
    token.spelling = StringRef(start, 3);
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
    return peek() == '=' ? twoChar(TokenKind::CompoundAssign)
                         : single(TokenKind::Tilde);
  case '=':
    return peek() == '=' ? twoChar(TokenKind::EqualsEquals)
                         : single(TokenKind::Equals);
  case '!':
    return peek() == '=' ? twoChar(TokenKind::NotEquals)
                         : single(TokenKind::ExclamationPoint);
  case '<':
    if (peek() == '<') {
      return cur + 2 < end && cur[2] == '='
                 ? threeChar(TokenKind::CompoundAssign)
                 : twoChar(TokenKind::ShiftLeft);
    }
    return peek() == '=' ? twoChar(TokenKind::LessEquals)
                         : single(TokenKind::Less);
  case '>':
    if (peek() == '>') {
      return cur + 2 < end && cur[2] == '='
                 ? threeChar(TokenKind::CompoundAssign)
                 : twoChar(TokenKind::ShiftRight);
    }
    return peek() == '=' ? twoChar(TokenKind::GreaterEquals)
                         : single(TokenKind::Greater);
  case '+':
    return peek() == '=' ? twoChar(TokenKind::CompoundAssign)
                         : single(TokenKind::Plus);
  case '-':
    if (peek() == '>') {
      return twoChar(TokenKind::Arrow);
    }
    return peek() == '=' ? twoChar(TokenKind::CompoundAssign)
                         : single(TokenKind::Minus);
  case '*':
    if (peek() == '*') {
      return cur + 2 < end && cur[2] == '='
                 ? threeChar(TokenKind::CompoundAssign)
                 : twoChar(TokenKind::DoubleAsterisk);
    }
    return peek() == '=' ? twoChar(TokenKind::CompoundAssign)
                         : single(TokenKind::Asterisk);
  case '/':
    return peek() == '=' ? twoChar(TokenKind::CompoundAssign)
                         : single(TokenKind::Slash);
  case '&':
    if (peek() == '&') {
      return twoChar(TokenKind::AmpAmp);
    }
    if (peek() == '=') {
      return twoChar(TokenKind::CompoundAssign);
    }
    return single(TokenKind::Amp);
  case '|':
    if (peek() == '|') {
      return twoChar(TokenKind::PipePipe);
    }
    if (peek() == '=') {
      return twoChar(TokenKind::CompoundAssign);
    }
    return single(TokenKind::Pipe);
  case '%':
    return peek() == '=' ? twoChar(TokenKind::CompoundAssign)
                         : single(TokenKind::Percent);
  case '^':
    if (peek() == '=') {
      return twoChar(TokenKind::CompoundAssign);
    }
    return single(TokenKind::Caret);
  default:
    break;
  }

  return single(TokenKind::Error);
}

} // namespace mlir::oq3::frontend::detail
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
