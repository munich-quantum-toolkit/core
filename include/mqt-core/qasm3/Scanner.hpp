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

#include "Token.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>

namespace qasm3 {
class Scanner {
  std::istream* is;
  std::unordered_map<std::string, Token::Kind> keywords;
  char ch = 0;
  size_t line = 1;
  size_t col = 0;

  [[nodiscard]] static bool isSpace(char c);

  [[nodiscard]] static bool isFirstIdChar(char c);

  [[nodiscard]] static bool isNum(char c);

  [[nodiscard]] static bool isHex(char c);

  [[nodiscard]] static bool hasTimingSuffix(char first, char second);

  static char readUtf8Codepoint(std::istream* in);

  void nextCh();

  [[nodiscard]] char peek() const;

  std::optional<Token> consumeWhitespaceAndComments();

  static bool isValidDigit(uint8_t base, char c);

  std::string consumeNumberLiteral(uint8_t base);

  static uint64_t parseIntegerLiteral(const std::string& str, uint8_t base);

  Token consumeNumberLiteral();

  Token consumeHardwareQubit();

  Token consumeString();

  Token consumeName();

  void error(const std::string& msg) const;

  void expect(char expected);

public:
  explicit Scanner(std::istream* in);

  ~Scanner() = default;

  Token next();
};
} // namespace qasm3
