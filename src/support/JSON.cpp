/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/JSON.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace mqt::json {
namespace {

class Parser {
  llvm::StringRef text_;
  llvm::StringRef source_;
  size_t position_ = 0;

  [[nodiscard]] llvm::Error fail(const llvm::Twine& message) const {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        source_ + ": invalid JSON at byte " + llvm::Twine(position_) + ": " +
            message);
  }

  [[nodiscard]] char peek() const {
    return position_ < text_.size() ? text_[position_] : '\0';
  }

  bool consume(const char c) {
    if (position_ == text_.size() || text_[position_] != c) {
      return false;
    }
    ++position_;
    return true;
  }

  void whitespace() {
    while (peek() == ' ' || peek() == '\n' || peek() == '\r' ||
           peek() == '\t') {
      ++position_;
    }
  }

  static bool digit(const char c) { return c >= '0' && c <= '9'; }

  llvm::Expected<uint16_t> hexCodeUnit() {
    if (text_.size() - position_ < 4) {
      return fail("incomplete Unicode escape");
    }
    uint16_t code = 0;
    const auto token = text_.substr(position_, 4);
    const auto parsed = std::from_chars(token.begin(), token.end(), code, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != token.end()) {
      return fail("invalid Unicode escape");
    }
    position_ += 4;
    return code;
  }

  llvm::Expected<llvm::json::Value> string() {
    const auto start = position_;
    ++position_;
    while (position_ < text_.size()) {
      const auto c = text_[position_++];
      if (c == '"') {
        auto decoded = llvm::json::parse(text_.slice(start, position_));
        if (!decoded) {
          return fail(llvm::toString(decoded.takeError()));
        }
        return decoded;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        return fail("unescaped control character in string");
      }
      if (c != '\\') {
        continue;
      }
      if (position_ == text_.size()) {
        break;
      }
      const auto escape = text_[position_++];
      if (escape != 'u') {
        if (llvm::StringRef("\"\\/bfnrt").contains(escape)) {
          continue;
        }
        return fail("invalid string escape");
      }
      auto code = hexCodeUnit();
      if (!code) {
        return code.takeError();
      }
      if (*code >= 0xD800 && *code <= 0xDBFF) {
        if (!consume('\\') || !consume('u')) {
          return fail("missing low Unicode surrogate");
        }
        auto low = hexCodeUnit();
        if (!low) {
          return low.takeError();
        }
        if (*low < 0xDC00 || *low > 0xDFFF) {
          return fail("invalid low Unicode surrogate");
        }
      } else if (*code >= 0xDC00 && *code <= 0xDFFF) {
        return fail("unpaired low Unicode surrogate");
      }
    }
    return fail("unterminated string");
  }

  llvm::Expected<llvm::json::Value> numeric() {
    const auto start = position_;
    const auto negative = consume('-');
    if (!consume('0')) {
      if (peek() < '1' || peek() > '9') {
        return fail("expected a decimal digit");
      }
      while (digit(peek())) {
        ++position_;
      }
    }
    bool integral = true;
    if (consume('.')) {
      integral = false;
      if (!digit(peek())) {
        return fail("expected a fractional digit");
      }
      while (digit(peek())) {
        ++position_;
      }
    }
    if (consume('e') || consume('E')) {
      integral = false;
      if (!consume('+')) {
        consume('-');
      }
      if (!digit(peek())) {
        return fail("expected an exponent digit");
      }
      while (digit(peek())) {
        ++position_;
      }
    }
    const auto token = text_.slice(start, position_);
    if (integral) {
      if (negative) {
        int64_t parsed = 0;
        if (std::from_chars(token.begin(), token.end(), parsed).ec ==
            std::errc{}) {
          return llvm::json::Value(parsed);
        }
      } else {
        uint64_t parsed = 0;
        if (std::from_chars(token.begin(), token.end(), parsed).ec ==
            std::errc{}) {
          return llvm::json::Value(parsed);
        }
      }
    }
    double parsed = 0;
    if (token.getAsDouble(parsed)) {
      return fail("number cannot be represented as a finite double");
    }
    if (!std::isfinite(parsed)) {
      return fail("number cannot be represented as a finite double");
    }
    return llvm::json::Value(parsed);
  }

  llvm::Expected<llvm::json::Value> value(const unsigned depth) {
    whitespace();
    if (peek() == '[' || peek() == '{') {
      if (depth == 256) {
        return fail("JSON nesting exceeds 256 containers");
      }
      const auto object = consume('{');
      if (!object) {
        consume('[');
      }
      llvm::json::Value result = object
                                     ? llvm::json::Value(llvm::json::Object{})
                                     : llvm::json::Value(llvm::json::Array{});
      const auto end = object ? '}' : ']';
      whitespace();
      if (consume(end)) {
        return result;
      }
      while (true) {
        std::string key;
        if (object) {
          whitespace();
          if (peek() != '"') {
            return fail("expected an object key");
          }
          auto parsedKey = string();
          if (!parsedKey) {
            return parsedKey.takeError();
          }
          key = parsedKey->getAsString()->str();
          if (result.getAsObject()->get(key) != nullptr) {
            return fail("duplicate key '" + llvm::Twine(key) + "'");
          }
          whitespace();
          if (!consume(':')) {
            return fail("expected a colon after object key");
          }
        }
        auto element = value(depth + 1);
        if (!element) {
          return element.takeError();
        }
        if (object) {
          result.getAsObject()->try_emplace(std::move(key),
                                            std::move(*element));
        } else {
          result.getAsArray()->push_back(std::move(*element));
        }
        whitespace();
        if (consume(end)) {
          return result;
        }
        if (!consume(',')) {
          return fail("expected a comma or container end");
        }
      }
    }
    if (peek() == '"') {
      return string();
    }
    if (peek() == '-' || digit(peek())) {
      return numeric();
    }
    if (text_.substr(position_).starts_with("true")) {
      position_ += 4;
      return llvm::json::Value(true);
    }
    if (text_.substr(position_).starts_with("false")) {
      position_ += 5;
      return llvm::json::Value(false);
    }
    if (text_.substr(position_).starts_with("null")) {
      position_ += 4;
      return llvm::json::Value(nullptr);
    }
    return fail("expected a JSON value");
  }

public:
  Parser(const llvm::StringRef text, const llvm::StringRef source)
      : text_(text), source_(source) {}

  llvm::Expected<llvm::json::Value> run() {
    if (!llvm::json::isUTF8(text_)) {
      return fail("invalid UTF-8");
    }
    if (text_.starts_with("\xEF\xBB\xBF")) {
      position_ = 3;
    }
    auto result = value(0);
    if (!result) {
      return result.takeError();
    }
    whitespace();
    if (position_ != text_.size()) {
      return fail("trailing characters after JSON value");
    }
    return result;
  }
};

} // namespace

llvm::Expected<llvm::json::Value> parse(const llvm::StringRef text,
                                        const llvm::StringRef source) {
  return Parser(text, source).run();
}

llvm::Expected<llvm::json::Value> readFile(const llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return llvm::createStringError(buffer.getError(),
                                   path + ": cannot read JSON file");
  }
  return parse((*buffer)->getBuffer(), path);
}

void number(llvm::json::OStream& stream, const double value) {
  if (!std::isfinite(value)) {
    stream.value(nullptr);
    return;
  }
  std::array<char, 32> buffer{};
  const auto result = std::to_chars(buffer.begin(), buffer.end(), value);
  assert(result.ec == std::errc{} && "a double fits in 32 decimal characters");
  const llvm::StringRef text(buffer.data(),
                             static_cast<size_t>(result.ptr - buffer.data()));
  stream.rawValue([&](llvm::raw_ostream& output) {
    output << text;
    if (!text.contains('.') && !text.contains('e')) {
      output << ".0";
    }
  });
}

void value(llvm::json::OStream& stream, const llvm::json::Value& input) {
  switch (input.kind()) {
  case llvm::json::Value::Number: {
    if (const auto integer = input.getAsUINT64()) {
      stream.value(*integer);
      return;
    }
    const auto floating = *input.getAsNumber();
    // Avoid LLVM's signed conversion at the rounded INT64_MAX boundary.
    if (floating < 0 &&
        floating >= static_cast<double>(std::numeric_limits<int64_t>::min())) {
      if (const auto integer = input.getAsInteger()) {
        stream.value(*integer);
        return;
      }
    }
    number(stream, floating);
    return;
  }
  case llvm::json::Value::Array:
    stream.array([&] {
      for (const auto& element : *input.getAsArray()) {
        value(stream, element);
      }
    });
    return;
  case llvm::json::Value::Object:
    stream.object([&] {
      for (const auto* field :
           llvm::json::sortedElements(*input.getAsObject())) {
        stream.attributeBegin(field->first);
        value(stream, field->second);
        stream.attributeEnd();
      }
    });
    return;
  default:
    stream.value(input);
  }
}

std::string serialize(const llvm::json::Value& input,
                      const unsigned indentation) {
  std::string output;
  llvm::raw_string_ostream raw(output);
  llvm::json::OStream stream(raw, indentation);
  value(stream, input);
  return output;
}

} // namespace mqt::json
