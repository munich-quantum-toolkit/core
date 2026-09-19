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

#include "nlohmann/detail/input/json_sax.hpp"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace mqt::detail {

/// SAX reports the parser diagnostic without throwing or rebuilding a DOM.
struct JSONDiagnostic : nlohmann::json_sax<nlohmann::json> {
  using Json = nlohmann::json;
  std::string message;
  bool null() override { return true; }
  bool boolean(bool /*value*/) override { return true; }
  bool number_integer(Json::number_integer_t /*value*/) override {
    return true;
  }
  bool number_unsigned(Json::number_unsigned_t /*value*/) override {
    return true;
  }
  bool number_float(Json::number_float_t /*value*/,
                    const std::string& /*text*/) override {
    return true;
  }
  bool string(std::string& /*value*/) override { return true; }
  bool binary(Json::binary_t& /*value*/) override { return true; }
  bool start_object(size_t /*size*/) override { return true; }
  bool key(std::string& /*value*/) override { return true; }
  bool end_object() override { return true; }
  bool start_array(size_t /*size*/) override { return true; }
  bool end_array() override { return true; }
  bool parse_error(size_t /*position*/, const std::string& /*token*/,
                   const Json::exception& error) override {
    message = error.what();
    return false;
  }
};

/// Valid inputs use the DOM parser once; only failures need a diagnostic pass.
[[nodiscard]] inline std::variant<nlohmann::json, std::string>
parseJSON(std::string_view text,
          nlohmann::json::parser_callback_t callback = nullptr) {
  auto value = nlohmann::json::parse(text.begin(), text.end(),
                                     std::move(callback), false);
  if (!value.is_discarded()) {
    return value;
  }
  JSONDiagnostic diagnostic;
  nlohmann::json::sax_parse(text.begin(), text.end(), &diagnostic);
  return std::move(diagnostic.message);
}

} // namespace mqt::detail
