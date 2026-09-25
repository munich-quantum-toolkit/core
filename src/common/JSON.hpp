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

#include "support/Diagnostics.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace mqt::detail {
/// JSON dependency exceptions never cross into exception-free callers.
[[nodiscard]] inline mlir::FailureOr<nlohmann::json>
parseJSON(std::string_view text, std::string_view source,
          nlohmann::json::parser_callback_t callback = nullptr,
          std::optional<int> status = std::nullopt) {
  try {
    return nlohmann::json::parse(text.begin(), text.end(), std::move(callback));
  } catch (const nlohmann::json::exception& error) {
    return emitError(std::string(source) + ": invalid JSON: " + error.what(),
                     ErrorCategory::InvalidArgument, status);
  }
}
} // namespace mqt::detail
