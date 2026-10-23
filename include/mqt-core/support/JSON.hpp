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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <string>

namespace mqt::json {

/// Parse strict UTF-8 JSON, rejecting duplicate keys and non-finite numbers.
///
/// Numeric conversion is independent of the process locale. Nesting is limited
/// to 256 containers to bound LLVM value destruction and serialization.
/// Returned strings own their storage independently of the input.
/// Allocation failures follow LLVM policy and are not guaranteed recoverable.
[[nodiscard]] llvm::Expected<llvm::json::Value>
parse(llvm::StringRef text, llvm::StringRef source = "<input>");

/// Read and parse a JSON file, retaining its name in error diagnostics.
[[nodiscard]] llvm::Expected<llvm::json::Value> readFile(llvm::StringRef path);

/// Emit a double using locale-independent, round-tripping decimal formatting.
/// Non-finite values are represented as null.
void number(llvm::json::OStream& stream, double value);

/// Emit a JSON value with sorted object keys and locale-independent numbers.
void value(llvm::json::OStream& stream, const llvm::json::Value& input);

/// Serialize JSON with sorted object keys; zero indentation produces compact
/// JSON.
[[nodiscard]] std::string serialize(const llvm::json::Value& input,
                                    unsigned indentation = 0);

} // namespace mqt::json
