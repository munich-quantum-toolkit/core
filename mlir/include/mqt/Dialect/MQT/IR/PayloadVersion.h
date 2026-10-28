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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"

#include <optional>
#include <string>

namespace mlir::mqt {

/// Numeric receiver-version interval with an inclusive lower bound.
///
/// Constraints support >=, <, and <=, joined by commas. Bare versions retain
/// the same-major ceiling convention. This is not a package-version resolver.
struct PayloadVersionRange {
  llvm::VersionTuple lower;
  std::optional<llvm::VersionTuple> upper;
  bool upperInclusive = false;

  /// Parse a nonempty constraint; reject malformed or empty intervals.
  [[nodiscard]] static std::optional<PayloadVersionRange>
  parse(llvm::StringRef text);

  /// Return a canonical constraint string.
  [[nodiscard]] std::string str() const;

  /// Whether the receiver accepts this concrete version.
  [[nodiscard]] bool contains(llvm::VersionTuple version) const;

  /// Whether all accepted versions belong to the other interval.
  [[nodiscard]] bool isSubsetOf(const PayloadVersionRange& other) const;

  /// Whether every accepted version belongs to this major family.
  [[nodiscard]] bool withinMajor(unsigned major) const;

  /// Select the lowest known version sufficient for the program and receiver.
  [[nodiscard]] std::optional<llvm::VersionTuple>
  select(llvm::ArrayRef<llvm::VersionTuple> known,
         llvm::VersionTuple required) const;

  friend bool operator==(const PayloadVersionRange&,
                         const PayloadVersionRange&) = default;
};

} // namespace mlir::mqt
