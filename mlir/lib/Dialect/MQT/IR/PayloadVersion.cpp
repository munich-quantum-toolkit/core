/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/PayloadVersion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"

#include <algorithm>
#include <optional>
#include <string>

namespace mlir::mqt {

static std::optional<llvm::VersionTuple> parseVersion(llvm::StringRef text) {
  llvm::VersionTuple version;
  if (text.empty() || version.tryParse(text) || version.getBuild() ||
      version.getAsString() != text) {
    return std::nullopt;
  }
  return version;
}

std::optional<PayloadVersionRange>
PayloadVersionRange::parse(llvm::StringRef text) {
  text = text.trim();
  if (const auto version = parseVersion(text)) {
    return PayloadVersionRange{
        llvm::VersionTuple(version->getMajor()),
        version,
        true,
    };
  }
  PayloadVersionRange result;
  while (!text.empty()) {
    const auto [part, rest] = text.split(',');
    auto clause = part.trim();
    const bool isLower = clause.consume_front(">=");
    const bool inclusive = !isLower && clause.consume_front("<=");
    if (!isLower && !inclusive && !clause.consume_front("<")) {
      return std::nullopt;
    }
    const auto version = parseVersion(clause.trim());
    if (!version) {
      return std::nullopt;
    }
    if (isLower) {
      result.lower = std::max(result.lower, *version);
    } else if (!result.upper || *version < *result.upper) {
      result.upper = *version;
      result.upperInclusive = inclusive;
    } else if (*version == *result.upper) {
      result.upperInclusive &= inclusive;
    }
    if (result.upper &&
        (result.lower > *result.upper ||
         (result.lower == *result.upper && !result.upperInclusive))) {
      return std::nullopt;
    }
    if (rest.empty()) {
      return text.ends_with(",") ? std::nullopt : std::optional(result);
    }
    text = rest;
  }
  return std::nullopt;
}

std::string PayloadVersionRange::str() const {
  const auto normalized = [](llvm::VersionTuple v) {
    return llvm::VersionTuple(v.getMajor(), v.getMinor().value_or(0),
                              v.getSubminor().value_or(0))
        .getAsString();
  };
  std::string result = ">=" + normalized(lower);
  if (upper) {
    result += upperInclusive ? ",<=" : ",<";
    result += normalized(*upper);
  }
  return result;
}

bool PayloadVersionRange::contains(llvm::VersionTuple version) const {
  return version >= lower &&
         (!upper || version < *upper || (upperInclusive && version == *upper));
}

bool PayloadVersionRange::isSubsetOf(const PayloadVersionRange& other) const {
  return lower >= other.lower &&
         (!other.upper ||
          (upper && (*upper < *other.upper ||
                     (*upper == *other.upper &&
                      (!upperInclusive || other.upperInclusive)))));
}

bool PayloadVersionRange::withinMajor(unsigned major) const {
  return isSubsetOf({llvm::VersionTuple(major), llvm::VersionTuple(major + 1)});
}

std::optional<llvm::VersionTuple>
PayloadVersionRange::select(llvm::ArrayRef<llvm::VersionTuple> known,
                            llvm::VersionTuple required) const {
  std::optional<llvm::VersionTuple> selected;
  for (const auto candidate : known) {
    if (candidate >= required && contains(candidate) &&
        (!selected || candidate < *selected)) {
      selected = candidate;
    }
  }
  return selected;
}

} // namespace mlir::mqt
