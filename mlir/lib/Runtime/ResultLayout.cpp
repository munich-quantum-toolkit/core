/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/ResultLayout.h"

#include "Readout.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {

bool ResultLayout::isIdentity() const noexcept {
  return bits.size() == rawWidth &&
         llvm::all_of(llvm::enumerate(bits), [](auto entry) {
           return entry.value() == entry.index();
         });
}

llvm::Expected<std::string>
ResultLayout::decodeBits(llvm::StringRef raw) const {
  if (raw.size() != rawWidth ||
      raw.find_first_not_of("01") != llvm::StringRef::npos) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "raw sample has an invalid width or digit");
  }
  std::string result(bits.size(), '0');
  for (auto [index, slot] : llvm::enumerate(bits)) {
    if (slot) {
      if (*slot >= rawWidth) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "result layout selects an invalid raw bit");
      }
      result[bits.size() - index - 1] = raw[rawWidth - *slot - 1];
    }
  }
  return result;
}

static llvm::Expected<std::string>
decodeSample(std::string raw, std::optional<size_t>& width,
             const std::optional<ResultLayout>& layout) {
  std::erase(raw, ' ');
  if (layout) {
    auto decoded = layout->decodeBits(raw);
    if (!decoded) {
      return decoded.takeError();
    }
    raw = std::move(*decoded);
  }
  if (!width) {
    width = raw.size();
  }
  if (raw.size() != *width ||
      raw.find_first_not_of("01") != std::string::npos) {
    return llvm::createStringError("Invalid QDMI sample width or binary digit");
  }
  return raw;
}

llvm::Expected<std::map<std::string, size_t>>
decodeCounts(std::map<std::string, size_t> counts, std::optional<size_t> width,
             size_t shots, const std::optional<ResultLayout>& layout) {
  std::map<std::string, size_t> result;
  size_t total = 0;
  while (!counts.empty()) {
    auto entry = counts.extract(counts.begin());
    auto decoded = decodeSample(std::move(entry.key()), width, layout);
    if (!decoded) {
      return decoded.takeError();
    }
    const auto count = entry.mapped();
    if (count > shots - total) {
      return llvm::createStringError("Invalid QDMI histogram: too many shots");
    }
    total += count;
    if (count != 0) {
      entry.key() = std::move(*decoded);
      auto inserted = result.insert(result.end(), std::move(entry));
      // NOLINTNEXTLINE(bugprone-use-after-move): Duplicates retain the node.
      if (!entry.empty()) {
        inserted->second += count;
      }
    }
  }
  if (total != shots) {
    return llvm::createStringError(
        "Invalid QDMI histogram: incorrect shot total");
  }
  return result;
}

llvm::Expected<std::vector<std::string>>
decodeShots(std::vector<std::string> samples, std::optional<size_t> width,
            size_t shots, const std::optional<ResultLayout>& layout) {
  if (samples.size() != shots) {
    return llvm::createStringError(
        "Invalid QDMI SHOTS result: incorrect shot total");
  }
  std::map<std::string, std::string> cache;
  for (auto& bits : samples) {
    auto found = cache.find(bits);
    if (found == cache.end()) {
      auto decoded = decodeSample(bits, width, layout);
      if (!decoded) {
        return decoded.takeError();
      }
      found = cache.emplace(bits, std::move(*decoded)).first;
    }
    bits = found->second;
  }
  return samples;
}

} // namespace mlir

namespace mqt::runtime {

llvm::Expected<Readout>
decodeOrderedShots(const std::vector<std::string>& samples,
                   std::optional<size_t> width, size_t shots,
                   const std::optional<mlir::ResultLayout>& layout) {
  if (samples.size() != shots) {
    return llvm::createStringError(
        "Invalid QDMI SHOTS result: incorrect shot total");
  }
  Readout result;
  result.order.reserve(shots);
  llvm::DenseMap<llvm::StringRef, size_t> rawIndices;
  llvm::StringMap<size_t> logicalIndices;
  for (const auto& bits : samples) {
    auto [raw, inserted] = rawIndices.try_emplace(bits, 0);
    if (inserted) {
      auto decoded = mlir::decodeSample(bits, width, layout);
      if (!decoded) {
        return decoded.takeError();
      }
      const auto [logical, unique] =
          logicalIndices.try_emplace(*decoded, result.outcomes.size());
      raw->second = logical->second;
      if (unique) {
        result.outcomes.emplace_back(std::move(*decoded), 0);
      }
    }
    ++result.outcomes[raw->second].second;
    result.order.push_back(raw->second);
  }
  return result;
}

} // namespace mqt::runtime
