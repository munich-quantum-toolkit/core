/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QDMIReadout.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mqt::runtime {

static void require(bool valid, const llvm::Twine& message) {
  if (!valid) {
    throw std::runtime_error(message.str());
  }
}

template <class T> static T unwrap(llvm::Expected<T> value) {
  if (!value) {
    throw std::runtime_error(llvm::toString(value.takeError()));
  }
  return std::move(*value);
}

static void validateShots(size_t shots) {
  require(shots > 0 && shots <= static_cast<size_t>(INT64_MAX),
          "QDMI requires a positive shot count that fits int64_t.");
}

static size_t product(size_t lhs, size_t rhs) {
  require(rhs == 0 || lhs <= SIZE_MAX / rhs,
          "Result dimensions overflow size_t.");
  return lhs * rhs;
}

static size_t dimension(size_t width) {
  require(width < std::numeric_limits<size_t>::digits,
          "Dense result dimension does not fit size_t.");
  return size_t{1} << width;
}

Readout readout(const qdmi::Job& job, size_t width, size_t shots,
                size_t programIndex,
                const std::optional<mlir::ResultLayout>& layout, bool samples) {
  validateShots(shots);
  const auto load = [&](bool loadSamples) -> llvm::Expected<Readout> {
    Readout result;
    if (loadSamples) {
      auto values = job.getShots(programIndex);
      if (!values) {
        return values.takeError();
      }
      if (values->empty()) {
        return result;
      }
      *values =
          unwrap(mlir::decodeShots(std::move(*values), width, shots, layout));
      result.reserve(values->size());
      for (auto& bits : *values) {
        result.emplace_back(std::move(bits), 1);
      }
    } else {
      auto counts = job.getCounts(programIndex);
      if (!counts) {
        return counts.takeError();
      }
      if (counts->empty()) {
        return result;
      }
      *counts =
          unwrap(mlir::decodeCounts(std::move(*counts), width, shots, layout));
      result.reserve(counts->size());
      while (!counts->empty()) {
        auto entry = counts->extract(counts->begin());
        result.emplace_back(std::move(entry.key()), entry.mapped());
      }
    }
    return result;
  };
  auto loaded = load(samples);
  if (!loaded || loaded->empty()) {
    const auto firstError = loaded ? "" : llvm::toString(loaded.takeError());
    loaded = load(!samples);
    if (!loaded) {
      throw std::runtime_error("Cannot read QDMI results; " +
                               std::string(samples ? "shots: " : "counts: ") +
                               firstError +
                               (samples ? "; counts: " : "; shots: ") +
                               llvm::toString(loaded.takeError()));
    }
  }
  auto result = std::move(*loaded);
  require(!result.empty(), "QDMI returned no samples for a nonempty job.");
  return result;
}

size_t basisIndex(const std::string& bits, llvm::ArrayRef<size_t> wires) {
  size_t index = 0;
  for (const auto wire : wires) {
    index =
        (index << 1U) | static_cast<size_t>(bits[bits.size() - wire - 1] - '0');
  }
  return index;
}

std::vector<int8_t>
decodeSamples(const qdmi::Job& job, size_t width, size_t shots,
              size_t programIndex,
              const std::optional<mlir::ResultLayout>& layout) {
  const auto result = readout(job, width, shots, programIndex, layout, true);
  std::vector<int8_t> values(product(shots, width));
  auto output = values.begin();
  for (const auto& [bits, count] : result) {
    for (size_t i = 0; i < count; ++i) {
      output = std::ranges::transform(
                   bits.rbegin(), bits.rend(), output,
                   [](char bit) { return static_cast<int8_t>(bit - '0'); })
                   .out;
    }
  }
  return values;
}

static void validateWires(const Readout& result, llvm::ArrayRef<size_t> wires) {
  require(!result.empty(), "Cannot project empty readout.");
  for (const auto wire : wires) {
    require(wire < result.front().first.size(),
            "Invalid logical readout wire.");
  }
}

std::vector<double> marginalProbabilities(const Readout& result,
                                          llvm::ArrayRef<size_t> wires,
                                          size_t shots) {
  validateShots(shots);
  validateWires(result, wires);
  std::vector<size_t> frequencies(dimension(wires.size()), 0);
  for (const auto& [bits, count] : result) {
    frequencies[basisIndex(bits, wires)] += count;
  }
  std::vector<double> probabilities;
  probabilities.reserve(frequencies.size());
  for (const auto count : frequencies) {
    probabilities.push_back(static_cast<double>(count) /
                            static_cast<double>(shots));
  }
  return probabilities;
}

std::map<std::string, size_t> marginalCounts(const Readout& result,
                                             llvm::ArrayRef<size_t> wires) {
  validateWires(result, wires);
  std::map<std::string, size_t> counts;
  for (const auto& [bits, count] : result) {
    std::string key;
    key.reserve(wires.size());
    for (const auto wire : wires) {
      key.push_back(bits[bits.size() - wire - 1]);
    }
    counts[std::move(key)] += count;
  }
  return counts;
}

} // namespace mqt::runtime
