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
#include "qdmi/QDMI.hpp"

#include "Qiskit.h"
#include "Result.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mqt::bindings::qiskit {
namespace {

struct ResultDecodingError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Results {
  std::vector<std::string> keys;
  std::vector<size_t> counts;
  std::vector<size_t> memory;
};

} // namespace

// Qiskit memory supports arbitrary widths, including an empty classical output.
static std::string hexadecimal(llvm::StringRef bits) {
  std::string result = "0x";
  const auto first = bits.find('1');
  if (first == llvm::StringRef::npos) {
    return "0x0";
  }
  bits = bits.drop_front(first);
  size_t digits = (bits.size() - 1) % 4 + 1;
  while (!bits.empty()) {
    unsigned value = 0;
    for (const char bit : bits.take_front(digits)) {
      value = (value << 1U) | static_cast<unsigned>(bit - '0');
    }
    result.push_back("0123456789abcdef"[value]);
    bits = bits.drop_front(digits);
    digits = 4;
  }
  return result;
}

template <typename T> static T decoded(llvm::Expected<T> result) {
  if (!result) {
    throw ResultDecodingError(llvm::toString(result.takeError()));
  }
  return std::move(*result);
}

static Results readout(const qdmi::Job& job, size_t width, size_t shots,
                       size_t programIndex,
                       const std::optional<mlir::ResultLayout>& layout,
                       bool memory) {
  Results result;
  if (memory) {
    const auto values = decoded(mlir::decodeShots(
        unwrap(job.getShots(programIndex)), width, shots, layout));
    llvm::DenseMap<llvm::StringRef, size_t> indices;
    result.memory.reserve(values.size());
    for (const auto& bits : values) {
      const auto [entry, inserted] =
          indices.try_emplace(bits, result.keys.size());
      const auto index = entry->second;
      if (inserted) {
        result.keys.push_back(hexadecimal(bits));
        result.counts.push_back(0);
      }
      ++result.counts[index];
      result.memory.push_back(index);
    }
  } else if (width != 0) {
    const auto counts = decoded(mlir::decodeCounts(
        unwrap(job.getCounts(programIndex)), width, shots, layout));
    result.keys.reserve(counts.size());
    result.counts.reserve(counts.size());
    for (const auto& [bits, count] : counts) {
      result.keys.push_back(hexadecimal(bits));
      result.counts.push_back(count);
    }
  }
  return result;
}

void registerResultBindings(nb::module_& m) {
  using namespace nb::literals;
  auto qiskit = m.def_submodule("_qiskit", "Native QDMI readout for Qiskit.");
  const nb::exception<ResultDecodingError> error(qiskit, "ResultDecodingError",
                                                 PyExc_ValueError);
  qiskit.def(
      "readout",
      [](const qdmi::Job& job, size_t width, size_t shots, size_t programIndex,
         const std::optional<mlir::ResultLayout>& layout, bool memory) {
        auto values = [&] {
          const nb::gil_scoped_release release;
          return readout(job, width, shots, programIndex, layout, memory);
        }();
        std::vector<nb::str> keys;
        keys.reserve(values.keys.size());
        nb::typed<nb::dict, nb::str, size_t> counts;
        for (size_t i = 0; i < values.keys.size(); ++i) {
          keys.emplace_back(values.keys[i].data(), values.keys[i].size());
          counts[keys.back()] = nb::int_(values.counts[i]);
        }
        nb::typed<nb::dict, nb::str, nb::object> result;
        result["counts"] = counts;
        if (memory) {
          nb::typed<nb::list, nb::str> samples;
          for (const auto index : values.memory) {
            samples.append(keys[index]);
          }
          result["memory"] = samples;
        }
        return result;
      },
      "job"_a, "width"_a, "shots"_a, "program_index"_a, "layout"_a,
      nb::kw_only(), "memory"_a);
}

} // namespace mqt::bindings::qiskit
