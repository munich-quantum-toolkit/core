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
#include "Readout.h"
#include "Result.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace mqt::bindings::qiskit {
namespace {

struct ResultDecodingError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Results {
  llvm::SmallVector<std::pair<std::string, size_t>> counts;
  llvm::SmallVector<size_t> memory;
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
    auto logical = decoded(runtime::decodeOrderedShots(
        unwrap(job.getShots(programIndex)), width, shots, layout));
    result = {std::move(logical.outcomes), std::move(logical.order)};
    for (auto& [bits, count] : result.counts) {
      bits = hexadecimal(bits);
    }
  } else if (width != 0) {
    const auto counts = decoded(mlir::decodeCounts(
        unwrap(job.getCounts(programIndex)), width, shots, layout));
    result.counts.reserve(counts.size());
    for (const auto& [bits, count] : counts) {
      result.counts.emplace_back(hexadecimal(bits), count);
    }
  }
  return result;
}

void registerResultBindings(nb::module_& m) {
  using namespace nb::literals;
  auto qiskit = m.def_submodule("_qiskit", "Native QDMI readout for Qiskit.");
  const nb::exception<ResultDecodingError> error(qiskit, "ResultDecodingError",
                                                 PyExc_ValueError);
  error.attr("__doc__") =
      "The provider result violates its declared shot count or output layout.";
  qiskit.def(
      "readout",
      [](const qdmi::Job& job, size_t width, size_t shots, size_t programIndex,
         const std::optional<mlir::ResultLayout>& layout, bool memory) {
        auto values = [&] {
          const nb::gil_scoped_release release;
          return readout(job, width, shots, programIndex, layout, memory);
        }();
        llvm::SmallVector<nb::str> keys;
        keys.reserve(values.counts.size());
        nb::typed<nb::dict, nb::str, size_t> counts;
        for (const auto& [bits, count] : values.counts) {
          keys.emplace_back(bits.data(), bits.size());
          counts[keys.back()] = nb::int_(count);
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
      nb::kw_only(), "memory"_a,
      R"doc(Return Qiskit hexadecimal counts and optional ordered memory.

Args:
    job: Completed provider job whose results are read.
    width: Number of logical classical output bits.
    shots: Expected number of shots.
    program_index: Program within a multi-program job.
    layout: Projection from provider output to logical classical bits.
    memory: Read ordered SHOTS and derive counts from those same samples.
        When false, read the provider histogram instead.

Returns:
    A dictionary with ``counts`` and, when requested, ``memory``. Keys use
    lowercase hexadecimal strings without a machine-word width limit.

Raises:
    ResultDecodingError: Results violate the declared shot count or layout.
    RuntimeError: Provider result retrieval fails. Ordered memory requires
        SHOTS support and never falls back to a histogram.
)doc");
}

} // namespace mqt::bindings::qiskit
