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

#include "bench/JSON.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include "llvm/Support/Error.h"

#include <memory>
#include <string_view>
#include <system_error>
#include <utility>

namespace mqt {

template <class T> T unwrapBenchmark(llvm::Expected<T> result) {
  if (!result) {
    auto error = result.takeError();
    bool overflow = false;
    llvm::visitErrors(error, [&](const llvm::ErrorInfoBase& info) {
      overflow = info.convertToErrorCode() == std::errc::value_too_large;
    });
    // Consume inside LLVM, whose RTTI setting may differ from nanobind's.
    const auto message = llvm::toString(std::move(error));
    if (overflow) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): nanobind API.
      nanobind::chain_error(
          nanobind::module_::import_("builtins").attr("OverflowError"), "%s",
          message.c_str());
      throw nanobind::python_error();
    }
    throw nanobind::value_error(message.c_str());
  }
  return std::move(*result);
}

template <class Benchmark, class... Args>
auto benchmarkInit(llvm::Expected<Benchmark> (*create)(Args...)) {
  return [create](Benchmark* self, Args... args) {
    std::construct_at(self, unwrapBenchmark(create(std::move(args)...)));
  };
}

template <class Benchmark, class T, class... Args>
auto benchmarkMethod(llvm::Expected<T> (Benchmark::*method)(Args...) const) {
  return [method](const Benchmark& self, Args... args) {
    return unwrapBenchmark((self.*method)(args...));
  };
}

template <class Benchmark, class T>
auto benchmarkOutcome(T (Benchmark::*method)(std::string_view) const) {
  return [method](const Benchmark& self, const std::string_view outcome) {
    if (outcome.size() != self.output().width) {
      throw nanobind::value_error(
          "outcome width does not match the benchmark output");
    }
    if (outcome.find_first_not_of("01") != std::string_view::npos) {
      throw nanobind::value_error("outcome must contain only '0' and '1'");
    }
    return (self.*method)(outcome);
  };
}

template <class Benchmark>
void registerBenchmarkIO(
    nanobind::class_<Benchmark>& benchmark,
    llvm::Expected<Benchmark> (*fromInstanceSpecification)(std::string_view,
                                                           std::string_view),
    llvm::Expected<Benchmark> (*fromManifest)(std::string_view,
                                              std::string_view)) {
  namespace nb = nanobind;
  using namespace nb::literals;

  benchmark
      .def(
          "generate",
          [](const Benchmark& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(nb::find(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro("instance_specification_json",
                   nb::overload_cast<const Benchmark&>(
                       &bench::toInstanceSpecificationJSON),
                   "The canonical instance specification JSON.")
      .def_prop_ro("manifest_json",
                   nb::overload_cast<const Benchmark&>(&bench::toManifestJSON),
                   "The canonical manifest JSON.")
      .def_prop_ro("case_id",
                   nb::overload_cast<const Benchmark&>(&bench::caseId),
                   "The stable semantic case ID.")
      .def_static(
          "from_instance_specification_json",
          [fromInstanceSpecification](std::string_view json,
                                      std::string_view source) {
            return unwrapBenchmark(fromInstanceSpecification(json, source));
          },
          "json"_a, nb::kw_only(), "source"_a = "<instance-specification>",
          "Parse a strict benchmark instance specification.")
      .def_static(
          "from_manifest_json",
          [fromManifest](std::string_view json, std::string_view source) {
            return unwrapBenchmark(fromManifest(json, source));
          },
          "json"_a, nb::kw_only(), "source"_a = "<manifest>",
          "Parse a strict benchmark manifest.");
}

} // namespace mqt
