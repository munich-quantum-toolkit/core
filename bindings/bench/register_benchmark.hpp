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

#include <string_view>

namespace mqt {

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
            auto result = fromInstanceSpecification(json, source);
            if (!result) {
              throw nb::value_error(llvm::toString(result.takeError()).c_str());
            }
            return std::move(*result);
          },
          "json"_a, nb::kw_only(), "source"_a = "<instance-specification>",
          "Parse a strict benchmark instance specification.")
      .def_static(
          "from_manifest_json",
          [fromManifest](std::string_view json, std::string_view source) {
            auto result = fromManifest(json, source);
            if (!result) {
              throw nb::value_error(llvm::toString(result.takeError()).c_str());
            }
            return std::move(*result);
          },
          "json"_a, nb::kw_only(), "source"_a = "<manifest>",
          "Parse a strict benchmark manifest.");
}

} // namespace mqt
