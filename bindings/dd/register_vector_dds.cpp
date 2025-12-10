/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Export.hpp"
#include "dd/Node.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>  // NOLINT(misc-include-cleaner)
#include <ranges>
#include <sstream>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using Vector = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<1>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
Vector getVector(const dd::vEdge& v, const dd::fp threshold = 0.) {
  auto vec = v.getVector(threshold);
  auto dataPtr = std::make_unique<std::complex<dd::fp>[]>(vec.size());
  std::ranges::copy(vec, dataPtr.get());
  auto* data = dataPtr.release();
  const nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<dd::fp>*>(ptr);
  });
  return Vector(data, {vec.size()}, owner);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVectorDDs(const nb::module_& m) {
  auto vec = nb::class_<dd::vEdge>(m, "VectorDD");

  vec.def("is_terminal", &dd::vEdge::isTerminal);
  vec.def("is_zero_terminal", &dd::vEdge::isZeroTerminal);

  vec.def("size", nb::overload_cast<>(&dd::vEdge::size, nb::const_));

  vec.def(
      "__getitem__",
      [](const dd::vEdge& v, const size_t idx) {
        return v.getValueByIndex(idx);
      },
      "index"_a);

  vec.def(
      "get_amplitude",
      [](const dd::vEdge& v, const size_t numQubits,
         const std::string& decisions) {
        return v.getValueByPath(numQubits, decisions);
      },
      "num_qubits"_a, "decisions"_a);

  vec.def("get_vector", &getVector, "threshold"_a = 0.);

  vec.def(
      "to_dot",
      [](const dd::vEdge& e, const bool colored = true,
         const bool edgeLabels = false, const bool classic = false,
         const bool memory = false, const bool formatAsPolar = true) {
        std::ostringstream os;
        dd::toDot(e, os, colored, edgeLabels, classic, memory, formatAsPolar);
        return os.str();
      },
      "colored"_a = true, "edge_labels"_a = false, "classic"_a = false,
      "memory"_a = false, "format_as_polar"_a = true);

  vec.def(
      "to_svg",
      [](const dd::vEdge& e, const std::string& filename,
         const bool colored = true, const bool edgeLabels = false,
         const bool classic = false, const bool memory = false,
         const bool formatAsPolar = true) {
        // replace the filename extension with .dot
        const auto dotFilename =
            filename.substr(0, filename.find_last_of('.')) + ".dot";
        dd::export2Dot(e, dotFilename, colored, edgeLabels, classic, memory,
                       true, formatAsPolar);
      },
      "filename"_a, "colored"_a = true, "edge_labels"_a = false,
      "classic"_a = false, "memory"_a = false, "format_as_polar"_a = true);
}

} // namespace mqt
