/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>  // NOLINT(misc-include-cleaner)
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
  auto vec = nb::class_<dd::vEdge>(
      m, "VectorDD", "A class representing a vector decision diagram (DD).");

  vec.def("is_terminal", &dd::vEdge::isTerminal,
          "Check if the DD is a terminal node.");

  vec.def("is_zero_terminal", &dd::vEdge::isZeroTerminal,
          "Check if the DD is a zero terminal node.");

  vec.def("size", nb::overload_cast<>(&dd::vEdge::size, nb::const_),
          "Get the size of the DD by traversing it once.");

  vec.def(
      "__getitem__",
      [](const dd::vEdge& v, nb::ssize_t idx) {
        const auto n = static_cast<nb::ssize_t>(v.size());
        if (idx < 0) {
          idx += n;
        }
        if (idx < 0 || idx >= n) {
          throw nb::index_error();
        }
        return v.getValueByIndex(static_cast<std::size_t>(idx));
      },
      "key"_a, "Get the amplitude of a basis state by index.");

  vec.def(
      "get_amplitude",
      [](const dd::vEdge& v, const size_t numQubits,
         const std::string& decisions) {
        return v.getValueByPath(numQubits, decisions);
      },
      "num_qubits"_a, "decisions"_a,
      R"pb(Get the amplitude of a basis state by decisions.

Args:
    num_qubits: The number of qubits.
    decisions: The decisions as a string of bits (`0` or `1`), where `decisions[i]` corresponds to the successor to follow at level `i` of the DD.
        Must be at least `num_qubits` long.

Returns:
    The amplitude of the basis state.)pb");

  vec.def("get_vector", &getVector, "threshold"_a = 0.,
          R"pb(Get the state vector represented by the DD.

Args:
    threshold: The threshold for not including amplitudes in the state vector. Defaults to 0.0.

Returns:
    The state vector.

Raises:
    MemoryError: If the memory allocation fails.)pb");

  vec.def(
      "to_dot",
      [](const dd::vEdge& e, const bool colored = true,
         const bool edgeLabels = false, const bool classic = false,
         const bool memory = false, const bool formatAsPolar = true) {
        std::ostringstream os;
        toDot(e, os, colored, edgeLabels, classic, memory, formatAsPolar);
        return os.str();
      },
      "colored"_a = true, "edge_labels"_a = false, "classic"_a = false,
      "memory"_a = false, "format_as_polar"_a = true,
      R"pb(Convert the DD to a DOT graph that can be plotted via Graphviz.

Args:
    colored: Whether to use colored edge weights
    edge_labels: Whether to include edge weights as labels.
    classic: Whether to use the classic DD visualization style.
    memory: Whether to include memory information. For debugging purposes only.
    format_as_polar: Whether to format the edge weights in polar coordinates.

Returns:
    The DOT graph.)pb");

  vec.def(
      "to_svg",
      [](const dd::vEdge& e, const std::string& filename,
         const bool colored = true, const bool edgeLabels = false,
         const bool classic = false, const bool memory = false,
         const bool formatAsPolar = true) {
        // replace the filename extension with .dot
        const auto dotFilename =
            filename.substr(0, filename.find_last_of('.')) + ".dot";
        export2Dot(e, dotFilename, colored, edgeLabels, classic, memory, true,
                   formatAsPolar);
      },
      "filename"_a, "colored"_a = true, "edge_labels"_a = false,
      "classic"_a = false, "memory"_a = false, "format_as_polar"_a = true,
      R"pb(Convert the DD to an SVG file that can be viewed in a browser.

Requires the `dot` command from Graphviz to be installed and available in the PATH.

Args:
    filename: The filename of the SVG file. Any file extension will be replaced by `.dot` and then `.svg`.
    colored: Whether to use colored edge weights.
    edge_labels: Whether to include edge weights as labels.
    classic: Whether to use the classic DD visualization style.
    memory: Whether to include memory information. For debugging purposes only.
    format_as_polar: Whether to format the edge weights in polar coordinates.)pb");
}

} // namespace mqt
