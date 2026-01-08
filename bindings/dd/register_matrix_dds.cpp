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

using Matrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
Matrix getMatrix(const dd::mEdge& m, const size_t numQubits,
                 const dd::fp threshold = 0.) {
  if (numQubits > 20U) {
    throw nb::value_error("num_qubits exceeds practical limit of 20");
  }

  if (numQubits == 0U) {
    auto dataPtr = std::make_unique<std::complex<dd::fp>>(m.w);
    auto* data = dataPtr.release();
    const nb::capsule owner(data, [](void* ptr) noexcept {
      delete static_cast<std::complex<dd::fp>*>(ptr);
    });
    return Matrix(data, {1, 1}, owner);
  }

  const auto dim = 1ULL << numQubits;
  auto dataPtr = std::make_unique<std::complex<dd::fp>[]>(dim * dim);
  m.traverseMatrix(
      std::complex<dd::fp>{1., 0.}, 0ULL, 0ULL,
      [&dataPtr, dim](const std::size_t i, const std::size_t j,
                      const std::complex<dd::fp>& c) {
        dataPtr[(i * dim) + j] = c;
      },
      numQubits, threshold);
  auto* data = dataPtr.release();
  const nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<dd::fp>*>(ptr);
  });
  return Matrix(data, {dim, dim}, owner);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerMatrixDDs(const nb::module_& m) {
  auto mat = nb::class_<dd::mEdge>(
      m, "MatrixDD", "A class representing a matrix decision diagram (DD).");

  mat.def("is_terminal", &dd::mEdge::isTerminal,
          "Check if the DD is a terminal node.");
  mat.def("is_zero_terminal", &dd::mEdge::isZeroTerminal,
          "Check if the DD is a zero terminal node.");

  mat.def("is_identity", &dd::mEdge::isIdentity<>,
          "up_to_global_phase"_a = true,
          R"pb(Check if the DD represents the identity matrix.

Args:
    up_to_global_phase: Whether to ignore global phase.

Returns:
    Whether the DD represents the identity matrix.)pb");

  mat.def("size", nb::overload_cast<>(&dd::mEdge::size, nb::const_),
          "Get the size of the DD by traversing it once.");

  mat.def("get_entry", &dd::mEdge::getValueByIndex<>, "num_qubits"_a, "row"_a,
          "col"_a, "Get the entry of the matrix by row and column index.");

  mat.def("get_entry_by_path", &dd::mEdge::getValueByPath, "num_qubits"_a,
          "decisions"_a, R"pb(Get the entry of the matrix by decisions.

Args:
    num_qubits: The number of qubits.
    decisions: The decisions as a string of `0`, `1`, `2`, or `3`, where `decisions[i]` corresponds to the successor to follow at level `i` of the DD.
        Must be at least `num_qubits` long.

Returns:
    The entry of the matrix.)pb");

  mat.def("get_matrix", &getMatrix, "num_qubits"_a, "threshold"_a = 0.,
          R"pb(Get the matrix represented by the DD.

Args:
    num_qubits: The number of qubits.
    threshold: The threshold for not including entries in the matrix. Defaults to 0.0.

Returns:
    The matrix.

Raises:
    MemoryError: If the memory allocation fails.)pb");

  mat.def(
      "to_dot",
      [](const dd::mEdge& e, const bool colored = true,
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

  mat.def(
      "to_svg",
      [](const dd::mEdge& e, const std::string& filename,
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
