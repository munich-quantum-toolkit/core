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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <string>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using Matrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
Matrix getMatrix(const dd::mEdge& m, const size_t numQubits,
                 const dd::fp threshold = 0.) {
  if (numQubits == 0U) {
    auto* data = new std::complex<dd::fp>[1];
    data[0] = static_cast<std::complex<dd::fp>>(m.w);
    nb::capsule owner(
        data, [](void* p) noexcept { delete[] (std::complex<dd::fp>*)p; });
    return Matrix(data, {1, 1}, owner);
  }

  const auto dim = 1ULL << numQubits;
  auto* data = new std::complex<dd::fp>[dim * dim];
  m.traverseMatrix(
      std::complex<dd::fp>{1., 0.}, 0ULL, 0ULL,
      [&data, dim](const std::size_t i, const std::size_t j,
                   const std::complex<dd::fp>& c) { data[(i * dim) + j] = c; },
      numQubits, threshold);
  nb::capsule owner(
      data, [](void* p) noexcept { delete[] (std::complex<dd::fp>*)p; });
  return Matrix(data, {dim, dim}, owner);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerMatrixDDs(const nb::module_& m) {
  auto mat = nb::class_<dd::mEdge>(m, "MatrixDD");

  mat.def("is_terminal", &dd::mEdge::isTerminal);
  mat.def("is_zero_terminal", &dd::mEdge::isZeroTerminal);
  mat.def("is_identity", &dd::mEdge::isIdentity<>,
          "up_to_global_phase"_a = true);

  mat.def("size", nb::overload_cast<>(&dd::mEdge::size, nb::const_));

  mat.def("get_entry", &dd::mEdge::getValueByIndex<>, "num_qubits"_a, "row"_a,
          "col"_a);
  mat.def("get_entry_by_path", &dd::mEdge::getValueByPath, "num_qubits"_a,
          "decisions"_a);

  mat.def("get_matrix", &getMatrix, "num_qubits"_a, "threshold"_a = 0.);

  mat.def(
      "to_dot",
      [](const dd::mEdge& e, const bool colored = true,
         const bool edgeLabels = false, const bool classic = false,
         const bool memory = false, const bool formatAsPolar = true) {
        std::ostringstream os;
        dd::toDot(e, os, colored, edgeLabels, classic, memory, formatAsPolar);
        return os.str();
      },
      "colored"_a = true, "edge_labels"_a = false, "classic"_a = false,
      "memory"_a = false, "format_as_polar"_a = true);

  mat.def(
      "to_svg",
      [](const dd::mEdge& e, const std::string& filename,
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
