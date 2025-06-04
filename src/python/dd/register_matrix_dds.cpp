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

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
// clang-format on

#include <cmath>
#include <complex>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

struct Matrix {
  std::vector<std::complex<dd::fp>> data;
  size_t n;
};

Matrix getMatrix(const dd::mEdge& m, const size_t numQubits,
                 const dd::fp threshold = 0.) {
  if (numQubits == 0U) {
    return Matrix{{static_cast<std::complex<dd::fp>>(m.w)}, 1};
  }
  const size_t dim = 1ULL << numQubits;
  auto data = std::vector<std::complex<dd::fp>>(dim * dim);
  m.traverseMatrix(
      std::complex<dd::fp>{1., 0.}, 0ULL, 0ULL,
      [&data, dim](const std::size_t i, const std::size_t j,
                   const std::complex<dd::fp>& c) { data[(i * dim) + j] = c; },
      numQubits, threshold);
  return Matrix{data, dim};
}

void registerMatrixDDs(const py::module& mod) {
  auto mat = py::class_<dd::mEdge>(mod, "MatrixDD");

  mat.def("is_terminal", &dd::mEdge::isTerminal);
  mat.def("is_zero_terminal", &dd::mEdge::isZeroTerminal);
  mat.def("is_identity", &dd::mEdge::isIdentity<>,
          "up_to_global_phase"_a = true);

  mat.def("size", py::overload_cast<>(&dd::mEdge::size, py::const_));

  mat.def("get_entry", &dd::mEdge::getValueByIndex<>, "num_qubits"_a, "row"_a,
          "col"_a);
  mat.def("get_entry_by_path", &dd::mEdge::getValueByPath, "num_qubits"_a,
          "decisions"_a);

  py::class_<Matrix>(mod, "Matrix", py::buffer_protocol())
      .def_buffer([](Matrix& matrix) -> py::buffer_info {
        return py::buffer_info(
            matrix.data.data(), sizeof(std::complex<dd::fp>),
            py::format_descriptor<std::complex<dd::fp>>::format(), 2,
            {matrix.n, matrix.n},
            {sizeof(std::complex<dd::fp>) * matrix.n,
             sizeof(std::complex<dd::fp>)});
      });

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
