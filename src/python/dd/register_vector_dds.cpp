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

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

struct Vector {
  dd::CVec v;
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
Vector getVector(const dd::vEdge& v, const dd::fp threshold = 0.) {
  return {v.getVector(threshold)};
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVectorDDs(const py::module& mod) {
  auto vec = py::class_<dd::vEdge>(mod, "VectorDD");

  vec.def("is_terminal", &dd::vEdge::isTerminal);
  vec.def("is_zero_terminal", &dd::vEdge::isZeroTerminal);

  vec.def("size", py::overload_cast<>(&dd::vEdge::size, py::const_));

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

  py::class_<Vector>(mod, "Vector", py::buffer_protocol())
      .def_buffer([](Vector& vector) -> py::buffer_info {
        return py::buffer_info(
            vector.v.data(), sizeof(std::complex<dd::fp>),
            // NOLINTNEXTLINE(misc-include-cleaner)
            py::format_descriptor<std::complex<dd::fp>>::format(), 1,
            {vector.v.size()}, {sizeof(std::complex<dd::fp>)});
      });

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
