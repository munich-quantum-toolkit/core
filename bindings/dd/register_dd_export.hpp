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

#include "dd/Edge.hpp"
#include "dd/Export.hpp"
#include "dd/Package.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h" /// NOLINT(misc-include-cleaner)

#include <ios>
#include <sstream>
#include <string>

namespace mqt {

template <class Node>
void registerDDExport(nanobind::class_<dd::Edge<Node>>& edgeClass) {
  namespace nb = nanobind;
  using namespace nb::literals;
  edgeClass.def(
      "to_bytes",
      [](const dd::Edge<Node>& e, const bool binary = true) {
        std::ostringstream os(std::ios::out | std::ios::binary);
        dd::serialize(e, os, binary);
        const auto data = os.view();
        return nb::bytes(data.data(), data.size());
      },
      "binary"_a = true, R"pb(Serialize the DD to bytes.

Args:
    binary: Whether to use the binary serialization format. Defaults to True.
        If False, the textual serialization format is used.

Returns:
    The serialized DD.

Notes:
    The binary format is not portable across different architectures or platforms.)pb");

  edgeClass.def_static(
      "from_bytes",
      [](dd::Package& p, const nb::bytes& data, const bool binary = true) {
        std::istringstream is(std::string(data.c_str(), data.size()),
                              std::ios::in | std::ios::binary);
        return p.deserialize<Node>(is, binary);
      },
      "dd_package"_a, "data"_a, "binary"_a = true,
      /// keep the DD package alive while the returned DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Deserialize a DD from bytes.

Args:
    dd_package: The DD package that owns the deserialized DD.
    data: The serialized DD.
    binary: Whether the data uses the binary serialization format. Defaults to True.
        If False, the textual serialization format is expected.

Returns:
    The deserialized DD.

Notes:
    The binary format is not portable across different architectures or platforms.)pb");

  edgeClass.def(
      "to_dot",
      [](const dd::Edge<Node>& e, const bool colored = true,
         const bool edgeLabels = false, const bool classic = false,
         const bool memory = false, const bool formatAsPolar = true) {
        std::ostringstream os;
        dd::toDot(e, os, colored, edgeLabels, classic, memory, formatAsPolar);
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

  edgeClass.def(
      "to_svg",
      [](const dd::Edge<Node>& e, const std::string& filename,
         const bool colored = true, const bool edgeLabels = false,
         const bool classic = false, const bool memory = false,
         const bool formatAsPolar = true) {
        /// replace the filename extension with .dot
        const auto dotFilename =
            filename.substr(0, filename.find_last_of('.')) + ".dot";
        dd::export2Dot(e, dotFilename, colored, edgeLabels, classic, memory,
                       true, formatAsPolar);
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
