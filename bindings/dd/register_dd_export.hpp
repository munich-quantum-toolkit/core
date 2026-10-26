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

#include "../Result.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h" // NOLINT(misc-include-cleaner)

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mqt {

template <class Node>
void registerDDExport(nanobind::class_<dd::Edge<Node>>& edgeClass) {
  namespace nb = nanobind;
  using namespace nb::literals;
  edgeClass.def(
      "to_bytes",
      [](const dd::Edge<Node>& e, const bool binary = true) {
        std::string data;
        llvm::raw_string_ostream os(data);
        dd::serialize(e, os, binary);
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
        return bindings::unwrap(p.deserialize<Node>(
            llvm::StringRef(data.c_str(), data.size()), binary));
      },
      "dd_package"_a, "data"_a, "binary"_a = true,
      // keep the DD package alive while the returned DD is alive.
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
        std::string data;
        llvm::raw_string_ostream os(data);
        dd::toDot(e, os, colored, edgeLabels, classic, memory, formatAsPolar);
        return data;
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
        llvm::SmallString<128> dotFilename(filename);
        llvm::sys::path::replace_extension(dotFilename, ".dot");
        nb::object pygraphviz;
        try {
          pygraphviz = nb::module_::import_("pygraphviz");
        } catch (const nb::python_error& error) {
          if (!error.matches(nb::module_::import_("builtins")
                                 .attr("ModuleNotFoundError")) ||
              !error.value().attr("name").equal(nb::str("pygraphviz"))) {
            throw;
          }
        }
        const auto usePyGraphviz =
            pygraphviz &&
            std::stoi(nb::cast<std::string>(pygraphviz.attr("__version__"))) >=
                2;
        bindings::unwrap(dd::exportToDot(e, dotFilename, colored, edgeLabels,
                                         classic, memory, !usePyGraphviz,
                                         formatAsPolar));
        if (usePyGraphviz) {
          auto svgFilename = dotFilename;
          llvm::sys::path::replace_extension(svgFilename, ".svg");
          pygraphviz
              .attr("AGraph")("filename"_a = nb::str(dotFilename.data(),
                                                     dotFilename.size()))
              .attr("draw")(nb::str(svgFilename.data(), svgFilename.size()),
                            "format"_a = "svg", "prog"_a = "dot");
        }
      },
      "filename"_a, "colored"_a = true, "edge_labels"_a = false,
      "classic"_a = false, "memory"_a = false, "format_as_polar"_a = true,
      R"pb(Convert the DD to an SVG file that can be viewed in a browser.

Uses PyGraphviz 2 or later when installed. Otherwise, requires the `dot` command
from Graphviz to be available in the PATH.

Args:
    filename: The filename of the SVG file. Any file extension will be replaced by `.dot` and then `.svg`.
    colored: Whether to use colored edge weights.
    edge_labels: Whether to include edge weights as labels.
    classic: Whether to use the classic DD visualization style.
    memory: Whether to include memory information. For debugging purposes only.
    format_as_polar: Whether to format the edge weights in polar coordinates.)pb");
}

} // namespace mqt
