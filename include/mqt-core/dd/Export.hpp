/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Export.hpp
/// Export decision diagrams to vector and matrix representations.

#pragma once

#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>

namespace dd {

std::string colorFromPhase(const Complex& a);

fp thicknessFromMagnitude(const Complex& a);

void printPhaseFormatted(llvm::raw_ostream& os, fp r);

std::string conditionalFormat(const Complex& a, bool formatAsPolar = true);

/// Render a DOT file as a sibling SVG using Graphviz.
///
/// Requires `dot` on PATH and returns an error if rendering fails.
[[nodiscard]] llvm::Error renderDotToSvg(llvm::StringRef dotFilename);

template <class Node>
static llvm::raw_ostream& header(const Edge<Node>& e, llvm::raw_ostream& os,
                                 bool edgeLabels, bool formatAsPolar = true) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
  os << "root->" << (e.isTerminal() ? "t" : "0");
  os << "[penwidth=\"" << thicknessFromMagnitude(e.w) << "\",tooltip=\""
     << conditionalFormat(e.w, formatAsPolar) << "\"";
  if (!e.w.exactlyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(e.w, formatAsPolar) << "</font>>";
  }

  os << "]\n";

  return os;
}
template <class Node>
static llvm::raw_ostream& coloredHeader(const Edge<Node>& e,
                                        llvm::raw_ostream& os, bool edgeLabels,
                                        bool formatAsPolar = true) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

  auto mag = thicknessFromMagnitude(e.w);
  auto color = colorFromPhase(e.w);
  os << "root->" << (e.isTerminal() ? "t" : "0");
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(e.w, formatAsPolar) << "\",color=\"" << color << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(e.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";
  return os;
}
template <class Node>
static llvm::raw_ostream& memoryHeader(const Edge<Node>& e,
                                       llvm::raw_ostream& os, bool edgeLabels) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

  auto mag = thicknessFromMagnitude(e.w);
  auto color = colorFromPhase(e.w);
  os << "root->" << (e.isTerminal() ? "t" : "0");
  os << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w.toString(false, 4)
     << "\" color=\"" << color << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;[";
    if (e.w.exactlyZero()) {
      os << "0";
    } else if (e.w.exactlyOne()) {
      os << "1";
    } else {
      if (RealNumber::exactlyZero(e.w.r)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(e.w.r)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(e.w.r)) {
        os << "1";
      } else {
        os << llvm::format_hex_no_prefix(
            reinterpret_cast<std::uintptr_t>(e.w.r), 1);
      }
      os << " ";
      if (RealNumber::exactlyZero(e.w.i)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(e.w.i)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(e.w.i)) {
        os << "1";
      } else {
        os << llvm::format_hex_no_prefix(
            reinterpret_cast<std::uintptr_t>(e.w.i), 1);
      }
    }
    os << "]</font>>";
  }
  os << "]\n";
  return os;
}

llvm::raw_ostream& modernNode(const mEdge& e, size_t nodeId,
                              llvm::raw_ostream& os, bool formatAsPolar = true);
llvm::raw_ostream& modernNode(const vEdge& e, size_t nodeId,
                              llvm::raw_ostream& os, bool formatAsPolar = true);
llvm::raw_ostream& classicNode(const mEdge& e, size_t nodeId,
                               llvm::raw_ostream& os,
                               bool formatAsPolar = true);
llvm::raw_ostream& classicNode(const vEdge& e, size_t nodeId,
                               llvm::raw_ostream& os,
                               bool formatAsPolar = true);
template <class Node>
static llvm::raw_ostream& memoryNode(const Edge<Node>& e, size_t nodeId,
                                     llvm::raw_ostream& os) {
  constexpr std::size_t n = std::tuple_size_v<decltype(e.p->e)>;
  os << nodeId << "[label=<";
  os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
  os << R"(<tr><td colspan=")" << n << R"(" border="1" sides="B">)"
     << llvm::format_hex_no_prefix(reinterpret_cast<std::uintptr_t>(e.p), 1)
     << "</td></tr>";
  os << "<tr>";
  for (std::size_t i = 0; i < n; ++i) {
    os << "<td port=\"" << i << R"(" href="javascript:;" border="0" tooltip=")"
       << e.p->e[i].w.toString(false, 4) << "\">";
    if (e.p->e[i].isZeroTerminal()) {
      os << "&nbsp;0 "
            "";
    } else {
      os << "<font color=\"white\">&nbsp;0 </font>";
    }
    os << "</td>";
  }
  os << "</tr>";
  os << "</table></font>>,tooltip=\""
     << llvm::format_hex_no_prefix(reinterpret_cast<std::uintptr_t>(e.p), 1)
     << "\"]\n";
  return os;
}

llvm::raw_ostream& bwEdge(const mEdge& to, size_t fromId, size_t toId,
                          std::uint16_t idx, llvm::raw_ostream& os,
                          bool edgeLabels = false, bool classic = false,
                          bool formatAsPolar = true);
llvm::raw_ostream& bwEdge(const vEdge& to, size_t fromId, size_t toId,
                          std::uint16_t idx, llvm::raw_ostream& os,
                          bool edgeLabels = false, bool classic = false,
                          bool formatAsPolar = true);
llvm::raw_ostream& coloredEdge(const mEdge& to, size_t fromId, size_t toId,
                               std::uint16_t idx, llvm::raw_ostream& os,
                               bool edgeLabels = false, bool classic = false,
                               bool formatAsPolar = true);
llvm::raw_ostream& coloredEdge(const vEdge& to, size_t fromId, size_t toId,
                               std::uint16_t idx, llvm::raw_ostream& os,
                               bool edgeLabels = false, bool classic = false,
                               bool formatAsPolar = true);
template <class Node>
static llvm::raw_ostream&
memoryEdge(const Edge<Node>& to, size_t fromId, size_t toId, std::uint16_t idx,
           llvm::raw_ostream& os, bool edgeLabels = false) {
  os << fromId << ":" << idx << ":s->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << toId;
  }

  auto mag = thicknessFromMagnitude(to.w);
  auto color = colorFromPhase(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w.toString(false, 4)
     << "\" color=\"" << color << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;[";
    if (to.w.exactlyOne()) {
      os << "1";
    } else {
      if (RealNumber::exactlyZero(to.w.r)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(to.w.r)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(to.w.r)) {
        os << "1";
      } else {
        os << llvm::format_hex_no_prefix(
            reinterpret_cast<std::uintptr_t>(to.w.r), 1);
      }
      os << " ";
      if (RealNumber::exactlyZero(to.w.i)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(to.w.i)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(to.w.i)) {
        os << "1";
      } else {
        os << llvm::format_hex_no_prefix(
            reinterpret_cast<std::uintptr_t>(to.w.i), 1);
      }
    }
    os << "]</font>>";
  }
  os << "]\n";

  return os;
}

/// Write DOT text; the caller owns the output stream and its error state.
template <class Node>
void toDot(const Edge<Node>& e, llvm::raw_ostream& os, bool colored = true,
           bool edgeLabels = false, bool classic = false, bool memory = false,
           bool formatAsPolar = true);

template <class Node>
[[nodiscard]] static llvm::Error
exportToDot(Edge<Node> basic, llvm::StringRef outputFilename,
            bool colored = true, bool edgeLabels = false, bool classic = false,
            bool memory = false, bool show = true, bool formatAsPolar = true) {
  int descriptor = -1;
  auto error = llvm::sys::fs::openFileForWrite(outputFilename, descriptor);
  if (error) {
    return llvm::createStringError(std::errc::io_error,
                                   "Cannot open DOT file: %s",
                                   error.message().c_str());
  }
  llvm::raw_fd_ostream output(descriptor, false);
  toDot(basic, output, colored, edgeLabels, classic, memory, formatAsPolar);
  output.flush();
  error = output.error();
  output.clear_error();
  if (const auto closeError =
          llvm::sys::Process::SafelyCloseFileDescriptor(descriptor);
      !error) {
    error = closeError;
  }
  if (error) {
    return llvm::createStringError(std::errc::io_error,
                                   "Cannot write DOT file: %s",
                                   error.message().c_str());
  }

  if (show) {
    return renderDotToSvg(outputFilename);
  }
  return llvm::Error::success();
}

///
/// Serialization
/// Note: do not rely on the binary format being portable across different
/// architectures/platforms
///

void serialize(const vEdge& basic, llvm::raw_ostream& os,
               bool writeBinary = false);

void serialize(const mEdge& basic, llvm::raw_ostream& os,
               bool writeBinary = false);
template <class Node>
[[nodiscard]] static llvm::Error serialize(const Edge<Node>& basic,
                                           llvm::StringRef outputFilename,
                                           bool writeBinary = false) {
  int descriptor = -1;
  auto error = llvm::sys::fs::openFileForWrite(outputFilename, descriptor);
  if (error) {
    return llvm::createStringError(std::errc::io_error,
                                   "Cannot open serialized file: %s",
                                   error.message().c_str());
  }
  llvm::raw_fd_ostream output(descriptor, false);
  serialize(basic, output, writeBinary);
  output.flush();
  error = output.error();
  output.clear_error();
  if (const auto closeError =
          llvm::sys::Process::SafelyCloseFileDescriptor(descriptor);
      !error) {
    error = closeError;
  }
  if (error) {
    return llvm::createStringError(std::errc::io_error,
                                   "Cannot write serialized file: %s",
                                   error.message().c_str());
  }
  return llvm::Error::success();
}

template <typename Node>
void exportEdgeWeights(const Edge<Node>& edge, llvm::raw_ostream& stream);

} // namespace dd
