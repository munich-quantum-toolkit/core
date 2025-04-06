/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dd {

std::string colorFromPhase(const Complex& a);

fp thicknessFromMagnitude(const Complex& a);

void printPhaseFormatted(std::ostream& os, fp r);

std::string conditionalFormat(const Complex& a, bool formatAsPolar = true);

template <class Node>
static std::ostream& header(const Edge<Node>& e, std::ostream& os,
                            bool edgeLabels, bool formatAsPolar = true) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
  auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
  os << "root->";
  if (e.isTerminal()) {
    os << "t";
  } else {
    os << toplabel;
  }
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
static std::ostream& coloredHeader(const Edge<Node>& e, std::ostream& os,
                                   bool edgeLabels, bool formatAsPolar = true) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

  auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
  auto mag = thicknessFromMagnitude(e.w);
  auto color = colorFromPhase(e.w);
  os << "root->";
  if (e.isTerminal()) {
    os << "t";
  } else {
    os << toplabel;
  }
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
static std::ostream& memoryHeader(const Edge<Node>& e, std::ostream& os,
                                  bool edgeLabels) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

  auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
  auto mag = thicknessFromMagnitude(e.w);
  auto color = colorFromPhase(e.w);
  os << "root->";
  if (e.isTerminal()) {
    os << "t";
  } else {
    os << toplabel;
  }
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
        os << std::hex << reinterpret_cast<std::uintptr_t>(e.w.r) << std::dec;
      }
      os << " ";
      if (RealNumber::exactlyZero(e.w.i)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(e.w.i)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(e.w.i)) {
        os << "1";
      } else {
        os << std::hex << reinterpret_cast<std::uintptr_t>(e.w.i) << std::dec;
      }
    }
    os << "]</font>>";
  }
  os << "]\n";
  return os;
}

std::ostream& modernNode(const mEdge& e, std::ostream& os,
                         bool formatAsPolar = true);
std::ostream& modernNode(const vEdge& e, std::ostream& os,
                         bool formatAsPolar = true);
std::ostream& classicNode(const mEdge& e, std::ostream& os,
                          bool formatAsPolar = true);
std::ostream& classicNode(const vEdge& e, std::ostream& os,
                          bool formatAsPolar = true);
template <class Node>
static std::ostream& memoryNode(const Edge<Node>& e, std::ostream& os) {
  constexpr std::size_t n = std::tuple_size_v<decltype(e.p->e)>;
  auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
                   1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[label=<";
  os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
  os << R"(<tr><td colspan=")" << n << R"(" border="1" sides="B">)" << std::hex
     << reinterpret_cast<std::uintptr_t>(e.p) << std::dec
     << " ref: " << e.p->ref << "</td></tr>";
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
  os << "</table></font>>,tooltip=\"" << std::hex
     << reinterpret_cast<std::uintptr_t>(e.p) << "\"]\n"
     << std::dec;
  return os;
}

std::ostream& bwEdge(const mEdge& from, const mEdge& to, std::uint16_t idx,
                     std::ostream& os, bool edgeLabels = false,
                     bool classic = false, bool formatAsPolar = true);
std::ostream& bwEdge(const vEdge& from, const vEdge& to, std::uint16_t idx,
                     std::ostream& os, bool edgeLabels = false,
                     bool classic = false, bool formatAsPolar = true);
std::ostream& coloredEdge(const mEdge& from, const mEdge& to, std::uint16_t idx,
                          std::ostream& os, bool edgeLabels = false,
                          bool classic = false, bool formatAsPolar = true);
std::ostream& coloredEdge(const vEdge& from, const vEdge& to, std::uint16_t idx,
                          std::ostream& os, bool edgeLabels = false,
                          bool classic = false, bool formatAsPolar = true);
template <class Node>
static std::ostream& memoryEdge(const Edge<Node>& from, const Edge<Node>& to,
                                std::uint16_t idx, std::ostream& os,
                                bool edgeLabels = false) {
  auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  auto tolabel = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":s->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
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
        os << std::hex << reinterpret_cast<std::uintptr_t>(to.w.r) << std::dec;
      }
      os << " ";
      if (RealNumber::exactlyZero(to.w.i)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(to.w.i)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(to.w.i)) {
        os << "1";
      } else {
        os << std::hex << reinterpret_cast<std::uintptr_t>(to.w.i) << std::dec;
      }
    }
    os << "]</font>>";
  }
  os << "]\n";

  return os;
}

template <class Node>
static void toDot(const Edge<Node>& e, std::ostream& os, bool colored = true,
                  bool edgeLabels = false, bool classic = false,
                  bool memory = false, bool formatAsPolar = true) {
  std::ostringstream oss{};
  // header, root and terminal declaration

  if (memory) {
    memoryHeader(e, oss, edgeLabels);
  } else if (colored) {
    coloredHeader(e, oss, edgeLabels, formatAsPolar);
  } else {
    header(e, oss, edgeLabels, formatAsPolar);
  }

  std::unordered_set<decltype(e.p)> nodes{};

  auto priocmp = [](const Edge<Node>* left, const Edge<Node>* right) {
    if (left->p == nullptr) {
      return right->p != nullptr;
    }
    if (right->p == nullptr) {
      return false;
    }
    return left->p->v < right->p->v;
  };

  std::priority_queue<const Edge<Node>*, std::vector<const Edge<Node>*>,
                      decltype(priocmp)>
      q(priocmp);
  q.push(&e);

  // bfs until finished
  while (!q.empty()) {
    auto node = q.top();
    q.pop();

    // base case
    if (node->isTerminal()) {
      continue;
    }

    // check if node has already been processed
    auto ret = nodes.emplace(node->p);
    if (!ret.second) {
      continue;
    }

    // node definition as HTML-like label (href="javascript:;" is used as
    // workaround to make tooltips work)
    if (memory) {
      memoryNode(*node, oss);
    } else if (classic) {
      classicNode(*node, oss, formatAsPolar);
    } else {
      modernNode(*node, oss, formatAsPolar);
    }

    // iterate over edges in reverse to guarantee correct processing order
    for (auto i = static_cast<std::int16_t>(node->p->e.size() - 1); i >= 0;
         --i) {
      auto& edge = node->p->e[static_cast<std::size_t>(i)];
      if ((!memory && edge.w.approximatelyZero()) || edge.w.exactlyZero()) {
        // potentially add zero stubs here
        continue;
      }

      // non-zero edge to be included
      q.push(&edge);

      if (memory) {
        memoryEdge(*node, edge, static_cast<std::uint16_t>(i), oss, edgeLabels);
      } else if (colored) {
        coloredEdge(*node, edge, static_cast<std::uint16_t>(i), oss, edgeLabels,
                    classic, formatAsPolar);
      } else {
        bwEdge(*node, edge, static_cast<std::uint16_t>(i), oss, edgeLabels,
               classic, formatAsPolar);
      }
    }
  }
  oss << "}\n";

  os << oss.str() << std::flush;
}

template <class Node>
[[maybe_unused]] static void
export2Dot(Edge<Node> basic, const std::string& outputFilename,
           bool colored = true, bool edgeLabels = false, bool classic = false,
           bool memory = false, bool show = true, bool formatAsPolar = true) {
  std::ofstream init(outputFilename);
  toDot(basic, init, colored, edgeLabels, classic, memory, formatAsPolar);
  init.close();

  if (show) {
    std::ostringstream oss;
    oss << "dot -Tsvg " << outputFilename << " -o "
        << outputFilename.substr(0, outputFilename.find_last_of('.')) << ".svg";
    const auto str =
        oss.str(); // required to avoid immediate deallocation of temporary
    const auto exitCode = std::system(str.c_str());
    if (exitCode != 0) {
      std::cerr << "Error: dot returned with exit code " << exitCode << "\n";
    }
  }
}

///
/// Serialization
/// Note: do not rely on the binary format being portable across different
/// architectures/platforms
///

void serialize(const vEdge& basic, std::ostream& os, bool writeBinary = false);

void serializeMatrix(const mEdge& basic, std::int64_t& idx,
                     std::unordered_map<mNode*, std::int64_t>& nodeIndex,
                     std::unordered_set<mNode*>& visited, std::ostream& os,
                     bool writeBinary = false);
void serialize(const mEdge& basic, std::ostream& os, bool writeBinary = false);
template <class Node>
static void serialize(const Edge<Node>& basic,
                      const std::string& outputFilename,
                      bool writeBinary = false) {
  std::ofstream ofs = std::ofstream(outputFilename, std::ios::binary);

  if (!ofs.good()) {
    throw std::invalid_argument("Cannot open file: " + outputFilename);
  }

  serialize(basic, ofs, writeBinary);
}

template <typename Node>
static void exportEdgeWeights(const Edge<Node>& edge, std::ostream& stream) {
  struct Priocmp {
    bool operator()(const Edge<Node>* left, const Edge<Node>* right) {
      if (left->p == nullptr) {
        return right->p != nullptr;
      }
      if (right->p == nullptr) {
        return false;
      }
      return left->p->v < right->p->v;
    }
  };
  stream << std::showpos << RealNumber::val(edge.w.r)
         << RealNumber::val(edge.w.i) << std::noshowpos << "i\n";

  std::unordered_set<decltype(edge.p)> nodes{};

  std::priority_queue<const Edge<Node>*, std::vector<const Edge<Node>*>,
                      Priocmp>
      q;
  q.push(&edge);

  // bfs until finished
  while (!q.empty()) {
    auto edgePtr = q.top();
    q.pop();

    // base case
    if (edgePtr->isTerminal()) {
      continue;
    }

    // check if edgePtr has already been processed
    if (auto ret = nodes.emplace(edgePtr->p); !ret.second) {
      continue;
    }

    // iterate over edges in reverse to guarantee correct processing order
    for (auto i = static_cast<std::int16_t>(edgePtr->p->e.size() - 1); i >= 0;
         --i) {
      auto& child = edgePtr->p->e[static_cast<std::size_t>(i)];
      if (child.w.approximatelyZero()) {
        // potentially add zero stubs here
        continue;
      }

      // non-zero child to be included
      q.push(&child);
      stream << std::showpos << RealNumber::val(child.w.r)
             << RealNumber::val(child.w.i) << std::noshowpos << "i\n";
    }
  }
}

} // namespace dd
