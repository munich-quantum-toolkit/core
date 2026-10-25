/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Export.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <stack>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dd {

void renderDotToSvg(llvm::StringRef dotFilename) {
  llvm::SmallString<128> input(dotFilename);
  if (const auto error = llvm::sys::fs::make_absolute(input)) {
    throw std::system_error(error, "Cannot resolve DOT filename");
  }
  auto output = input;
  llvm::sys::path::replace_extension(output, ".svg");
  auto program = llvm::sys::findProgramByName("dot");
  if (!program) {
    throw std::system_error(program.getError(), "Cannot find Graphviz dot");
  }
  const std::array<llvm::StringRef, 5> args{
      *program, "-Tsvg", input, "-o", output,
  };
  std::string message;
  const auto exitCode = llvm::sys::ExecuteAndWait(*program, args, std::nullopt,
                                                  {}, 0, 0, &message);
  if (exitCode < 0) {
    throw std::runtime_error("Cannot run Graphviz dot: " + message);
  }
  if (exitCode != 0) {
    throw std::runtime_error("Graphviz dot exited with code " +
                             std::to_string(exitCode));
  }
}

std::string colorFromPhase(const Complex& a) {
  auto phase = ComplexNumbers::arg(a) / (2 * PI);
  if (phase < 0) {
    phase += 1.;
  }
  return llvm::formatv("{0:F3} 0.667 0.750", phase).str();
}
fp thicknessFromMagnitude(const Complex& a) {
  return 3.0 * std::max(ComplexNumbers::mag(a), 0.10);
}
void printPhaseFormatted(llvm::raw_ostream& os, fp r) {
  const auto tol = RealNumber::eps;

  r /= PI;
  // special case treatment for +-i
  os << "ℯ(" << (std::signbit(r) ? "-" : "") << "iπ";

  const auto absr = std::abs(r);
  auto fraction = ComplexValue::getLowestFraction(absr);
  auto approx =
      static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  auto error = std::abs(absr - approx);

  if (error < tol) { // suitable fraction a/b found
    if (fraction.first == 1U && fraction.second == 1U) {
      os << ")";
    } else if (fraction.second == 1U) {
      os << " " << fraction.first << ")";
    } else if (fraction.first == 1U) {
      os << "/" << fraction.second << ")";
    } else {
      os << " " << fraction.first << "/" << fraction.second << ")";
    }
    return;
  }

  auto const abssqrt = absr / SQRT2_2;
  fraction = ComplexValue::getLowestFraction(abssqrt);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  error = std::abs(abssqrt - approx);

  if (error < tol) { // suitable fraction a/(b * sqrt(2)) found
    if (fraction.first == 1U && fraction.second == 1U) {
      os << "/√2)";
    } else if (fraction.second == 1U) {
      os << " " << fraction.first << "/√2)";
    } else if (fraction.first == 1U) {
      os << "/(" << fraction.second << "√2))";
    } else {
      os << " " << fraction.first << "/(" << fraction.second << "√2))";
    }
    return;
  }

  auto const abspi = absr / PI;
  fraction = ComplexValue::getLowestFraction(abspi);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  error = std::abs(abspi - approx);

  if (error < tol) { // suitable fraction a/b π found
    if (fraction.first == 1U && fraction.second == 1U) {
      os << " π)";
    } else if (fraction.second == 1U) {
      os << " " << fraction.first << "π)";
    } else if (fraction.first == 1U) {
      os << " π/" << fraction.second << ")";
    } else {
      os << " " << fraction.first << "π/" << fraction.second << ")";
    }
    return;
  }

  // default
  os << " " << ComplexValue::toString(absr, 0., false) << ")";
}
std::string conditionalFormat(const Complex& a, const bool formatAsPolar) {
  if (!formatAsPolar) {
    return a.toString();
  }

  const auto mag = ComplexNumbers::mag(a);
  const auto phase = ComplexNumbers::arg(a);

  if (RealNumber::approximatelyZero(mag)) {
    return "0";
  }

  std::string result;
  llvm::raw_string_ostream ss(result);
  // magnitude is (almost) 1
  if (RealNumber::approximatelyEquals(mag, 1.)) {
    if (RealNumber::approximatelyZero(phase)) {
      return "1";
    }
    if (RealNumber::approximatelyEquals(phase, PI_2)) {
      return "i";
    }
    if (RealNumber::approximatelyEquals(phase, -PI_2)) {
      return "-i";
    }
    if (RealNumber::approximatelyEquals(phase, PI)) {
      return "-1";
    }
    printPhaseFormatted(ss, phase);
    return result;
  }

  if (RealNumber::approximatelyEquals(std::abs(phase), PI)) {
    ss << "-";
    ComplexValue::printFormatted(ss, mag);
    return result;
  }
  if (RealNumber::approximatelyZero(phase)) {
    ComplexValue::printFormatted(ss, mag);
    return result;
  }

  ComplexValue::printFormatted(ss, mag);
  ss << " ";
  printPhaseFormatted(ss, phase);

  return result;
}
llvm::raw_ostream& modernNode(const mEdge& e, const size_t nodeId,
                              llvm::raw_ostream& os, const bool formatAsPolar) {
  os << nodeId << "[label=<";
  os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
  os << R"(<tr><td colspan="2" rowspan="2" port="0" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar) << "\">"
     << (e.p->e[0].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << R"(</td><td sides="R"></td><td sides="L"></td>)"
     << R"(<td colspan="2" rowspan="2" port="1" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar) << "\">"
     << (e.p->e[1].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << R"(</td></tr>)";
  os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
  os << R"(<tr><td colspan="2" sides="B"></td><td colspan="2" rowspan="2" border="0"><font point-size="24">q<sub><font point-size="16">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td><td colspan="2" sides="B"></td></tr>)";
  os << R"(<tr><td sides="T" colspan="2"></td><td sides="T" colspan="2"></td></tr>)";
  os << R"(<tr><td colspan="2" rowspan="2" port="2" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[2].w, formatAsPolar) << "\">"
     << (e.p->e[2].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << R"(</td><td sides="R"></td><td sides="L"></td>)"
     << R"(<td colspan="2" rowspan="2" port="3" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[3].w, formatAsPolar) << "\">"
     << (e.p->e[3].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << "</td></tr>";
  os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
  os << "</table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v)
     << "\"]\n";
  return os;
}
llvm::raw_ostream& modernNode(const vEdge& e, const size_t nodeId,
                              llvm::raw_ostream& os, const bool formatAsPolar) {
  os << nodeId << "[label=<";
  os << R"(<font point-size="8"><table border="1" cellspacing="0" cellpadding="0" style="rounded">)";
  os << R"(<tr><td colspan="2" border="0" cellpadding="1"><font point-size="20">q<sub><font point-size="12">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td></tr><tr>)";
  os << R"(<td height="6" width="14" port="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar)
     << R"(" href="javascript:;" sides="RT">)"
     << (e.p->e[0].w.approximatelyZero()
             ? "&nbsp;0 "
             : R"(<font color="white">&nbsp;0 </font>)")
     << "</td>";
  os << R"(<td height="6" width="14" port="1" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar)
     << R"(" href="javascript:;" sides="LT">)"
     << (e.p->e[1].w.approximatelyZero()
             ? "&nbsp;0 "
             : R"(<font color="white">&nbsp;0 </font>)")
     << "</td>";
  os << "</tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v)
     << "\"]\n";
  return os;
}
llvm::raw_ostream& classicNode(const mEdge& e, const size_t nodeId,
                               llvm::raw_ostream& os,
                               const bool formatAsPolar) {
  os << nodeId << "[shape=circle, width=0.53, fixedsize=true, label=<";
  os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
  os << R"(<tr><td colspan="4"><font point-size="18">q<sub><font point-size="10">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td></tr><tr>)";
  os << R"(<td port="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[0].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "<td></td><td></td>";
  os << R"(<td port="3" tooltip=")"
     << conditionalFormat(e.p->e[3].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[3].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "</tr><tr><td></td>";
  os << R"(<td port="1" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[1].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << R"(<td port="2" tooltip=")"
     << conditionalFormat(e.p->e[2].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[2].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "<td></td></tr></table></font>>,tooltip=\"q"
     << static_cast<std::size_t>(e.p->v) << "\"]\n";
  return os;
}
llvm::raw_ostream& classicNode(const vEdge& e, const size_t nodeId,
                               llvm::raw_ostream& os,
                               const bool formatAsPolar) {
  os << nodeId << "[shape=circle, width=0.46, fixedsize=true, label=<";
  os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
  os << R"(<tr><td colspan="2"><font point-size="18">q<sub><font point-size="10">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td></tr><tr>)";
  os << R"(<td port="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[0].w.approximatelyZero()) {
    os << R"(<font point-size="10">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << R"(<td port="1" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[1].w.approximatelyZero()) {
    os << R"(<font point-size="10">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "</tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v)
     << "\"]\n";
  return os;
}
namespace {
template <class Node>
llvm::raw_ostream& renderEdge(const Edge<Node>& to, const size_t fromId,
                              const size_t toId, const std::uint16_t idx,
                              llvm::raw_ostream& os, const bool colored,
                              const bool edgeLabels, const bool classic,
                              const bool formatAsPolar) {
  os << fromId << ":" << idx << ":";
  if constexpr (IsVector<Node>) {
    os << (idx == 0 ? "sw" : "se");
  } else if (classic) {
    if (idx == 0) {
      os << "sw";
    } else if (idx == 1 || idx == 2) {
      os << "s";
    } else {
      os << "se";
    }
  } else {
    if (idx == 0) {
      os << "sw";
    } else if (idx == 1) {
      os << "se";
    } else {
      os << 's';
    }
  }
  os << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << toId;
  }

  const auto mag = thicknessFromMagnitude(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\"";
  if (colored) {
    os << " color=\"" << colorFromPhase(to.w) << "\"";
  } else if (!to.w.exactlyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
} // namespace

llvm::raw_ostream& bwEdge(const mEdge& to, const size_t fromId,
                          const size_t toId, const std::uint16_t idx,
                          llvm::raw_ostream& os, const bool edgeLabels,
                          const bool classic, const bool formatAsPolar) {
  return renderEdge(to, fromId, toId, idx, os, false, edgeLabels, classic,
                    formatAsPolar);
}
llvm::raw_ostream& bwEdge(const vEdge& to, const size_t fromId,
                          const size_t toId, const std::uint16_t idx,
                          llvm::raw_ostream& os, const bool edgeLabels,
                          const bool classic, const bool formatAsPolar) {
  return renderEdge(to, fromId, toId, idx, os, false, edgeLabels, classic,
                    formatAsPolar);
}
llvm::raw_ostream& coloredEdge(const mEdge& to, const size_t fromId,
                               const size_t toId, const std::uint16_t idx,
                               llvm::raw_ostream& os, const bool edgeLabels,
                               const bool classic, const bool formatAsPolar) {
  return renderEdge(to, fromId, toId, idx, os, true, edgeLabels, classic,
                    formatAsPolar);
}
llvm::raw_ostream& coloredEdge(const vEdge& to, const size_t fromId,
                               const size_t toId, const std::uint16_t idx,
                               llvm::raw_ostream& os, const bool edgeLabels,
                               const bool classic, const bool formatAsPolar) {
  return renderEdge(to, fromId, toId, idx, os, true, edgeLabels, classic,
                    formatAsPolar);
}
void serialize(const vEdge& basic, llvm::raw_ostream& os,
               const bool writeBinary) {
  if (writeBinary) {
    os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION),
             sizeof(decltype(SERIALIZATION_VERSION)));
    basic.w.writeBinary(os);
  } else {
    os << SERIALIZATION_VERSION << "\n";
    os << basic.w.toString(false, 16) << "\n";
  }
  std::unordered_map<vNode*, std::int64_t> nodeIndex{};

  // POST ORDER TRAVERSAL USING ONE STACK
  // https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
  std::stack<const vEdge*> stack{};

  const auto* node = &basic;
  if (!node->isTerminal()) {
    std::int64_t nextIndex = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      while (node != nullptr && !node->isTerminal()) {
        for (auto i = static_cast<std::size_t>(RADIX - 1); i > 0; --i) {
          auto const& edge = node->p->e.at(i);
          if (edge.isTerminal()) {
            continue;
          }
          if (edge.w.approximatelyZero()) {
            continue;
          }
          if (nodeIndex.contains(edge.p)) {
            continue;
          }

          stack.push(&edge);
        }
        stack.push(node);
        node = &node->p->e[0]; // NOLINT(readability-container-data-pointer)
      }
      node = stack.top();
      stack.pop();

      bool hasChild = false;
      for (auto i = 1U; i < RADIX && !hasChild; ++i) {
        auto const& edge = node->p->e.at(i);
        if (edge.w.approximatelyZero()) {
          continue;
        }
        if (nodeIndex.contains(edge.p)) {
          continue;
        }
        if (!stack.empty()) {
          hasChild = edge.p == stack.top()->p;
        }
      }

      if (hasChild) {
        const auto* const temp = stack.top();
        stack.pop();
        stack.push(node);
        node = temp;
      } else {
        if (nodeIndex.contains(node->p)) {
          node = nullptr;
          continue;
        }
        nodeIndex[node->p] = nextIndex;
        nextIndex++;

        if (writeBinary) {
          os.write(reinterpret_cast<const char*>(&nodeIndex[node->p]),
                   sizeof(decltype(nodeIndex[node->p])));
          os.write(reinterpret_cast<const char*>(&node->p->v),
                   sizeof(decltype(node->p->v)));

          for (auto i = 0U; i < RADIX; ++i) {
            auto const& edge = node->p->e.at(i);
            std::int64_t edgeIdx = edge.isTerminal() ? -1 : nodeIndex[edge.p];
            os.write(reinterpret_cast<const char*>(&edgeIdx),
                     sizeof(decltype(edgeIdx)));
            edge.w.writeBinary(os);
          }
        } else {
          os << nodeIndex[node->p] << " "
             << static_cast<std::size_t>(node->p->v);

          for (auto i = 0U; i < RADIX; ++i) {
            os << " (";
            auto const& edge = node->p->e.at(i);
            if (!edge.w.approximatelyZero()) {
              const std::int64_t edgeIdx =
                  edge.isTerminal() ? -1 : nodeIndex[edge.p];
              os << edgeIdx << " " << edge.w.toString(false, 16);
            }
            os << ")";
          }
          os << "\n";
        }
        node = nullptr;
      }
    } while (!stack.empty());
  }
}
namespace {
void serializeMatrix(const mEdge& basic, std::int64_t& idx,
                     std::unordered_map<mNode*, std::int64_t>& nodeIndex,
                     std::unordered_set<mNode*>& visited, llvm::raw_ostream& os,
                     const bool writeBinary) {
  if (!basic.isTerminal()) {
    for (auto const& e : basic.p->e) {
      if (visited.insert(e.p).second) {
        serializeMatrix(e, idx, nodeIndex, visited, os, writeBinary);
      }
    }

    if (!nodeIndex.contains(basic.p)) {
      nodeIndex[basic.p] = idx;
      ++idx;
    }

    if (writeBinary) {
      os.write(reinterpret_cast<const char*>(&nodeIndex[basic.p]),
               sizeof(decltype(nodeIndex[basic.p])));
      os.write(reinterpret_cast<const char*>(&basic.p->v),
               sizeof(decltype(basic.p->v)));

      for (auto const& edge : basic.p->e) {
        std::int64_t edgeIdx = edge.isTerminal() ? -1 : nodeIndex[edge.p];
        os.write(reinterpret_cast<const char*>(&edgeIdx),
                 sizeof(decltype(edgeIdx)));
        edge.w.writeBinary(os);
      }
    } else {
      os << nodeIndex[basic.p] << " " << static_cast<std::size_t>(basic.p->v);

      for (auto const& edge : basic.p->e) {
        os << " (";
        if (!edge.w.approximatelyZero()) {
          const std::int64_t edgeIdx =
              edge.isTerminal() ? -1 : nodeIndex[edge.p];
          os << edgeIdx << " " << edge.w.toString(false, 16);
        }
        os << ")";
      }
      os << "\n";
    }
  }
}
} // namespace

void serialize(const mEdge& basic, llvm::raw_ostream& os,
               const bool writeBinary) {
  if (writeBinary) {
    os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION),
             sizeof(decltype(SERIALIZATION_VERSION)));
    basic.w.writeBinary(os);
  } else {
    os << SERIALIZATION_VERSION << "\n";
    os << basic.w.toString(false, std::numeric_limits<fp>::max_digits10)
       << "\n";
  }
  std::int64_t idx = 0;
  std::unordered_map<mNode*, std::int64_t> nodeIndex{};
  std::unordered_set<mNode*> visited{};
  serializeMatrix(basic, idx, nodeIndex, visited, os, writeBinary);
}

template <class Node>
void toDot(const Edge<Node>& e, llvm::raw_ostream& os, bool colored,
           bool edgeLabels, bool classic, bool memory, bool formatAsPolar) {
  // header, root and terminal declaration

  if (memory) {
    memoryHeader(e, os, edgeLabels);
  } else if (colored) {
    coloredHeader(e, os, edgeLabels, formatAsPolar);
  } else {
    header(e, os, edgeLabels, formatAsPolar);
  }

  // Assign IDs on discovery, independent of allocation addresses.
  std::unordered_map<const Node*, size_t> nodes{{e.p, 0}};
  std::queue<const Edge<Node>*> q;
  q.push(&e);

  // bfs until finished
  while (!q.empty()) {
    auto node = q.front();
    q.pop();

    // base case
    if (node->isTerminal()) {
      continue;
    }

    const auto nodeId = nodes.at(node->p);

    // node definition as HTML-like label (href="javascript:;" is used as
    // workaround to make tooltips work)
    if (memory) {
      memoryNode(*node, nodeId, os);
    } else if (classic) {
      classicNode(*node, nodeId, os, formatAsPolar);
    } else {
      modernNode(*node, nodeId, os, formatAsPolar);
    }

    // iterate over edges in reverse to guarantee correct processing order
    for (auto i = static_cast<std::int16_t>(node->p->e.size() - 1); i >= 0;
         --i) {
      const auto& edge = node->p->e[static_cast<std::size_t>(i)];
      if ((!memory && edge.w.approximatelyZero()) || edge.w.exactlyZero()) {
        // potentially add zero stubs here
        continue;
      }

      size_t childId = 0;
      if (!edge.isTerminal()) {
        const auto [it, inserted] = nodes.try_emplace(edge.p, nodes.size());
        childId = it->second;
        if (inserted) {
          q.push(&edge);
        }
      }

      if (memory) {
        memoryEdge(edge, nodeId, childId, static_cast<std::uint16_t>(i), os,
                   edgeLabels);
      } else if (colored) {
        coloredEdge(edge, nodeId, childId, static_cast<std::uint16_t>(i), os,
                    edgeLabels, classic, formatAsPolar);
      } else {
        bwEdge(edge, nodeId, childId, static_cast<std::uint16_t>(i), os,
               edgeLabels, classic, formatAsPolar);
      }
    }
  }
  os << "}\n";
}

template <typename Node>
void exportEdgeWeights(const Edge<Node>& edge, llvm::raw_ostream& stream) {
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
  stream << llvm::format("%+g%+gi\n", RealNumber::val(edge.w.r),
                         RealNumber::val(edge.w.i));

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
    if (const auto ret = nodes.emplace(edgePtr->p); !ret.second) {
      continue;
    }

    // iterate over edges in reverse to guarantee correct processing order
    for (auto i = static_cast<std::int16_t>(edgePtr->p->e.size() - 1); i >= 0;
         --i) {
      const auto& child = edgePtr->p->e[static_cast<std::size_t>(i)];
      if (child.w.approximatelyZero()) {
        // potentially add zero stubs here
        continue;
      }

      // non-zero child to be included
      q.push(&child);
      stream << llvm::format("%+g%+gi\n", RealNumber::val(child.w.r),
                             RealNumber::val(child.w.i));
    }
  }
}

template void toDot(const vEdge&, llvm::raw_ostream&, bool, bool, bool, bool,
                    bool);
template void toDot(const mEdge&, llvm::raw_ostream&, bool, bool, bool, bool,
                    bool);
template void exportEdgeWeights(const vEdge&, llvm::raw_ostream&);
template void exportEdgeWeights(const mEdge&, llvm::raw_ostream&);

} // namespace dd
