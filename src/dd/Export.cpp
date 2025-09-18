/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Export.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dd {

std::string colorFromPhase(const Complex& a) {
  auto phase = ComplexNumbers::arg(a) / (2 * PI);
  if (phase < 0) {
    phase += 1.;
  }
  std::ostringstream oss{};
  oss << std::fixed << std::setprecision(3) << phase << " " << 0.667 << " "
      << 0.75;
  return oss.str();
}
fp thicknessFromMagnitude(const Complex& a) {
  return 3.0 * std::max(ComplexNumbers::mag(a), 0.10);
}
void printPhaseFormatted(std::ostream& os, fp r) {
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

  auto abssqrt = absr / SQRT2_2;
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

  auto abspi = absr / PI;
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
  os << " " << absr << ")";
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

  std::ostringstream ss{};
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
    return ss.str();
  }

  if (RealNumber::approximatelyEquals(std::abs(phase), PI)) {
    ss << "-";
    ComplexValue::printFormatted(ss, mag);
    return ss.str();
  }
  if (RealNumber::approximatelyZero(phase)) {
    ComplexValue::printFormatted(ss, mag);
    return ss.str();
  }

  ComplexValue::printFormatted(ss, mag);
  ss << " ";
  printPhaseFormatted(ss, phase);

  return ss.str();
}
std::ostream& modernNode(const mEdge& e, std::ostream& os,
                         const bool formatAsPolar) {
  const auto nodelabel =
      (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
      1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[label=<";
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
std::ostream& modernNode(const vEdge& e, std::ostream& os,
                         const bool formatAsPolar) {
  const auto nodelabel =
      (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
      1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[label=<";
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
std::ostream& classicNode(const mEdge& e, std::ostream& os,
                          const bool formatAsPolar) {
  const auto nodelabel =
      (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
      1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[shape=circle, width=0.53, fixedsize=true, label=<";
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
std::ostream& classicNode(const vEdge& e, std::ostream& os,
                          const bool formatAsPolar) {
  const auto nodelabel =
      (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
      1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[shape=circle, width=0.46, fixedsize=true, label=<";
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
std::ostream& bwEdge(const mEdge& from, const mEdge& to,
                     const std::uint16_t idx, std::ostream& os,
                     const bool edgeLabels, const bool classic,
                     const bool formatAsPolar) {
  const auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  const auto tolabel =
      (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  if (classic) {
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
    os << tolabel;
  }

  const auto mag = thicknessFromMagnitude(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\"";
  if (!to.w.exactlyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
std::ostream& bwEdge(const vEdge& from, const vEdge& to,
                     const std::uint16_t idx, std::ostream& os,
                     const bool edgeLabels, [[maybe_unused]] const bool classic,
                     const bool formatAsPolar) {
  const auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  const auto tolabel =
      (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  os << (idx == 0 ? "sw" : "se") << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  const auto mag = thicknessFromMagnitude(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\"";
  if (!to.w.exactlyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
std::ostream& coloredEdge(const mEdge& from, const mEdge& to,
                          const std::uint16_t idx, std::ostream& os,
                          const bool edgeLabels, const bool classic,
                          const bool formatAsPolar) {
  const auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  const auto tolabel =
      (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  if (classic) {
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
    os << tolabel;
  }

  const auto mag = thicknessFromMagnitude(to.w);
  const auto color = colorFromPhase(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\" color=\"" << color
     << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
std::ostream& coloredEdge(const vEdge& from, const vEdge& to,
                          const std::uint16_t idx, std::ostream& os,
                          const bool edgeLabels,
                          [[maybe_unused]] const bool classic,
                          const bool formatAsPolar) {
  const auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  const auto tolabel =
      (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  os << (idx == 0 ? "sw" : "se") << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  const auto mag = thicknessFromMagnitude(to.w);
  const auto color = colorFromPhase(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\" color=\"" << color
     << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
void serialize(const vEdge& basic, std::ostream& os, const bool writeBinary) {
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
          auto& edge = node->p->e.at(i);
          if (edge.isTerminal()) {
            continue;
          }
          if (edge.w.approximatelyZero()) {
            continue;
          }
          if (nodeIndex.contains(edge.p)) {
            continue;
          }

          // non-zero edge to be included
          stack.push(&edge);
        }
        stack.push(node);
        node = &node->p->e[0]; // NOLINT(readability-container-data-pointer)
      }
      node = stack.top();
      stack.pop();

      bool hasChild = false;
      for (auto i = 1U; i < RADIX && !hasChild; ++i) {
        auto& edge = node->p->e.at(i);
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

          // iterate over edges in reverse to guarantee correct processing order
          for (auto i = 0U; i < RADIX; ++i) {
            auto& edge = node->p->e.at(i);
            std::int64_t edgeIdx = edge.isTerminal() ? -1 : nodeIndex[edge.p];
            os.write(reinterpret_cast<const char*>(&edgeIdx),
                     sizeof(decltype(edgeIdx)));
            edge.w.writeBinary(os);
          }
        } else {
          os << nodeIndex[node->p] << " "
             << static_cast<std::size_t>(node->p->v);

          // iterate over edges in reverse to guarantee correct processing order
          for (auto i = 0U; i < RADIX; ++i) {
            os << " (";
            auto& edge = node->p->e.at(i);
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
void serializeMatrix(const mEdge& basic, std::int64_t& idx,
                     std::unordered_map<mNode*, std::int64_t>& nodeIndex,
                     std::unordered_set<mNode*>& visited, std::ostream& os,
                     const bool writeBinary) {
  if (!basic.isTerminal()) {
    for (auto& e : basic.p->e) {
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

      // iterate over edges in reverse to guarantee correct processing order
      for (auto& edge : basic.p->e) {
        std::int64_t edgeIdx = edge.isTerminal() ? -1 : nodeIndex[edge.p];
        os.write(reinterpret_cast<const char*>(&edgeIdx),
                 sizeof(decltype(edgeIdx)));
        edge.w.writeBinary(os);
      }
    } else {
      os << nodeIndex[basic.p] << " " << static_cast<std::size_t>(basic.p->v);

      // iterate over edges in reverse to guarantee correct processing order
      for (auto& edge : basic.p->e) {
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
void serialize(const mEdge& basic, std::ostream& os, const bool writeBinary) {
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
} // namespace dd
