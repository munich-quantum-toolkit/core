/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/CachedEdge.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <regex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace dd {

template <class Node, std::size_t N>
CachedEdge<Node>
Package::deserializeNode(const std::int64_t index, const Qubit v,
                         std::array<std::int64_t, N>& edgeIdx,
                         const std::array<ComplexValue, N>& edgeWeight,
                         std::unordered_map<std::int64_t, Node*>& nodes) {
  if (index == -1) {
    return CachedEdge<Node>::zero();
  }

  std::array<CachedEdge<Node>, N> edges{};
  for (auto i = 0U; i < N; ++i) {
    if (edgeIdx[i] == -2) {
      edges[i] = CachedEdge<Node>::zero();
    } else {
      if (edgeIdx[i] == -1) {
        edges[i] = CachedEdge<Node>::one();
      } else {
        edges[i].p = nodes[edgeIdx[i]];
      }
      edges[i].w = edgeWeight[i];
    }
  }
  // reset
  edgeIdx.fill(-2);

  auto r = makeDDNode(v, edges);
  nodes[index] = r.p;
  return r;
}

template <class Node, class Edge, std::size_t N>
Edge Package::deserialize(std::istream& is, const bool readBinary) {
  auto result = CachedEdge<Node>::one();
  ComplexValue rootweight{};

  std::unordered_map<std::int64_t, Node*> nodes{};
  std::int64_t nodeIndex{};
  Qubit v{};
  std::array<ComplexValue, N> edgeWeights{};
  std::array<std::int64_t, N> edgeIndices{};
  edgeIndices.fill(-2);

  if (readBinary) {
    std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
    is.read(reinterpret_cast<char*>(&version),
            sizeof(decltype(SERIALIZATION_VERSION)));
    if (version != SERIALIZATION_VERSION) {
      throw std::runtime_error(
          "Wrong Version of serialization file version. version of file: " +
          std::to_string(version) +
          "; current version: " + std::to_string(SERIALIZATION_VERSION));
    }

    if (!is.eof()) {
      rootweight.readBinary(is);
    }

    while (is.read(reinterpret_cast<char*>(&nodeIndex),
                   sizeof(decltype(nodeIndex)))) {
      is.read(reinterpret_cast<char*>(&v), sizeof(decltype(v)));
      for (std::size_t i = 0U; i < N; i++) {
        is.read(reinterpret_cast<char*>(&edgeIndices[i]),
                sizeof(decltype(edgeIndices[i])));
        edgeWeights[i].readBinary(is);
      }
      result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
    }
  } else {
    std::string version;
    std::getline(is, version);
    if (std::cmp_not_equal(std::stoi(version), SERIALIZATION_VERSION)) {
      throw std::runtime_error(
          "Wrong Version of serialization file version. version of file: " +
          version +
          "; current version: " + std::to_string(SERIALIZATION_VERSION));
    }

    const std::string complexRealRegex =
        R"(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![ \d\.]*(?:[eE][+-])?\d*[iI]))?)";
    const std::string complexImagRegex =
        R"(( ?[+-]? ?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?[iI])?)";
    const std::string edgeRegex =
        " \\(((-?\\d+) (" + complexRealRegex + complexImagRegex + "))?\\)";
    const std::regex complexWeightRegex(complexRealRegex + complexImagRegex);

    std::string lineConstruct = "(\\d+) (\\d+)";
    for (std::size_t i = 0U; i < N; ++i) {
      lineConstruct += "(?:" + edgeRegex + ")";
    }
    lineConstruct += " *(?:#.*)?";
    const std::regex lineRegex(lineConstruct);
    std::smatch m;

    std::string line;
    if (std::getline(is, line)) {
      if (!std::regex_match(line, m, complexWeightRegex)) {
        throw std::runtime_error("Regex did not match second line: " + line);
      }
      rootweight.fromString(m.str(1), m.str(2));
    }

    while (std::getline(is, line)) {
      if (line.empty() || line.size() == 1) {
        continue;
      }

      if (!std::regex_match(line, m, lineRegex)) {
        throw std::runtime_error("Regex did not match line: " + line);
      }

      // match 1: node_idx
      // match 2: qubit_idx

      // repeats for every edge
      // match 3: edge content
      // match 4: edge_target_idx
      // match 5: real + imag (without i)
      // match 6: real
      // match 7: imag (without i)
      nodeIndex = std::stoi(m.str(1));
      v = static_cast<Qubit>(std::stoi(m.str(2)));

      for (auto edgeIdx = 3U, i = 0U; i < N; i++, edgeIdx += 5) {
        if (m.str(edgeIdx).empty()) {
          continue;
        }

        edgeIndices[i] = std::stoi(m.str(edgeIdx + 1));
        edgeWeights[i].fromString(m.str(edgeIdx + 3), m.str(edgeIdx + 4));
      }

      result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
    }
  }
  return cn.lookup(CachedEdge<Node>{result.p, result.w * rootweight});
}

template <class Node, class Edge>
Edge Package::deserialize(const std::string& inputFilename,
                          const bool readBinary) {
  auto ifs = std::ifstream(inputFilename, std::ios::binary);

  if (!ifs.good()) {
    throw std::invalid_argument("Cannot open serialized file: " +
                                inputFilename);
  }

  return deserialize<Node>(ifs, readBinary);
}

template vEdge Package::deserialize<vNode>(std::istream&, bool);
template mEdge Package::deserialize<mNode>(std::istream&, bool);
template vEdge Package::deserialize<vNode>(const std::string&, bool);
template mEdge Package::deserialize<mNode>(const std::string&, bool);

} // namespace dd
