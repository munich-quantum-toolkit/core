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
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
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
  if (index < 0 || v >= qubits() || nodes.contains(index)) {
    throw std::runtime_error("Invalid serialized DD node index or qubit.");
  }

  std::array<CachedEdge<Node>, N> edges{};
  for (auto i = 0U; i < N; ++i) {
    if (edgeIdx[i] == -2) {
      edges[i] = CachedEdge<Node>::zero();
    } else {
      if (edgeIdx[i] == -1) {
        edges[i] = CachedEdge<Node>::one();
      } else {
        const auto child = nodes.find(edgeIdx[i]);
        if (child == nodes.end() ||
            (!Node::isTerminal(child->second) && child->second->v >= v)) {
          throw std::runtime_error(
              "Serialized DD edges must refer to preceding nodes on lower "
              "qubits.");
        }
        edges[i].p = child->second;
      }
      if (!std::isfinite(edgeWeight[i].r) || !std::isfinite(edgeWeight[i].i)) {
        throw std::runtime_error("Serialized DD weights must be finite.");
      }
      edges[i].w = edgeWeight[i];
    }
    if constexpr (IsVector<Node>) {
      if (!edges[i].w.exactlyZero() &&
          (edges[i].isTerminal() ? v != 0 : edges[i].p->v + 1 != v)) {
        throw std::runtime_error(
            "Serialized vector DD edges must follow consecutive qubit "
            "levels.");
      }
    }
  }
  // reset
  edgeIdx.fill(-2);

  auto r = makeDDNode(v, edges);
  nodes[index] = r.p;
  return r;
}

template <class Node, class Edge, size_t N>
Edge Package::deserialize(llvm::StringRef data, const bool readBinary) {
  auto result = CachedEdge<Node>::one();
  ComplexValue rootweight{};
  std::unordered_map<int64_t, Node*> nodes;
  const auto invalid = [] {
    return std::runtime_error("Invalid or truncated serialized DD.");
  };
  const auto read = [&data]<class T>(T& value) {
    if (data.size() < sizeof(T)) {
      return false;
    }
    const auto bytes = data.take_front(sizeof(T));
    std::memcpy(&value, bytes.data(), bytes.size());
    data = data.drop_front(sizeof(T));
    return true;
  };
  const auto readLine = [&data](std::string& line) {
    if (data.empty()) {
      return false;
    }
    const auto parts = data.split('\n');
    line = parts.first.str();
    data = parts.second;
    return true;
  };
  if (readBinary) {
    std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
    if (!read(version) || version != SERIALIZATION_VERSION ||
        !read(rootweight.r) || !read(rootweight.i) ||
        !std::isfinite(rootweight.r) || !std::isfinite(rootweight.i)) {
      throw invalid();
    }
    while (!data.empty()) {
      int64_t index{};
      Qubit wire{};
      std::array<int64_t, N> indices{};
      std::array<ComplexValue, N> weights{};
      if (!read(index) || !read(wire)) {
        throw invalid();
      }
      for (size_t i = 0; i < N; ++i) {
        if (!read(indices[i]) || !read(weights[i].r) || !read(weights[i].i)) {
          throw invalid();
        }
      }
      result = deserializeNode(index, wire, indices, weights, nodes);
    }
  } else {
    std::int64_t nodeIndex{};
    Qubit v{};
    std::array<ComplexValue, N> edgeWeights{};
    std::array<std::int64_t, N> edgeIndices{};
    edgeIndices.fill(-2);
    std::string version;
    if (!readLine(version)) {
      throw std::runtime_error("Missing serialized DD version.");
    }
    size_t versionEnd = 0;
    if (std::cmp_not_equal(std::stoi(version, &versionEnd),
                           SERIALIZATION_VERSION) ||
        versionEnd != version.size()) {
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
    static const std::regex COMPLEX_WEIGHT_REGEX(complexRealRegex +
                                                 complexImagRegex);

    std::string lineConstruct = "(\\d+) (\\d+)";
    for (std::size_t i = 0U; i < N; ++i) {
      lineConstruct += "(?:" + edgeRegex + ")";
    }
    lineConstruct += " *(?:#.*)?";
    static const std::regex LINE_REGEX(lineConstruct);
    std::smatch m;

    std::string line;
    if (!readLine(line) || line.empty()) {
      throw std::runtime_error("Missing serialized DD root weight.");
    }
    if (!std::regex_match(line, m, COMPLEX_WEIGHT_REGEX)) {
      throw std::runtime_error("Regex did not match second line: " + line);
    }
    rootweight.fromString(m.str(1), m.str(2));

    while (readLine(line)) {
      if (line.empty()) {
        continue;
      }

      if (!std::regex_match(line, m, LINE_REGEX)) {
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
      nodeIndex = std::stoll(m.str(1));
      const auto qubit = std::stoull(m.str(2));
      if (qubit >= qubits()) {
        throw std::runtime_error("Invalid serialized DD qubit.");
      }
      v = static_cast<Qubit>(qubit);

      for (auto edgeIdx = 3U, i = 0U; i < N; i++, edgeIdx += 5) {
        if (m.str(edgeIdx).empty()) {
          continue;
        }

        if (m.str(edgeIdx + 2).empty()) {
          throw std::runtime_error("Missing serialized DD edge weight.");
        }
        edgeIndices[i] = std::stoll(m.str(edgeIdx + 1));
        edgeWeights[i].fromString(m.str(edgeIdx + 3), m.str(edgeIdx + 4));
      }

      result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
    }
  }
  if (!std::isfinite(rootweight.r) || !std::isfinite(rootweight.i)) {
    throw std::runtime_error("Serialized DD weights must be finite.");
  }
  return cn.lookup(CachedEdge<Node>{result.p, result.w * rootweight});
}

template <class Node, class Edge>
Edge Package::deserializeFile(llvm::StringRef inputFilename,
                              const bool readBinary) {
  auto input = llvm::MemoryBuffer::getFile(inputFilename);
  if (!input) {
    throw std::invalid_argument(
        "Cannot open serialized file: " + inputFilename.str() + ": " +
        input.getError().message());
  }
  return deserialize<Node>((*input)->getBuffer(), readBinary);
}

template vEdge Package::deserialize<vNode>(llvm::StringRef, bool);
template mEdge Package::deserialize<mNode>(llvm::StringRef, bool);
template vEdge Package::deserializeFile<vNode>(llvm::StringRef, bool);
template mEdge Package::deserializeFile<mNode>(llvm::StringRef, bool);

} // namespace dd
