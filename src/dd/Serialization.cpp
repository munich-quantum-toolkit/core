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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

using llvm::createStringError;
using llvm::Expected;
using llvm::StringRef;

namespace dd {

namespace {
bool splitWeight(StringRef text, StringRef& real, StringRef& imaginary) {
  static constexpr auto NUMBER =
      R"(([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?)";
  static const llvm::Regex IMAGINARY_ONLY("^ ?[+-]? ?(" + std::string(NUMBER) +
                                          ")?[iI]$");
  static const llvm::Regex COMPLEX("^([+-]?" + std::string(NUMBER) +
                                   ")( ?[+-] ?(" + NUMBER + ")?[iI])?$");
  if (IMAGINARY_ONLY.match(text)) {
    real = {};
    imaginary = text;
    return true;
  }
  llvm::SmallVector<StringRef, 10> parts;
  if (!COMPLEX.match(text, &parts)) {
    return false;
  }
  real = parts[1];
  imaginary = parts[5];
  return true;
}
} // namespace

template <class Node, size_t N>
Expected<CachedEdge<Node>>
Package::deserializeNode(const int64_t index, const Qubit v,
                         const std::array<int64_t, N>& edgeIdx,
                         const std::array<ComplexValue, N>& edgeWeight,
                         std::unordered_map<int64_t, Node*>& nodes) {
  if (index < 0 || v >= qubits() || nodes.contains(index)) {
    return createStringError(
        std::make_error_code(std::errc::state_not_recoverable),
        "Invalid serialized DD node index or qubit.");
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
          return createStringError(
              std::make_error_code(std::errc::state_not_recoverable),
              "Serialized DD edges must refer to preceding nodes on "
              "lower qubits.");
        }
        edges[i].p = child->second;
      }
      if (!std::isfinite(edgeWeight[i].r) || !std::isfinite(edgeWeight[i].i)) {
        return createStringError(
            std::make_error_code(std::errc::state_not_recoverable),
            "Serialized DD weights must be finite.");
      }
      edges[i].w = edgeWeight[i];
    }
    if constexpr (IsVector<Node>) {
      if (!edges[i].w.exactlyZero() &&
          (edges[i].isTerminal() ? v != 0 : edges[i].p->v + 1 != v)) {
        return createStringError(
            std::make_error_code(std::errc::state_not_recoverable),
            "Serialized vector DD edges must follow consecutive qubit "
            "levels.");
      }
    }
  }
  auto r = makeDDNode(v, edges);
  nodes[index] = r.p;
  return r;
}

template <class Node, class Edge, size_t N>
Expected<Edge> Package::deserialize(StringRef data, const bool readBinary) {
  auto result = CachedEdge<Node>::one();
  ComplexValue rootweight{};
  std::unordered_map<int64_t, Node*> nodes;
  const auto invalid = [] {
    return createStringError(std::errc::state_not_recoverable,
                             "Invalid or truncated serialized DD.");
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
  const auto readLine = [&data](StringRef& line) {
    if (data.empty()) {
      return false;
    }
    const auto parts = data.split('\n');
    line = parts.first;
    data = parts.second;
    return true;
  };
  if (readBinary) {
    std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
    if (!read(version) || version != SERIALIZATION_VERSION ||
        !read(rootweight.r) || !read(rootweight.i) ||
        !std::isfinite(rootweight.r) || !std::isfinite(rootweight.i)) {
      return invalid();
    }
    while (!data.empty()) {
      int64_t index{};
      Qubit wire{};
      std::array<int64_t, N> indices{};
      std::array<ComplexValue, N> weights{};
      if (!read(index) || !read(wire)) {
        return invalid();
      }
      for (size_t i = 0; i < N; ++i) {
        if (!read(indices[i]) || !read(weights[i].r) || !read(weights[i].i)) {
          return invalid();
        }
      }
      auto node = deserializeNode(index, wire, indices, weights, nodes);
      if (!node) {
        return node.takeError();
      }
      result = (*node);
    }
  } else {
    StringRef line;
    if (!readLine(line)) {
      return invalid();
    }
    auto versionText = line.ltrim(" \t\n\v\f\r");
    const bool hadPlus = versionText.consume_front("+");
    if (versionText.empty() || (hadPlus && versionText.starts_with("-"))) {
      return createStringError(std::errc::invalid_argument,
                               "Invalid serialized DD version.");
    }
    int version{};
    const auto parsed =
        std::from_chars(versionText.begin(), versionText.end(), version);
    if (parsed.ec != std::errc{}) {
      return createStringError(parsed.ec, "Invalid serialized DD version.");
    }
    if (parsed.ptr != versionText.end() ||
        std::cmp_not_equal(version, SERIALIZATION_VERSION)) {
      return invalid();
    }

    static const llvm::Regex LINE_REGEX([] {
      std::string pattern = "^([0-9]+) ([0-9]+)";
      for (size_t i = 0; i < N; ++i) {
        pattern += R"( \(((-?[0-9]+) ([^()]*))?\))";
      }
      return pattern + " *(#[^\r]*)?$";
    }());
    llvm::SmallVector<StringRef, 16> match;
    StringRef real;
    StringRef imaginary;
    if (!readLine(line) || !splitWeight(line, real, imaginary)) {
      return invalid();
    }
    if (auto error = rootweight.fromString(real, imaginary)) {
      return error;
    }
    while (readLine(line)) {
      if (line.empty()) {
        continue;
      }
      if (!LINE_REGEX.match(line, &match)) {
        return invalid();
      }
      int64_t index{};
      uint64_t wire{};
      if (match[1].getAsInteger(10, index) || match[2].getAsInteger(10, wire)) {
        return createStringError(std::errc::result_out_of_range,
                                 "Serialized DD index is out of range.");
      }
      if (wire >= qubits()) {
        return invalid();
      }
      std::array<int64_t, N> indices{};
      indices.fill(-2);
      std::array<ComplexValue, N> weights{};
      for (size_t i = 0, group = 3; i < N; ++i, group += 3) {
        if (match[group].empty()) {
          continue;
        }
        if (!splitWeight(match[group + 2], real, imaginary)) {
          return invalid();
        }
        if (match[group + 1].getAsInteger(10, indices[i])) {
          return createStringError(std::errc::result_out_of_range,
                                   "Serialized DD edge index is out of range.");
        }
        if (auto error = weights[i].fromString(real, imaginary)) {
          return error;
        }
      }
      auto node = deserializeNode(index, static_cast<Qubit>(wire), indices,
                                  weights, nodes);
      if (!node) {
        return node.takeError();
      }
      result = *node;
    }
  }
  return cn.lookup(CachedEdge<Node>{result.p, result.w * rootweight});
}

template <class Node, class Edge>
Expected<Edge> Package::deserializeFile(StringRef inputFilename,
                                        const bool readBinary) {
  auto input = llvm::MemoryBuffer::getFile(inputFilename);
  if (!input) {
    return createStringError(
        std::errc::io_error, "Cannot open serialized file: %s: %s",
        inputFilename.str().c_str(), input.getError().message().c_str());
  }
  return deserialize<Node>((*input)->getBuffer(), readBinary);
}

template Expected<vEdge> Package::deserialize<vNode>(StringRef, bool);
template Expected<mEdge> Package::deserialize<mNode>(StringRef, bool);
template Expected<vEdge> Package::deserializeFile<vNode>(StringRef, bool);
template Expected<mEdge> Package::deserializeFile<mNode>(StringRef, bool);

} // namespace dd
