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
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include "support/Diagnostics.hpp"

#include "llvm/Support/LogicalResult.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>

namespace dd {

template <class Node, size_t N>
llvm::FailureOr<CachedEdge<Node>>
Package::deserializeNode(const int64_t index, const Qubit v,
                         const std::array<int64_t, N>& edgeIdx,
                         const std::array<ComplexValue, N>& edgeWeight,
                         std::unordered_map<int64_t, Node*>& nodes) {
  if (index < 0 || v >= qubits() || nodes.contains(index)) {
    return ::mqt::emitError("Invalid serialized DD node index or qubit.",
                            ::mqt::ErrorCategory::InvalidArgument);
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
          return ::mqt::emitError(
              "Serialized DD edges must refer to preceding nodes on "
              "lower qubits.",
              ::mqt::ErrorCategory::InvalidArgument);
        }
        edges[i].p = child->second;
      }
      if (!std::isfinite(edgeWeight[i].r) || !std::isfinite(edgeWeight[i].i)) {
        return ::mqt::emitError("Serialized DD weights must be finite.",
                                ::mqt::ErrorCategory::InvalidArgument);
      }
      edges[i].w = edgeWeight[i];
    }
    if constexpr (IsVector<Node>) {
      if (!edges[i].w.exactlyZero() &&
          (edges[i].isTerminal() ? v != 0 : edges[i].p->v + 1 != v)) {
        return ::mqt::emitError(
            "Serialized vector DD edges must follow consecutive qubit "
            "levels.",
            ::mqt::ErrorCategory::InvalidArgument);
      }
    }
  }
  auto r = makeDDNode(v, edges);
  nodes[index] = r.p;
  return r;
}

template <class Node, class Edge, size_t N>
llvm::FailureOr<Edge> Package::deserialize(std::istream& is,
                                           const bool readBinary) {
  if (is.exceptions() != std::ios::goodbit) {
    return ::mqt::emitError(
        "DD deserialization requires a stream without exception flags.",
        ::mqt::ErrorCategory::IO);
  }
  auto result = CachedEdge<Node>::one();
  ComplexValue rootweight{};
  std::unordered_map<int64_t, Node*> nodes;
  const auto invalid = [] {
    return ::mqt::emitError("Invalid or truncated serialized DD.",
                            ::mqt::ErrorCategory::InvalidArgument);
  };
  const auto readInteger = []<typename T>(std::string_view& input, T& value) {
    if (input.empty()) {
      return false;
    }
    const auto parsed =
        std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{}) {
      return false;
    }
    input.remove_prefix(static_cast<size_t>(parsed.ptr - input.data()));
    return true;
  };
  if (readBinary) {
    std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!is || version != SERIALIZATION_VERSION) {
      return invalid();
    }
    rootweight.readBinary(is);
    if (!is || !std::isfinite(rootweight.r) || !std::isfinite(rootweight.i)) {
      return invalid();
    }
    while (true) {
      int64_t index{};
      is.read(reinterpret_cast<char*>(&index), sizeof(index));
      if (!is) {
        if (is.eof() && is.gcount() == 0) {
          break;
        }
        return invalid();
      }
      Qubit wire{};
      is.read(reinterpret_cast<char*>(&wire), sizeof(wire));
      std::array<int64_t, N> indices{};
      std::array<ComplexValue, N> weights{};
      for (size_t i = 0; i < N; ++i) {
        is.read(reinterpret_cast<char*>(&indices[i]), sizeof(indices[i]));
        weights[i].readBinary(is);
      }
      if (!is) {
        return invalid();
      }
      auto node = deserializeNode(index, wire, indices, weights, nodes);
      if (llvm::failed(node)) {
        return llvm::failure();
      }
      result = (*node);
    }
  } else {
    std::string line;
    if (!std::getline(is, line)) {
      return invalid();
    }
    std::string_view text = line;
    uint64_t version{};
    if (!readInteger(text, version) || !text.empty() ||
        version != SERIALIZATION_VERSION) {
      return invalid();
    }
    if (!std::getline(is, line) || line.empty()) {
      return invalid();
    }
    auto weight = ComplexValue::parse(line);
    if (llvm::failed(weight)) {
      return llvm::failure();
    }
    rootweight = (*weight);
    while (std::getline(is, line)) {
      if (line.empty()) {
        continue;
      }
      text = line;
      int64_t index{};
      size_t wire{};
      if (!readInteger(text, index) || index < 0 || !text.starts_with(' ')) {
        return invalid();
      }
      text.remove_prefix(1);
      if (!readInteger(text, wire) || wire >= qubits()) {
        return invalid();
      }
      std::array<int64_t, N> indices{};
      indices.fill(-2);
      std::array<ComplexValue, N> weights{};
      for (size_t i = 0; i < N; ++i) {
        if (!text.starts_with(" (")) {
          return invalid();
        }
        text.remove_prefix(2);
        const auto end = text.find(')');
        if (end == std::string_view::npos) {
          return invalid();
        }
        auto edge = text.substr(0, end);
        text.remove_prefix(end + 1);
        if (edge.empty()) {
          continue;
        }
        if (!readInteger(edge, indices[i]) || !edge.starts_with(' ')) {
          return invalid();
        }
        edge.remove_prefix(1);
        if (edge.empty()) {
          return invalid();
        }
        auto edgeWeight = ComplexValue::parse(edge);
        if (llvm::failed(edgeWeight)) {
          return llvm::failure();
        }
        weights[i] = (*edgeWeight);
      }
      while (text.starts_with(' ')) {
        text.remove_prefix(1);
      }
      if (!text.empty() && !text.starts_with('#')) {
        return invalid();
      }
      auto node = deserializeNode(index, static_cast<Qubit>(wire), indices,
                                  weights, nodes);
      if (llvm::failed(node)) {
        return llvm::failure();
      }
      result = (*node);
    }
  }
  if (is.bad()) {
    return ::mqt::emitError("Cannot read serialized DD.",
                            ::mqt::ErrorCategory::IO);
  }
  return cn.lookup(CachedEdge<Node>{result.p, result.w * rootweight});
}

template <class Node, class Edge>
llvm::FailureOr<Edge> Package::deserialize(const std::string& inputFilename,
                                           const bool readBinary) {
  auto input = std::ifstream(inputFilename, std::ios::binary);
  if (!input) {
    return ::mqt::emitError("Cannot open serialized file: " + inputFilename,
                            ::mqt::ErrorCategory::IO);
  }
  return deserialize<Node>(input, readBinary);
}

template llvm::FailureOr<vEdge> Package::deserialize<vNode>(std::istream&,
                                                            bool);
template llvm::FailureOr<mEdge> Package::deserialize<mNode>(std::istream&,
                                                            bool);
template llvm::FailureOr<vEdge> Package::deserialize<vNode>(const std::string&,
                                                            bool);
template llvm::FailureOr<mEdge> Package::deserialize<mNode>(const std::string&,
                                                            bool);

} // namespace dd
