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

#include "ZXDefinitions.hpp"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace zx {

/**
 * @brief Struct to represent a (half-)edge in a ZX-diagram
 */
struct Edge {
  Vertex to;
  EdgeType type;

  Edge() = default;
  Edge(const Vertex t, const EdgeType typo) : to(t), type(typo) {};
  void toggle() {
    if (type == EdgeType::Simple) {
      type = EdgeType::Hadamard;
    } else {
      type = EdgeType::Simple;
    }
  }
};

/**
 * @brief Struct storing all data corresponding to a vertex in a ZX-diagram
 */
struct VertexData {
  Col col;
  Qubit qubit;
  PiExpression phase;
  VertexType type;
};

/**
 * @brief Class to represent a collection of vertices in a ZX-diagram.
 * @details The ZXDiagram class stores vertices in a vector of optional
 * VertexData objects to allow for fast deletion without changing the indices of
 * the other vertices. The Vertices class provides an iterator to iterate over
 * all vertices in the diagram. This avoids the need to iterate over all indices
 * and check if the vertex at that index is valid.
 */
class Vertices {
public:
  explicit Vertices(const std::vector<std::optional<VertexData>>& verts)
      : vertices(verts) {};

  class VertexIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::int32_t;
    using value_type = std::pair<Vertex, const VertexData&>;
    using pointer = value_type*;
    using reference = value_type&;

    explicit VertexIterator(const std::vector<std::optional<VertexData>>& verts)
        : currentPos(verts.begin()), vertices(verts) {
      nextValidVertex();
    }
    VertexIterator(const std::vector<std::optional<VertexData>>& verts,
                   Vertex vertex);

    value_type operator*() const {
      assert(currentPos->has_value());
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      return {v, currentPos->value()};
    }

    // Prefix increment
    VertexIterator operator++();

    // Postfix increment
    VertexIterator operator++(int);

    friend bool operator==(const VertexIterator& a, const VertexIterator& b);
    friend bool operator!=(const VertexIterator& a, const VertexIterator& b);

  private:
    Vertex v = 0;
    std::vector<std::optional<VertexData>>::const_iterator currentPos;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::optional<VertexData>>& vertices;

    void nextValidVertex();
  };

  using iterator = VertexIterator;

  iterator begin() { return VertexIterator(vertices); }
  iterator end() { return {vertices, vertices.size()}; }

private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::optional<VertexData>>& vertices;
};

/**
 * @brief Class to represent a collection of edges in a ZX-diagram.
 * @details The ZXDiagram class stores edges in a vector of vectors of Edge
 * objects. The Edges class provides an iterator to iterate over all edges in
 * the diagram. This avoids the need to iterate over all vertices and check if
 * the vertex has an edge to another vertex.
 */
class Edges {
public:
  Edges(const std::vector<std::vector<Edge>>& edgs,
        const std::vector<std::optional<VertexData>>& verts)
      : edges(edgs), vertices(verts) {};

  /**
   * @brief Class wrapping an iterator to iterate over all edges in a
   * ZX-diagram.
   */
  class EdgeIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::int32_t;
    using value_type = std::pair<Vertex, Vertex>;
    using pointer = value_type*;
    using reference = value_type&;

    EdgeIterator(const std::vector<std::vector<Edge>>& es,
                 const std::vector<std::optional<VertexData>>& verts);

    EdgeIterator(const std::vector<std::vector<Edge>>& es,
                 const std::vector<std::optional<VertexData>>& verts,
                 Vertex vertex);

    value_type operator*() const { return {v, currentPos->to}; }

    // Prefix increment
    EdgeIterator operator++();

    // Postfix increment
    EdgeIterator operator++(int);

    friend bool operator==(const EdgeIterator& a, const EdgeIterator& b);
    friend bool operator!=(const EdgeIterator& a, const EdgeIterator& b);

  private:
    Vertex v;
    std::vector<Edge>::const_iterator currentPos;
    std::vector<std::vector<Edge>>::const_iterator edgesPos;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::vector<Edge>>& edges;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::optional<VertexData>>& vertices;

    void checkNextVertex();
  };

  using iterator = EdgeIterator;

  iterator begin() { return {edges, vertices}; }
  iterator end() { return {edges, vertices, edges.size()}; }

private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::vector<Edge>>& edges;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::optional<VertexData>>& vertices;
};

/**
 * @brief Check whether a PiExpression is a constant multiple of pi
 * @param expr PiExpression to check
 * @return true if the expression is a constant multiple of pi, false otherwise
 */
bool isPauli(const PiExpression& expr);

/**
 * @brief Check whether a PiExpression is a constant multiple of pi/2
 * @param expr PiExpression to check
 * @return true if the expression is a constant multiple of pi/2, false
 * otherwise
 */
bool isClifford(const PiExpression& expr);

/**
 * @brief Check whether a PiExpression is a constant multiple of pi/2 but not a
 * multiple of pi
 * @param expr PiExpression to check
 * @return true if the expression is a constant multiple of pi/2 but not a
 * multiple of pi, false otherwise
 */
bool isProperClifford(const PiExpression& expr);

/**
 * @brief Round phase to the nearest multiple of pi/2. The phase has to be
 * non-symbolic.
 * @param expr PiExpression to round
 * @param tolerance Tolerance for rounding
 */
void roundToClifford(PiExpression& expr, fp tolerance);
} // namespace zx
