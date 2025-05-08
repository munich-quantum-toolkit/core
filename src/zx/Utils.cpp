/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "zx/Utils.hpp"

#include "zx/ZXDefinitions.hpp"

#include <optional>
#include <vector>

namespace zx {
Vertices::VertexIterator::VertexIterator(
    const std::vector<std::optional<VertexData>>& verts, const Vertex vertex)
    : v(vertex), currentPos(verts.begin()), vertices(verts) {
  if (v >= vertices.size()) {
    currentPos = vertices.end();
    v = vertices.size();
  } else {
    currentPos = vertices.begin() + static_cast<int>(v);
    nextValidVertex();
  }
}
// Prefix increment
Vertices::VertexIterator Vertices::VertexIterator::operator++() {
  Vertices::VertexIterator it = *this;
  ++currentPos;
  ++v;
  nextValidVertex();
  return it;
}

// Postfix increment
Vertices::VertexIterator Vertices::VertexIterator::operator++(int) {
  ++currentPos;
  ++v;
  nextValidVertex();
  return *this;
}

bool operator==(const Vertices::VertexIterator& a,
                const Vertices::VertexIterator& b) {
  return a.currentPos == b.currentPos;
}
bool operator!=(const Vertices::VertexIterator& a,
                const Vertices::VertexIterator& b) {
  return !(a == b);
}

void Vertices::VertexIterator::nextValidVertex() {
  while (currentPos != vertices.end() && !currentPos->has_value()) {
    ++v;
    ++currentPos;
  }
}

Edges::EdgeIterator::EdgeIterator(
    const std::vector<std::vector<Edge>>& es,
    const std::vector<std::optional<VertexData>>& verts)
    : v(0), currentPos(es[0].begin()), edgesPos(es.begin()), edges(es),
      vertices(verts) {
  if (!vertices.empty()) {
    while (v < edges.size() && !vertices[v].has_value()) {
      ++v;
    }
    if (v < edges.size()) {
      currentPos = edges[v].begin();
      edgesPos = edges.begin() + static_cast<int>(v);
      checkNextVertex();
    } else {
      currentPos = edges.back().end();
      edgesPos = edges.end();
    }
  } else {
    currentPos = edges.back().end();
    edgesPos = edges.end();
    v = edges.size();
  }
}

Edges::EdgeIterator::EdgeIterator(
    const std::vector<std::vector<Edge>>& es,
    const std::vector<std::optional<VertexData>>& verts, const Vertex vertex)
    : v(vertex), edges(es), vertices(verts) {
  if (v >= edges.size()) {
    currentPos = edges.back().end();
    edgesPos = edges.end();
    this->v = edges.size();
  } else {
    currentPos = edges[v].begin();
    edgesPos = edges.begin() + static_cast<int>(v);
  }
}

// Prefix increment
Edges::EdgeIterator Edges::EdgeIterator::operator++() {
  Edges::EdgeIterator it = *this;
  currentPos++;
  checkNextVertex();
  return it;
}

void Edges::EdgeIterator::checkNextVertex() {
  while (currentPos != edges[v].end() &&
         currentPos->to < v) { // make sure to not iterate over an edge twice
    ++currentPos;
  }

  while (currentPos == edges[v].end() && v < edges.size()) {
    ++v;
    while (v < edges.size() && !vertices[v].has_value()) {
      ++v;
    }

    if (v == edges.size()) {
      currentPos = edges.back().end();
      edgesPos = edges.end();
      --v;
      return;
    }
    currentPos = edges[v].begin();
    edgesPos = edges.begin() + static_cast<int>(v);
    while (currentPos != edges[v].end() &&
           currentPos->to < v) { // make sure to not iterate over an edge twice
      ++currentPos;
    }
  }
}
// Postfix increment
Edges::EdgeIterator Edges::EdgeIterator::operator++(int) {
  ++currentPos;
  checkNextVertex();
  return *this;
}

bool operator==(const Edges::EdgeIterator& a, const Edges::EdgeIterator& b) {
  return a.edgesPos == b.edgesPos && a.currentPos == b.currentPos;
}
bool operator!=(const Edges::EdgeIterator& a, const Edges::EdgeIterator& b) {
  return !(a == b);
}

bool isPauli(const PiExpression& expr) {
  return expr.isConstant() && expr.getConst().isInteger();
}
bool isClifford(const PiExpression& expr) {
  return expr.isConstant() &&
         (expr.getConst().isInteger() || expr.getConst().getDenom() == 2);
}
bool isProperClifford(const PiExpression& expr) {
  return expr.isConstant() && expr.getConst().getDenom() == 2;
}

void roundToClifford(PiExpression& expr, const fp tolerance) {
  if (!expr.isConstant()) {
    return;
  }

  if (expr.getConst().isCloseDivPi(0, tolerance)) {
    expr.setConst(PiRational(0, 1));
  } else if (expr.getConst().isCloseDivPi(0.5, tolerance)) {
    expr.setConst(PiRational(1, 2));
  } else if (expr.getConst().isCloseDivPi(-0.5, tolerance)) {
    expr.setConst(PiRational(-1, 2));
  } else if (expr.getConst().isCloseDivPi(1, tolerance)) {
    expr.setConst(PiRational(1, 1));
  }
}
} // namespace zx
