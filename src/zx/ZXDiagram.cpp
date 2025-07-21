/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "zx/ZXDiagram.hpp"

#include "ir/operations/Expression.hpp"
#include "zx/Rational.hpp"
#include "zx/Utils.hpp"
#include "zx/ZXDefinitions.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace zx {

ZXDiagram::ZXDiagram(const std::size_t nqubits) {
  auto qubitVertices = initGraph(nqubits);
  closeGraph(qubitVertices);
}

void ZXDiagram::addEdge(const Vertex from, const Vertex to,
                        const EdgeType type) {
  edges[from].emplace_back(to, type);
  edges[to].emplace_back(from, type);
  ++nedges;
}

void ZXDiagram::addEdgeParallelAware(const Vertex from, const Vertex to,
                                     const EdgeType eType) {
  if (from == to) {
    if (type(from) != VertexType::Boundary && eType == EdgeType::Hadamard) {
      addPhase(from, PiExpression(PiRational(1, 1)));
    }
    return;
  }

  const auto edgeIt = getEdgePtr(from, to);

  if (edgeIt == edges[from].end()) {
    addEdge(from, to, eType);
    return;
  }

  if (type(from) == VertexType::Boundary || type(to) == VertexType::Boundary) {
    return;
  }

  if (type(from) == type(to)) {
    if (edgeIt->type == EdgeType::Hadamard && eType == EdgeType::Hadamard) {
      edges[from].erase(edgeIt);
      removeHalfEdge(to, from);
      --nedges;
    } else if (edgeIt->type == EdgeType::Hadamard &&
               eType == EdgeType::Simple) {
      edgeIt->type = EdgeType::Simple;
      getEdgePtr(to, from)->toggle();
      addPhase(from, PiExpression(PiRational(1, 1)));
    } else if (edgeIt->type == EdgeType::Simple &&
               eType == EdgeType::Hadamard) {
      addPhase(from, PiExpression(PiRational(1, 1)));
    }
  } else {
    if (edgeIt->type == EdgeType::Simple && eType == EdgeType::Simple) {
      edges[from].erase(edgeIt);
      removeHalfEdge(to, from);
      --nedges;
    } else if (edgeIt->type == EdgeType::Hadamard &&
               eType == EdgeType::Simple) {
      addPhase(from, PiExpression(PiRational(1, 1)));
    } else if (edgeIt->type == EdgeType::Simple &&
               eType == EdgeType::Hadamard) {
      edgeIt->type = EdgeType::Hadamard;
      getEdgePtr(to, from)->toggle();
      addPhase(from, PiExpression(PiRational(1, 1)));
    }
  }
}

void ZXDiagram::removeEdge(const Vertex from, const Vertex to) {
  removeHalfEdge(from, to);
  removeHalfEdge(to, from);
  --nedges;
}

void ZXDiagram::removeHalfEdge(const Vertex from, const Vertex to) {
  auto& incident = edges[from];
  incident.erase(std::remove_if(incident.begin(), incident.end(),
                                [&](auto& edge) { return edge.to == to; }),
                 incident.end());
}

Vertex ZXDiagram::addVertex(const VertexData& data) {
  ++nvertices;
  if (!deleted.empty()) {
    const auto v = deleted.back();
    deleted.pop_back();
    vertices[v] = data;
    edges[v].clear();
    return v;
  }
  vertices.emplace_back(data);
  edges.emplace_back();

  return nvertices - 1;
}

Vertex ZXDiagram::addVertex(const Qubit qubit, const Col col,
                            const PiExpression& phase, const VertexType type) {
  return addVertex({col, qubit, phase, type});
}

void ZXDiagram::addQubit() {
  auto in = addVertex(static_cast<zx::Qubit>(getNQubits()) + 1, 0,
                      PiExpression(), VertexType::Boundary);
  auto out = addVertex(static_cast<zx::Qubit>(getNQubits()) + 1, 0,
                       PiExpression(), VertexType::Boundary);
  inputs.emplace_back(in);
  outputs.emplace_back(out);
}
void ZXDiagram::addQubits(const Qubit n) {
  for (zx::Qubit i = 0; i < n; ++i) {
    addQubit();
  }
}

void ZXDiagram::removeVertex(const Vertex toRemove) {
  deleted.push_back(toRemove);
  vertices[toRemove].reset();
  --nvertices;

  for (const auto& [to, _] : incidentEdges(toRemove)) {
    removeHalfEdge(to, toRemove);
    --nedges;
  }
}

[[nodiscard]] bool ZXDiagram::connected(const Vertex from,
                                        const Vertex to) const {
  if (isDeleted(from) || isDeleted(to)) {
    return false;
  }

  const auto& incident = edges[from];
  const auto edge = std::find_if(incident.begin(), incident.end(),
                                 [&](const auto& e) { return e.to == to; });
  return edge != incident.end();
}

[[nodiscard]] std::optional<Edge> ZXDiagram::getEdge(const Vertex from,
                                                     const Vertex to) const {
  std::optional<Edge> ret;
  const auto& incident = edges[from];
  const auto edge = std::find_if(incident.begin(), incident.end(),
                                 [&](const auto& e) { return e.to == to; });
  if (edge != incident.end()) {
    ret = *edge;
  }
  return ret;
}

std::vector<Edge>::iterator ZXDiagram::getEdgePtr(const Vertex from,
                                                  const Vertex to) {
  auto& incident = edges[from];
  return std::find_if(incident.begin(), incident.end(),
                      [&](const auto& e) { return e.to == to; });
}

[[nodiscard]] std::vector<std::pair<Vertex, const VertexData&>>
ZXDiagram::getVertices() const {
  Vertices verts(vertices);

  return {verts.begin(), verts.end()};
}

[[nodiscard]] std::vector<std::pair<Vertex, Vertex>>
ZXDiagram::getEdges() const {
  Edges es(edges, vertices);
  return {es.begin(), es.end()};
}

bool ZXDiagram::isInput(const Vertex v) const {
  return std::find(inputs.begin(), inputs.end(), v) != inputs.end();
}
bool ZXDiagram::isOutput(const Vertex v) const {
  return std::find(outputs.begin(), outputs.end(), v) != outputs.end();
}

void ZXDiagram::toGraphlike() {
  const auto nverts = vertices.size();
  for (Vertex v = 0U; v < nverts; ++v) {
    auto& vertex = vertices[v];
    if (!vertex.has_value()) {
      continue;
    }
    if (vertex.value().type == VertexType::X) {
      for (auto& edge : edges[v]) {
        edge.toggle();
        // toggle corresponding edge in other direction
        getEdgePtr(edge.to, v)->toggle();
      }

      vertex.value().type = VertexType::Z;
    }
  }
}

[[nodiscard]] ZXDiagram ZXDiagram::adjoint() const {
  ZXDiagram copy = *this;
  copy.invert();
  return copy;
}

ZXDiagram& ZXDiagram::invert() {
  const auto h = inputs;
  inputs = outputs;
  outputs = h;

  for (auto& data : vertices) {
    if (data.has_value()) {
      data.value().phase = -data.value().phase;
    }
  }
  return *this;
}

ZXDiagram& ZXDiagram::concat(const ZXDiagram& rhs) {
  if (rhs.getNQubits() != this->getNQubits()) {
    throw ZXException(
        "Cannot concatenate Diagrams with differing number of qubits!");
  }

  std::unordered_map<Vertex, Vertex> newVs;
  const auto nverts = rhs.vertices.size();
  for (std::size_t i = 0; i < nverts; ++i) {
    const auto& vertex = rhs.vertices[i];
    if (!vertex.has_value() || rhs.isInput(i)) {
      continue;
    }

    const auto newV = addVertex(vertex.value());
    newVs[i] = newV;
  }

  for (std::size_t i = 0; i < nverts; ++i) { // add new edges
    if (!rhs.vertices[i].has_value() || rhs.isInput(i)) {
      continue;
    }

    for (const auto& [to, type] : rhs.edges[i]) {
      if (!rhs.isInput(to)) {
        if (i < to) { // make sure not to add edge twice
          addEdge(newVs[i], newVs[to], type);
        }
      } else {
        const auto outV = outputs[static_cast<std::size_t>(rhs.qubit(to))];
        for (const auto& [interior_v, interior_type] :
             edges[outV]) { // redirect edges going to outputs
          if (interior_type == type) {
            addEdge(interior_v, newVs[i], EdgeType::Simple);
          } else {
            addEdge(interior_v, newVs[i], EdgeType::Hadamard);
          }
        }
      }
    }
  } // add new edges

  const auto nOutputs = outputs.size();
  for (size_t i = 0; i < nOutputs; ++i) {
    removeVertex(outputs[i]);
    outputs[i] = newVs[rhs.outputs[i]];
  }

  this->addGlobalPhase(-rhs.globalPhase);
  return *this;
}

bool ZXDiagram::isIdentity() const {
  if (nedges != inputs.size() || !globalPhase.isZero()) {
    return false;
  }

  const auto nInputs = inputs.size();
  for (size_t i = 0; i < nInputs; ++i) {
    if (!connected(inputs[i], outputs[i])) {
      return false;
    }
  }
  return true;
}

std::vector<Vertex> ZXDiagram::initGraph(const std::size_t nqubits) {
  std::vector<Vertex> qubitVertices(nqubits, 0);

  const auto nVerts = qubitVertices.size();
  for (size_t i = 0; i < nVerts; ++i) {
    const auto v = addVertex(
        {1, static_cast<Qubit>(i), PiExpression(), VertexType::Boundary});
    qubitVertices[i] = v;
    inputs.push_back(v);
  }

  return qubitVertices;
}

void ZXDiagram::closeGraph(const std::vector<Vertex>& qubitVertices) {
  for (const Vertex v : qubitVertices) {
    const auto& vData = vertices[v];
    if (!vData.has_value()) {
      continue;
    }

    const Vertex newV = addVertex(
        {vData->col + 1, vData->qubit, PiExpression(), VertexType::Boundary});
    addEdge(v, newV);
    outputs.push_back(newV);
  }
}

void ZXDiagram::makeAncilla(const Qubit qubit) { makeAncilla(qubit, qubit); }

void ZXDiagram::makeAncilla(const Qubit in, const Qubit out) {
  const auto inV = inputs[static_cast<std::size_t>(in)];
  const auto outV = outputs[static_cast<std::size_t>(out)];
  inputs.erase(inputs.begin() + in);
  outputs.erase(outputs.begin() + out);

  setType(inV, VertexType::X);
  setType(outV, VertexType::X);
}

void ZXDiagram::approximateCliffords(const fp tolerance) {
  for (auto& v : vertices) {
    if (v.has_value()) {
      roundToClifford(v.value().phase, tolerance);
    }
  }
}

void ZXDiagram::removeDisconnectedSpiders() {
  auto connectedToBoundary = [this](const Vertex v) {
    std::unordered_set<Vertex> visited{};
    std::vector<Vertex> stack{};
    stack.push_back(v);

    while (!stack.empty()) {
      auto w = stack.back();
      stack.pop_back();

      if (visited.contains(w)) {
        continue;
      }

      visited.emplace(w);

      if (isInput(w) || isOutput(w)) {
        return true;
      }

      for (const auto [to, _] : incidentEdges(w)) {
        stack.push_back(to);
      }
    }
    return false;
  };

  const auto nVerts = vertices.size();
  for (Vertex v = 0; v < nVerts; ++v) {
    if (!isDeleted(v) && !connectedToBoundary(v)) {
      removeVertex(v);
    }
  }
}

void ZXDiagram::addGlobalPhase(const PiExpression& phase) {
  globalPhase += phase;
}

gf2Mat ZXDiagram::getAdjMat() const {
  gf2Mat adjMat{nvertices, gf2Vec(nvertices, false)};
  for (const auto& [from, to] : getEdges()) {
    adjMat[from][to] = true;
    adjMat[to][from] = true;
  }
  for (std::size_t i = 0; i < adjMat.size(); ++i) {
    adjMat[i][i] = true;
  }
  return adjMat;
}

std::vector<Vertex>
ZXDiagram::getConnectedSet(const std::vector<Vertex>& s,
                           const std::vector<Vertex>& exclude) const {
  std::vector<Vertex> connected;
  for (const auto v : s) {
    for (const auto& [to, _] : edges[v]) {
      if (isIn(to, exclude)) {
        continue;
      }

      const auto& p = std::lower_bound(connected.begin(), connected.end(), to);
      if (p == connected.end()) {
        connected.emplace_back(to);
        continue;
      }
      if (*p != to) {
        connected.insert(p, to);
      }
    }
  }
  return connected;
}

bool ZXDiagram::isIn(const Vertex& v, const std::vector<Vertex>& vertices) {
  return std::find(vertices.begin(), vertices.end(), v) != vertices.end();
}
} // namespace zx
