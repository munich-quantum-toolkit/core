/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "zx/Rules.hpp"

#include "zx/Rational.hpp"
#include "zx/Utils.hpp"
#include "zx/ZXDefinitions.hpp"
#include "zx/ZXDiagram.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace zx {

namespace {
bool isPauli(const ZXDiagram& diag, const Vertex v) {
  return isPauli(diag.phase(v));
}

bool isInterior(const ZXDiagram& diag, const Vertex v) {
  const auto& edges = diag.incidentEdges(v);
  return std::ranges::all_of(edges, [&](auto& edge) {
    return diag.degree(edge.to) > 1 && diag.type(edge.to) == VertexType::Z;
  });
}

void extractGadget(ZXDiagram& diag, const Vertex v) {
  const auto& vData = diag.getVData(v);
  if (!vData.has_value()) {
    return;
  }
  const Vertex phaseVert = diag.addVertex(vData->qubit, -2, vData->phase);
  const Vertex idVert = diag.addVertex(vData->qubit, -1);
  diag.setPhase(v, PiExpression(PiRational(0, 1)));
  diag.addHadamardEdge(v, idVert);
  diag.addHadamardEdge(idVert, phaseVert);
}

void extractPauliGadget(ZXDiagram& diag, const Vertex v) {
  if (isPauli(diag.phase(v))) {
    return;
  }

  extractGadget(diag, v);
}

void ensureInterior(ZXDiagram& diag, const Vertex v) {
  const auto edges = diag.incidentEdges(v);
  const auto& vData = diag.getVData(v);
  if (!vData.has_value()) {
    return;
  }

  for (const auto& [to, type] : edges) {
    if (!diag.isBoundaryVertex(to)) {
      continue;
    }

    const Vertex newV = diag.addVertex(vData->qubit, vData->col,
                                       PiExpression(PiRational(0, 1)));
    const auto boundaryEdgeType = type == zx::EdgeType::Simple
                                      ? zx::EdgeType::Hadamard
                                      : zx::EdgeType::Simple;

    diag.addEdge(v, newV, EdgeType::Hadamard);
    diag.addEdge(to, newV, boundaryEdgeType);
    diag.removeEdge(v, to);
  }
}

void ensurePauliVertex(ZXDiagram& diag, const Vertex v) {
  extractPauliGadget(diag, v);
  ensureInterior(diag, v);
}
} // namespace

bool checkIdSimp(const ZXDiagram& diag, const Vertex v) {
  return diag.degree(v) == 2 && diag.phase(v).isZero() &&
         !diag.isBoundaryVertex(v);
}

void removeId(ZXDiagram& diag, const Vertex v) {
  auto edges = diag.incidentEdges(v);
  const Vertex v0 = edges[0].to;
  const Vertex v1 = edges[1].to;

  EdgeType type = EdgeType::Simple;
  if (edges[0].type != edges[1].type) {
    type = EdgeType::Hadamard;
  }
  diag.addEdgeParallelAware(v0, v1, type);
  diag.removeVertex(v);
}

bool checkSpiderFusion(const ZXDiagram& diag, const Vertex v0,
                       const Vertex v1) {
  const auto edgeOpt = diag.getEdge(v0, v1);
  return v0 != v1 && diag.type(v0) == diag.type(v1) &&
         edgeOpt.value_or(Edge{0, EdgeType::Hadamard}).type ==
             EdgeType::Simple &&
         diag.type(v0) != VertexType::Boundary;
}

void fuseSpiders(ZXDiagram& diag, const Vertex v0, const Vertex v1) {
  diag.addPhase(v0, diag.phase(v1));
  for (const auto& [to, type] : diag.incidentEdges(v1)) {
    if (v0 != to) {
      diag.addEdgeParallelAware(v0, to, type);
    }
  }
  diag.removeVertex(v1);
}

bool checkLocalComp(const ZXDiagram& diag, const Vertex v) {
  const auto vData = diag.getVData(v).value_or(VertexData{
      .col = 0, .qubit = 0, .phase = PiExpression(), .type = VertexType::X});
  if (vData.type != VertexType::Z || !isProperClifford(vData.phase)) {
    return false;
  }

  const auto& edges = diag.incidentEdges(v);
  return std::ranges::all_of(edges, [&](auto& edge) {
    return edge.type == EdgeType::Hadamard &&
           diag.type(edge.to) == VertexType::Z;
  });
}

void localComp(ZXDiagram& diag, const Vertex v) { // TODO:scalars
  const auto phase = -diag.phase(v);
  const auto& edges = diag.incidentEdges(v);
  const auto nedges = edges.size();

  for (std::size_t i = 0U; i < nedges; ++i) {
    const auto& [n0, _] = edges[i];
    diag.addPhase(n0, phase);
    for (size_t j = i + 1; j < nedges; ++j) {
      const auto& [n1, _u] = edges[j];
      diag.addEdgeParallelAware(n0, n1, EdgeType::Hadamard);
    }
  }
  diag.addGlobalPhase(
      PiExpression(PiRational{diag.phase(v).getConst().getNum(), 4}));
  diag.removeVertex(v);
}

bool checkPivotPauli(const ZXDiagram& diag, const Vertex v0, const Vertex v1) {
  const auto v0Data = diag.getVData(v0).value_or(VertexData{
      .col = 0, .qubit = 0, .phase = PiExpression(), .type = VertexType::X});
  const auto v1Data = diag.getVData(v1).value_or(VertexData{
      .col = 0, .qubit = 0, .phase = PiExpression(), .type = VertexType::X});

  if (v0Data.type != VertexType::Z || // maybe problem if there is a self-loop?
      v1Data.type != VertexType::Z || !isPauli(diag, v0) ||
      !isPauli(diag, v1)) {
    return false;
  }

  const auto edgeOpt = diag.getEdge(v0, v1);
  if (!edgeOpt.has_value() || edgeOpt.value().type != EdgeType::Hadamard) {
    return false;
  }

  const auto& v0Edges = diag.incidentEdges(v0);
  auto isValidEdge = [&](const Edge& e) {
    return diag.type(e.to) == VertexType::Z && e.type == EdgeType::Hadamard;
  };

  if (!std::ranges::all_of(v0Edges, isValidEdge)) {
    return false;
  }

  const auto& v1Edges = diag.incidentEdges(v1);

  return std::ranges::all_of(v1Edges, isValidEdge);
}

void pivotPauli(ZXDiagram& diag, const Vertex v0,
                const Vertex v1) { // TODO: phases

  const auto v0Phase = diag.phase(v0);
  const auto v1Phase = diag.phase(v1);

  if (!v0Phase.isZero() && !v1Phase.isZero()) {
    diag.addGlobalPhase(PiExpression(PiRational(1, 1)));
  }

  const auto& v0Edges = diag.incidentEdges(v0);
  const auto& v1Edges = diag.incidentEdges(v1);

  for (const auto& [neighbor_v0, _] : v0Edges) {
    if (neighbor_v0 == v1) {
      continue;
    }

    diag.addPhase(neighbor_v0, v1Phase);
    for (const auto& [neighbor_v1, type] : v1Edges) {
      if (neighbor_v1 != v0) {
        diag.addEdgeParallelAware(neighbor_v0, neighbor_v1, EdgeType::Hadamard);
      }
    }
  }

  for (const auto& [neighbor_v1, _] : v1Edges) {
    diag.addPhase(neighbor_v1, v0Phase);
  }

  diag.removeVertex(v0);
  diag.removeVertex(v1);
}

bool checkPivot(const ZXDiagram& diag, const Vertex v0, const Vertex v1) {
  const auto v0Type = diag.type(v0);
  const auto v1Type = diag.type(v1);

  if (v0 == v1 || v0Type != VertexType::Z || v1Type != VertexType::Z) {
    return false;
  }

  const auto edgeOpt = diag.getEdge(v0, v1);
  if (!edgeOpt.has_value() || edgeOpt.value().type != EdgeType::Hadamard) {
    return false;
  }

  const auto& v0Edges = diag.incidentEdges(v0);
  const auto isInvalidEdge = [&](const Edge& e) {
    const auto toType = diag.type(e.to);
    return (toType != VertexType::Z || e.type != EdgeType::Hadamard) &&
           toType != VertexType::Boundary;
  };

  if (std::ranges::any_of(v0Edges, isInvalidEdge)) {
    return false;
  }

  const auto& v1Edges = diag.incidentEdges(v1);
  if (std::ranges::any_of(v1Edges, isInvalidEdge)) {
    return false;
  }

  auto isInteriorPauli = [&](const Vertex v) {
    return isInterior(diag, v) && isPauli(diag, v);
  };

  return (isInteriorPauli(v0) || isInteriorPauli(v1));
}

void pivot(ZXDiagram& diag, const Vertex v0, const Vertex v1) {
  ensurePauliVertex(diag, v0);
  ensurePauliVertex(diag, v1);

  pivotPauli(diag, v0, v1);
}

bool checkPivotGadget(const ZXDiagram& diag, const Vertex v0, const Vertex v1) {
  const auto& p0 = diag.phase(v0);
  const auto& p1 = diag.phase(v1);
  if (!isPauli(p0)) {
    if (!isPauli(p1)) {
      return false;
    }
  } else if (isPauli(p1)) {
    return false;
  }
  if (!isInterior(diag, v0) || !isInterior(diag, v1)) {
    return false;
  }

  return checkPivot(diag, v0, v1);
}

void pivotGadget(ZXDiagram& diag, const Vertex v0, const Vertex v1) {
  if (isPauli(diag, v0)) {
    extractGadget(diag, v1);
  } else {
    extractGadget(diag, v0);
  }
  pivotPauli(diag, v0, v1);
}

bool checkAndFuseGadget(ZXDiagram& diag, const Vertex v) {
  if (diag.degree(v) != 1 || diag.isBoundaryVertex(v)) {
    return false;
  }

  const auto id0 = diag.incidentEdges(v)[0].to;
  const auto id0Etype = diag.incidentEdges(v)[0].type;
  if (!isPauli(diag, id0) || diag.degree(id0) < 2 ||
      id0Etype != zx::EdgeType::Hadamard) {
    return false;
  }

  if (diag.degree(id0) == 2) {
    const auto& [v0, v0Etype] = diag.incidentEdges(id0)[0].to == v
                                    ? diag.incidentEdges(id0)[1]
                                    : diag.incidentEdges(id0)[0];
    if (v0Etype != EdgeType::Hadamard) {
      return false;
    }

    if (diag.phase(id0).isZero()) {
      diag.addPhase(v0, diag.phase(v));
    } else {
      diag.addPhase(v0, -diag.phase(v));
    }
    diag.removeVertex(v);
    diag.removeVertex(id0);
    return true;
  }

  std::optional<Vertex> n0;
  for (const auto& [n, etype] : diag.incidentEdges(id0)) {
    if (n == v) {
      continue;
    }

    if (etype != zx::EdgeType::Hadamard) {
      return false;
    }
    n0 = n;
  }

  std::optional<Vertex> id1;
  std::optional<Vertex> phaseSpider;

  bool foundGadget = false;
  for (const auto& [n, etype] : diag.incidentEdges(n0.value())) {
    if (n == id0) {
      continue;
    }

    if (etype != zx::EdgeType::Hadamard || diag.isDeleted(n) ||
        !isPauli(diag.phase(n)) || diag.degree(n) != diag.degree(id0) ||
        diag.connected(n, id0)) {
      continue;
    }

    foundGadget = true;
    id1 = n;

    for (const auto& [nn, nnEtype] :
         diag.incidentEdges(id1.value())) { // Todo: maybe problem with parallel
                                            // edge? There shouldn't be any
      if (nnEtype != zx::EdgeType::Hadamard || diag.isDeleted(nn)) {
        // not a phase gadget
        foundGadget = false;
        break;
      }

      if (diag.degree(nn) == 1 && !diag.isBoundaryVertex(nn)) {
        foundGadget = true;
        phaseSpider = nn;
        continue;
      }

      if (std::find_if(diag.incidentEdges(nn).begin(),
                       diag.incidentEdges(nn).end(), [&](const Edge e) {
                         return e.to == id0;
                       }) == diag.incidentEdges(nn).end()) {
        foundGadget = false;
        break;
      }
    }

    if (foundGadget) {
      break;
    }
  }

  if (!foundGadget || !phaseSpider.has_value()) {
    return false;
  }

  if (!diag.phase(id0).isZero()) {
    diag.setPhase(v, -diag.phase(v));
    diag.setPhase(id0, PiExpression(PiRational(0, 1)));
  }
  if (diag.phase(id1.value()).isZero()) {
    diag.addPhase(v, diag.phase(phaseSpider.value()));
  } else {
    diag.addPhase(v, -diag.phase(phaseSpider.value()));
  }
  diag.removeVertex(phaseSpider.value());
  diag.removeVertex(id1.value());
  return true;
}

} // namespace zx
