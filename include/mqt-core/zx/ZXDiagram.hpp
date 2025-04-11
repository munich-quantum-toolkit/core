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

#include "Utils.hpp"
#include "ZXDefinitions.hpp"

#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace zx {

/**
 * @brief Class representing a ZX-diagram.
 *
 * @details Labelled undirected graph. Every node is either of type
 * VertexType::Z, VertexType::X or VertexType::Boundary. The boundary vertices
 * are the inputs and outputs of the diagram. The graph is stored as an
 * adjacency list. The vertices are stored in a vector of optional VertexData.
 * The optional is empty if the vertex has been deleted.
 * The scalars of the diagram are SymbolicPiExpressions, meaning that all
 * computations can be done symbolically.
 */
class ZXDiagram {
public:
  ZXDiagram() = default;
  /**
   * @brief Create an nqubit identity diagram.
   * @param nqubits number of qubits
   */
  explicit ZXDiagram(std::size_t nqubits);

  /**
   * @brief Add an edge to the diagram.
   * @param from first vertex of the edge
   * @param to second vertex of the edge
   * @param type type of the edge (Simple or Hadamard)
   */
  void addEdge(Vertex from, Vertex to, EdgeType type = EdgeType::Simple);

  /**
   * @brief Add a Hadamard edge to the diagram.
   * @param from first vertex of the edge
   * @param to second vertex of the edge
   */
  void addHadamardEdge(const Vertex from, const Vertex to) {
    addEdge(from, to, EdgeType::Hadamard);
  };

  /**
   * @brief Add edge to diagram, simplifying the diagram if this results in
   * parallel edges.
   * @details In the ZX-calculus, parallel edges and self-loops can usually be
   * simplified. Two parallel Simple edges between two opposite-colored
   * vertices cancel out for example. This function adds an edge to the diagram
   * and then simplifies the diagram if necessary.
   * @param from first vertex of the edge
   * @param to second vertex of the edge
   * @param eType type of the edge (Simple or Hadamard)
   */
  void addEdgeParallelAware(Vertex from, Vertex to,
                            EdgeType eType = EdgeType::Simple);

  /**
   * @brief Remove Edge from the diagram.
   * @param from first vertex of the edge
   * @param to second vertex of the edge
   */
  void removeEdge(Vertex from, Vertex to);

  /**
   * @brief Add a vertex to the diagram.
   * @param data data of the vertex
   * @return the vertex that was added
   */
  Vertex addVertex(const VertexData& data);

  /**
   * @brief Add a vertex to the diagram.
   * @param qubit qubit index of the vertex
   * @param col column index of the vertex
   * @param phase phase of the vertex
   * @param type type of the vertex
   * @return the vertex that was added
   */
  Vertex addVertex(Qubit qubit, Col col = 0,
                   const PiExpression& phase = PiExpression(),
                   VertexType type = VertexType::Z);

  /**
   * @brief Add a bare qubit to the diagram.
   */
  void addQubit();

  /**
   * @brief Add multiple bare qubits to the diagram.
   * @param n number of qubits to add
   */
  void addQubits(zx::Qubit n);

  /**
   * @brief Remove a vertex from the diagram. Also removes all edges incident to
   * the vertex.
   * @param toRemove vertex to remove
   */
  void removeVertex(Vertex toRemove);

  /**
   * @brief Get number of deleted vertices.
   * @details Deleted vertices are not removed from the vertices vector, but
   * are left as optional in the array.
   * @return number of deleted vertices
   */
  [[nodiscard]] std::size_t getNdeleted() const { return deleted.size(); }

  /**
   * @brief Get number of vertices.
   * @return number of vertices
   */
  [[nodiscard]] std::size_t getNVertices() const { return nvertices; }

  /**
   * @brief Get number of edges.
   * @return number of edges
   */
  [[nodiscard]] std::size_t getNEdges() const { return nedges; }

  /**
   * @brief Get number of qubits.
   * @return number of qubits
   */
  [[nodiscard]] std::size_t getNQubits() const { return inputs.size(); }

  /**
   * @brief Check whether there is a path between two vertices.
   * @param from first vertex
   * @param to second vertex
   * @return true if there is a path between the two vertices, false otherwise
   */
  [[nodiscard]] bool connected(Vertex from, Vertex to) const;

  /**
   * @brief Get the edge between two vertices.
   * @param from first vertex
   * @param to second vertex
   * @return the edge between the two vertices, if it exists
   */
  [[nodiscard]] std::optional<Edge> getEdge(Vertex from, Vertex to) const;

  /**
   * @brief Get all edges incident to a vertex.
   * @param v vertex
   * @return vector of edges incident to the vertex
   */
  [[nodiscard]] const std::vector<Edge>& incidentEdges(const Vertex v) const {
    return edges[v];
  }

  /**
   * @brief Get the n-th incident edge of a vertex. Bounds are not checked for
   * performance reasons!
   * @param v vertex
   * @param n index of the incident edge
   * @return the n-th incident edge of the vertex
   */
  [[nodiscard]] const Edge& incidentEdge(const Vertex v, const std::size_t n) {
    return edges[v][n];
  }

  /**
   * @brief Get the degree of a vertex.
   * @param v vertex
   * @return degree of the vertex
   */
  [[nodiscard]] std::size_t degree(const Vertex v) const {
    return edges[v].size();
  }

  /**
   * @brief Get the phase of a vertex
   * @param v vertex
   * @return phase of the vertex
   */
  [[nodiscard]] const PiExpression& phase(const Vertex v) const {
    const auto& vertex = vertices[v];
    assert(vertex.has_value());
    return vertex->phase; // NOLINT(bugprone-unchecked-optional-access)
  }

  /**
   * @brief Get the qubit of a vertex
   * @param v vertex
   * @return qubit of the vertex
   */
  [[nodiscard]] Qubit qubit(const Vertex v) const {
    const auto& vertex = vertices[v];
    assert(vertex.has_value());
    return vertex->qubit; // NOLINT(bugprone-unchecked-optional-access)
  }

  /**
   * @brief Get the type of a vertex
   * @param v vertex
   * @return type of the vertex
   */
  [[nodiscard]] VertexType type(const Vertex v) const {
    const auto& vertex = vertices[v];
    assert(vertex.has_value());
    return vertex->type; // NOLINT(bugprone-unchecked-optional-access)
  }

  /**
   * @brief Get vertex data if vertex has not been deleted.
   * @param v vertex
   * @return vertex data if vertex has not been deleted
   */
  [[nodiscard]] std::optional<VertexData> getVData(const Vertex v) const {
    return vertices[v];
  }

  /**
   * @brief Get all vertices of the diagram.
   * @return vector of vertices
   */
  [[nodiscard]] std::vector<std::pair<Vertex, const VertexData&>>
  getVertices() const;

  /**
   * @brief Get all edges of the diagram.
   * @return vector of edges
   */
  [[nodiscard]] std::vector<std::pair<Vertex, Vertex>> getEdges() const;

  /**
   * @brief Get all input vertices of the diagram.
   * @return vector of input vertices
   */
  [[nodiscard]] const std::vector<Vertex>& getInputs() const { return inputs; }

  /**
   * @brief Get i-th input vertex of the diagram. Bounds are not checked!
   * @param i index of the input vertex
   * @return i-th input vertex
   */
  [[nodiscard]] Vertex getInput(const std::size_t i) const { return inputs[i]; }

  /**
   * @brief Get all output vertices of the diagram.
   * @return vector of output vertices
   */
  [[nodiscard]] const std::vector<Vertex>& getOutputs() const {
    return outputs;
  }

  /**
   * @brief Get i-th output vertex of the diagram. Bounds are not checked!
   * @param i index of the output vertex
   * @return i-th output vertex
   */
  [[nodiscard]] Vertex getOutput(const std::size_t i) const {
    return outputs[i];
  }

  /**
   * @brief Check whether vertex has been deleted.
   * @param v vertex
   * @return true if vertex has been deleted, false otherwise
   */
  [[nodiscard]] bool isDeleted(const Vertex v) const {
    return !vertices[v].has_value();
  }

  /**
   * @brief Check whether vertex is a boundary vertex.
   * @param v vertex
   * @return true if vertex is a boundary vertex, false otherwise
   */
  [[nodiscard]] bool isBoundaryVertex(const Vertex v) const {
    return type(v) == VertexType::Boundary;
  }

  /**
   * @brief Check whether vertex is an input vertex.
   * @param v vertex
   * @return true if vertex is an input vertex, false otherwise
   */
  [[nodiscard]] bool isInput(Vertex v) const;

  /**
   * @brief Check whether vertex is an output vertex.
   * @param v vertex
   * @return true if vertex is an output vertex, false otherwise
   */
  [[nodiscard]] bool isOutput(Vertex v) const;

  /**
   * @brief Add phase to a vertex. The phase is added to the existing phase.
   * @param v vertex
   * @param phase phase to add
   */
  void addPhase(const Vertex v, const PiExpression& phase) {
    auto& vertex = vertices[v];
    if (vertex.has_value()) {
      vertex->phase += phase;
    }
  }

  /**
   * @brief Set phase of a vertex. Previous value is overwritten.
   * @param v vertex
   * @param phase phase to set
   */
  void setPhase(const Vertex v, const PiExpression& phase) {
    auto& vertex = vertices[v];
    if (vertex.has_value()) {
      vertex->phase = phase;
    }
  }

  /**
   * @brief Set type of a vertex.
   * @param v vertex
   * @param type type to set
   */
  void setType(const Vertex v, const VertexType type) {
    auto& vertex = vertices[v];
    if (vertex.has_value()) {
      vertex->type = type;
    }
  }

  /**
   * @brief Transform the diagram to a graph-like diagram. Modifies the diagram
   * in place.
   * @details A graph-like diagram is a diagram where all vertices are of type Z
   * and all edges are hadamard edges. Every ZX-diagram can be rewritten into
   * this form by adding Hadamard edges and changing the type of the vertices.
   */
  void toGraphlike();

  /**
   * @brief check whether the diagram is the identity diagram.
   * @details The identity diagram is a diagram where every input is connected
   * to one output without any additional vertices or hadamard edges in between.
   * @return true if the diagram is the identity diagram, false otherwise
   */
  [[nodiscard]] bool isIdentity() const;

  /**
   * @brief Get the adjoint of the diagram.
   * @details The adjoint of a ZX-diagram is the diagram with inputs and outputs
   * swapped and all phases negated.
   * @return the adjoint of the diagram
   */
  [[nodiscard]] ZXDiagram adjoint() const;

  /**
   * @brief Invert the diagram in place.
   * @details Diagram is replaced by its adjoint.
   * @return reference to the inverted diagram
   */
  ZXDiagram& invert();

  /**
   * @brief Concatenate two diagrams. Modifies the first diagram in place.
   * @details The second diagram is added to the first diagram. The inputs of
   * the second diagram are connected to the outputs of the first diagram. The
   * number of qubits of the two diagrams must be the same.
   * @param rhs second diagram
   * @return reference to the concatenated diagram
   */
  ZXDiagram& concat(const ZXDiagram& rhs);

  /**
   * @brief Same as concat(const ZXDiagram&)
   * @param rhs second diagram
   * @return reference to the concatenated diagram
   */
  ZXDiagram& operator+=(const ZXDiagram& rhs) { return this->concat(rhs); }

  /**
   * @brief Convert a qubit to an ancilla initialized and post-selected on |0>.
   * @param qubit qubit to convert
   */
  void makeAncilla(Qubit qubit);

  /**
   * @brief Convert a qubit to an ancilla initialized and post-selected on |0>.
   * @param in qubit initialized in |0>
   * @param out qubit post-selected in |0>
   */
  void makeAncilla(Qubit in, Qubit out);

  /**
   * @brief Round phases in the diagram to multiples of pi/2. Modifies the
   * diagram in place.
   * @param tolerance tolerance for rounding
   */
  void approximateCliffords(fp tolerance);

  /**
   * @brief Remove disconnected spiders from the diagram. Modifies the diagram
   * in place.
   */
  void removeDisconnectedSpiders();

  /**
   * @brief Add a global phase to the diagram. Adds the phase to the current
   * global phase.
   * @param phase phase to add
   */
  void addGlobalPhase(const PiExpression& phase);

  /**
   * @brief Get the global phase of the diagram.
   * @return global phase of the diagram
   */
  [[nodiscard]] PiExpression getGlobalPhase() const { return globalPhase; }

  /**
   * @brief Check whether the global phase of the diagram is zero.
   * @return true if the global phase is zero, false otherwise
   */
  [[nodiscard]] bool globalPhaseIsZero() const { return globalPhase.isZero(); }

  /**
   * @brief Get the adjacency matrix of the diagram.
   * @return adjacency matrix of the diagram
   */
  [[nodiscard]] gf2Mat getAdjMat() const;

  /**
   * @brief Get the connected set of a set of vertices.
   * @details The connected set of a set of vertices is the set of all vertices
   * that are connected to the input set.
   * @param s set of vertices
   * @param exclude set of vertices to exclude from the connected set
   * @return connected set of the input set
   */
  [[nodiscard]] std::vector<Vertex>
  getConnectedSet(const std::vector<Vertex>& s,
                  const std::vector<Vertex>& exclude = {}) const;

  /**
   * @brief Check whether a vertex is in a vector of vertices.
   * @param v vertex
   * @param vertices vector of vertices
   * @return true if the vertex is in the vector, false otherwise
   */
  static bool isIn(const Vertex& v, const std::vector<Vertex>& vertices);

private:
  std::vector<std::vector<Edge>> edges;
  std::vector<std::optional<VertexData>> vertices;
  std::vector<Vertex> deleted;
  std::vector<Vertex> inputs;
  std::vector<Vertex> outputs;
  std::size_t nvertices = 0;
  std::size_t nedges = 0;
  PiExpression globalPhase;

  std::vector<Vertex> initGraph(std::size_t nqubits);
  void closeGraph(const std::vector<Vertex>& qubitVertices);

  void removeHalfEdge(Vertex from, Vertex to);

  std::vector<Edge>::iterator getEdgePtr(Vertex from, Vertex to);
};
} // namespace zx
