/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "zx/Rational.hpp"
#include "zx/Simplify.hpp"
#include "zx/ZXDefinitions.hpp"
#include "zx/ZXDiagram.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

namespace zx {
class ZXDiagramTest : public ::testing::Test {
public:
  ZXDiagram diag;

  /*
   * Diagram should look like this:
   * {0: (Boundary, Phase=0)} ---H--- {4: (Z, Phase=0)} ---- {5: (Z, Phase = 0)}
   * ----- {1: (Boundary, Phase = 0)}
   *                                                                 |
   *                                                                 |
   *                                                                 |
   * {2: (Boundary, Phase=0)} ------------------------------ {6: (X, Phase = 0)}
   * ----- {3: (Boundary, Phase = 0)}
   */
protected:
  void SetUp() override {
    diag = ZXDiagram();
    diag.addQubits(2);
    diag.addVertex(0, 0, PiExpression(), VertexType::Z);
    diag.addEdge(0, 4, EdgeType::Hadamard);
    diag.addVertex(0, 0, PiExpression(), VertexType::Z);
    diag.addEdge(4, 5);
    diag.addVertex(0, 0, PiExpression(), VertexType::X);
    diag.addEdge(2, 6);
    diag.addEdge(5, 6);
    diag.addEdge(5, 1);
    diag.addEdge(6, 3);
  }
};

TEST_F(ZXDiagramTest, createDiagram) {
  EXPECT_EQ(diag.getNVertices(), 7);
  EXPECT_EQ(diag.getNEdges(), 6);

  auto inputs = diag.getInputs();
  EXPECT_EQ(inputs[0], 0);
  EXPECT_EQ(inputs[1], 2);

  auto outputs = diag.getOutputs();
  EXPECT_EQ(outputs[0], 1);
  EXPECT_EQ(outputs[1], 3);

  constexpr auto edges =
      std::array{std::pair{0U, 4U}, std::pair{4U, 5U}, std::pair{2U, 6U},
                 std::pair{5U, 6U}, std::pair{5U, 1U}, std::pair{6U, 3U}};
  constexpr auto expectedEdgeTypes =
      std::array{EdgeType::Hadamard, EdgeType::Simple, EdgeType::Simple,
                 EdgeType::Simple,   EdgeType::Simple, EdgeType::Simple};
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const auto& [v1, v2] = edges[i];
    const auto& edge = diag.getEdge(v1, v2);
    const auto hasValue = edge.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(edge->type, expectedEdgeTypes[i]);
    }
  }

  constexpr auto expectedVertexTypes = std::array{
      VertexType::Boundary, VertexType::Boundary, VertexType::Boundary,
      VertexType::Boundary, VertexType::Z,        VertexType::Z,
      VertexType::X};
  const auto nVerts = diag.getNVertices();
  for (std::size_t i = 0; i < nVerts; ++i) {
    const auto& vData = diag.getVData(i);
    const auto hasValue = vData.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(vData->type, expectedVertexTypes[i]);
      EXPECT_TRUE(vData->phase.isZero());
    }
  }
}

TEST_F(ZXDiagramTest, deletions) {
  diag.removeVertex(5);
  EXPECT_EQ(diag.getNVertices(), 6);
  EXPECT_EQ(diag.getNEdges(), 3);
  EXPECT_FALSE(diag.getVData(5).has_value());

  diag.removeEdge(0, 4);
  EXPECT_EQ(diag.getNVertices(), 6);
  EXPECT_EQ(diag.getNEdges(), 2);
}

TEST_F(ZXDiagramTest, graphLike) {
  diag.toGraphlike();

  constexpr auto edges =
      std::array{std::pair{0U, 4U}, std::pair{4U, 5U}, std::pair{2U, 6U},
                 std::pair{5U, 6U}, std::pair{5U, 1U}, std::pair{6U, 3U}};
  constexpr auto expectedEdgeTypes =
      std::array{EdgeType::Hadamard, EdgeType::Simple, EdgeType::Hadamard,
                 EdgeType::Hadamard, EdgeType::Simple, EdgeType::Hadamard};
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const auto& [v1, v2] = edges[i];
    const auto& edge = diag.getEdge(v1, v2);
    const auto hasValue = edge.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(edge->type, expectedEdgeTypes[i]);
    }
  }

  constexpr auto expectedVertexTypes = std::array{
      VertexType::Boundary, VertexType::Boundary, VertexType::Boundary,
      VertexType::Boundary, VertexType::Z,        VertexType::Z,
      VertexType::Z};
  const auto nVerts = diag.getNVertices();
  for (std::size_t i = 0; i < nVerts; ++i) {
    const auto& vData = diag.getVData(i);
    const auto hasValue = vData.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(vData->type, expectedVertexTypes[i]);
      EXPECT_TRUE(vData->phase.isZero());
    }
  }
}

TEST_F(ZXDiagramTest, adjoint) {
  diag = diag.adjoint();

  constexpr auto edges =
      std::array{std::pair{0U, 4U}, std::pair{4U, 5U}, std::pair{2U, 6U},
                 std::pair{5U, 6U}, std::pair{5U, 1U}, std::pair{6U, 3U}};
  constexpr auto expectedEdgeTypes =
      std::array{EdgeType::Hadamard, EdgeType::Simple, EdgeType::Simple,
                 EdgeType::Simple,   EdgeType::Simple, EdgeType::Simple};
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const auto& [v1, v2] = edges[i];
    const auto& edge = diag.getEdge(v1, v2);
    const auto hasValue = edge.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(edge->type, expectedEdgeTypes[i]);
    }
  }

  constexpr auto expectedVertexTypes = std::array{
      VertexType::Boundary, VertexType::Boundary, VertexType::Boundary,
      VertexType::Boundary, VertexType::Z,        VertexType::Z,
      VertexType::X};
  const auto nVerts = diag.getNVertices();
  for (std::size_t i = 0; i < nVerts; ++i) {
    const auto& vData = diag.getVData(i);
    const auto hasValue = vData.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(vData->type, expectedVertexTypes[i]);
      EXPECT_TRUE(vData->phase.isZero());
    }
  }
}

TEST_F(ZXDiagramTest, approximate) {
  ZXDiagram almostId(3);

  almostId.removeEdge(0, 3);
  const auto v = almostId.addVertex(0, 1, PiExpression(PiRational(1e-8)));
  almostId.addEdge(0, v);
  almostId.addEdge(v, 3);

  EXPECT_FALSE(almostId.phase(v).isZero());

  almostId.approximateCliffords(1e-7);

  EXPECT_TRUE(almostId.phase(v).isZero());
}

TEST_F(ZXDiagramTest, ancilla) {
  ZXDiagram cx(2);
  cx.removeEdge(0, 2);
  cx.removeEdge(1, 3);
  const auto tar = cx.addVertex(0, 0, PiExpression{}, VertexType::X);

  const auto ctrl = cx.addVertex(1);

  cx.addEdge(tar, ctrl);
  cx.addEdge(0, tar);
  cx.addEdge(tar, 2);
  cx.addEdge(1, ctrl);
  cx.addEdge(ctrl, 3);
  EXPECT_EQ(cx.getInputs().size(), 2);
  EXPECT_EQ(cx.getOutputs().size(), 2);
  EXPECT_EQ(cx.getNVertices(), 6);

  cx.makeAncilla(1);
  EXPECT_EQ(cx.getInputs().size(), 1);
  EXPECT_EQ(cx.getOutputs().size(), 1);
  EXPECT_EQ(cx.getNVertices(), 6);

  fullReduce(cx);

  EXPECT_EQ(cx.getNEdges(), 1);
  for (const auto& [v, data] : cx.getVertices()) {
    std::cout << v << " " << (data.type == VertexType::Boundary) << "\n";
  }
  EXPECT_EQ(cx.getNVertices(), 2);
  EXPECT_TRUE(cx.isIdentity());
}

TEST_F(ZXDiagramTest, RemoveScalarSubDiagram) {
  ZXDiagram idWithScal(1);

  const auto v = idWithScal.addVertex(1);
  const auto w = idWithScal.addVertex(2);
  idWithScal.addEdge(v, w);

  fullReduce(idWithScal);

  EXPECT_EQ(idWithScal.getNVertices(), 2);
  EXPECT_EQ(idWithScal.getNEdges(), 1);
  EXPECT_TRUE(idWithScal.isDeleted(v));
  EXPECT_TRUE(idWithScal.isDeleted(w));
}

TEST_F(ZXDiagramTest, AdjMat) {
  diag = ZXDiagram(3);

  const auto& adj = diag.getAdjMat();

  for (const auto& [v, _] : diag.getVertices()) {
    for (const auto& [w, wd] : diag.getVertices()) {
      if (diag.connected(v, w) || v == w) {
        EXPECT_TRUE(adj[v][w]);
        EXPECT_TRUE(adj[w][v]);
      } else {
        EXPECT_FALSE(adj[v][w]);
        EXPECT_FALSE(adj[w][v]);
      }
    }
  }
}

TEST_F(ZXDiagramTest, ConnectedSet) {
  diag = ZXDiagram(3);
  auto connected = diag.getConnectedSet(diag.getInputs());

  for (const auto& v : connected) {
    EXPECT_TRUE(diag.isIn(v, diag.getOutputs()));
  }

  connected = diag.getConnectedSet(diag.getInputs(), {4});

  EXPECT_TRUE(diag.isIn(3, connected));
  EXPECT_FALSE(diag.isIn(4, connected));
  EXPECT_TRUE(diag.isIn(5, connected));
}

TEST_F(ZXDiagramTest, EdgeTypePrinting) {
  diag = ZXDiagram(3);
  diag.addEdge(0, 1, EdgeType::Simple);

  const auto& edge = diag.getEdge(0, 1);
  ASSERT_NE(edge, std::nullopt);
  if (edge) {
    std::stringstream ss;
    ss << edge->type;
    EXPECT_EQ(ss.str(), "Simple");
  }

  // Change the type to Hadamard
  diag.addHadamardEdge(1, 2);
  const auto& edge2 = diag.getEdge(1, 2);
  ASSERT_NE(edge2, std::nullopt);
  if (edge2) {
    std::stringstream ss2;
    ss2 << edge2->type;
    EXPECT_EQ(ss2.str(), "Hadamard");
  }
}

TEST_F(ZXDiagramTest, VertexTypePrinting) {
  diag = ZXDiagram(1);
  const auto boundary = diag.getInput(0);
  const auto boundaryData = diag.getVData(boundary);
  ASSERT_NE(boundaryData, std::nullopt);
  if (boundaryData) {
    EXPECT_EQ(boundaryData->type, VertexType::Boundary);
    std::stringstream ss;
    ss << boundaryData->type;
    EXPECT_EQ(ss.str(), "Boundary");
  }
  const auto z = diag.addVertex(0, 0, PiExpression(), VertexType::Z);
  const auto zData = diag.getVData(z);
  ASSERT_NE(zData, std::nullopt);
  if (zData) {
    EXPECT_EQ(zData->type, VertexType::Z);
    std::stringstream ss;
    ss << zData->type;
    EXPECT_EQ(ss.str(), "Z");
  }
  const auto x = diag.addVertex(0, 0, PiExpression(), VertexType::X);
  const auto xData = diag.getVData(x);
  ASSERT_NE(xData, std::nullopt);
  if (xData) {
    EXPECT_EQ(xData->type, VertexType::X);
    std::stringstream ss;
    ss << xData->type;
    EXPECT_EQ(ss.str(), "X");
  }
}
} // namespace zx
