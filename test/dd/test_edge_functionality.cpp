/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>

namespace dd {

///-----------------------------------------------------------------------------
///                     \n Tests for vector DDs \n
///-----------------------------------------------------------------------------

TEST(VectorFunctionality, GetValueByPathTerminal) {
  EXPECT_EQ(vEdge::zero().getValueByPath(0, "0"), 0.);
  EXPECT_EQ(vEdge::one().getValueByPath(0, "0"), 1.);
}

TEST(VectorFunctionality, GetValueByIndexTerminal) {
  EXPECT_EQ(vEdge::zero().getValueByIndex(0), 0.);
  EXPECT_EQ(vEdge::one().getValueByIndex(0), 1.);
}

TEST(VectorFunctionality, GetValueByIndexEndianness) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);

  for (std::size_t i = 0U; i < state.size(); ++i) {
    EXPECT_EQ(state[i], stateDD.getValueByIndex(i));
  }
}

TEST(VectorFunctionality, GetVectorTerminal) {
  EXPECT_EQ(vEdge::zero().getVector(), CVec{0.});
  EXPECT_EQ(vEdge::one().getVector(), CVec{1.});
}

TEST(VectorFunctionality, GetVectorRoundtrip) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);
  const auto stateVec = stateDD.getVector();
  EXPECT_EQ(stateVec, state);
}

TEST(VectorFunctionality, GetVectorTolerance) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);
  const auto stateVec = stateDD.getVector(std::sqrt(0.1));
  EXPECT_EQ(stateVec, state);
  const auto stateVec2 = stateDD.getVector(std::sqrt(0.1) + RealNumber::eps);
  EXPECT_NE(stateVec2, state);
  EXPECT_EQ(stateVec2[0], 0.);
}

TEST(VectorFunctionality, GetSparseVectorTerminal) {
  const auto zero = SparseCVec{{0, 0}};
  EXPECT_EQ(vEdge::zero().getSparseVector(), zero);
  const auto one = SparseCVec{{0, 1}};
  EXPECT_EQ(vEdge::one().getSparseVector(), one);
}

TEST(VectorFunctionality, GetSparseVectorConsistency) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);
  const auto stateSparseVec = stateDD.getSparseVector();
  const auto stateVec = stateDD.getVector();
  for (const auto& [index, value] : stateSparseVec) {
    EXPECT_EQ(value, stateVec[index]);
  }
}

TEST(VectorFunctionality, GetSparseVectorTolerance) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);
  const auto stateSparseVec = stateDD.getSparseVector(std::sqrt(0.1));
  for (const auto& [index, value] : stateSparseVec) {
    EXPECT_EQ(value, state[index]);
  }
  const auto stateSparseVec2 =
      stateDD.getSparseVector(std::sqrt(0.1) + RealNumber::eps);
  EXPECT_NE(stateSparseVec2, stateSparseVec);
  EXPECT_EQ(stateSparseVec2.count(0), 0);
}

TEST(VectorFunctionality, PrintVectorTerminal) {
  testing::internal::CaptureStdout();
  vEdge::zero().printVector();
  const auto zeroStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(zeroStr, "0: (0,0)\n");
  testing::internal::CaptureStdout();
  vEdge::one().printVector();
  const auto oneStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(oneStr, "0: (1,0)\n");
}

TEST(VectorFunctionality, PrintVector) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);
  testing::internal::CaptureStdout();
  stateDD.printVector();
  const auto stateStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(stateStr,
            "00: (0.316,0)\n01: (0.447,0)\n10: (0.548,0)\n11: (0.632,0)\n");
}

TEST(VectorFunctionality, AddToVectorTerminal) {
  CVec vec = {0.};
  vEdge::one().addToVector(vec);
  EXPECT_EQ(vec, CVec{1.});
}

TEST(VectorFunctionality, AddToVector) {
  CVec vec = {0., 0., 0., 0.};

  auto dd = std::make_unique<Package>(2);
  const CVec state = {std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                      std::sqrt(0.4)};
  const auto stateDD = makeStateFromVector(state, *dd);
  stateDD.addToVector(vec);
  EXPECT_EQ(vec, state);
}

TEST(VectorFunctionality, SizeTerminal) {
  EXPECT_EQ(vEdge::zero().size(), 1);
  EXPECT_EQ(vEdge::one().size(), 1);
}

TEST(VectorFunctionality, SizeBellState) {
  auto dd = std::make_unique<Package>(2);
  const CVec state = {SQRT2_2, 0., 0., SQRT2_2};
  const auto bell = makeStateFromVector(state, *dd);
  EXPECT_EQ(bell.size(), 4);
}

///-----------------------------------------------------------------------------
///                     \n Tests for matrix DDs \n
///-----------------------------------------------------------------------------

TEST(MatrixFunctionality, GetValueByPathTerminal) {
  EXPECT_EQ(mEdge::zero().getValueByPath(0, "0"), 0.);
  EXPECT_EQ(mEdge::one().getValueByPath(0, "0"), 1.);
}

TEST(MatrixFunctionality, GetValueByIndexTerminal) {
  EXPECT_EQ(mEdge::zero().getValueByIndex(0, 0, 0), 0.);
  EXPECT_EQ(mEdge::one().getValueByIndex(0, 0, 0), 1.);
}

TEST(MatrixFunctionality, GetValueByIndexEndianness) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)}};
  // clang-format on

  const auto matDD = dd->makeDDFromMatrix(mat);

  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = matDD.getValueByIndex(dd->qubits(), i, j);
      const auto ref = mat[i][j];
      EXPECT_NEAR(ref.real(), val.real(), 1e-10);
      EXPECT_NEAR(ref.imag(), val.imag(), 1e-10);
    }
  }
}

TEST(MatrixFunctionality, GetMatrixTerminal) {
  EXPECT_EQ(mEdge::zero().getMatrix(0), CMat{{0.}});
  EXPECT_EQ(mEdge::one().getMatrix(0), CMat{{1.}});
}

TEST(MatrixFunctionality, GetMatrixRoundtrip) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)}};
  // clang-format on

  const auto matDD = dd->makeDDFromMatrix(mat);
  const auto matVec = matDD.getMatrix(dd->qubits());
  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = matDD.getValueByIndex(dd->qubits(), i, j);
      const auto ref = mat[i][j];
      EXPECT_NEAR(ref.real(), val.real(), 1e-10);
      EXPECT_NEAR(ref.imag(), val.imag(), 1e-10);
    }
  }
}

TEST(MatrixFunctionality, GetMatrixTolerance) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)}};
  // clang-format on

  const auto matDD = dd->makeDDFromMatrix(mat);
  const auto matVec = matDD.getMatrix(dd->qubits(), std::sqrt(0.1));
  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = matDD.getValueByIndex(dd->qubits(), i, j);
      const auto ref = mat[i][j];
      EXPECT_NEAR(ref.real(), val.real(), 1e-10);
      EXPECT_NEAR(ref.imag(), val.imag(), 1e-10);
    }
  }
  const auto matVec2 =
      matDD.getMatrix(dd->qubits(), std::sqrt(0.1) + RealNumber::eps);
  EXPECT_NE(matVec2, matVec);
  EXPECT_EQ(matVec2[0][0], 0.);
  EXPECT_EQ(matVec2[1][3], 0.);
  EXPECT_EQ(matVec2[2][2], 0.);
  EXPECT_EQ(matVec2[3][1], 0.);
}

TEST(MatrixFunctionality, GetSparseMatrixTerminal) {
  const auto zero = SparseCMat{{{0, 0}, 0.}};
  EXPECT_EQ(mEdge::zero().getSparseMatrix(0), zero);
  const auto one = SparseCMat{{{0, 0}, 1.}};
  EXPECT_EQ(mEdge::one().getSparseMatrix(0), one);
}

TEST(MatrixFunctionality, GetSparseMatrixConsistency) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)}};
  // clang-format on

  const auto matDD = dd->makeDDFromMatrix(mat);
  const auto matSparse = matDD.getSparseMatrix(dd->qubits());
  const auto matDense = matDD.getMatrix(dd->qubits());
  for (const auto& [index, value] : matSparse) {
    const auto val = matDense.at(index.first).at(index.second);
    EXPECT_NEAR(value.real(), val.real(), 1e-10);
    EXPECT_NEAR(value.imag(), val.imag(), 1e-10);
  }
}

TEST(MatrixFunctionality, GetSparseMatrixTolerance) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)}};
  // clang-format on

  const auto matDD = dd->makeDDFromMatrix(mat);
  const auto matSparse = matDD.getSparseMatrix(dd->qubits(), std::sqrt(0.1));
  const auto matDense = matDD.getMatrix(dd->qubits());
  for (const auto& [index, value] : matSparse) {
    const auto val = matDense.at(index.first).at(index.second);
    EXPECT_NEAR(value.real(), val.real(), 1e-10);
    EXPECT_NEAR(value.imag(), val.imag(), 1e-10);
  }
  const auto matSparse2 =
      matDD.getSparseMatrix(dd->qubits(), std::sqrt(0.1) + RealNumber::eps);
  EXPECT_NE(matSparse2, matSparse);
  EXPECT_EQ(matSparse2.count({0, 0}), 0);
  EXPECT_EQ(matSparse2.count({1, 3}), 0);
  EXPECT_EQ(matSparse2.count({2, 2}), 0);
  EXPECT_EQ(matSparse2.count({3, 1}), 0);
}

TEST(MatrixFunctionality, PrintMatrixTerminal) {
  testing::internal::CaptureStdout();
  mEdge::zero().printMatrix(0);
  const auto zeroStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(zeroStr, "(0,0)\n");
  testing::internal::CaptureStdout();
  mEdge::one().printMatrix(0);
  const auto oneStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(oneStr, "(1,0)\n");
}

TEST(MatrixFunctionality, PrintMatrix) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)}};
  // clang-format on

  const auto matDD = dd->makeDDFromMatrix(mat);
  testing::internal::CaptureStdout();
  matDD.printMatrix(dd->qubits());
  const auto matStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(matStr, "(0.316,-0) (0.447,-0) (0.548,0) (0.632,0) \n"
                    "(-0.447,0) (-0.548,0) (0.632,0) (0.316,0) \n"
                    "(-0.548,0) (-0.632,0) (0.316,0) (0.447,0) \n"
                    "(-0.632,0) (-0.316,0) (-0.447,0) (0.548,0) \n");
}

TEST(MatrixFunctionality, SizeTerminal) {
  EXPECT_EQ(mEdge::zero().size(), 1);
  EXPECT_EQ(mEdge::one().size(), 1);
}

TEST(MatrixFunctionality, SizeBellState) {
  auto dd = std::make_unique<Package>(2);
  // clang-format off
  const CMat mat = {
    {SQRT2_2, 0., 0., SQRT2_2},
    {0., SQRT2_2, SQRT2_2, 0.},
    {0., SQRT2_2, -SQRT2_2, 0.},
    {SQRT2_2, 0., 0., -SQRT2_2}};
  // clang-format on

  const auto bell = dd->makeDDFromMatrix(mat);
  EXPECT_EQ(bell.size(), 3);
}

} // namespace dd
