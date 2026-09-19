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
#include "dd/Error.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include "TestUtils.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace dd {

///-----------------------------------------------------------------------------
///                     \n Tests for vector DDs \n
///-----------------------------------------------------------------------------

TEST(VectorFunctionality, GetValueByPathTerminal) {
  EXPECT_EQ(test::value(vEdge::zero().getValueByPath(0, "0")), 0.);
  EXPECT_EQ(test::value(vEdge::one().getValueByPath(0, "0")), 1.);
}

TEST(VectorFunctionality, GetValueByIndexTerminal) {
  EXPECT_EQ(test::value(vEdge::zero().getValueByIndex(0)), 0.);
  EXPECT_EQ(test::value(vEdge::one().getValueByIndex(0)), 1.);
}

TEST(VectorFunctionality, GetValueByIndexEndianness) {
  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));

  for (std::size_t i = 0U; i < state.size(); ++i) {
    EXPECT_EQ(state[i], test::value(stateDD.getValueByIndex(i)));
  }
}

TEST(VectorFunctionality, WideIndices) {
  constexpr auto digits = std::numeric_limits<size_t>::digits;
  auto dd = test::value(Package::create(digits + 1U));
  const auto ones =
      test::value(makeBasisState(digits, std::vector<bool>(digits, true), *dd));
  EXPECT_EQ(
      test::value(ones.getValueByIndex(std::numeric_limits<size_t>::max())),
      1.);
  const auto zero = test::value(makeZeroState(digits + 1U, *dd));
  EXPECT_EQ(test::value(zero.getValueByIndex(0)), 1.);
  EXPECT_EQ(
      test::value(zero.getValueByIndex(std::numeric_limits<size_t>::max())),
      0.);
  EXPECT_EQ(test::errorKind(vEdge::one().getValueByIndex(1)),
            Error::Kind::OutOfRange);
  EXPECT_EQ(
      test::errorKind(test::value(makeZeroState(3, *dd)).getValueByIndex(8)),
      Error::Kind::OutOfRange);
}

TEST(VectorFunctionality, InvalidPaths) {
  auto dd = test::value(Package::create(2));
  const auto zero = test::value(makeZeroState(2, *dd));
  EXPECT_EQ(test::errorKind(zero.getValueByPath(2, "2")),
            Error::Kind::OutOfRange);
  for (const auto* path : {"20", "91", "/0", "x0"}) {
    EXPECT_EQ(test::errorKind(zero.getValueByPath(2, path)),
              Error::Kind::InvalidArgument);
  }
  EXPECT_EQ(test::value(zero.getValueByPath(2, "00ignored")), 1.);
}

TEST(MatrixFunctionality, WideIndices) {
  constexpr auto digits = std::numeric_limits<size_t>::digits;
  auto dd = test::value(Package::create(digits + 1U));
  const auto gate =
      test::value(dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, 0));
  EXPECT_EQ(test::value(gate.getValueByIndex(digits + 1U, 0, 1)),
            std::complex<fp>(0, -1));
  EXPECT_EQ(test::value(gate.getValueByIndex(digits + 1U, 1, 0)),
            std::complex<fp>(0, 1));
  EXPECT_EQ(test::value(gate.getValueByIndex(digits + 1U, 2, 1)), 0.);
  const auto highGate = test::value(
      dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, digits));
  EXPECT_EQ(test::value(highGate.getValueByIndex(digits + 1U, 0, 0)), 0.);
  EXPECT_EQ(test::value(mEdge::one().getValueByIndex(
                digits, std::numeric_limits<size_t>::max(),
                std::numeric_limits<size_t>::max())),
            1.);
  EXPECT_EQ(test::errorKind(gate.getValueByIndex(1, 2, 0)),
            Error::Kind::OutOfRange);
  EXPECT_EQ(test::errorKind(mEdge::one().getValueByIndex(1, 0, 2)),
            Error::Kind::OutOfRange);
}

TEST(MatrixFunctionality, InvalidPaths) {
  EXPECT_EQ(test::errorKind(mEdge::one().getValueByPath(1, "")),
            Error::Kind::OutOfRange);
  for (const auto* path : {"4", "9", "/", "x"}) {
    EXPECT_EQ(test::errorKind(mEdge::one().getValueByPath(1, path)),
              Error::Kind::InvalidArgument);
  }
  EXPECT_EQ(test::value(mEdge::one().getValueByPath(1, "3ignored")), 1.);
}

TEST(EdgeFunctionality, NonpositiveExportThresholds) {
  auto dd = test::value(Package::create(1));
  const auto vector =
      test::value(makeStateFromVector(CVec{0.6, {0., 0.8}}, *dd));
  const auto matrix =
      test::value(dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, 0));
  for (const auto threshold : {0., -1., std::numeric_limits<fp>::quiet_NaN()}) {
    EXPECT_EQ(vector.getVector(threshold), vector.getVector());
    EXPECT_EQ(vector.getSparseVector(threshold), vector.getSparseVector());
    EXPECT_EQ(matrix.getMatrix(1, threshold), matrix.getMatrix(1));
    EXPECT_EQ(matrix.getSparseMatrix(1, threshold), matrix.getSparseMatrix(1));
  }
}

TEST(VectorFunctionality, GetVectorTerminal) {
  EXPECT_EQ(vEdge::zero().getVector(), CVec{0.});
  EXPECT_EQ(vEdge::one().getVector(), CVec{1.});
}

TEST(VectorFunctionality, GetVectorRoundtrip) {
  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));
  const auto stateVec = stateDD.getVector();
  EXPECT_EQ(stateVec, state);
}

TEST(VectorFunctionality, GetVectorTolerance) {
  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));
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
  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));
  const auto stateSparseVec = stateDD.getSparseVector();
  const auto stateVec = stateDD.getVector();
  for (const auto& [index, value] : stateSparseVec) {
    EXPECT_EQ(value, stateVec[index]);
  }
}

TEST(VectorFunctionality, GetSparseVectorTolerance) {
  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));
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
  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));
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

  auto dd = test::value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = test::value(makeStateFromVector(state, *dd));
  stateDD.addToVector(vec);
  EXPECT_EQ(vec, state);
}

TEST(VectorFunctionality, SizeTerminal) {
  EXPECT_EQ(vEdge::zero().size(), 1);
  EXPECT_EQ(vEdge::one().size(), 1);
}

TEST(VectorFunctionality, SizeBellState) {
  auto dd = test::value(Package::create(2));
  const CVec state = {SQRT2_2, 0., 0., SQRT2_2};
  const auto bell = test::value(makeStateFromVector(state, *dd));
  EXPECT_EQ(bell.size(), 4);
}

///-----------------------------------------------------------------------------
///                     \n Tests for matrix DDs \n
///-----------------------------------------------------------------------------

TEST(MatrixFunctionality, GetValueByPathTerminal) {
  EXPECT_EQ(test::value(mEdge::zero().getValueByPath(0, "0")), 0.);
  EXPECT_EQ(test::value(mEdge::one().getValueByPath(0, "0")), 1.);
}

TEST(MatrixFunctionality, GetValueByIndexTerminal) {
  EXPECT_EQ(test::value(mEdge::zero().getValueByIndex(0, 0, 0)), 0.);
  EXPECT_EQ(test::value(mEdge::one().getValueByIndex(0, 0, 0)), 1.);
}

TEST(MatrixFunctionality, ScalarAccessPreservesImplicitIdentities) {
  auto packageOwner = test::value(Package::create(3));
  auto& package = *packageOwner;
  const auto phase = package.cn.lookup(0.5, -0.5);
  auto gate = test::value(package.makeGateDD(GateMatrix{0., 1., 1., 0.}, 1));
  gate.w = phase;
  for (const auto& matrix : {mEdge::zero(), mEdge::terminal(phase), gate}) {
    const auto dense = matrix.getMatrix(3);
    for (size_t row = 0; row < dense.size(); ++row) {
      for (size_t col = 0; col < dense.size(); ++col) {
        std::string path(3, '0');
        for (size_t bit = 0; bit < path.size(); ++bit) {
          path[bit] = static_cast<char>('0' + (2 * ((row >> bit) & 1U)) +
                                        ((col >> bit) & 1U));
        }
        EXPECT_EQ(test::value(matrix.getValueByIndex(3, row, col)),
                  dense[row][col]);
        EXPECT_EQ(test::value(matrix.getValueByPath(3, path)), dense[row][col]);
      }
    }
  }
}

TEST(MatrixFunctionality, GetValueByIndexEndianness) {
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = test::value(dd->makeDDFromMatrix(mat));

  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = test::value(matDD.getValueByIndex(dd->qubits(), i, j));
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
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = test::value(dd->makeDDFromMatrix(mat));
  const auto matVec = matDD.getMatrix(dd->qubits());
  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = test::value(matDD.getValueByIndex(dd->qubits(), i, j));
      const auto ref = mat[i][j];
      EXPECT_NEAR(ref.real(), val.real(), 1e-10);
      EXPECT_NEAR(ref.imag(), val.imag(), 1e-10);
    }
  }
}

TEST(MatrixFunctionality, GetMatrixTolerance) {
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = test::value(dd->makeDDFromMatrix(mat));
  const auto matVec = matDD.getMatrix(dd->qubits(), std::sqrt(0.1));
  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = test::value(matDD.getValueByIndex(dd->qubits(), i, j));
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
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = test::value(dd->makeDDFromMatrix(mat));
  const auto matSparse = matDD.getSparseMatrix(dd->qubits());
  const auto matDense = matDD.getMatrix(dd->qubits());
  for (const auto& [index, value] : matSparse) {
    const auto val = matDense.at(index.first).at(index.second);
    EXPECT_NEAR(value.real(), val.real(), 1e-10);
    EXPECT_NEAR(value.imag(), val.imag(), 1e-10);
  }
}

TEST(MatrixFunctionality, GetSparseMatrixTolerance) {
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = test::value(dd->makeDDFromMatrix(mat));
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
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = test::value(dd->makeDDFromMatrix(mat));
  testing::internal::CaptureStdout();
  matDD.printMatrix(dd->qubits());
  const auto matStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(matStr, "(0.316,-0) (0.447,-0) (0.548,0) (0.632,0) \n"
                    "(-0.447,0) (-0.548,0) (0.632,0) (0.316,0) \n"
                    "(-0.548,0) (-0.632,0) (0.316,0) (0.447,0) \n"
                    "(-0.632,0) (-0.316,0) (-0.447,0) (0.548,0) \n");
}

TEST(MatrixFunctionality, TraversalUsesOneCallbackAcrossIdentityLevels) {
  size_t visited = 0;
  mEdge::one().traverseMatrix(
      {0., 1.}, 0, 0,
      [&visited,
       ordinal = size_t{0}](const size_t i, const size_t j,
                            const std::complex<fp>& amplitude) mutable {
        EXPECT_EQ(i, j);
        EXPECT_EQ(i, ordinal++);
        EXPECT_EQ(amplitude, (std::complex<fp>{0., 1.}));
        ++visited;
      },
      4);
  EXPECT_EQ(visited, 16U);
}

TEST(MatrixFunctionality, SizeTerminal) {
  EXPECT_EQ(mEdge::zero().size(), 1);
  EXPECT_EQ(mEdge::one().size(), 1);
}

TEST(MatrixFunctionality, SizeBellState) {
  auto dd = test::value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {SQRT2_2, 0., 0., SQRT2_2},
    {0., SQRT2_2, SQRT2_2, 0.},
    {0., SQRT2_2, -SQRT2_2, 0.},
    {SQRT2_2, 0., 0., -SQRT2_2},};
  // clang-format on

  const auto bell = test::value(dd->makeDDFromMatrix(mat));
  EXPECT_EQ(bell.size(), 3);
}

} // namespace dd
