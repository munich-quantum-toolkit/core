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
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

using mqt::test::errorCode;
using mqt::test::value;

namespace dd {

//-----------------------------------------------------------------------------
//                     \n Tests for vector DDs \n
//-----------------------------------------------------------------------------

TEST(VectorFunctionality, GetValueByPathTerminal) {
  EXPECT_EQ(value(vEdge::zero().getValueByPath(0, "0")), 0.);
  EXPECT_EQ(value(vEdge::one().getValueByPath(0, "0")), 1.);
}

TEST(VectorFunctionality, GetValueByIndexTerminal) {
  EXPECT_EQ(value(dd::getValueByIndex(vEdge::zero(), 0)), 0.);
  EXPECT_EQ(value(dd::getValueByIndex(vEdge::one(), 0)), 1.);
}

TEST(VectorFunctionality, GetValueByIndexEndianness) {
  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));

  for (std::size_t i = 0U; i < state.size(); ++i) {
    EXPECT_EQ(state[i], value(dd::getValueByIndex(stateDD, i)));
  }
}

TEST(VectorFunctionality, WideIndices) {
  constexpr auto digits = std::numeric_limits<size_t>::digits;
  auto dd = value(Package::create(digits + 1U));
  const auto ones =
      value(makeBasisState(digits, std::vector<bool>(digits, true), *dd));
  EXPECT_EQ(
      value(dd::getValueByIndex(ones, std::numeric_limits<size_t>::max())), 1.);
  const auto zero = value(makeZeroState(digits + 1U, *dd));
  EXPECT_EQ(value(dd::getValueByIndex(zero, 0)), 1.);
  EXPECT_EQ(
      value(dd::getValueByIndex(zero, std::numeric_limits<size_t>::max())), 0.);
  EXPECT_EQ(errorCode([&] { return dd::getValueByIndex(vEdge::one(), 1); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] {
              return dd::getValueByIndex(value(makeZeroState(3, *dd)), 8);
            }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] {
              return dd::traverseVector(zero,
                                        [](size_t, const std::complex<fp>&) {});
            }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return dd::getVector(ones, 2.); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return dd::getSparseVector(zero, 2.); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return dd::printVector(ones); }),
            std::errc::result_out_of_range);
  CVec amplitudes(1);
  EXPECT_EQ(errorCode([&] { return dd::addToVector(ones, amplitudes); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return dd::addToVector(zero, amplitudes); }),
            std::errc::result_out_of_range);
}

TEST(VectorFunctionality, InvalidPaths) {
  auto dd = value(Package::create(2));
  const auto zero = value(makeZeroState(2, *dd));
  EXPECT_EQ(errorCode([&] { return zero.getValueByPath(1, "0"); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return zero.getValueByPath(2, "2"); }),
            std::errc::result_out_of_range);
  for (const auto* path : {"20", "91", "/0", "x0"}) {
    EXPECT_EQ(errorCode([&] { return zero.getValueByPath(2, path); }),
              std::errc::invalid_argument);
  }
  EXPECT_EQ(value(zero.getValueByPath(2, "00ignored")), 1.);
}

TEST(MatrixFunctionality, WideIndices) {
  constexpr auto digits = std::numeric_limits<size_t>::digits;
  auto dd = value(Package::create(digits + 1U));
  const auto gate =
      value(dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, 0));
  const auto twoQubit =
      value(dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, 1));
  for (const size_t width : {0U, 1U}) {
    EXPECT_EQ(errorCode([&] {
                return twoQubit.getValueByPath(width, std::string(width, '0'));
              }),
              std::errc::result_out_of_range);
    EXPECT_EQ(
        errorCode([&] { return dd::getValueByIndex(twoQubit, width, 0, 0); }),
        std::errc::result_out_of_range);
    EXPECT_EQ(errorCode([&] { return dd::getMatrix(twoQubit, width); }),
              std::errc::result_out_of_range);
    EXPECT_EQ(errorCode([&] { return dd::getSparseMatrix(twoQubit, width); }),
              std::errc::result_out_of_range);
    EXPECT_EQ(errorCode([&] { return dd::printMatrix(twoQubit, width); }),
              std::errc::result_out_of_range);
    EXPECT_EQ(errorCode([&] {
                return dd::traverseMatrix(
                    twoQubit, 1., 0, 0,
                    [](size_t, size_t, const std::complex<fp>&) {}, width);
              }),
              std::errc::result_out_of_range);
  }
  EXPECT_EQ(value(dd::getValueByIndex(gate, digits + 1U, 0, 1)),
            std::complex<fp>(0, -1));
  EXPECT_EQ(value(dd::getValueByIndex(gate, digits + 1U, 1, 0)),
            std::complex<fp>(0, 1));
  EXPECT_EQ(value(dd::getValueByIndex(gate, digits + 1U, 2, 1)), 0.);
  const auto highGate =
      value(dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, digits));
  EXPECT_EQ(value(dd::getValueByIndex(highGate, digits + 1U, 0, 0)), 0.);
  EXPECT_EQ(value(dd::getValueByIndex(mEdge::one(), digits,
                                      std::numeric_limits<size_t>::max(),
                                      std::numeric_limits<size_t>::max())),
            1.);
  EXPECT_EQ(errorCode([&] { return dd::getValueByIndex(gate, 1, 2, 0); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(
      errorCode([&] { return dd::getValueByIndex(mEdge::one(), 1, 0, 2); }),
      std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return dd::getMatrix(mEdge::zero(), digits, 2.); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] {
              return dd::getSparseMatrix(mEdge::zero(), digits + 1U, 2.);
            }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] {
              return dd::traverseMatrix(
                  mEdge::zero(), 1., 0, 0,
                  [](size_t, size_t, const std::complex<fp>&) {}, digits + 1U,
                  2.);
            }),
            std::errc::result_out_of_range);
  EXPECT_EQ(errorCode([&] { return dd::printMatrix(mEdge::zero(), digits); }),
            std::errc::result_out_of_range);
}

TEST(MatrixFunctionality, InvalidPaths) {
  EXPECT_EQ(errorCode([&] { return mEdge::one().getValueByPath(1, ""); }),
            std::errc::result_out_of_range);
  for (const auto* path : {"4", "9", "/", "x"}) {
    EXPECT_EQ(errorCode([&] { return mEdge::one().getValueByPath(1, path); }),
              std::errc::invalid_argument);
  }
  EXPECT_EQ(value(mEdge::one().getValueByPath(1, "3ignored")), 1.);
}

TEST(EdgeFunctionality, NonpositiveExportThresholds) {
  auto dd = value(Package::create(1));
  const auto vector = value(makeStateFromVector(CVec{0.6, {0., 0.8}}, *dd));
  const auto matrix =
      value(dd->makeGateDD(GateMatrix{0., {0., -1.}, {0., 1.}, 0.}, 0));
  for (const auto threshold : {0., -1., std::numeric_limits<fp>::quiet_NaN()}) {
    EXPECT_EQ(value(dd::getVector(vector, threshold)),
              value(dd::getVector(vector)));
    EXPECT_EQ(value(dd::getSparseVector(vector, threshold)),
              value(dd::getSparseVector(vector)));
    EXPECT_EQ(value(dd::getMatrix(matrix, 1, threshold)),
              value(dd::getMatrix(matrix, 1)));
    EXPECT_EQ(value(dd::getSparseMatrix(matrix, 1, threshold)),
              value(dd::getSparseMatrix(matrix, 1)));
  }
}

TEST(VectorFunctionality, GetVectorTerminal) {
  EXPECT_EQ(value(dd::getVector(vEdge::zero())), CVec{0.});
  EXPECT_EQ(value(dd::getVector(vEdge::one())), CVec{1.});
}

TEST(VectorFunctionality, GetVectorRoundtrip) {
  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));
  const auto stateVec = value(dd::getVector(stateDD));
  EXPECT_EQ(stateVec, state);
}

TEST(VectorFunctionality, GetVectorTolerance) {
  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));
  const auto stateVec = value(dd::getVector(stateDD, std::sqrt(0.1)));
  EXPECT_EQ(stateVec, state);
  const auto stateVec2 =
      value(dd::getVector(stateDD, std::sqrt(0.1) + RealNumber::eps));
  EXPECT_NE(stateVec2, state);
  EXPECT_EQ(stateVec2[0], 0.);
}

TEST(VectorFunctionality, GetSparseVectorTerminal) {
  const auto zero = SparseCVec{{0, 0}};
  EXPECT_EQ(value(dd::getSparseVector(vEdge::zero())), zero);
  const auto one = SparseCVec{{0, 1}};
  EXPECT_EQ(value(dd::getSparseVector(vEdge::one())), one);
}

TEST(VectorFunctionality, TraversalPreservesBasisOrderAndThreshold) {
  auto package = value(Package::create(2));
  const auto state = value(
      makeStateFromVector({0.5, 0., {0., -0.5}, std::sqrt(0.5)}, *package));
  std::vector<size_t> indices;
  CVec amplitudes;
  const auto collect = [&](const size_t index, const std::complex<fp>& value) {
    indices.push_back(index);
    amplitudes.push_back(value);
  };
  value(dd::traverseVector(state, collect));
  EXPECT_EQ(indices, (std::vector<size_t>{0, 2, 3}));
  ASSERT_EQ(amplitudes.size(), 3U);
  EXPECT_NEAR(std::abs(amplitudes[0] - 0.5), 0., 1e-12);
  EXPECT_NEAR(std::abs(amplitudes[1] - std::complex<fp>(0., -0.5)), 0., 1e-12);
  EXPECT_NEAR(std::abs(amplitudes[2] - std::sqrt(0.5)), 0., 1e-12);

  indices.clear();
  value(dd::traverseVector(state, collect, 0.6));
  EXPECT_EQ(indices, (std::vector<size_t>{3}));
}

TEST(VectorFunctionality, GetSparseVectorConsistency) {
  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));
  const auto stateSparseVec = value(dd::getSparseVector(stateDD));
  const auto stateVec = value(dd::getVector(stateDD));
  for (const auto& [index, value] : stateSparseVec) {
    EXPECT_EQ(value, stateVec[index]);
  }
}

TEST(VectorFunctionality, GetSparseVectorTolerance) {
  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));
  const auto stateSparseVec =
      value(dd::getSparseVector(stateDD, std::sqrt(0.1)));
  for (const auto& [index, value] : stateSparseVec) {
    EXPECT_EQ(value, state[index]);
  }
  const auto stateSparseVec2 =
      value(dd::getSparseVector(stateDD, std::sqrt(0.1) + RealNumber::eps));
  EXPECT_NE(stateSparseVec2, stateSparseVec);
  EXPECT_EQ(stateSparseVec2.count(0), 0);
}

TEST(VectorFunctionality, PrintVectorTerminal) {
  llvm::outs().flush();
  testing::internal::CaptureStdout();
  value(dd::printVector(vEdge::zero()));
  const auto zeroStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(zeroStr, "0: (0,0)\n");
  llvm::outs().flush();
  testing::internal::CaptureStdout();
  value(dd::printVector(vEdge::one()));
  const auto oneStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(oneStr, "0: (1,0)\n");
}

TEST(VectorFunctionality, PrintVector) {
  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));
  llvm::outs().flush();
  testing::internal::CaptureStdout();
  value(dd::printVector(stateDD));
  const auto stateStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(stateStr,
            "00: (0.316,0)\n01: (0.447,0)\n10: (0.548,0)\n11: (0.632,0)\n");
}

TEST(VectorFunctionality, AddToVectorTerminal) {
  CVec vec = {0.};
  value(dd::addToVector(vEdge::one(), vec));
  EXPECT_EQ(vec, CVec{1.});
  CVec empty;
  EXPECT_EQ(errorCode([&] { return dd::addToVector(vEdge::one(), empty); }),
            std::errc::result_out_of_range);
}

TEST(VectorFunctionality, AddToVector) {
  CVec vec = {0., 0., 0., 0.};

  auto dd = value(Package::create(2));
  const CVec state = {
      std::sqrt(0.1),
      std::sqrt(0.2),
      std::sqrt(0.3),
      std::sqrt(0.4),
  };
  const auto stateDD = value(makeStateFromVector(state, *dd));
  const CVec initial = {1., 2.};
  auto tooSmall = initial;
  EXPECT_EQ(errorCode([&] { return dd::addToVector(stateDD, tooSmall); }),
            std::errc::result_out_of_range);
  EXPECT_EQ(tooSmall, initial);
  value(dd::addToVector(stateDD, vec));
  EXPECT_EQ(vec, state);
}

TEST(VectorFunctionality, SizeTerminal) {
  EXPECT_EQ(vEdge::zero().size(), 1);
  EXPECT_EQ(vEdge::one().size(), 1);
}

TEST(VectorFunctionality, SizeBellState) {
  auto dd = value(Package::create(2));
  const CVec state = {SQRT2_2, 0., 0., SQRT2_2};
  const auto bell = value(makeStateFromVector(state, *dd));
  EXPECT_EQ(bell.size(), 4);
}

//-----------------------------------------------------------------------------
//                     \n Tests for matrix DDs \n
//-----------------------------------------------------------------------------

TEST(MatrixFunctionality, GetValueByPathTerminal) {
  EXPECT_EQ(value(mEdge::zero().getValueByPath(0, "0")), 0.);
  EXPECT_EQ(value(mEdge::one().getValueByPath(0, "0")), 1.);
}

TEST(MatrixFunctionality, GetValueByIndexTerminal) {
  EXPECT_EQ(value(dd::getValueByIndex(mEdge::zero(), 0, 0, 0)), 0.);
  EXPECT_EQ(value(dd::getValueByIndex(mEdge::one(), 0, 0, 0)), 1.);
}

TEST(MatrixFunctionality, ScalarAccessPreservesImplicitIdentities) {
  auto packageOwner = value(Package::create(3));
  auto& package = *packageOwner;
  const auto phase = package.cn.lookup(0.5, -0.5);
  auto gate = value(package.makeGateDD(GateMatrix{0., 1., 1., 0.}, 1));
  gate.w = phase;
  for (const auto& matrix : {mEdge::zero(), mEdge::terminal(phase), gate}) {
    const auto dense = value(dd::getMatrix(matrix, 3));
    for (size_t row = 0; row < dense.size(); ++row) {
      for (size_t col = 0; col < dense.size(); ++col) {
        std::string path(3, '0');
        for (size_t bit = 0; bit < path.size(); ++bit) {
          path[bit] = static_cast<char>('0' + (2 * ((row >> bit) & 1U)) +
                                        ((col >> bit) & 1U));
        }
        EXPECT_EQ(value(dd::getValueByIndex(matrix, 3, row, col)),
                  dense[row][col]);
        EXPECT_EQ(value(matrix.getValueByPath(3, path)), dense[row][col]);
      }
    }
  }
}

TEST(MatrixFunctionality, GetValueByIndexEndianness) {
  auto dd = value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = value(dd->makeDDFromMatrix(mat));

  for (std::size_t i = 0U; i < mat.size(); ++i) {
    for (std::size_t j = 0U; j < mat.size(); ++j) {
      const auto val = value(dd::getValueByIndex(matDD, dd->qubits(), i, j));
      const auto ref = mat[i][j];
      EXPECT_NEAR(ref.real(), val.real(), 1e-10);
      EXPECT_NEAR(ref.imag(), val.imag(), 1e-10);
    }
  }
}

TEST(MatrixFunctionality, GetMatrixTerminal) {
  EXPECT_EQ(value(dd::getMatrix(mEdge::zero(), 0)), CMat{{0.}});
  EXPECT_EQ(value(dd::getMatrix(mEdge::one(), 0)), CMat{{1.}});
}

TEST(MatrixFunctionality, GetMatrixTolerance) {
  auto dd = value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = value(dd->makeDDFromMatrix(mat));
  const auto matVec = value(dd::getMatrix(matDD, dd->qubits(), std::sqrt(0.1)));
  const auto matVec2 = value(
      dd::getMatrix(matDD, dd->qubits(), std::sqrt(0.1) + RealNumber::eps));
  EXPECT_NE(matVec2, matVec);
  EXPECT_EQ(matVec2[0][0], 0.);
  EXPECT_EQ(matVec2[1][3], 0.);
  EXPECT_EQ(matVec2[2][2], 0.);
  EXPECT_EQ(matVec2[3][1], 0.);
}

TEST(MatrixFunctionality, GetSparseMatrixTerminal) {
  const auto zero = SparseCMat{{{0, 0}, 0.}};
  EXPECT_EQ(value(dd::getSparseMatrix(mEdge::zero(), 0)), zero);
  const auto one = SparseCMat{{{0, 0}, 1.}};
  EXPECT_EQ(value(dd::getSparseMatrix(mEdge::one(), 0)), one);
}

TEST(MatrixFunctionality, GetSparseMatrixConsistency) {
  auto dd = value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = value(dd->makeDDFromMatrix(mat));
  const auto matSparse = value(dd::getSparseMatrix(matDD, dd->qubits()));
  const auto matDense = value(dd::getMatrix(matDD, dd->qubits()));
  for (const auto& [index, value] : matSparse) {
    const auto val = matDense.at(index.first).at(index.second);
    EXPECT_NEAR(value.real(), val.real(), 1e-10);
    EXPECT_NEAR(value.imag(), val.imag(), 1e-10);
  }
}

TEST(MatrixFunctionality, GetSparseMatrixTolerance) {
  auto dd = value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = value(dd->makeDDFromMatrix(mat));
  const auto matSparse =
      value(dd::getSparseMatrix(matDD, dd->qubits(), std::sqrt(0.1)));
  const auto matDense = value(dd::getMatrix(matDD, dd->qubits()));
  for (const auto& [index, value] : matSparse) {
    const auto val = matDense.at(index.first).at(index.second);
    EXPECT_NEAR(value.real(), val.real(), 1e-10);
    EXPECT_NEAR(value.imag(), val.imag(), 1e-10);
  }
  const auto matSparse2 = value(dd::getSparseMatrix(
      matDD, dd->qubits(), std::sqrt(0.1) + RealNumber::eps));
  EXPECT_NE(matSparse2, matSparse);
  EXPECT_EQ(matSparse2.count({0, 0}), 0);
  EXPECT_EQ(matSparse2.count({1, 3}), 0);
  EXPECT_EQ(matSparse2.count({2, 2}), 0);
  EXPECT_EQ(matSparse2.count({3, 1}), 0);
}

TEST(MatrixFunctionality, PrintMatrixTerminal) {
  llvm::outs().flush();
  testing::internal::CaptureStdout();
  value(dd::printMatrix(mEdge::zero(), 0));
  const auto zeroStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(zeroStr, "(0,0)\n");
  llvm::outs().flush();
  testing::internal::CaptureStdout();
  value(dd::printMatrix(mEdge::one(), 0));
  const auto oneStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(oneStr, "(1,0)\n");
}

TEST(MatrixFunctionality, PrintMatrix) {
  auto dd = value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {std::sqrt(0.1),  std::sqrt(0.2),  std::sqrt(0.3),  std::sqrt(0.4)},
    {-std::sqrt(0.2), -std::sqrt(0.3), std::sqrt(0.4),  std::sqrt(0.1)},
    {-std::sqrt(0.3), -std::sqrt(0.4), std::sqrt(0.1),  std::sqrt(0.2)},
    {-std::sqrt(0.4), -std::sqrt(0.1), -std::sqrt(0.2), std::sqrt(0.3)},};
  // clang-format on

  const auto matDD = value(dd->makeDDFromMatrix(mat));
  llvm::outs().flush();
  testing::internal::CaptureStdout();
  value(dd::printMatrix(matDD, dd->qubits()));
  const auto matStr = testing::internal::GetCapturedStdout();
  EXPECT_EQ(matStr, "(0.316,-0) (0.447,-0) (0.548,0) (0.632,0) \n"
                    "(-0.447,0) (-0.548,0) (0.632,0) (0.316,0) \n"
                    "(-0.548,0) (-0.632,0) (0.316,0) (0.447,0) \n"
                    "(-0.632,0) (-0.316,0) (-0.447,0) (0.548,0) \n");
}

TEST(MatrixFunctionality, TraversalUsesOneCallbackAcrossIdentityLevels) {
  size_t visited = 0;
  value(dd::traverseMatrix(
      mEdge::one(), {0., 1.}, 0, 0,
      [&visited,
       ordinal = size_t{0}](const size_t i, const size_t j,
                            const std::complex<fp>& amplitude) mutable {
        EXPECT_EQ(i, j);
        EXPECT_EQ(i, ordinal++);
        EXPECT_EQ(amplitude, (std::complex<fp>{0., 1.}));
        ++visited;
      },
      4));
  EXPECT_EQ(visited, 16U);
}

TEST(MatrixFunctionality, SizeTerminal) {
  EXPECT_EQ(mEdge::zero().size(), 1);
  EXPECT_EQ(mEdge::one().size(), 1);
}

TEST(MatrixFunctionality, SizeBellState) {
  auto dd = value(Package::create(2));
  // clang-format off
  const CMat mat = {
    {SQRT2_2, 0., 0., SQRT2_2},
    {0., SQRT2_2, SQRT2_2, 0.},
    {0., SQRT2_2, -SQRT2_2, 0.},
    {SQRT2_2, 0., 0., -SQRT2_2},};
  // clang-format on

  const auto bell = value(dd->makeDDFromMatrix(mat));
  EXPECT_EQ(bell.size(), 3);
}

} // namespace dd
