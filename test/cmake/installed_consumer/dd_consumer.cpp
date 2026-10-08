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
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
void check(const bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
} // namespace

int main() {
  try {
    dd::Package package(1);
    auto* vectorNode = package.vMemoryManager.get<dd::vNode>();
    vectorNode->v = 0;
    const auto vector = dd::normalize(
        vectorNode, std::array{dd::vEdge::zero(), dd::vEdge::one()},
        package.vMemoryManager, package.cn);
    check(dd::getVector(vector) == dd::CVec{0., 1.}, "Dense vector mismatch");
    check(dd::getSparseVector(vector) == dd::SparseCVec{{1, 1.}},
          "Sparse vector mismatch");
    check(dd::getValueByIndex(vector, 0) == 0. &&
              dd::getValueByIndex(vector, 1) == 1.,
          "Vector index mismatch");
    dd::CVec sum{1., 1.};
    dd::addToVector(vector, sum);
    check(sum == dd::CVec{1., 2.}, "Vector accumulation mismatch");
    dd::printVector(vector);

    auto* matrixNode = package.mMemoryManager.get<dd::mNode>();
    matrixNode->v = 0;
    const auto matrix = dd::normalize(matrixNode,
                                      std::array{
                                          dd::mEdge::zero(),
                                          dd::mEdge::one(),
                                          dd::mEdge::one(),
                                          dd::mEdge::zero(),
                                      },
                                      package.mMemoryManager, package.cn);
    check(dd::getMatrix(matrix, 1) == dd::CMat{{0., 1.}, {1., 0.}},
          "Dense matrix mismatch");
    check(dd::getSparseMatrix(matrix, 1) ==
              dd::SparseCMat{{{0, 1}, 1.}, {{1, 0}, 1.}},
          "Sparse matrix mismatch");
    check(dd::getValueByIndex(matrix, 1, 0, 1) == 1. &&
              dd::getValueByIndex(matrix, 1, 0, 0) == 0.,
          "Matrix index mismatch");
    check(dd::isIdentity(dd::mEdge::one()) && !dd::isIdentity(matrix),
          "Matrix identity mismatch");
    dd::printMatrix(matrix, 1);

    size_t visited = 0;
    dd::traverseMatrix(
        matrix, {0., 1.}, 0, 0,
        [&visited,
         ordinal = size_t{0}](const size_t row, const size_t column,
                              const std::complex<dd::fp>& value) mutable {
          check(row == ordinal++ && column == (row ^ 1U) &&
                    value == std::complex<dd::fp>{0., 1.},
                "Matrix traversal mismatch");
          ++visited;
        },
        2);
    check(visited == 4, "Matrix traversal missed identity levels");

    auto* cachedVectorNode = package.vMemoryManager.get<dd::vNode>();
    cachedVectorNode->v = 0;
    const auto cachedVector =
        dd::normalize(cachedVectorNode,
                      std::array{
                          dd::vCachedEdge::terminal(dd::ComplexValue{2.}),
                          dd::vCachedEdge::zero(),
                      },
                      package.vMemoryManager, package.cn);
    check(dd::getVector(package.cn.lookup(cachedVector)) == dd::CVec{2., 0.},
          "Cached vector normalization mismatch");

    auto* cachedMatrixNode = package.mMemoryManager.get<dd::mNode>();
    cachedMatrixNode->v = 0;
    const auto scaledOne = dd::mCachedEdge::terminal(dd::ComplexValue{2.});
    const auto cachedMatrix = dd::normalize(cachedMatrixNode,
                                            std::array{
                                                dd::mCachedEdge::zero(),
                                                scaledOne,
                                                scaledOne,
                                                dd::mCachedEdge::zero(),
                                            },
                                            package.mMemoryManager, package.cn);
    check(dd::getMatrix(package.cn.lookup(cachedMatrix), 1) ==
              dd::CMat{{0., 2.}, {2., 0.}},
          "Cached matrix normalization mismatch");
    check(dd::isIdentity(dd::mCachedEdge::one()) &&
              !dd::isIdentity(cachedMatrix) &&
              !dd::isIdentity(scaledOne, false),
          "Cached matrix identity mismatch");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
