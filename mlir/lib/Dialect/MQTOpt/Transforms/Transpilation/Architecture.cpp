/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"

#include "ir/Definitions.hpp"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace mqt::ir::opt {
[[nodiscard]] llvm::SmallVector<std::size_t>
Architecture::shortestPathBetween(qc::Qubit u, qc::Qubit v) const {
  llvm::SmallVector<std::size_t> path;

  if (prev_[u][v] == UINT64_MAX) {
    throw std::domain_error("No path between qubits " + std::to_string(u) +
                            " and " + std::to_string(v));
  }

  path.push_back(v);
  while (u != v) {
    v = prev_[u][v];
    path.push_back(v);
  }

  return path;
}

[[nodiscard]] std::size_t Architecture::distanceBetween(qc::Qubit u,
                                                        qc::Qubit v) const {
  if (dist_[u][v] == UINT64_MAX) {
    throw std::domain_error("No path between qubits " + std::to_string(u) +
                            " and " + std::to_string(v));
  }
  return dist_[u][v];
}

void Architecture::floydWarshallWithPathReconstruction() {
  for (const auto& [u, v] : couplingMap_) {
    dist_[u][v] = 1;
    prev_[u][v] = u;
  }
  for (std::size_t v = 0; v < nqubits(); ++v) {
    dist_[v][v] = 0;
    prev_[v][v] = v;
  }

  for (std::size_t k = 0; k < nqubits(); ++k) {
    for (std::size_t i = 0; i < nqubits(); ++i) {
      for (std::size_t j = 0; j < nqubits(); ++j) {
        if (dist_[i][k] == UINT64_MAX || dist_[k][j] == UINT64_MAX) {
          continue; // avoid overflow with "infinite" distances
        }
        const std::size_t sum = dist_[i][k] + dist_[k][j];
        if (dist_[i][j] > sum) {
          dist_[i][j] = sum;
          prev_[i][j] = prev_[k][j];
        }
      }
    }
  }
}

[[nodiscard]] llvm::SmallVector<qc::Qubit>
Architecture::neighboursOf(qc::Qubit u) const {
  llvm::SmallVector<qc::Qubit> n;
  for (const auto [i, j] : couplingMap_) {
    if (i == u) {
      n.push_back(j);
    }
  }
  return n;
}

std::unique_ptr<Architecture> getArchitecture(const ArchitectureName& name) {
  switch (name) {
  case ArchitectureName::MQTTest: {
    // 0 -- 1
    // |    |
    // 2 -- 3
    // |    |
    // 4 -- 5

    const Architecture::CouplingMap couplingMap{
        {0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 1}, {2, 3},
        {3, 2}, {2, 4}, {4, 2}, {3, 5}, {5, 3}, {4, 5}, {5, 4}};

    return std::make_unique<Architecture>("MQT-Test", 6, couplingMap);
  }
  case ArchitectureName::IBMFalcon: {
    const Architecture::CouplingMap couplingMap{
        {0, 1},     {0, 14},    {1, 0},     {1, 2},     {2, 1},     {2, 3},
        {3, 2},     {3, 4},     {4, 3},     {4, 5},     {4, 15},    {5, 4},
        {5, 6},     {6, 5},     {6, 7},     {7, 6},     {7, 8},     {8, 7},
        {8, 16},    {9, 10},    {10, 9},    {10, 11},   {11, 10},   {11, 12},
        {12, 11},   {12, 13},   {12, 17},   {13, 12},   {14, 0},    {14, 18},
        {15, 4},    {15, 22},   {16, 8},    {16, 26},   {17, 12},   {17, 30},
        {18, 14},   {18, 19},   {19, 18},   {19, 20},   {20, 19},   {20, 21},
        {20, 33},   {21, 20},   {21, 22},   {22, 15},   {22, 21},   {22, 23},
        {23, 22},   {23, 24},   {24, 23},   {24, 25},   {24, 34},   {25, 24},
        {25, 26},   {26, 16},   {26, 25},   {26, 27},   {27, 26},   {27, 28},
        {28, 27},   {28, 29},   {28, 35},   {29, 28},   {29, 30},   {30, 17},
        {30, 29},   {30, 31},   {31, 30},   {31, 32},   {32, 31},   {32, 36},
        {33, 20},   {33, 39},   {34, 24},   {34, 43},   {35, 28},   {35, 47},
        {36, 32},   {36, 51},   {37, 38},   {37, 52},   {38, 37},   {38, 39},
        {39, 33},   {39, 38},   {39, 40},   {40, 39},   {40, 41},   {41, 40},
        {41, 42},   {41, 53},   {42, 41},   {42, 43},   {43, 34},   {43, 42},
        {43, 44},   {44, 43},   {44, 45},   {45, 44},   {45, 46},   {45, 54},
        {46, 45},   {46, 47},   {47, 35},   {47, 46},   {47, 48},   {48, 47},
        {48, 49},   {49, 48},   {49, 50},   {49, 55},   {50, 49},   {50, 51},
        {51, 36},   {51, 50},   {52, 37},   {52, 56},   {53, 41},   {53, 60},
        {54, 45},   {54, 64},   {55, 49},   {55, 68},   {56, 52},   {56, 57},
        {57, 56},   {57, 58},   {58, 57},   {58, 59},   {58, 71},   {59, 58},
        {59, 60},   {60, 53},   {60, 59},   {60, 61},   {61, 60},   {61, 62},
        {62, 61},   {62, 63},   {62, 72},   {63, 62},   {63, 64},   {64, 54},
        {64, 63},   {64, 65},   {65, 64},   {65, 66},   {66, 65},   {66, 67},
        {66, 73},   {67, 66},   {67, 68},   {68, 55},   {68, 67},   {68, 69},
        {69, 68},   {69, 70},   {70, 69},   {70, 74},   {71, 58},   {71, 77},
        {72, 62},   {72, 81},   {73, 66},   {73, 85},   {74, 70},   {74, 89},
        {75, 76},   {75, 90},   {76, 75},   {76, 77},   {77, 71},   {77, 76},
        {77, 78},   {78, 77},   {78, 79},   {79, 78},   {79, 80},   {79, 91},
        {80, 79},   {80, 81},   {81, 72},   {81, 80},   {81, 82},   {82, 81},
        {82, 83},   {83, 82},   {83, 84},   {83, 92},   {84, 83},   {84, 85},
        {85, 73},   {85, 84},   {85, 86},   {86, 85},   {86, 87},   {87, 86},
        {87, 88},   {87, 93},   {88, 87},   {88, 89},   {89, 74},   {89, 88},
        {90, 75},   {90, 94},   {91, 79},   {91, 98},   {92, 83},   {92, 102},
        {93, 87},   {93, 106},  {94, 90},   {94, 95},   {95, 94},   {95, 96},
        {96, 95},   {96, 97},   {96, 109},  {97, 96},   {97, 98},   {98, 91},
        {98, 97},   {98, 99},   {99, 98},   {99, 100},  {100, 99},  {100, 101},
        {100, 110}, {101, 100}, {101, 102}, {102, 92},  {102, 101}, {102, 103},
        {103, 102}, {103, 104}, {104, 103}, {104, 105}, {104, 111}, {105, 104},
        {105, 106}, {106, 93},  {106, 105}, {106, 107}, {107, 106}, {107, 108},
        {108, 107}, {108, 112}, {109, 96},  {110, 100}, {110, 118}, {111, 104},
        {111, 122}, {112, 108}, {112, 126}, {113, 114}, {114, 113}, {114, 115},
        {115, 114}, {115, 116}, {116, 115}, {116, 117}, {117, 116}, {117, 118},
        {118, 110}, {118, 117}, {118, 119}, {119, 118}, {119, 120}, {120, 119},
        {120, 121}, {121, 120}, {121, 122}, {122, 111}, {122, 121}, {122, 123},
        {123, 122}, {123, 124}, {124, 123}, {124, 125}, {125, 124}, {125, 126},
        {126, 112}, {126, 125}};

    return std::make_unique<Architecture>("MQT-Test", 127, couplingMap);
  }
  }

  throw std::invalid_argument("Unsupported architecture.");
}
}; // namespace mqt::ir::opt
