/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/PackageStatistics.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"

#include "StatisticsJson.hpp"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <ostream>
#include <string>
#include <utility>

namespace dd {
namespace {

constexpr auto V_NODE_MEMORY_MIB =
    static_cast<double>(sizeof(vNode)) / static_cast<double>(1ULL << 20U);
constexpr auto M_NODE_MEMORY_MIB =
    static_cast<double>(sizeof(mNode)) / static_cast<double>(1ULL << 20U);
constexpr auto REAL_NUMBER_MEMORY_MIB =
    static_cast<double>(sizeof(RealNumber)) / static_cast<double>(1ULL << 20U);

double estimateMemoryMiB(const Package::ActiveCounts& counts) {
  return (static_cast<double>(counts.vector) * V_NODE_MEMORY_MIB) +
         (static_cast<double>(counts.matrix) * M_NODE_MEMORY_MIB) +
         (static_cast<double>(counts.reals) * REAL_NUMBER_MEMORY_MIB);
}
} // namespace

double computeActiveMemoryMiB(Package& package) {
  return estimateMemoryMiB(package.computeActiveCounts());
}

double computePeakMemoryMiB(const Package& package) {
  return estimateMemoryMiB({
      .vector = package.vMemoryManager.getStats().peakNumUsed,
      .matrix = package.mMemoryManager.getStats().peakNumUsed,
      .reals = package.cMemoryManager.getStats().peakNumUsed,
  });
}

namespace {
[[nodiscard]] llvm::json::Value getDataStructureStatistics();

[[nodiscard]] llvm::json::Value
getStatistics(Package& package, const bool includeIndividualTables) {
  llvm::json::Object j;

  j["data_structure"] = getDataStructureStatistics();

  const auto activeCounts = package.computeActiveCounts();

  llvm::json::Object vector;
  auto& vectorUniqueTable = vector["unique_table"];
  vectorUniqueTable = toJson(package.vUniqueTable, includeIndividualTables);
  if (vectorUniqueTable != "unused") {
    (*vectorUniqueTable.getAsObject()->getObject(
        "total"))["num_active_entries"] = activeCounts.vector;
  }
  vector["memory_manager"] = toJson(package.vMemoryManager.getStats());

  llvm::json::Object matrix;
  auto& matrixUniqueTable = matrix["unique_table"];
  matrixUniqueTable = toJson(package.mUniqueTable, includeIndividualTables);
  if (matrixUniqueTable != "unused") {
    (*matrixUniqueTable.getAsObject()->getObject(
        "total"))["num_active_entries"] = activeCounts.matrix;
  }
  matrix["memory_manager"] = toJson(package.mMemoryManager.getStats());

  llvm::json::Object realNumbers;
  auto& realNumbersUniqueTable = realNumbers["unique_table"];
  realNumbersUniqueTable = toJson(package.cUniqueTable.getStats());
  if (realNumbersUniqueTable != "unused") {
    (*realNumbersUniqueTable.getAsObject())["num_active_entries"] =
        activeCounts.reals;
  }
  realNumbers["memory_manager"] = toJson(package.cMemoryManager.getStats());

  llvm::json::Object computeTables;
  computeTables["vector_add"] = toJson(package.vectorAdd.getStats());
  computeTables["matrix_add"] = toJson(package.matrixAdd.getStats());
  computeTables["matrix_conjugate_transpose"] =
      toJson(package.conjugateMatrixTranspose.getStats());
  computeTables["matrix_vector_mult"] =
      toJson(package.matrixVectorMultiplication.getStats());
  computeTables["matrix_matrix_mult"] =
      toJson(package.matrixMatrixMultiplication.getStats());
  computeTables["vector_kronecker"] =
      toJson(package.vectorKronecker.getStats());
  computeTables["matrix_kronecker"] =
      toJson(package.matrixKronecker.getStats());
  computeTables["vector_inner_product"] =
      toJson(package.vectorInnerProduct.getStats());

  j["vector"] = std::move(vector);
  j["matrix"] = std::move(matrix);
  j["real_numbers"] = std::move(realNumbers);
  j["compute_tables"] = std::move(computeTables);

  j["active_memory_mib"] = estimateMemoryMiB(activeCounts);
  j["peak_memory_mib"] = computePeakMemoryMiB(package);

  return j;
}

template <class T> [[nodiscard]] llvm::json::Object typeStatistics() {
  return llvm::json::Object{{"size_B", sizeof(T)}, {"alignment_B", alignof(T)}};
}

[[nodiscard]] llvm::json::Value getDataStructureStatistics() {
  return llvm::json::Object{
      {"Package", typeStatistics<Package>()},
      {"vNode", typeStatistics<vNode>()},
      {"mNode", typeStatistics<mNode>()},
      {"vEdge", typeStatistics<Edge<vNode>>()},
      {"mEdge", typeStatistics<Edge<mNode>>()},
      {"RealNumber", typeStatistics<RealNumber>()},
      {"ComplexValue", typeStatistics<ComplexValue>()},
      {"Complex", typeStatistics<Complex>()},
      {"ComplexNumbers", typeStatistics<ComplexNumbers>()},
      {
          "ComplexTableEntries",
          llvm::json::Object{
              {
                  "vector_add",
                  typeStatistics<decltype(Package::vectorAdd)::Entry>(),
              },
              {
                  "matrix_add",
                  typeStatistics<decltype(Package::matrixAdd)::Entry>(),
              },
              {
                  "conjugate_matrix_transpose",
                  typeStatistics<
                      decltype(Package::conjugateMatrixTranspose)::Entry>(),
              },
              {
                  "matrix_vector_mult",
                  typeStatistics<
                      decltype(Package::matrixVectorMultiplication)::Entry>(),
              },
              {
                  "matrix_matrix_mult",
                  typeStatistics<
                      decltype(Package::matrixMatrixMultiplication)::Entry>(),
              },
              {
                  "vector_kronecker",
                  typeStatistics<decltype(Package::vectorKronecker)::Entry>(),
              },
              {
                  "matrix_kronecker",
                  typeStatistics<decltype(Package::matrixKronecker)::Entry>(),
              },
              {
                  "vector_inner_product",
                  typeStatistics<
                      decltype(Package::vectorInnerProduct)::Entry>(),
              },
          },
      },
  };
}

} // namespace

std::string getDataStructureStatisticsString() {
  return llvm::formatv("{0:2}", getDataStructureStatistics()).str();
}

std::string getStatisticsString(Package& package,
                                const bool includeIndividualTables) {
  return llvm::formatv("{0:2}", getStatistics(package, includeIndividualTables))
      .str();
}

void printStatistics(Package& package, std::ostream& os) {
  os << getStatisticsString(package);
}
} // namespace dd
