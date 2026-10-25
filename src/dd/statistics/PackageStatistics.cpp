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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <ostream>
#include <string>

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

void writeStatistics(llvm::json::OStream& json, Package& package,
                     const bool includeIndividualTables) {
  const auto activeCounts = package.computeActiveCounts();
  const auto attribute = [&](llvm::StringRef name, const auto& statistics) {
    json.attributeBegin(name);
    writeJson(json, statistics);
    json.attributeEnd();
  };
  json.object([&] {
    json.attribute("data_structure", getDataStructureStatistics());
    json.attributeObject("vector", [&] {
      json.attributeBegin("unique_table");
      writeJson(json, package.vUniqueTable, includeIndividualTables,
                activeCounts.vector);
      json.attributeEnd();
      attribute("memory_manager", package.vMemoryManager.getStats());
    });
    json.attributeObject("matrix", [&] {
      json.attributeBegin("unique_table");
      writeJson(json, package.mUniqueTable, includeIndividualTables,
                activeCounts.matrix);
      json.attributeEnd();
      attribute("memory_manager", package.mMemoryManager.getStats());
    });
    json.attributeObject("real_numbers", [&] {
      json.attributeBegin("unique_table");
      writeJson(json, package.cUniqueTable.getStats(), activeCounts.reals);
      json.attributeEnd();
      attribute("memory_manager", package.cMemoryManager.getStats());
    });
    json.attributeObject("compute_tables", [&] {
      attribute("vector_add", package.vectorAdd.getStats());
      attribute("matrix_add", package.matrixAdd.getStats());
      attribute("matrix_conjugate_transpose",
                package.conjugateMatrixTranspose.getStats());
      attribute("matrix_vector_mult",
                package.matrixVectorMultiplication.getStats());
      attribute("matrix_matrix_mult",
                package.matrixMatrixMultiplication.getStats());
      attribute("vector_kronecker", package.vectorKronecker.getStats());
      attribute("matrix_kronecker", package.matrixKronecker.getStats());
      attribute("vector_inner_product", package.vectorInnerProduct.getStats());
    });
    writeStatistic(json, "active_memory_mib", estimateMemoryMiB(activeCounts));
    writeStatistic(json, "peak_memory_mib", computePeakMemoryMiB(package));
  });
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
  std::string result;
  llvm::raw_string_ostream out(result);
  llvm::json::OStream json(out, 2);
  writeStatistics(json, package, includeIndividualTables);
  return result;
}

void printStatistics(Package& package, std::ostream& os) {
  os << getStatisticsString(package);
}
} // namespace dd
