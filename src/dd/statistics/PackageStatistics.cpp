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

#include <nlohmann/json.hpp>
#include <ostream>
#include <string>

namespace dd {

static constexpr auto V_NODE_MEMORY_MIB =
    static_cast<double>(sizeof(vNode)) / static_cast<double>(1ULL << 20U);
static constexpr auto M_NODE_MEMORY_MIB =
    static_cast<double>(sizeof(mNode)) / static_cast<double>(1ULL << 20U);
static constexpr auto REAL_NUMBER_MEMORY_MIB =
    static_cast<double>(sizeof(RealNumber)) / static_cast<double>(1ULL << 20U);

static constexpr auto V_EDGE_MEMORY_MIB =
    static_cast<double>(sizeof(Edge<vNode>)) / static_cast<double>(1ULL << 20U);
static constexpr auto M_EDGE_MEMORY_MIB =
    static_cast<double>(sizeof(Edge<mNode>)) / static_cast<double>(1ULL << 20U);

double computeActiveMemoryMiB(Package& package) {
  const auto [vectorNodes, matrixNodes, realNumbers] =
      package.computeActiveCounts();
  const auto vActiveEntries = static_cast<double>(vectorNodes);
  const auto mActiveEntries = static_cast<double>(matrixNodes);

  const auto vMemoryForNodes = vActiveEntries * V_NODE_MEMORY_MIB;
  const auto mMemoryForNodes = mActiveEntries * M_NODE_MEMORY_MIB;
  const auto memoryForNodes = vMemoryForNodes + mMemoryForNodes;

  const auto vMemoryForEdges = vActiveEntries * V_EDGE_MEMORY_MIB;
  const auto mMemoryForEdges = mActiveEntries * M_EDGE_MEMORY_MIB;
  const auto memoryForEdges = vMemoryForEdges + mMemoryForEdges;

  const auto activeRealNumbers = static_cast<double>(realNumbers);
  const auto memoryForRealNumbers = activeRealNumbers * REAL_NUMBER_MEMORY_MIB;

  return memoryForNodes + memoryForEdges + memoryForRealNumbers;
}

double computePeakMemoryMiB(const Package& package) {
  const auto vPeakUsedEntries =
      static_cast<double>(package.vMemoryManager.getStats().peakNumUsed);
  const auto mPeakUsedEntries =
      static_cast<double>(package.mMemoryManager.getStats().peakNumUsed);

  const auto vMemoryForNodes = vPeakUsedEntries * V_NODE_MEMORY_MIB;
  const auto mMemoryForNodes = mPeakUsedEntries * M_NODE_MEMORY_MIB;
  const auto memoryForNodes = vMemoryForNodes + mMemoryForNodes;

  const auto vMemoryForEdges = vPeakUsedEntries * V_EDGE_MEMORY_MIB;
  const auto mMemoryForEdges = mPeakUsedEntries * M_EDGE_MEMORY_MIB;
  const auto memoryForEdges = vMemoryForEdges + mMemoryForEdges;

  const auto peakRealNumbers =
      static_cast<double>(package.cMemoryManager.getStats().peakNumUsed);
  const auto memoryForRealNumbers = peakRealNumbers * REAL_NUMBER_MEMORY_MIB;

  return memoryForNodes + memoryForEdges + memoryForRealNumbers;
}

nlohmann::basic_json<> getStatistics(Package& package,
                                     const bool includeIndividualTables) {
  nlohmann::basic_json<> j;

  j["data_structure"] = getDataStructureStatistics();

  const auto [activeVectorNodes, activeMatrixNodes, activeRealNumbers] =
      package.computeActiveCounts();

  auto& vector = j["vector"];
  auto& vectorUniqueTable = vector["unique_table"];
  vectorUniqueTable =
      package.vUniqueTable.getStatsJson(includeIndividualTables);
  if (vectorUniqueTable != "unused") {
    vectorUniqueTable["total"]["num_active_entries"] = activeVectorNodes;
  }
  vector["memory_manager"] = package.vMemoryManager.getStats().json();

  auto& matrix = j["matrix"];
  auto& matrixUniqueTable = matrix["unique_table"];
  matrixUniqueTable =
      package.mUniqueTable.getStatsJson(includeIndividualTables);
  if (matrixUniqueTable != "unused") {
    matrixUniqueTable["total"]["num_active_entries"] = activeMatrixNodes;
  }
  matrix["memory_manager"] = package.mMemoryManager.getStats().json();

  auto& realNumbers = j["real_numbers"];
  auto& realNumbersUniqueTable = realNumbers["unique_table"];
  realNumbersUniqueTable = package.cUniqueTable.getStats().json();
  if (realNumbersUniqueTable != "unused") {
    realNumbersUniqueTable["num_active_entries"] = activeRealNumbers;
  }
  realNumbers["memory_manager"] = package.cMemoryManager.getStats().json();

  auto& computeTables = j["compute_tables"];
  computeTables["vector_add"] = package.vectorAdd.getStats().json();
  computeTables["matrix_add"] = package.matrixAdd.getStats().json();
  computeTables["matrix_conjugate_transpose"] =
      package.conjugateMatrixTranspose.getStats().json();
  computeTables["matrix_vector_mult"] =
      package.matrixVectorMultiplication.getStats().json();
  computeTables["matrix_matrix_mult"] =
      package.matrixMatrixMultiplication.getStats().json();
  computeTables["vector_kronecker"] = package.vectorKronecker.getStats().json();
  computeTables["matrix_kronecker"] = package.matrixKronecker.getStats().json();
  computeTables["vector_inner_product"] =
      package.vectorInnerProduct.getStats().json();

  j["active_memory_mib"] = computeActiveMemoryMiB(package);
  j["peak_memory_mib"] = computePeakMemoryMiB(package);

  return j;
}

nlohmann::basic_json<> getDataStructureStatistics() {
  nlohmann::basic_json<> j;

  // Information about key data structures
  // For every entry, we store the size in bytes and the alignment in bytes
  auto& ddPackage = j["Package"];
  ddPackage["size_B"] = sizeof(Package);
  ddPackage["alignment_B"] = alignof(Package);

  auto& vectorNode = j["vNode"];
  vectorNode["size_B"] = sizeof(vNode);
  vectorNode["alignment_B"] = alignof(vNode);

  auto& matrixNode = j["mNode"];
  matrixNode["size_B"] = sizeof(mNode);
  matrixNode["alignment_B"] = alignof(mNode);

  auto& vectorEdge = j["vEdge"];
  vectorEdge["size_B"] = sizeof(Edge<vNode>);
  vectorEdge["alignment_B"] = alignof(Edge<vNode>);

  auto& matrixEdge = j["mEdge"];
  matrixEdge["size_B"] = sizeof(Edge<mNode>);
  matrixEdge["alignment_B"] = alignof(Edge<mNode>);

  auto& realNumber = j["RealNumber"];
  realNumber["size_B"] = sizeof(RealNumber);
  realNumber["alignment_B"] = alignof(RealNumber);

  auto& complexValue = j["ComplexValue"];
  complexValue["size_B"] = sizeof(ComplexValue);
  complexValue["alignment_B"] = alignof(ComplexValue);

  auto& complex = j["Complex"];
  complex["size_B"] = sizeof(Complex);
  complex["alignment_B"] = alignof(Complex);

  auto& complexNumbers = j["ComplexNumbers"];
  complexNumbers["size_B"] = sizeof(ComplexNumbers);
  complexNumbers["alignment_B"] = alignof(ComplexNumbers);

  // Information about all the compute table entries
  // For every entry, we store the size in bytes and the alignment in bytes
  auto& ctEntries = j["ComplexTableEntries"];
  auto& vectorAdd = ctEntries["vector_add"];
  vectorAdd["size_B"] = sizeof(typename decltype(Package::vectorAdd)::Entry);
  vectorAdd["alignment_B"] =
      alignof(typename decltype(Package::vectorAdd)::Entry);

  auto& matrixAdd = ctEntries["matrix_add"];
  matrixAdd["size_B"] = sizeof(typename decltype(Package::matrixAdd)::Entry);
  matrixAdd["alignment_B"] =
      alignof(typename decltype(Package::matrixAdd)::Entry);

  auto& conjugateMatrixTranspose = ctEntries["conjugate_matrix_transpose"];
  conjugateMatrixTranspose["size_B"] =
      sizeof(typename decltype(Package::conjugateMatrixTranspose)::Entry);
  conjugateMatrixTranspose["alignment_B"] =
      alignof(typename decltype(Package::conjugateMatrixTranspose)::Entry);

  auto& matrixVectorMult = ctEntries["matrix_vector_mult"];
  matrixVectorMult["size_B"] =
      sizeof(typename decltype(Package::matrixVectorMultiplication)::Entry);
  matrixVectorMult["alignment_B"] =
      alignof(typename decltype(Package::matrixVectorMultiplication)::Entry);

  auto& matrixMatrixMult = ctEntries["matrix_matrix_mult"];
  matrixMatrixMult["size_B"] =
      sizeof(typename decltype(Package::matrixMatrixMultiplication)::Entry);
  matrixMatrixMult["alignment_B"] =
      alignof(typename decltype(Package::matrixMatrixMultiplication)::Entry);

  auto& vectorKronecker = ctEntries["vector_kronecker"];
  vectorKronecker["size_B"] =
      sizeof(typename decltype(Package::vectorKronecker)::Entry);
  vectorKronecker["alignment_B"] =
      alignof(typename decltype(Package::vectorKronecker)::Entry);

  auto& matrixKronecker = ctEntries["matrix_kronecker"];
  matrixKronecker["size_B"] =
      sizeof(typename decltype(Package::matrixKronecker)::Entry);
  matrixKronecker["alignment_B"] =
      alignof(typename decltype(Package::matrixKronecker)::Entry);

  auto& vectorInnerProduct = ctEntries["vector_inner_product"];
  vectorInnerProduct["size_B"] =
      sizeof(typename decltype(Package::vectorInnerProduct)::Entry);
  vectorInnerProduct["alignment_B"] =
      alignof(typename decltype(Package::vectorInnerProduct)::Entry);

  return j;
}

std::string getStatisticsString(Package& package) {
  return getStatistics(package).dump(2U);
}

void printStatistics(Package& package, std::ostream& os) {
  os << getStatisticsString(package);
}
} // namespace dd
