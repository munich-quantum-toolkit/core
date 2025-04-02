/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
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
static constexpr auto D_NODE_MEMORY_MIB =
    static_cast<double>(sizeof(dNode)) / static_cast<double>(1ULL << 20U);
static constexpr auto REAL_NUMBER_MEMORY_MIB =
    static_cast<double>(sizeof(RealNumber)) / static_cast<double>(1ULL << 20U);

static constexpr auto V_EDGE_MEMORY_MIB =
    static_cast<double>(sizeof(Edge<vNode>)) / static_cast<double>(1ULL << 20U);
static constexpr auto M_EDGE_MEMORY_MIB =
    static_cast<double>(sizeof(Edge<mNode>)) / static_cast<double>(1ULL << 20U);
static constexpr auto D_EDGE_MEMORY_MIB =
    static_cast<double>(sizeof(Edge<dNode>)) / static_cast<double>(1ULL << 20U);

double computeActiveMemoryMiB(const Package& package) {
  const auto vActiveEntries =
      static_cast<double>(package.vUniqueTable.getNumActiveEntries());
  const auto mActiveEntries =
      static_cast<double>(package.mUniqueTable.getNumActiveEntries());
  const auto dActiveEntries =
      static_cast<double>(package.dUniqueTable.getNumActiveEntries());

  const auto vMemoryForNodes = vActiveEntries * V_NODE_MEMORY_MIB;
  const auto mMemoryForNodes = mActiveEntries * M_NODE_MEMORY_MIB;
  const auto dMemoryForNodes = dActiveEntries * D_NODE_MEMORY_MIB;
  const auto memoryForNodes =
      vMemoryForNodes + mMemoryForNodes + dMemoryForNodes;

  const auto vMemoryForEdges = vActiveEntries * V_EDGE_MEMORY_MIB;
  const auto mMemoryForEdges = mActiveEntries * M_EDGE_MEMORY_MIB;
  const auto dMemoryForEdges = dActiveEntries * D_EDGE_MEMORY_MIB;
  const auto memoryForEdges =
      vMemoryForEdges + mMemoryForEdges + dMemoryForEdges;

  const auto activeRealNumbers =
      static_cast<double>(package.cUniqueTable.getStats().numActiveEntries);
  const auto memoryForRealNumbers = activeRealNumbers * REAL_NUMBER_MEMORY_MIB;

  return memoryForNodes + memoryForEdges + memoryForRealNumbers;
}

double computePeakMemoryMiB(const Package& package) {
  const auto vPeakUsedEntries =
      static_cast<double>(package.vMemoryManager.getStats().peakNumUsed);
  const auto mPeakUsedEntries =
      static_cast<double>(package.mMemoryManager.getStats().peakNumUsed);
  const auto dPeakUsedEntries =
      static_cast<double>(package.dMemoryManager.getStats().peakNumUsed);

  const auto vMemoryForNodes = vPeakUsedEntries * V_NODE_MEMORY_MIB;
  const auto mMemoryForNodes = mPeakUsedEntries * M_NODE_MEMORY_MIB;
  const auto dMemoryForNodes = dPeakUsedEntries * D_NODE_MEMORY_MIB;
  const auto memoryForNodes =
      vMemoryForNodes + mMemoryForNodes + dMemoryForNodes;

  const auto vMemoryForEdges = vPeakUsedEntries * V_EDGE_MEMORY_MIB;
  const auto mMemoryForEdges = mPeakUsedEntries * M_EDGE_MEMORY_MIB;
  const auto dMemoryForEdges = dPeakUsedEntries * D_EDGE_MEMORY_MIB;
  const auto memoryForEdges =
      vMemoryForEdges + mMemoryForEdges + dMemoryForEdges;

  const auto peakRealNumbers =
      static_cast<double>(package.cMemoryManager.getStats().peakNumUsed);
  const auto memoryForRealNumbers = peakRealNumbers * REAL_NUMBER_MEMORY_MIB;

  return memoryForNodes + memoryForEdges + memoryForRealNumbers;
}

nlohmann::basic_json<> getStatistics(const Package& package,
                                     const bool includeIndividualTables) {
  nlohmann::basic_json<> j;

  j["data_structure"] = getDataStructureStatistics();

  auto& vector = j["vector"];
  vector["unique_table"] =
      package.vUniqueTable.getStatsJson(includeIndividualTables);
  vector["memory_manager"] = package.vMemoryManager.getStats().json();

  auto& matrix = j["matrix"];
  matrix["unique_table"] =
      package.mUniqueTable.getStatsJson(includeIndividualTables);
  matrix["memory_manager"] = package.mMemoryManager.getStats().json();

  auto& densityMatrix = j["density_matrix"];
  densityMatrix["unique_table"] =
      package.dUniqueTable.getStatsJson(includeIndividualTables);
  densityMatrix["memory_manager"] = package.dMemoryManager.getStats().json();

  auto& realNumbers = j["real_numbers"];
  realNumbers["unique_table"] = package.cUniqueTable.getStats().json();
  realNumbers["memory_manager"] = package.cMemoryManager.getStats().json();

  auto& computeTables = j["compute_tables"];
  computeTables["vector_add"] = package.vectorAdd.getStats().json();
  computeTables["matrix_add"] = package.matrixAdd.getStats().json();
  computeTables["density_matrix_add"] = package.densityAdd.getStats().json();
  computeTables["matrix_conjugate_transpose"] =
      package.conjugateMatrixTranspose.getStats().json();
  computeTables["matrix_vector_mult"] =
      package.matrixVectorMultiplication.getStats().json();
  computeTables["matrix_matrix_mult"] =
      package.matrixMatrixMultiplication.getStats().json();
  computeTables["density_density_mult"] =
      package.densityDensityMultiplication.getStats().json();
  computeTables["vector_kronecker"] = package.vectorKronecker.getStats().json();
  computeTables["matrix_kronecker"] = package.matrixKronecker.getStats().json();
  computeTables["vector_inner_product"] =
      package.vectorInnerProduct.getStats().json();
  computeTables["stochastic_noise_operations"] =
      package.stochasticNoiseOperationCache.getStats().json();
  computeTables["density_noise_operations"] =
      package.densityNoise.getStats().json();

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

  auto& densityNode = j["dNode"];
  densityNode["size_B"] = sizeof(dNode);
  densityNode["alignment_B"] = alignof(dNode);

  auto& vectorEdge = j["vEdge"];
  vectorEdge["size_B"] = sizeof(Edge<vNode>);
  vectorEdge["alignment_B"] = alignof(Edge<vNode>);

  auto& matrixEdge = j["mEdge"];
  matrixEdge["size_B"] = sizeof(Edge<mNode>);
  matrixEdge["alignment_B"] = alignof(Edge<mNode>);

  auto& densityEdge = j["dEdge"];
  densityEdge["size_B"] = sizeof(Edge<dNode>);
  densityEdge["alignment_B"] = alignof(Edge<dNode>);

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

  auto& densityAdd = ctEntries["density_add"];
  densityAdd["size_B"] = sizeof(typename decltype(Package::densityAdd)::Entry);
  densityAdd["alignment_B"] =
      alignof(typename decltype(Package::densityAdd)::Entry);

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

  auto& densityDensityMult = ctEntries["density_density_mult"];
  densityDensityMult["size_B"] =
      sizeof(typename decltype(Package::densityDensityMultiplication)::Entry);
  densityDensityMult["alignment_B"] =
      alignof(typename decltype(Package::densityDensityMultiplication)::Entry);

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

std::string getStatisticsString(const Package& package) {
  return getStatistics(package).dump(2U);
}

void printStatistics(const Package& package, std::ostream& os) {
  os << getStatisticsString(package);
}
} // namespace dd
