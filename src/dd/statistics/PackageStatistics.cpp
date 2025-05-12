/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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

static constexpr auto REAL_NUMBER_MEMORY_MIB =
    static_cast<double>(sizeof(RealNumber)) / static_cast<double>(1ULL << 20U);

double computeActiveMemoryMiB(const Package& package) {
  const auto vectorMem = package.vectors().computeActiveMemoryMiB();
  const auto matrixMem = package.matrices().computeActiveMemoryMiB();
  const auto densityMem = package.densities().computeActiveMemoryMiB();

  const auto activeRealNumbers =
      static_cast<double>(package.cUt.getStats().numActiveEntries);
  const auto memoryForRealNumbers = activeRealNumbers * REAL_NUMBER_MEMORY_MIB;

  return vectorMem + matrixMem + densityMem + memoryForRealNumbers;
}

double computePeakMemoryMiB(const Package& package) {
  const auto vectorMem = package.vectors().computePeakMemoryMiB();
  const auto matrixMem = package.matrices().computePeakMemoryMiB();
  const auto densityMem = package.densities().computePeakMemoryMiB();

  const auto peakRealNumbers =
      static_cast<double>(package.cMm.getStats().peakNumUsed);
  const auto memoryForRealNumbers = peakRealNumbers * REAL_NUMBER_MEMORY_MIB;

  return vectorMem + matrixMem + densityMem + memoryForRealNumbers;
}

nlohmann::basic_json<> getStatistics(const Package& package,
                                     const bool includeIndividualTables) {
  nlohmann::basic_json<> j;

  j["data_structure"] = getDataStructureStatistics();

  auto& vector = j["vector"];
  package.vectors().addStatsJson(vector, includeIndividualTables);

  auto& matrix = j["matrix"];
  package.matrices().addStatsJson(matrix, includeIndividualTables);

  auto& densityMatrix = j["density_matrix"];
  package.densities().addStatsJson(densityMatrix, includeIndividualTables);

  auto& realNumbers = j["real_numbers"];
  realNumbers["unique_table"] = package.cUt.getStats().json();
  realNumbers["memory_manager"] = package.cMm.getStats().json();

  // TODO: auto& computeTables = j["compute_tables"];
  // TODO: computeTables["vector_add"] = package.vectorAdd.getStats().json();
  // TODO: computeTables["matrix_add"] = package.matrixAdd.getStats().json();
  // TODO: computeTables["density_matrix_add"] =
  // package.densityAdd.getStats().json();
  // TODO: computeTables["matrix_conjugate_transpose"] =
  // TODO:     package.conjugateMatrixTranspose.getStats().json();
  // TODO: computeTables["matrix_vector_mult"] =
  // TODO:     package.matrixVectorMultiplication.getStats().json();
  // TODO: computeTables["matrix_matrix_mult"] =
  // TODO:     package.matrixMatrixMultiplication.getStats().json();
  // TODO: computeTables["density_density_mult"] =
  // TODO:     package.densityDensityMultiplication.getStats().json();
  // TODO: computeTables["vector_kronecker"] =
  // package.vectorKronecker.getStats().json();
  // TODO: computeTables["matrix_kronecker"] =
  // package.matrixKronecker.getStats().json();
  // TODO: computeTables["vector_inner_product"] =
  // TODO:     package.vectorInnerProduct.getStats().json();
  // TODO: computeTables["stochastic_noise_operations"] =
  // TODO:     package.stochasticNoiseOperationCache.getStats().json();
  // TODO: computeTables["density_noise_operations"] =
  // TODO:     package.densityNoise.getStats().json();

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
  // TODO: auto& vectorAdd = ctEntries["vector_add"];
  // TODO: vectorAdd["size_B"] = sizeof(typename
  // decltype(Package::vectorAdd)::Entry);
  // TODO: vectorAdd["alignment_B"] =
  // TODO:     alignof(typename decltype(Package::vectorAdd)::Entry);
  //  TODO:
  // TODO: auto& matrixAdd = ctEntries["matrix_add"];
  // TODO: matrixAdd["size_B"] = sizeof(typename
  // decltype(Package::matrixAdd)::Entry);
  // TODO: matrixAdd["alignment_B"] =
  // TODO:     alignof(typename decltype(Package::matrixAdd)::Entry);
  //  TODO:
  // TODO: auto& densityAdd = ctEntries["density_add"];
  // TODO: densityAdd["size_B"] = sizeof(typename
  // decltype(Package::densityAdd)::Entry);
  // TODO: densityAdd["alignment_B"] =
  // TODO:     alignof(typename decltype(Package::densityAdd)::Entry);

  // TODO: auto& conjugateMatrixTranspose =
  // ctEntries["conjugate_matrix_transpose"];
  // TODO: conjugateMatrixTranspose["size_B"] =
  // TODO:     sizeof(typename
  // decltype(Package::conjugateMatrixTranspose)::Entry);
  // TODO: conjugateMatrixTranspose["alignment_B"] =
  // TODO:     alignof(typename
  // decltype(Package::conjugateMatrixTranspose)::Entry);

  // auto& matrixVectorMult = ctEntries["matrix_vector_mult"];
  // matrixVectorMult["size_B"] =
  //     sizeof(typename decltype(Package::matrixVectorMultiplication)::Entry);
  // matrixVectorMult["alignment_B"] =
  //     alignof(typename decltype(Package::matrixVectorMultiplication)::Entry);
  //
  // auto& matrixMatrixMult = ctEntries["matrix_matrix_mult"];
  // matrixMatrixMult["size_B"] =
  //     sizeof(typename decltype(Package::matrixMatrixMultiplication)::Entry);
  // matrixMatrixMult["alignment_B"] =
  //     alignof(typename decltype(Package::matrixMatrixMultiplication)::Entry);
  //
  // auto& densityDensityMult = ctEntries["density_density_mult"];
  // densityDensityMult["size_B"] =
  //     sizeof(typename
  //     decltype(Package::densityDensityMultiplication)::Entry);
  // densityDensityMult["alignment_B"] =
  //     alignof(typename
  //     decltype(Package::densityDensityMultiplication)::Entry);
  //
  // auto& vectorKronecker = ctEntries["vector_kronecker"];
  // vectorKronecker["size_B"] =
  //     sizeof(typename decltype(Package::vectorKronecker)::Entry);
  // vectorKronecker["alignment_B"] =
  //     alignof(typename decltype(Package::vectorKronecker)::Entry);
  //
  // auto& matrixKronecker = ctEntries["matrix_kronecker"];
  // matrixKronecker["size_B"] =
  //     sizeof(typename decltype(Package::matrixKronecker)::Entry);
  // matrixKronecker["alignment_B"] =
  //     alignof(typename decltype(Package::matrixKronecker)::Entry);
  //
  // auto& vectorInnerProduct = ctEntries["vector_inner_product"];
  // vectorInnerProduct["size_B"] =
  //     sizeof(typename decltype(Package::vectorInnerProduct)::Entry);
  // vectorInnerProduct["alignment_B"] =
  //     alignof(typename decltype(Package::vectorInnerProduct)::Entry);
  //
  return j;
}

std::string getStatisticsString(const Package& package) {
  return getStatistics(package).dump(2U);
}

void printStatistics(const Package& package, std::ostream& os) {
  os << getStatisticsString(package);
}
} // namespace dd
