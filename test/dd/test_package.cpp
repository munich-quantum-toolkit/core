/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/Export.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace qc::literals;

namespace dd {
TEST(DDPackageTest, RequestInvalidPackageSize) {
  EXPECT_THROW(auto dd =
                   std::make_unique<Package>(Package::MAX_POSSIBLE_QUBITS + 2),
               std::invalid_argument);
}

TEST(DDPackageTest, TrivialTest) {
  auto dd = std::make_unique<Package>(2);
  EXPECT_EQ(dd->qubits(), 2);

  auto xGate = getDD(qc::StandardOperation(0, qc::X), *dd);
  auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);

  ASSERT_EQ(hGate.getValueByPath(1, "0"), SQRT2_2);

  auto zeroState = dd->makeZeroState(1);
  auto hState = dd->multiply(hGate, zeroState);
  auto oneState = dd->multiply(xGate, zeroState);

  ASSERT_EQ(dd->fidelity(zeroState, oneState), 0.0);
  // repeat the same calculation — triggering compute table hit
  ASSERT_EQ(dd->fidelity(zeroState, oneState), 0.0);
  ASSERT_NEAR(dd->fidelity(zeroState, hState), 0.5, RealNumber::eps);
  ASSERT_NEAR(dd->fidelity(oneState, hState), 0.5, RealNumber::eps);
}

TEST(DDPackageTest, BellState) {
  auto dd = std::make_unique<Package>(2);

  auto hGate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  bellState.printVector();

  // repeated calculation is practically for free
  auto bellState2 = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  EXPECT_EQ(bellState, bellState2);

  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "00"), SQRT2_2);
  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "01"), 0.);
  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "10"), 0.);
  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "11"), SQRT2_2);

  ASSERT_EQ(bellState.getValueByIndex(0), SQRT2_2);
  ASSERT_EQ(bellState.getValueByIndex(1), 0.);
  ASSERT_EQ(bellState.getValueByIndex(2), 0.);
  ASSERT_EQ(bellState.getValueByIndex(3), SQRT2_2);

  auto goalState = CVec{{SQRT2_2, 0.}, {0., 0.}, {0., 0.}, {SQRT2_2, 0.}};
  ASSERT_EQ(bellState.getVector(), goalState);

  ASSERT_DOUBLE_EQ(dd->fidelity(zeroState, bellState), 0.5);

  export2Dot(bellState, "bell_state_colored_labels.dot", true, true, false,
             false, false);
  export2Dot(bellState, "bell_state_colored_labels_classic.dot", true, true,
             true, false, false);
  export2Dot(bellState, "bell_state_mono_labels.dot", false, true, false, false,
             false);
  export2Dot(bellState, "bell_state_mono_labels_classic.dot", false, true, true,
             false, false);
  export2Dot(bellState, "bell_state_colored.dot", true, false, false, false,
             false);
  export2Dot(bellState, "bell_state_colored_classic.dot", true, false, true,
             false, false);
  export2Dot(bellState, "bell_state_mono.dot", false, false, false, false,
             false);
  export2Dot(bellState, "bell_state_mono_classic.dot", false, false, true,
             false, false);
  export2Dot(bellState, "bell_state_memory.dot", false, true, true, true,
             false);
  exportEdgeWeights(bellState, std::cout);

  const auto filenames = {
      "bell_state_colored_labels.dot", "bell_state_colored_labels_classic.dot",
      "bell_state_mono_labels.dot",    "bell_state_mono_labels_classic.dot",
      "bell_state_colored.dot",        "bell_state_colored_classic.dot",
      "bell_state_mono.dot",           "bell_state_mono_classic.dot",
      "bell_state_memory.dot"};

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  printStatistics(*dd);
}

TEST(DDPackageTest, QFTState) {
  // Create a package with 3 qubits
  auto dd = std::make_unique<Package>(3);

  // Simulate a QFT on 3 qubits
  auto h0Gate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto s0Gate = getDD(qc::StandardOperation(1_pc, 0, qc::S), *dd);
  auto t0Gate = getDD(qc::StandardOperation(2_pc, 0, qc::T), *dd);
  auto h1Gate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto s1Gate = getDD(qc::StandardOperation(2_pc, 1, qc::S), *dd);
  auto h2Gate = getDD(qc::StandardOperation(2, qc::H), *dd);
  auto swapGate =
      getDD(qc::StandardOperation(qc::Targets{0, 2}, qc::SWAP), *dd);

  auto qftOp = dd->multiply(s0Gate, h0Gate);
  qftOp = dd->multiply(t0Gate, qftOp);
  qftOp = dd->multiply(h1Gate, qftOp);
  qftOp = dd->multiply(s1Gate, qftOp);
  qftOp = dd->multiply(h2Gate, qftOp);

  qftOp = dd->multiply(swapGate, qftOp);
  auto qftState = dd->multiply(qftOp, dd->makeZeroState(3));

  qftState.printVector();

  for (size_t qubit = 0; qubit < 7; ++qubit) {
    ASSERT_NEAR(qftState.getValueByIndex(qubit).real(), 0.5 * SQRT2_2,
                RealNumber::eps);
    ASSERT_EQ(qftState.getValueByIndex(qubit).imag(), 0);
  }

  // export in all different variations
  export2Dot(qftState, "qft_state_colored_labels.dot", true, true, false, false,
             false);
  export2Dot(qftState, "qft_state_colored_labels_classic.dot", true, true, true,
             false, false);
  export2Dot(qftState, "qft_state_mono_labels.dot", false, true, false, false,
             false);
  export2Dot(qftState, "qft_state_mono_labels_classic.dot", false, true, true,
             false, false);
  export2Dot(qftState, "qft_state_colored.dot", true, false, false, false,
             false);
  export2Dot(qftState, "qft_state_colored_classic.dot", true, false, true,
             false, false);
  export2Dot(qftState, "qft_state_mono.dot", false, false, false, false, false);
  export2Dot(qftState, "qft_state_mono_classic.dot", false, false, true, false,
             false);
  export2Dot(qftState, "qft_state_memory.dot", false, true, true, true, false);
  exportEdgeWeights(qftState, std::cout);

  export2Dot(qftOp, "qft_op_polar_colored_labels.dot", true, true, false, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_colored_labels_classic.dot", true, true, true,
             false, false, true);
  export2Dot(qftOp, "qft_op_polar_mono_labels.dot", false, true, false, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_mono_labels_classic.dot", false, true, true,
             false, false, true);
  export2Dot(qftOp, "qft_op_polar_colored.dot", true, false, false, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_colored_classic.dot", true, false, true,
             false, false, true);
  export2Dot(qftOp, "qft_op_polar_mono.dot", false, false, false, false, false,
             true);
  export2Dot(qftOp, "qft_op_polar_mono_classic.dot", false, false, true, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_memory.dot", false, true, true, true, false,
             true);

  export2Dot(qftOp, "qft_op_rectangular_colored_labels.dot", true, true, false,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_colored_labels_classic.dot", true, true,
             true, false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono_labels.dot", false, true, false,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono_labels_classic.dot", false, true,
             true, false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_colored.dot", true, false, false, false,
             false, false);
  export2Dot(qftOp, "qft_op_rectangular_colored_classic.dot", true, false, true,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono.dot", false, false, false, false,
             false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono_classic.dot", false, false, true,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_memory.dot", false, true, true, true,
             false, false);

  const auto filenames = {"qft_state_colored_labels.dot",
                          "qft_state_colored_labels_classic.dot",
                          "qft_state_mono_labels.dot",
                          "qft_state_mono_labels_classic.dot",
                          "qft_state_colored.dot",
                          "qft_state_colored_classic.dot",
                          "qft_state_mono.dot",
                          "qft_state_mono_classic.dot",
                          "qft_state_memory.dot",
                          "qft_op_polar_colored_labels.dot",
                          "qft_op_polar_colored_labels_classic.dot",
                          "qft_op_polar_mono_labels.dot",
                          "qft_op_polar_mono_labels_classic.dot",
                          "qft_op_polar_colored.dot",
                          "qft_op_polar_colored_classic.dot",
                          "qft_op_polar_mono.dot",
                          "qft_op_polar_mono_classic.dot",
                          "qft_op_polar_memory.dot",
                          "qft_op_rectangular_colored_labels.dot",
                          "qft_op_rectangular_colored_labels_classic.dot",
                          "qft_op_rectangular_mono_labels.dot",
                          "qft_op_rectangular_mono_labels_classic.dot",
                          "qft_op_rectangular_colored.dot",
                          "qft_op_rectangular_colored_classic.dot",
                          "qft_op_rectangular_mono.dot",
                          "qft_op_rectangular_mono_classic.dot",
                          "qft_op_rectangular_memory.dot"};

  // cleanup files
  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }
  printStatistics(*dd);
}

TEST(DDPackageTest, CorruptedBellState) {
  auto dd = std::make_unique<Package>(2);

  auto hGate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  bellState.w = dd->cn.lookup(0.5, 0);
  // prints a warning
  std::mt19937_64 mt; // NOLINT(cert-msc51-cpp)
  std::cout << dd->measureAll(bellState, false, mt) << "\n";

  bellState.w = Complex::zero();

  ASSERT_THROW(dd->measureAll(bellState, false, mt), std::runtime_error);

  ASSERT_THROW(dd->measureOneCollapsing(bellState, 0, mt), std::runtime_error);
}

TEST(DDPackageTest, InvalidStandardOperation) {
  auto dd = std::make_unique<Package>();
  const std::vector<std::pair<qc::Targets, qc::OpType>> invalidOps{
      {{qc::Targets{}, qc::I},
       {qc::Targets{0, 1}, qc::I},
       {qc::Targets{}, qc::SWAP},
       {qc::Targets{0}, qc::SWAP},
       {qc::Targets{0, 1, 2}, qc::SWAP},
       {qc::Targets{0, 1}, qc::OpTypeEnd}}};
  for (const auto& [targets, type] : invalidOps) {
    ASSERT_THROW(getDD(qc::StandardOperation(targets, type), *dd),
                 std::invalid_argument);
  }
  ASSERT_THROW(opToSingleQubitGateMatrix(qc::SWAP), std::invalid_argument);
  ASSERT_THROW(opToSingleQubitGateMatrix(qc::OpTypeEnd), std::invalid_argument);
  ASSERT_THROW(opToTwoQubitGateMatrix(qc::I), std::invalid_argument);
  ASSERT_THROW(opToTwoQubitGateMatrix(qc::OpTypeEnd), std::invalid_argument);
}

TEST(DDPackageTest, PrintNoneGateType) {
  std::ostringstream oss;
  oss << qc::None;
  EXPECT_EQ(oss.str(), "none");
}

TEST(DDPackageTest, NegativeControl) {
  auto dd = std::make_unique<Package>(2);

  auto xGate = getDD(qc::StandardOperation(1_nc, 0, qc::X), *dd);
  auto zeroState = dd->makeZeroState(2);
  auto state01 = dd->multiply(xGate, zeroState);
  EXPECT_EQ(state01.getValueByIndex(0b01).real(), 1.);
}

TEST(DDPackageTest, IdentityTrace) {
  auto dd = std::make_unique<Package>(4);
  auto fullTrace = dd->trace(Package::makeIdent(), 4);

  ASSERT_EQ(fullTrace.r, 1.);
}

TEST(DDPackageTest, CNotKronTrace) {
  auto dd = std::make_unique<Package>(4);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);
  auto cxGateKron = dd->kronecker(cxGate, cxGate, 2);
  auto fullTrace = dd->trace(cxGateKron, 4);
  ASSERT_EQ(fullTrace, 0.25);
}

TEST(DDPackageTest, PartialIdentityTrace) {
  auto dd = std::make_unique<Package>(2);
  auto tr = dd->partialTrace(Package::makeIdent(), {false, true});
  auto mul = dd->multiply(tr, tr);
  EXPECT_EQ(RealNumber::val(mul.w.r), 1.);
}

TEST(DDPackageTest, PartialSWapMatTrace) {
  auto dd = std::make_unique<Package>(2);
  auto swapGate =
      getDD(qc::StandardOperation(qc::Targets{0, 1}, qc::SWAP), *dd);
  auto ptr = dd->partialTrace(swapGate, {true, false});
  auto fullTrace = dd->trace(ptr, 1);
  auto fullTraceOriginal = dd->trace(swapGate, 2);
  EXPECT_EQ(RealNumber::val(ptr.w.r), 0.5);
  // Check that successively tracing out subsystems is the same as computing the
  // full trace from the beginning
  EXPECT_EQ(fullTrace, fullTraceOriginal);
}

TEST(DDPackageTest, PartialTraceKeepInnerQubits) {
  // Check that the partial trace computation is correct when tracing out the
  // outer qubits only. This test shows that we should avoid storing
  // non-eliminated nodes in the compute table, as this would prevent their
  // proper elimination in subsequent trace calls.

  constexpr std::size_t numQubits = 8;
  auto dd = std::make_unique<Package>(numQubits);
  const auto swapGate =
      getDD(qc::StandardOperation(qc::Targets{0, 1}, qc::SWAP), *dd);
  auto swapKron = swapGate;
  for (std::size_t i = 0; i < 3; ++i) {
    swapKron = dd->kronecker(swapKron, swapGate, 2);
  }
  auto fullTraceOriginal = dd->trace(swapKron, numQubits);
  auto ptr = dd->partialTrace(
      swapKron, {true, true, false, false, false, false, true, true});
  auto fullTrace = dd->trace(ptr, 4);
  EXPECT_EQ(RealNumber::val(ptr.w.r), 0.25);
  EXPECT_EQ(fullTrace.r, 0.0625);
  // Check that successively tracing out subsystems is the same as computing the
  // full trace from the beginning
  EXPECT_EQ(fullTrace, fullTraceOriginal);
}

TEST(DDPackageTest, TraceComplexity) {
  // Check that the full trace computation scales with the number of nodes
  // instead of paths in the DD due to the usage of a compute table
  for (std::size_t numQubits = 1; numQubits <= 10; ++numQubits) {
    auto dd = std::make_unique<Package>(numQubits);
    auto& computeTable = dd->getTraceComputeTable<mNode>();
    const auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
    auto hKron = hGate;
    for (std::size_t i = 0; i < numQubits - 1; ++i) {
      hKron = dd->kronecker(hKron, hGate, 1);
    }
    dd->trace(hKron, numQubits);
    const auto& stats = computeTable.getStats();
    ASSERT_EQ(stats.lookups, (2 * numQubits) - 1);
    ASSERT_EQ(stats.hits, numQubits - 1);
  }
}

TEST(DDPackageTest, KeepBottomQubitsPartialTraceComplexity) {
  // Check that during the trace computation, once a level is reached
  // where the remaining qubits should not be eliminated, the function does not
  // recurse further but immediately returns the current CachedEdge<Node>.
  constexpr std::size_t numQubits = 8;
  auto dd = std::make_unique<Package>(numQubits);
  auto& uniqueTable = dd->getUniqueTable<mNode>();
  const auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto hKron = hGate;
  for (std::size_t i = 0; i < numQubits - 1; ++i) {
    hKron = dd->kronecker(hKron, hGate, 1);
  }

  constexpr std::size_t maxNodeVal = 6;
  std::array<std::size_t, maxNodeVal> lookupValues{};

  for (std::size_t i = 0; i < maxNodeVal; ++i) {
    // Store the number of lookups performed so far for the six bottom qubits
    lookupValues[i] = uniqueTable.getStats(i).lookups;
  }
  dd->partialTrace(hKron,
                   {false, false, false, false, false, false, true, true});
  for (std::size_t i = 0; i < maxNodeVal; ++i) {
    // Check that the partial trace computation performs no additional lookups
    // on the bottom qubits that are not eliminated
    ASSERT_EQ(uniqueTable.getStats(i).lookups, lookupValues[i]);
  }
}

TEST(DDPackageTest, PartialTraceComplexity) {
  // In the worst case, the partial trace computation scales with the number of
  // paths in the DD. This situation arises particularly when tracing out the
  // bottom qubits.
  constexpr std::size_t numQubits = 9;
  auto dd = std::make_unique<Package>(numQubits);
  auto& uniqueTable = dd->getUniqueTable<mNode>();
  const auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto hKron = hGate;
  for (std::size_t i = 0; i < numQubits - 2; ++i) {
    hKron = dd->kronecker(hKron, hGate, 1);
  }
  hKron = dd->kronecker(hKron, Package::makeIdent(), 1);

  constexpr std::size_t maxNodeVal = 6;
  std::array<std::size_t, maxNodeVal + 1> lookupValues{};
  for (std::size_t i = 1; i <= maxNodeVal; ++i) {
    // Store the number of lookups performed so far for levels 1 through 6
    lookupValues[i] = uniqueTable.getStats(i).lookups;
  }

  dd->partialTrace(
      hKron, {true, false, false, false, false, false, false, true, true});
  for (std::size_t i = 1; i < maxNodeVal; ++i) {
    // Check that the number of lookups scales with the number of paths in the
    // DD
    ASSERT_EQ(uniqueTable.getStats(i).lookups,
              lookupValues[i] +
                  static_cast<std::size_t>(std::pow(4, (maxNodeVal - i))));
  }
}

TEST(DDPackageTest, StateGenerationManipulation) {
  constexpr std::size_t nqubits = 6;
  auto dd = std::make_unique<Package>(nqubits);
  auto b = std::vector<bool>(nqubits, false);
  b[0] = b[1] = true;
  auto e = dd->makeBasisState(nqubits, b);
  auto f = dd->makeBasisState(nqubits, {BasisStates::zero, BasisStates::one,
                                        BasisStates::plus, BasisStates::minus,
                                        BasisStates::left, BasisStates::right});
  dd->vUniqueTable.print<vNode>();
  dd->decRef(e);
  dd->decRef(f);
}

TEST(DDPackageTest, VectorSerializationTest) {
  auto dd = std::make_unique<Package>(2);

  auto hGate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  serialize(bellState, "bell_state.dd", false);
  auto deserializedBellState = dd->deserialize<vNode>("bell_state.dd", false);
  EXPECT_EQ(bellState, deserializedBellState);
  std::filesystem::remove("bell_state.dd");

  serialize(bellState, "bell_state_binary.dd", true);
  deserializedBellState = dd->deserialize<vNode>("bell_state_binary.dd", true);
  EXPECT_EQ(bellState, deserializedBellState);
  std::filesystem::remove("bell_state_binary.dd");
}

TEST(DDPackageTest, BellMatrix) {
  auto dd = std::make_unique<Package>(2);

  auto hGate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);

  auto bellMatrix = dd->multiply(cxGate, hGate);

  bellMatrix.printMatrix(dd->qubits());

  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "00"), SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "02"), 0.);
  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "20"), 0.);
  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "22"), SQRT2_2);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 0), SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 0), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 0), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 0), SQRT2_2);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 1), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 1), SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 1), SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 1), 0.);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 2), SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 2), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 2), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 2), -SQRT2_2);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 3), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 3), SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 3), -SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 3), 0.);

  auto goalRow0 = CVec{{SQRT2_2, 0.}, {0., 0.}, {SQRT2_2, 0.}, {0., 0.}};
  auto goalRow1 = CVec{{0., 0.}, {SQRT2_2, 0.}, {0., 0.}, {SQRT2_2, 0.}};
  auto goalRow2 = CVec{{0., 0.}, {SQRT2_2, 0.}, {0., 0.}, {-SQRT2_2, 0.}};
  auto goalRow3 = CVec{{SQRT2_2, 0.}, {0., 0.}, {-SQRT2_2, 0.}, {0., 0.}};
  auto goalMatrix = CMat{goalRow0, goalRow1, goalRow2, goalRow3};
  ASSERT_EQ(bellMatrix.getMatrix(dd->qubits()), goalMatrix);

  export2Dot(bellMatrix, "bell_matrix_colored_labels.dot", true, true, false,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_colored_labels_classic.dot", true, true,
             true, false, false);
  export2Dot(bellMatrix, "bell_matrix_mono_labels.dot", false, true, false,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_mono_labels_classic.dot", false, true,
             true, false, false);
  export2Dot(bellMatrix, "bell_matrix_colored.dot", true, false, false, false,
             false);
  export2Dot(bellMatrix, "bell_matrix_colored_classic.dot", true, false, true,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_mono.dot", false, false, false, false,
             false);
  export2Dot(bellMatrix, "bell_matrix_mono_classic.dot", false, false, true,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_memory.dot", false, true, true, true,
             false);

  const auto filenames = {"bell_matrix_colored_labels.dot",
                          "bell_matrix_colored_labels_classic.dot",
                          "bell_matrix_mono_labels.dot",
                          "bell_matrix_mono_labels_classic.dot",
                          "bell_matrix_colored.dot",
                          "bell_matrix_colored_classic.dot",
                          "bell_matrix_mono.dot",
                          "bell_matrix_mono_classic.dot",
                          "bell_matrix_memory.dot"};

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  printStatistics(*dd);
}

TEST(DDPackageTest, MatrixSerializationTest) {
  auto dd = std::make_unique<Package>(2);

  auto hGate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);

  auto bellMatrix = dd->multiply(cxGate, hGate);

  serialize(bellMatrix, "bell_matrix.dd", false);
  auto deserializedBellMatrix = dd->deserialize<mNode>("bell_matrix.dd", false);
  EXPECT_EQ(bellMatrix, deserializedBellMatrix);
  std::filesystem::remove("bell_matrix.dd");

  serialize(bellMatrix, "bell_matrix_binary.dd", true);
  deserializedBellMatrix =
      dd->deserialize<mNode>("bell_matrix_binary.dd", true);
  EXPECT_EQ(bellMatrix, deserializedBellMatrix);
  std::filesystem::remove("bell_matrix_binary.dd");
}

TEST(DDPackageTest, SerializationErrors) {
  auto dd = std::make_unique<Package>(2);

  auto hGate = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);
  auto zeroState = dd->makeZeroState(2);
  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  // test non-existing file
  EXPECT_THROW(serialize(bellState, "./path/that/does/not/exist/filename.dd"),
               std::invalid_argument);
  EXPECT_THROW(
      dd->deserialize<vNode>("./path/that/does/not/exist/filename.dd", true),
      std::invalid_argument);

  // test wrong version number
  std::stringstream ss{};
  ss << 2 << "\n";
  EXPECT_THROW(dd->deserialize<vNode>(ss, false), std::runtime_error);
  ss << 2 << "\n";
  EXPECT_THROW(dd->deserialize<mNode>(ss, false), std::runtime_error);

  ss.str("");
  std::remove_const_t<decltype(SERIALIZATION_VERSION)> version = 2;
  ss.write(reinterpret_cast<const char*>(&version),
           sizeof(decltype(SERIALIZATION_VERSION)));
  EXPECT_THROW(dd->deserialize<vNode>(ss, true), std::runtime_error);
  ss.write(reinterpret_cast<const char*>(&version),
           sizeof(decltype(SERIALIZATION_VERSION)));
  EXPECT_THROW(dd->deserialize<mNode>(ss, true), std::runtime_error);

  // test wrong format
  ss.str("");
  ss << "1\n";
  ss << "not_complex\n";
  EXPECT_THROW(dd->deserialize<vNode>(ss), std::runtime_error);
  ss << "1\n";
  ss << "not_complex\n";
  EXPECT_THROW(dd->deserialize<mNode>(ss), std::runtime_error);

  ss.str("");
  ss << "1\n";
  ss << "1.0\n";
  ss << "no_node_here\n";
  EXPECT_THROW(dd->deserialize<vNode>(ss), std::runtime_error);
  ss << "1\n";
  ss << "1.0\n";
  ss << "no_node_here\n";
  EXPECT_THROW(dd->deserialize<mNode>(ss), std::runtime_error);
}

TEST(DDPackageTest, Ancillaries) {
  auto dd = std::make_unique<Package>(4);
  auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(0_pc, 1, qc::X), *dd);
  auto bellMatrix = dd->multiply(cxGate, hGate);

  dd->incRef(bellMatrix);
  auto reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, false, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, true, true});
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());

  EXPECT_EQ(reducedBellMatrix.p->e[0].p->e[0].p, bellMatrix.p);
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[3].isZeroTerminal());

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, true, true}, false);
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());

  EXPECT_EQ(reducedBellMatrix.p->e[0].p->e[0].p, bellMatrix.p);
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[3].isZeroTerminal());
}

TEST(DDPackageTest, GarbageVector) {
  auto dd = std::make_unique<Package>(4);
  auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(0_pc, 1, qc::X), *dd);
  auto zeroState = dd->makeZeroState(2);
  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  std::cout << "Bell State:\n";
  bellState.printVector();

  dd->incRef(bellState);
  auto reducedBellState =
      dd->reduceGarbage(bellState, {false, false, false, false});
  EXPECT_EQ(bellState, reducedBellState);
  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {false, false, true, false});
  EXPECT_EQ(bellState, reducedBellState);

  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {false, true, false, false});
  auto vec = reducedBellState.getVector();
  std::cout << "Reduced Bell State (q1 garbage):\n";
  reducedBellState.printVector();
  EXPECT_EQ(vec[2], 0.);
  EXPECT_EQ(vec[3], 0.);

  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {true, false, false, false});
  std::cout << "Reduced Bell State (q0 garbage):\n";
  reducedBellState.printVector();
  vec = reducedBellState.getVector();
  EXPECT_EQ(vec[1], 0.);
  EXPECT_EQ(vec[3], 0.);
}

TEST(DDPackageTest, GarbageMatrix) {
  auto dd = std::make_unique<Package>(4);
  auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto cxGate = getDD(qc::StandardOperation(0_pc, 1, qc::X), *dd);
  auto bellMatrix = dd->multiply(cxGate, hGate);

  dd->incRef(bellMatrix);
  auto reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, false, false, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);
  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, false, true, false});
  EXPECT_NE(bellMatrix, reducedBellMatrix);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, true, false, false});
  auto mat = reducedBellMatrix.getMatrix(2);
  auto zero = CVec{{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}};
  EXPECT_EQ(mat[2], zero);
  EXPECT_EQ(mat[3], zero);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {true, false, false, false});
  mat = reducedBellMatrix.getMatrix(2);
  EXPECT_EQ(mat[1], zero);
  EXPECT_EQ(mat[3], zero);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, true, false, false}, false);
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());
}

TEST(DDPackageTest, ReduceGarbageVector) {
  auto dd = std::make_unique<Package>(3);
  auto xGate = getDD(qc::StandardOperation(2, qc::X), *dd);
  auto hGate = getDD(qc::StandardOperation(2, qc::H), *dd);
  auto zeroState = dd->makeZeroState(3);
  auto initialState = dd->multiply(dd->multiply(hGate, xGate), zeroState);
  std::cout << "Initial State:\n";
  initialState.printVector();

  dd->incRef(initialState);
  auto reducedState = dd->reduceGarbage(initialState, {false, true, true});
  std::cout << "After reduceGarbage():\n";
  reducedState.printVector();
  EXPECT_EQ(reducedState, dd->makeZeroState(3));

  dd->incRef(initialState);
  auto reducedState2 =
      dd->reduceGarbage(initialState, {false, true, true}, true);

  EXPECT_EQ(reducedState2, dd->makeZeroState(3));
}

TEST(DDPackageTest, ReduceGarbageVectorTGate) {
  constexpr auto nqubits = 2U;
  const auto dd = std::make_unique<Package>(nqubits);
  const auto xGate0 = getDD(qc::StandardOperation(0, qc::X), *dd);
  const auto xGate1 = getDD(qc::StandardOperation(1, qc::X), *dd);
  const auto tdgGate0 = getDD(qc::StandardOperation(0, qc::Tdg), *dd);

  auto zeroState = dd->makeZeroState(nqubits);
  auto initialState = dd->multiply(
      dd->multiply(tdgGate0, dd->multiply(xGate0, xGate1)), zeroState);
  std::cout << "Initial State:\n";
  initialState.printVector();

  dd->incRef(initialState);
  auto reducedState = dd->reduceGarbage(initialState, {false, false}, true);
  std::cout << "After reduceGarbage():\n";
  reducedState.printVector();
  EXPECT_EQ(reducedState,
            dd->multiply(dd->multiply(xGate0, xGate1), zeroState));
}

TEST(DDPackageTest, ReduceGarbageMatrix) {
  auto dd = std::make_unique<Package>(3);
  auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto cNotGate = getDD(qc::StandardOperation(qc::Controls{0}, 1, qc::X), *dd);

  auto initialState = dd->multiply(hGate, cNotGate);

  std::cout << "Initial State:\n";
  initialState.printMatrix(dd->qubits());

  dd->incRef(initialState);
  auto reducedState1 =
      dd->reduceGarbage(initialState, {false, true, true}, true, true);
  std::cout << "After reduceGarbage(q1 and q2 are garbage):\n";
  reducedState1.printMatrix(dd->qubits());

  auto expectedMatrix1 = CMat{
      {SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2},
      {SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2, SQRT2_2},
      {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(reducedState1.getMatrix(dd->qubits()), expectedMatrix1);

  dd->incRef(initialState);
  auto reducedState2 =
      dd->reduceGarbage(initialState, {true, false, false}, true, true);
  std::cout << "After reduceGarbage(q0 is garbage):\n";
  reducedState2.printMatrix(dd->qubits());

  auto expectedMatrix2 =
      CMat{{1, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
           {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
           {0, 0, 0, 0, 1, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 0},
           {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(reducedState2.getMatrix(dd->qubits()), expectedMatrix2);
}

TEST(DDPackageTest, ReduceGarbageMatrix2) {
  constexpr auto nqubits = 3U;
  const auto dd = std::make_unique<Package>(nqubits);
  const auto controlledSwapGate = getDD(
      qc::StandardOperation(qc::Controls{1}, qc::Targets{0, 2}, qc::SWAP), *dd);
  const auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  const auto zGate = getDD(qc::StandardOperation(2, qc::Z), *dd);
  const auto xGate = getDD(qc::StandardOperation(1, qc::X), *dd);
  const auto controlledHGate =
      getDD(qc::StandardOperation(qc::Controls{1}, 0, qc::H), *dd);

  auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  auto c2 = dd->multiply(controlledHGate, xGate);

  std::cout << "c1:\n";
  c1.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c1);
  auto c1Reduced = dd->reduceGarbage(c1, {false, true, true}, true, true);
  c1Reduced.printMatrix(dd->qubits());

  std::cout << "c2:\n";
  c2.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c2);
  auto c2Reduced = dd->reduceGarbage(c2, {false, true, true}, true, true);
  c2Reduced.printMatrix(dd->qubits());

  EXPECT_EQ(c1Reduced, c2Reduced);
}

TEST(DDPackageTest, ReduceGarbageMatrixNoGarbage) {
  constexpr auto nqubits = 2U;
  const auto dd = std::make_unique<Package>(nqubits);
  const auto tdgGate0 = getDD(qc::StandardOperation(0, qc::Tdg), *dd);
  const auto tdgGate1 = getDD(qc::StandardOperation(1, qc::Tdg), *dd);

  auto c1 = Package::makeIdent();
  auto c2 = dd->multiply(tdgGate0, tdgGate1);

  std::cout << "c2:\n";
  c2.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c2);
  auto c2Reduced = dd->reduceGarbage(c2, {false, false}, true, true);
  c2Reduced.printMatrix(dd->qubits());

  EXPECT_EQ(c1, c2Reduced);
}

TEST(DDPackageTest, ReduceGarbageMatrixTGate) {
  constexpr auto nqubits = 2U;
  const auto dd = std::make_unique<Package>(nqubits);
  const auto tdgGate0 = getDD(qc::StandardOperation(0, qc::Tdg), *dd);
  const auto tdgGate1 = getDD(qc::StandardOperation(1, qc::Tdg), *dd);

  auto c1 = Package::makeIdent();
  auto c2 = dd->multiply(tdgGate0, tdgGate1);

  std::cout << "c1:\n";
  c1.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c1);
  auto c1Reduced = dd->reduceGarbage(c1, {false, true}, true, true);
  c1Reduced.printMatrix(dd->qubits());

  std::cout << "c2:\n";
  c2.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c2);
  auto c2Reduced = dd->reduceGarbage(c2, {false, true}, true, true);
  c2Reduced.printMatrix(dd->qubits());

  EXPECT_EQ(c1Reduced, c2Reduced);
}

TEST(DDPackageTest, InvalidMakeBasisStateAndGate) {
  auto nqubits = 2U;
  auto dd = std::make_unique<Package>(nqubits);
  auto basisState = std::vector<BasisStates>{BasisStates::zero};
  EXPECT_THROW(dd->makeBasisState(nqubits, basisState), std::runtime_error);
  EXPECT_THROW(dd->makeZeroState(3), std::runtime_error);
  EXPECT_THROW(dd->makeBasisState(3, {true, true, true}), std::runtime_error);
  EXPECT_THROW(dd->makeBasisState(
                   3, {BasisStates::one, BasisStates::one, BasisStates::one}),
               std::runtime_error);
  EXPECT_THROW(getDD(qc::StandardOperation(3, qc::X), *dd), std::runtime_error);
}

TEST(DDPackageTest, PackageReset) {
  auto dd = std::make_unique<Package>(1);

  // one node in unique table of variable 0
  auto xGate = getDD(qc::StandardOperation(0, qc::X), *dd);

  const auto& unique = dd->mUniqueTable.getTables();
  const auto& table = unique[0];
  auto ihash = dd->mUniqueTable.hash(*xGate.p);
  const auto* node = table[ihash];
  std::cout << ihash << ": " << reinterpret_cast<uintptr_t>(xGate.p) << "\n";
  // node should be the first in this unique table bucket
  EXPECT_EQ(node, xGate.p);
  dd->reset();
  // after clearing the tables, they should be empty
  EXPECT_EQ(table[ihash], nullptr);
  xGate = getDD(qc::StandardOperation(0, qc::X), *dd);
  const auto* node2 = table[ihash];
  // after recreating the DD, it should receive the same node
  EXPECT_EQ(node2, node);
}

TEST(DDPackageTest, ResetClearsRoots) {
  auto dd = std::make_unique<Package>(2);

  auto vec = dd->makeZeroState(2);
  auto mat = getDD(qc::StandardOperation(0, qc::X), *dd);
  auto dens = dd->makeZeroDensityOperator(2);

  dd->incRef(vec);
  dd->incRef(mat);
  dd->incRef(dens);

  EXPECT_EQ(dd->vectorRoots.size(), 1U);
  EXPECT_EQ(dd->matrixRoots.size(), 1U);
  EXPECT_EQ(dd->densityRoots.size(), 1U);

  dd->reset();

  EXPECT_TRUE(dd->vectorRoots.empty());
  EXPECT_TRUE(dd->matrixRoots.empty());
  EXPECT_TRUE(dd->densityRoots.empty());
}

TEST(DDPackageTest, DuplicateIncRefDoesNotLeaveStaleRoot) {
  auto dd = std::make_unique<Package>(1);

  // vector root
  auto vec = dd->makeZeroState(1);
  EXPECT_EQ(dd->vectorRoots.size(), 1U);
  dd->incRef(vec);
  EXPECT_EQ(dd->vectorRoots.at(vec), 2U);
  dd->decRef(vec);
  EXPECT_EQ(dd->vectorRoots.at(vec), 1U);
  dd->decRef(vec);
  EXPECT_TRUE(dd->vectorRoots.empty());
  EXPECT_THROW(dd->decRef(vec), std::invalid_argument);

  // matrix root
  auto mat = getDD(qc::StandardOperation(0, qc::X), *dd);
  dd->incRef(mat);
  dd->incRef(mat);
  EXPECT_EQ(dd->matrixRoots.size(), 1U);
  EXPECT_EQ(dd->matrixRoots.at(mat), 2U);
  dd->decRef(mat);
  EXPECT_EQ(dd->matrixRoots.at(mat), 1U);
  dd->decRef(mat);
  EXPECT_TRUE(dd->matrixRoots.empty());

  // density root
  auto dens = dd->makeZeroDensityOperator(1);
  EXPECT_EQ(dd->densityRoots.size(), 1U);
  dd->incRef(dens);
  EXPECT_EQ(dd->densityRoots.at(dens), 2U);
  dd->decRef(dens);
  EXPECT_EQ(dd->densityRoots.at(dens), 1U);
  dd->decRef(dens);
  EXPECT_TRUE(dd->densityRoots.empty());
}

TEST(DDPackageTest, Inverse) {
  auto dd = std::make_unique<Package>(1);
  auto x = getDD(qc::StandardOperation(0, qc::X), *dd);
  auto xdag = dd->conjugateTranspose(x);
  EXPECT_EQ(x, xdag);
  dd->garbageCollect();
  // Mark-and-sweep does not run if no unique table exceeded its threshold
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 1);
  dd->incRef(x);
  dd->garbageCollect(true);
  // For mark-and-sweep the node is still part of the root set and therefore
  // remains reachable
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 1);
  dd->decRef(x);
  dd->garbageCollect(true);
  // After removing the node from the root set it is reclaimed
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 0);
}

TEST(DDPackageTest, IncRefTwiceThenDecRefTwice) {
  auto dd = std::make_unique<Package>(1);

  auto x = getDD(qc::StandardOperation(0, qc::X), *dd);

  // add the same edge twice
  dd->incRef(x);
  dd->incRef(x);
  EXPECT_EQ(dd->matrixRoots.size(), 1U);
  EXPECT_EQ(dd->matrixRoots.at(x), 2U);

  dd->garbageCollect(true);

  // node should survive collection since it is still in the root set
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 1);

  // remove the edge twice
  dd->decRef(x);
  EXPECT_EQ(dd->matrixRoots.at(x), 1U);
  dd->decRef(x);

  dd->garbageCollect(true);

  // node should now be reclaimed
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 0);
}

TEST(DDPackageTest, UniqueTableAllocation) {
  auto dd = std::make_unique<Package>(1);

  auto allocs = dd->vMemoryManager.getStats().numAllocated;
  std::cout << allocs << "\n";
  std::vector<vNode*> nodes{allocs};
  // get all the nodes that are pre-allocated
  for (auto i = 0U; i < allocs; ++i) {
    nodes[i] = dd->vMemoryManager.get<vNode>();
  }

  // trigger new allocation
  const auto* node = dd->vMemoryManager.get<vNode>();
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(dd->vMemoryManager.getStats().numAllocated,
            (1. + MemoryManager::GROWTH_FACTOR) * static_cast<double>(allocs));

  // clearing the unique table should reduce the allocated size to the original
  // size
  dd->vMemoryManager.reset();
  EXPECT_EQ(dd->vMemoryManager.getStats().numAllocated, allocs);
}

TEST(DDPackageTest, SpecialCaseTerminal) {
  auto dd = std::make_unique<Package>(2);
  auto one = vEdge::one();
  export2Dot(one, "oneColored.dot", true, false, false, false, false);
  export2Dot(one, "oneClassic.dot", false, false, false, false, false);
  export2Dot(one, "oneMemory.dot", true, true, false, true, false);

  const auto filenames = {
      "oneColored.dot",
      "oneClassic.dot",
      "oneMemory.dot",
  };

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  EXPECT_EQ(dd->vUniqueTable.lookup(one.p), one.p);

  auto zero = vEdge::zero();
  EXPECT_TRUE(dd->kronecker(zero, one, 0).isZeroTerminal());
  EXPECT_TRUE(dd->kronecker(one, one, 0).isOneTerminal());

  EXPECT_EQ(one.getValueByPath(0, ""), 1.);
  EXPECT_EQ(one.getValueByIndex(0), 1.);
  EXPECT_EQ(mEdge::one().getValueByIndex(0, 0, 0), 1.);

  EXPECT_EQ(dd->innerProduct(zero, zero), ComplexValue(0.));
}

TEST(DDPackageTest, KroneckerProduct) {
  auto dd = std::make_unique<Package>(2);
  auto x = getDD(qc::StandardOperation(0, qc::X), *dd);
  auto kronecker = dd->kronecker(x, x, 1);
  EXPECT_EQ(kronecker.p->v, 1);
  EXPECT_TRUE(kronecker.p->e[0].isZeroTerminal());
  EXPECT_EQ(kronecker.p->e[0], kronecker.p->e[3]);
  EXPECT_EQ(kronecker.p->e[1], kronecker.p->e[2]);
  EXPECT_EQ(kronecker.p->e[1].p->v, 0);
  EXPECT_TRUE(kronecker.p->e[1].p->e[0].isZeroTerminal());
  EXPECT_EQ(kronecker.p->e[1].p->e[0], kronecker.p->e[1].p->e[3]);
  EXPECT_TRUE(kronecker.p->e[1].p->e[1].isOneTerminal());
  EXPECT_EQ(kronecker.p->e[1].p->e[1], kronecker.p->e[1].p->e[2]);

  auto kronecker2 = dd->kronecker(x, x, 1);
  EXPECT_EQ(kronecker, kronecker2);
}

TEST(DDPackageTest, KroneckerProductVectors) {
  auto dd = std::make_unique<Package>(2);
  auto zeroState = dd->makeZeroState(1);
  auto kronecker = dd->kronecker(zeroState, zeroState, 1);

  auto expected = dd->makeZeroState(2);
  EXPECT_EQ(kronecker, expected);
}

TEST(DDPackageTest, KroneckerIdentityHandling) {
  auto dd = std::make_unique<Package>(3U);
  // create a Hadamard gate on the middle qubit
  auto h = getDD(qc::StandardOperation(1U, qc::H), *dd);
  // create a single qubit identity
  auto id = Package::makeIdent();
  // kronecker both DDs
  const auto combined = dd->kronecker(h, id, 1);
  const auto matrix = combined.getMatrix(dd->qubits());
  const auto expectedMatrix = CMat{
      {SQRT2_2, 0, 0, 0, SQRT2_2, 0, 0, 0},
      {0, SQRT2_2, 0, 0, 0, SQRT2_2, 0, 0},
      {0, 0, SQRT2_2, 0, 0, 0, SQRT2_2, 0},
      {0, 0, 0, SQRT2_2, 0, 0, 0, SQRT2_2},
      {SQRT2_2, 0, 0, 0, -SQRT2_2, 0, 0, 0},
      {0, SQRT2_2, 0, 0, 0, -SQRT2_2, 0, 0},
      {0, 0, SQRT2_2, 0, 0, 0, -SQRT2_2, 0},
      {0, 0, 0, SQRT2_2, 0, 0, 0, -SQRT2_2},
  };
  EXPECT_EQ(matrix, expectedMatrix);
}

TEST(DDPackageTest, NearZeroNormalize) {
  auto dd = std::make_unique<Package>(2);
  const fp nearZero = RealNumber::eps / 10;
  vEdge ve{};
  ve.p = dd->vMemoryManager.get<vNode>();
  ve.p->v = 1;
  ve.w = Complex::one();
  std::array<vCachedEdge, RADIX> edges{};
  for (auto& edge : edges) {
    edge.p = dd->vMemoryManager.get<vNode>();
    edge.p->v = 0;
    edge.w = nearZero;
    edge.p->e = {vEdge::one(), vEdge::one()};
  }
  auto veNormalizedCached =
      vCachedEdge::normalize(ve.p, edges, dd->vMemoryManager, dd->cn);
  EXPECT_EQ(veNormalizedCached, vCachedEdge::zero());

  std::array<vEdge, RADIX> edges2{};
  for (auto& edge : edges2) {
    edge.p = dd->vMemoryManager.get<vNode>();
    edge.p->v = 0;
    edge.w = dd->cn.lookup(nearZero);
    edge.p->e = {vEdge::one(), vEdge::one()};
  }
  auto veNormalized =
      vEdge::normalize(ve.p, edges2, dd->vMemoryManager, dd->cn);
  EXPECT_TRUE(veNormalized.isZeroTerminal());

  mEdge me{};
  me.p = dd->mMemoryManager.get<mNode>();
  me.p->v = 1;
  me.w = Complex::one();
  std::array<mCachedEdge, NEDGE> edges3{};
  for (auto& edge : edges3) {
    edge.p = dd->mMemoryManager.get<mNode>();
    edge.p->v = 0;
    edge.w = nearZero;
    edge.p->e = {mEdge::one(), mEdge::one(), mEdge::one(), mEdge::one()};
  }
  auto meNormalizedCached =
      mCachedEdge::normalize(me.p, edges3, dd->mMemoryManager, dd->cn);
  EXPECT_EQ(meNormalizedCached, mCachedEdge::zero());

  me.p = dd->mMemoryManager.get<mNode>();
  std::array<mEdge, 4> edges4{};
  for (auto& edge : edges4) {
    edge.p = dd->mMemoryManager.get<mNode>();
    edge.p->v = 0;
    edge.w = dd->cn.lookup(nearZero, 0.);
    edge.p->e = {mEdge::one(), mEdge::one(), mEdge::one(), mEdge::one()};
  }
  auto meNormalized =
      mEdge::normalize(me.p, edges4, dd->mMemoryManager, dd->cn);
  EXPECT_TRUE(meNormalized.isZeroTerminal());
}

TEST(DDPackageTest, DestructiveMeasurementAll) {
  auto dd = std::make_unique<Package>(4);
  auto hGate0 = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto hGate1 = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(ms

  const CVec vBefore = plusState.getVector();

  ASSERT_EQ(vBefore[0], vBefore[1]);
  ASSERT_EQ(vBefore[0], vBefore[2]);
  ASSERT_EQ(vBefore[0], vBefore[3]);

  const std::string m = dd->measureAll(plusState, true, mt);

  const CVec vAfter = plusState.getVector();
  const int i = std::stoi(m, nullptr, 2);

  ASSERT_EQ(vAfter[static_cast<std::size_t>(i)], 1.);
}

TEST(DDPackageTest, DestructiveMeasurementOne) {
  auto dd = std::make_unique<Package>(4);
  auto hGate0 = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto hGate1 = getDD(qc::StandardOperation(1, qc::H), *dd);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(cert-msc51-cpp)

  const char m = dd->measureOneCollapsing(plusState, 0, mt);
  const CVec vAfter = plusState.getVector();

  ASSERT_EQ(m, '0');
  ASSERT_EQ(vAfter[0], SQRT2_2);
  ASSERT_EQ(vAfter[2], SQRT2_2);
  ASSERT_EQ(vAfter[1], 0.);
  ASSERT_EQ(vAfter[3], 0.);
}

TEST(DDPackageTest, ExportPolarPhaseFormatted) {
  std::ostringstream phaseString;

  // zero case
  printPhaseFormatted(phaseString, 0);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 0)");
  phaseString.str("");

  // one cases
  printPhaseFormatted(phaseString, 0.5 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/2)");
  phaseString.str("");

  printPhaseFormatted(phaseString, -0.5 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(-iπ/2)");
  phaseString.str("");

  printPhaseFormatted(phaseString, PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ)");
  phaseString.str("");

  printPhaseFormatted(phaseString, -PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(-iπ)");
  phaseString.str("");

  // 1/sqrt(2) cases
  printPhaseFormatted(phaseString, SQRT2_2 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/√2)");
  phaseString.str("");

  printPhaseFormatted(phaseString, 2 * SQRT2_2 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 2/√2)");
  phaseString.str("");

  printPhaseFormatted(phaseString, 0.5 * SQRT2_2 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/(2√2))");
  phaseString.str("");

  printPhaseFormatted(phaseString, 0.75 * SQRT2_2 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 3/(4√2))");
  phaseString.str("");

  // pi cases mhhh pie
  printPhaseFormatted(phaseString, PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ)");
  phaseString.str("");

  printPhaseFormatted(phaseString, 2 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 2)");
  phaseString.str("");

  printPhaseFormatted(phaseString, 0.5 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/2)");
  phaseString.str("");

  printPhaseFormatted(phaseString, 0.75 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 3/4)");
  phaseString.str("");

  printPhaseFormatted(phaseString, 0.25 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/4)");
  phaseString.str("");

  // general case
  printPhaseFormatted(phaseString, 0.12345 * PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 0.12345)");
  phaseString.str("");
}

TEST(DDPackageTest, BasicNumericInstabilityTest) {
  constexpr auto zero = 0.0;
  constexpr auto half = 0.5;
  constexpr auto one = 1.0;
  constexpr auto two = 2.0;

  std::cout << std::setprecision(std::numeric_limits<fp>::max_digits10);

  std::cout << "The 1/sqrt(2) constant used in this package is " << SQRT2_2
            << ", which is the closest floating point value to the actual "
               "value of 1/sqrt(2).\n";
  std::cout << "Computing std::sqrt(0.5) actually computes this value, i.e. "
            << std::sqrt(half) << "\n";
  EXPECT_EQ(SQRT2_2, std::sqrt(half));

  std::cout << "However, computing 1/std::sqrt(2.) leads to "
            << one / std::sqrt(two)
            << ", which differs by 1 ULP from std::sqrt(0.5)\n";
  EXPECT_EQ(one / std::sqrt(two), std::nextafter(std::sqrt(half), zero));

  std::cout << "In the same fashion, computing std::sqrt(2.) leads to "
            << std::sqrt(two) << ", while computing 1/std::sqrt(0.5) leads to "
            << one / std::sqrt(half) << ", which differ by exactly 1 ULP\n";
  EXPECT_EQ(std::sqrt(two), std::nextafter(one / std::sqrt(half), two));

  std::cout << "Another inaccuracy occurs when computing 1/sqrt(2) * "
               "1/sqrt(2), which should equal to 0.5 but is off by 1 ULP: "
            << std::sqrt(half) * std::sqrt(half) << "\n";
  EXPECT_EQ(std::sqrt(half) * std::sqrt(half), std::nextafter(half, one));

  std::cout << "This inaccuracy even persists when computing std::sqrt(0.5) * "
               "std::sqrt(0.5): "
            << std::sqrt(half) * std::sqrt(half) << "\n";
  EXPECT_EQ(std::sqrt(half) * std::sqrt(half), std::nextafter(half, one));

  std::cout << "Interestingly, calculating powers of SQRT2_2 can be "
               "conducted very precisely, i.e., with an error of only 1 ULP.\n";
  fp accumulator = SQRT2_2 * SQRT2_2;
  constexpr std::size_t nq = 64;
  for (std::size_t i = 1; i < nq; i += 2) {
    const std::size_t power = (i + 1) / 2;
    const std::size_t denom = static_cast<std::size_t>(1U) << power;
    const fp target = 1. / static_cast<double>(denom);
    const fp diff = std::abs(target - accumulator);
    const auto ulps = ulpDistance(accumulator, target);
    std::cout << accumulator << ", numerical error: " << diff
              << ", ulps: " << ulps << "\n";
    EXPECT_EQ(ulps, 1);
    accumulator *= SQRT2_2;
    accumulator *= SQRT2_2;
  }
}

TEST(DDPackageTest, BasicNumericStabilityTest) {
  using limits = std::numeric_limits<fp>;

  auto dd = std::make_unique<Package>(1);
  auto tol = RealNumber::eps;
  ComplexNumbers::setTolerance(limits::epsilon());
  auto state = dd->makeZeroState(1);
  auto h = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto state1 = dd->multiply(h, state);
  auto z = getDD(qc::StandardOperation(0, qc::Z), *dd);
  auto result = dd->multiply(z, state1);

  const auto topWeight = result.w.toString(false, limits::max_digits10);
  const auto leftWeight =
      result.p->e[0].w.toString(false, limits::max_digits10);
  const auto rightWeight =
      result.p->e[1].w.toString(false, limits::max_digits10);
  std::cout << topWeight << " | " << leftWeight << " | " << rightWeight << "\n";
  EXPECT_EQ(topWeight, "1");
  std::ostringstream oss{};
  oss << std::setprecision(limits::max_digits10) << SQRT2_2;
  EXPECT_EQ(leftWeight, oss.str());
  oss.str("");
  oss << -SQRT2_2;
  EXPECT_EQ(rightWeight, oss.str());
  // restore tolerance
  ComplexNumbers::setTolerance(tol);
}

TEST(DDPackageTest, NormalizationNumericStabilityTest) {
  auto dd = std::make_unique<Package>(1);
  for (std::size_t x = 23; x <= 50; ++x) {
    const auto lambda = PI / static_cast<fp>(1ULL << x);
    std::cout << std::setprecision(17) << "x: " << x << " | lambda: " << lambda
              << " | cos(lambda): " << std::cos(lambda)
              << " | sin(lambda): " << std::sin(lambda) << "\n";
    auto p = getDD(qc::StandardOperation(0, qc::P, {lambda}), *dd);
    auto pdag = getDD(qc::StandardOperation(0, qc::P, {-lambda}), *dd);
    auto result = dd->multiply(p, pdag);
    EXPECT_TRUE(result.isIdentity());
    dd->cUniqueTable.clear();
    dd->cMemoryManager.reset();
  }
}

TEST(DDPackageTest, FidelityOfMeasurementOutcomes) {
  const auto dd = std::make_unique<Package>(3);

  const auto hGate = getDD(qc::StandardOperation(2, qc::H), *dd);
  const auto cxGate1 = getDD(qc::StandardOperation(2_pc, 1, qc::X), *dd);
  const auto cxGate2 = getDD(qc::StandardOperation(1_pc, 0, qc::X), *dd);
  const auto zeroState = dd->makeZeroState(3);

  const auto ghzState = dd->multiply(
      cxGate2, dd->multiply(cxGate1, dd->multiply(hGate, zeroState)));

  SparsePVec probs{};
  probs[0] = 0.5;
  probs[7] = 0.5;
  const auto fidelity = Package::fidelityOfMeasurementOutcomes(ghzState, probs);
  EXPECT_NEAR(fidelity, 1.0, RealNumber::eps);
}

TEST(DDPackageTest, CloseToIdentity) {
  auto dd = std::make_unique<Package>(3);
  auto id = Package::makeIdent();
  EXPECT_TRUE(dd->isCloseToIdentity(id));
  mEdge close{};
  close.p = id.p;
  close.w = dd->cn.lookup(1e-11, 0);
  auto id2 =
      dd->makeDDNode(1, std::array{id, mEdge::zero(), mEdge::zero(), close});
  EXPECT_TRUE(dd->isCloseToIdentity(id2));

  auto noId =
      dd->makeDDNode(1, std::array{mEdge::zero(), id, mEdge::zero(), close});
  EXPECT_FALSE(dd->isCloseToIdentity(noId));

  mEdge notClose{};
  notClose.p = id.p;
  notClose.w = dd->cn.lookup(1e-9, 0);
  auto noId2 = dd->makeDDNode(
      1, std::array{notClose, mEdge::zero(), mEdge::zero(), close});
  EXPECT_FALSE(dd->isCloseToIdentity(noId2));

  auto noId3 = dd->makeDDNode(
      1, std::array{close, mEdge::zero(), mEdge::zero(), notClose});
  EXPECT_FALSE(dd->isCloseToIdentity(noId3));

  auto notClose2 = dd->makeDDNode(
      0, std::array{mEdge::zero(), mEdge::one(), mEdge::one(), mEdge::zero()});
  auto notClose3 = dd->makeDDNode(
      1, std::array{notClose2, mEdge::zero(), mEdge::zero(), notClose2});
  EXPECT_FALSE(dd->isCloseToIdentity(notClose3));
}

TEST(DDPackageTest, CloseToIdentityWithGarbageAtTheBeginning) {
  constexpr fp tol = 1.0E-10;
  constexpr auto nqubits = 3U;
  auto dd = std::make_unique<Package>(nqubits);
  auto controlledSwapGate = getDD(
      qc::StandardOperation(qc::Controls{1}, qc::Targets{0, 2}, qc::SWAP), *dd);
  auto hGate = getDD(qc::StandardOperation(0, qc::H), *dd);
  auto zGate = getDD(qc::StandardOperation(2, qc::Z), *dd);
  auto xGate = getDD(qc::StandardOperation(1, qc::X), *dd);
  auto controlledHGate =
      getDD(qc::StandardOperation(qc::Controls{1}, 0, qc::H), *dd);

  auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  auto c2 = dd->multiply(controlledHGate, xGate);

  auto c1MultipliedWithC2 = dd->multiply(c1, dd->conjugateTranspose(c2));

  EXPECT_TRUE(dd->isCloseToIdentity(c1MultipliedWithC2, tol,
                                    {false, true, true}, false));
  EXPECT_FALSE(dd->isCloseToIdentity(c1MultipliedWithC2, tol,
                                     {false, false, true}, false));
}

TEST(DDPackageTest, CloseToIdentityWithGarbageAtTheEnd) {
  constexpr fp tol = 1.0E-10;
  constexpr auto nqubits = 3U;
  const auto dd = std::make_unique<Package>(nqubits);

  const auto controlledSwapGate = getDD(
      qc::StandardOperation(qc::Controls{1}, qc::Targets{0, 2}, qc::SWAP), *dd);
  const auto xGate = getDD(qc::StandardOperation(1, qc::X), *dd);

  const auto hGate2 = getDD(qc::StandardOperation(2, qc::H), *dd);
  const auto zGate2 = getDD(qc::StandardOperation(0, qc::Z), *dd);
  const auto controlledHGate2 =
      getDD(qc::StandardOperation(qc::Controls{1}, 2, qc::H), *dd);

  const auto c3 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate2, dd->multiply(zGate2, controlledSwapGate)));
  const auto c4 = dd->multiply(controlledHGate2, xGate);

  const auto c3MultipliedWithC4 = dd->multiply(c3, dd->conjugateTranspose(c4));

  EXPECT_FALSE(dd->isCloseToIdentity(c3MultipliedWithC4, tol,
                                     {false, true, true}, false));
  EXPECT_FALSE(dd->isCloseToIdentity(c3MultipliedWithC4, tol,
                                     {true, false, true}, false));
  EXPECT_TRUE(dd->isCloseToIdentity(c3MultipliedWithC4, tol,
                                    {true, true, false}, false));
}

TEST(DDPackageTest, CloseToIdentityWithGarbageInTheMiddle) {
  constexpr fp tol = 1.0E-10;
  constexpr auto nqubits = 3U;
  const auto dd = std::make_unique<Package>(nqubits);

  const auto zGate = getDD(qc::StandardOperation(2, qc::Z), *dd);

  const auto controlledSwapGate3 = getDD(
      qc::StandardOperation(qc::Controls{0}, qc::Targets{1, 2}, qc::SWAP), *dd);
  const auto hGate3 = getDD(qc::StandardOperation(1, qc::H), *dd);
  const auto xGate3 = getDD(qc::StandardOperation(0, qc::X), *dd);
  const auto controlledHGate3 =
      getDD(qc::StandardOperation(qc::Controls{0}, 1, qc::H), *dd);

  const auto c5 = dd->multiply(
      controlledSwapGate3,
      dd->multiply(hGate3, dd->multiply(zGate, controlledSwapGate3)));
  const auto c6 = dd->multiply(controlledHGate3, xGate3);

  const auto c5MultipliedWithC6 = dd->multiply(c5, dd->conjugateTranspose(c6));

  EXPECT_FALSE(dd->isCloseToIdentity(c5MultipliedWithC6, tol,
                                     {false, true, true}, false));
  EXPECT_FALSE(dd->isCloseToIdentity(c5MultipliedWithC6, tol,
                                     {true, true, false}, false));
  EXPECT_TRUE(dd->isCloseToIdentity(c5MultipliedWithC6, tol,
                                    {true, false, true}, false));
}

TEST(DDPackageTest, dNodeMultiply) {
  // Multiply dNode with mNode (MxMxM)
  constexpr auto nrQubits = 3U;
  const auto dd = std::make_unique<Package>(
      nrQubits, DENSITY_MATRIX_SIMULATOR_DD_PACKAGE_CONFIG);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(dd->qubits());
  std::vector<mEdge> operations = {};
  operations.emplace_back(getDD(qc::StandardOperation(0, qc::H), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(1, qc::H), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(2, qc::H), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(2, qc::Z), *dd));

  for (const auto& op : operations) {
    dd->applyOperationToDensity(state, op);
  }

  const auto stateDensityMatrix = state.getMatrix(dd->qubits());

  for (const auto& stateVector : stateDensityMatrix) {
    for (const auto& cValue : stateVector) {
      std::cout << "r:" << cValue.real() << " i:" << cValue.imag();
    }
    std::cout << "\n";
  }

  for (std::size_t i = 0; i < (1 << nrQubits); i++) {
    for (std::size_t j = 0; j < (1 << nrQubits); j++) {
      EXPECT_EQ(std::abs(stateDensityMatrix[i][j].imag()), 0);
      if ((i < 4 && j < 4) || (i >= 4 && j >= 4)) {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() > 0);
      } else {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() < 0);
      }
      EXPECT_TRUE(std::abs(std::abs(stateDensityMatrix[i][j]) - 0.125) <
                  0.000001);
    }
  }

  const auto probVector = state.getSparseProbabilityVector(nrQubits, 0.001);
  for (const auto& [s, prob] : probVector) {
    constexpr auto tolerance = 1e-10;
    std::cout << s << ": " << prob << "\n";
    EXPECT_NEAR(prob, 0.125, tolerance);
  }
}

TEST(DDPackageTest, dNodeMultiply2) {
  // Multiply dNode with mNode (MxMxM)
  constexpr auto nrQubits = 3U;
  const auto dd = std::make_unique<Package>(
      nrQubits, DENSITY_MATRIX_SIMULATOR_DD_PACKAGE_CONFIG);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(dd->qubits());
  std::vector<mEdge> operations = {};
  operations.emplace_back(getDD(qc::StandardOperation(0, qc::H), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(1, qc::H), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(2, qc::H), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(2, qc::Z), *dd));

  for (const auto& op : operations) {
    dd->applyOperationToDensity(state, op);
  }
  operations[0].printMatrix(dd->qubits());

  const auto stateDensityMatrix = state.getMatrix(dd->qubits());

  for (std::size_t i = 0; i < (1 << nrQubits); i++) {
    for (std::size_t j = 0; j < (1 << nrQubits); j++) {
      EXPECT_TRUE(std::abs(stateDensityMatrix[i][j].imag()) == 0);
      if ((i < 4 && j < 4) || (i >= 4 && j >= 4)) {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() > 0);
      } else {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() < 0);
      }
      EXPECT_TRUE(std::abs(std::abs(stateDensityMatrix[i][j]) - 0.125) <
                  0.000001);
    }
  }
  const auto probVector = state.getSparseProbabilityVector(nrQubits, 0.001);
  for (const auto& [s, prob] : probVector) {
    constexpr double tolerance = 1e-10;
    std::cout << s << ": " << prob << "\n";
    EXPECT_NEAR(prob, 0.125, tolerance);
  }
}

TEST(DDPackageTest, dNodeMulCache1) {
  // Make caching test with dNodes
  constexpr auto nrQubits = 1U;
  const auto dd = std::make_unique<Package>(
      nrQubits, DENSITY_MATRIX_SIMULATOR_DD_PACKAGE_CONFIG);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(nrQubits);

  const auto operation = getDD(qc::StandardOperation(0, qc::H), *dd);
  dd->applyOperationToDensity(state, operation);

  state = dd->makeZeroDensityOperator(nrQubits);
  auto& computeTable = dd->getMultiplicationComputeTable<dNode>();

  const auto& densityMatrix0 =
      densityFromMatrixEdge(dd->conjugateTranspose(operation));

  const auto* cachedResult =
      computeTable.lookup(state.p, densityMatrix0.p, false);
  ASSERT_NE(cachedResult, nullptr);
  ASSERT_NE(cachedResult->p, nullptr);
  state = dd->multiply(state, densityMatrix0, false);
  ASSERT_NE(state.p, nullptr);
  ASSERT_EQ(state.p, cachedResult->p);

  const auto densityMatrix1 = densityFromMatrixEdge(operation);
  const auto* cachedResult1 =
      computeTable.lookup(densityMatrix1.p, state.p, true);
  ASSERT_NE(cachedResult1, nullptr);
  ASSERT_NE(cachedResult1->p, nullptr);
  const auto state2 = dd->multiply(densityMatrix1, state, true);
  ASSERT_NE(state2.p, nullptr);
  ASSERT_EQ(state2.p, cachedResult1->p);

  // try a repeated lookup
  const auto* cachedResult2 =
      computeTable.lookup(densityMatrix1.p, state.p, true);
  ASSERT_NE(cachedResult2, nullptr);
  ASSERT_NE(cachedResult2->p, nullptr);
  ASSERT_EQ(cachedResult2->p, cachedResult1->p);

  computeTable.clear();
  const auto* cachedResult3 =
      computeTable.lookup(densityMatrix1.p, state.p, true);
  ASSERT_EQ(cachedResult3, nullptr);
}

TEST(DDPackageTest, dNoiseCache) {
  // Test the flags for dnode, vnode and mnodes
  constexpr auto nrQubits = 1U;
  const auto dd = std::make_unique<Package>(nrQubits);
  // Make zero density matrix
  const auto initialState = dd->makeZeroDensityOperator(nrQubits);

  // nothing pre-cached
  const std::vector<Qubit> target = {0};
  const auto cachedNoise = dd->densityNoise.lookup(initialState, target);
  ASSERT_EQ(cachedNoise.p, nullptr);

  auto state = initialState;
  const auto operation = getDD(qc::StandardOperation(0, qc::X), *dd);
  dd->applyOperationToDensity(state, operation);
  dd->densityNoise.insert(initialState, state, target);

  // noise pre-cached
  const auto cachedNoise1 = dd->densityNoise.lookup(initialState, target);
  ASSERT_NE(cachedNoise1.p, nullptr);
  ASSERT_EQ(cachedNoise1.p, state.p);

  // no noise pre-cached after clear
  dd->densityNoise.clear();
  const auto cachedNoise2 = dd->densityNoise.lookup(initialState, target);
  ASSERT_EQ(cachedNoise2.p, nullptr);
}

TEST(DDPackageTest, calCulpDistance) {
  constexpr auto nrQubits = 1U;
  auto dd = std::make_unique<Package>(nrQubits);
  const auto tmp0 = ulpDistance(1 + 1e-12, 1);
  const auto tmp1 = ulpDistance(1, 1);
  EXPECT_TRUE(tmp0 > 0);
  EXPECT_EQ(tmp1, 0);
}

TEST(DDPackageTest, dStochCache) {
  constexpr auto nrQubits = 4U;
  const auto dd = std::make_unique<Package>(
      nrQubits, STOCHASTIC_NOISE_SIMULATOR_DD_PACKAGE_CONFIG);

  std::vector<mEdge> operations = {};
  operations.emplace_back(getDD(qc::StandardOperation(0, qc::X), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(1, qc::Z), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(2, qc::Y), *dd));
  operations.emplace_back(getDD(qc::StandardOperation(3, qc::H), *dd));

  dd->stochasticNoiseOperationCache.insert(
      0, 0, operations[0]); // insert X operations with target 0
  dd->stochasticNoiseOperationCache.insert(
      1, 1, operations[1]); // insert Z operations with target 1
  dd->stochasticNoiseOperationCache.insert(
      2, 2, operations[2]); // insert Y operations with target 2
  dd->stochasticNoiseOperationCache.insert(
      3, 3, operations[3]); // insert H operations with target 3

  for (std::uint8_t i = 0; i < 4; i++) {
    for (Qubit j = 0; j < 4; j++) {
      const auto* op = dd->stochasticNoiseOperationCache.lookup(i, j);
      if (static_cast<Qubit>(i) == j) {
        EXPECT_TRUE(op != nullptr && op->p == operations[i].p);
      } else {
        EXPECT_EQ(op, nullptr);
      }
    }
  }

  dd->stochasticNoiseOperationCache.clear();
  for (std::uint8_t i = 0; i < 4; i++) {
    for (Qubit j = 0; j < 4; j++) {
      auto* op = dd->stochasticNoiseOperationCache.lookup(i, j);
      EXPECT_EQ(op, nullptr);
    }
  }
}

TEST(DDPackageTest, stateFromVectorBell) {
  const auto dd = std::make_unique<Package>(2);
  const auto v = std::vector<std::complex<fp>>{SQRT2_2, 0, 0, SQRT2_2};
  const auto s = dd->makeStateFromVector(v);
  ASSERT_NE(s.p, nullptr);
  EXPECT_EQ(s.p->v, 1);
  EXPECT_EQ(s.p->e[0].w.r->value, SQRT2_2);
  EXPECT_EQ(s.p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].w.r->value, SQRT2_2);
  EXPECT_EQ(s.p->e[1].w.i->value, 0);
  ASSERT_NE(s.p->e[0].p, nullptr);
  EXPECT_EQ(s.p->e[0].p->e[0].w.r->value, 1);
  EXPECT_EQ(s.p->e[0].p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[1].w.r->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[1].w.i->value, 0);
  ASSERT_NE(s.p->e[1].p, nullptr);
  EXPECT_EQ(s.p->e[1].p->e[0].w.r->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[1].w.r->value, 1);
  EXPECT_EQ(s.p->e[1].p->e[1].w.i->value, 0);
}

TEST(DDPackageTest, stateFromVectorEmpty) {
  auto dd = std::make_unique<Package>(1);
  auto v = std::vector<std::complex<fp>>{};
  EXPECT_TRUE(dd->makeStateFromVector(v).isOneTerminal());
}

TEST(DDPackageTest, stateFromVectorNoPowerOfTwo) {
  auto dd = std::make_unique<Package>(3);
  auto v = std::vector<std::complex<fp>>{1, 2, 3, 4, 5};
  EXPECT_THROW(dd->makeStateFromVector(v), std::invalid_argument);
}

TEST(DDPackageTest, stateFromScalar) {
  const auto dd = std::make_unique<Package>(1);
  const auto s = dd->makeStateFromVector({1});
  EXPECT_TRUE(s.isTerminal());
  EXPECT_EQ(s.w.r->value, 1);
  EXPECT_EQ(s.w.i->value, 0);
}

TEST(DDPackageTest, expectationValueGlobalOperators) {
  constexpr Qubit maxQubits = 3;
  const auto dd = std::make_unique<Package>(maxQubits);
  for (Qubit nrQubits = 1; nrQubits < maxQubits + 1; ++nrQubits) {
    const auto zeroState = dd->makeZeroState(nrQubits);

    // Definition global operators
    const auto singleSiteX = getDD(qc::StandardOperation(0, qc::X), *dd);
    auto globalX = singleSiteX;

    const auto singleSiteZ = getDD(qc::StandardOperation(0, qc::Z), *dd);
    auto globalZ = singleSiteZ;

    const auto singleSiteHadamard = getDD(qc::StandardOperation(0, qc::H), *dd);
    auto globalHadamard = singleSiteHadamard;

    for (Qubit i = 1; i < nrQubits; ++i) {
      globalX = dd->kronecker(globalX, singleSiteX, 1);
      globalZ = dd->kronecker(globalZ, singleSiteZ, 1);
      globalHadamard = dd->kronecker(globalHadamard, singleSiteHadamard, 1);
    }

    // Global Expectation values
    EXPECT_EQ(dd->expectationValue(globalX, zeroState), 0);
    EXPECT_EQ(dd->expectationValue(globalZ, zeroState), 1);
    EXPECT_EQ(dd->expectationValue(globalHadamard, zeroState),
              std::pow(SQRT2_2, nrQubits));
  }
}

TEST(DDPackageTest, expectationValueLocalOperators) {
  constexpr Qubit maxQubits = 3;
  const auto dd = std::make_unique<Package>(maxQubits);
  for (Qubit nrQubits = 1; nrQubits < maxQubits + 1; ++nrQubits) {
    const auto zeroState = dd->makeZeroState(nrQubits);

    // Local expectation values at each site
    for (Qubit site = 0; site < nrQubits - 1; ++site) {
      // Definition local operators
      auto xGate = getDD(qc::StandardOperation(site, qc::X), *dd);
      auto zGate = getDD(qc::StandardOperation(site, qc::Z), *dd);
      auto hadamard = getDD(qc::StandardOperation(site, qc::H), *dd);

      EXPECT_EQ(dd->expectationValue(xGate, zeroState), 0);
      EXPECT_EQ(dd->expectationValue(zGate, zeroState), 1);
      EXPECT_EQ(dd->expectationValue(hadamard, zeroState), SQRT2_2);
    }
  }
}

TEST(DDPackageTest, expectationValueExceptions) {
  constexpr auto nrQubits = 2U;

  const auto dd = std::make_unique<Package>(nrQubits);
  const auto zeroState = dd->makeZeroState(nrQubits - 1);
  const auto xGate = getDD(qc::StandardOperation(1, qc::X), *dd);

  EXPECT_ANY_THROW(dd->expectationValue(xGate, zeroState));
}

TEST(DDPackageTest, DDFromSingleQubitMatrix) {
  const auto inputMatrix = CMat{{SQRT2_2, SQRT2_2}, {SQRT2_2, -SQRT2_2}};

  constexpr auto nrQubits = 1U;
  const auto dd = std::make_unique<Package>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromTwoQubitMatrix) {
  const auto inputMatrix =
      CMat{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};

  constexpr auto nrQubits = 2U;
  const auto dd = std::make_unique<Package>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromTwoQubitAsymmetricalMatrix) {
  const auto inputMatrix = CMat{{SQRT2_2, SQRT2_2, 0, 0},
                                {-SQRT2_2, SQRT2_2, 0, 0},
                                {0, 0, SQRT2_2, -SQRT2_2},
                                {0, 0, SQRT2_2, SQRT2_2}};

  constexpr auto nrQubits = 2U;
  const auto dd = std::make_unique<Package>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromThreeQubitMatrix) {
  const auto inputMatrix =
      CMat{{1, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0},
           {0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0},
           {0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0},
           {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 1, 0}};

  constexpr auto nrQubits = 3U;
  const auto dd = std::make_unique<Package>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);

  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromEmptyMatrix) {
  const auto inputMatrix = CMat{};

  constexpr auto nrQubits = 3U;
  const auto dd = std::make_unique<Package>(nrQubits);
  EXPECT_TRUE(dd->makeDDFromMatrix(inputMatrix).isOneTerminal());
}

TEST(DDPackageTest, DDFromNonPowerOfTwoMatrix) {
  auto inputMatrix = CMat{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

  constexpr auto nrQubits = 3U;
  const auto dd = std::make_unique<Package>(nrQubits);
  EXPECT_THROW(dd->makeDDFromMatrix(inputMatrix), std::invalid_argument);
}

TEST(DDPackageTest, DDFromNonSquareMatrix) {
  const auto inputMatrix = CMat{{0, 1, 2, 3}, {4, 5, 6, 7}};

  constexpr auto nrQubits = 3U;
  const auto dd = std::make_unique<Package>(nrQubits);
  EXPECT_THROW(dd->makeDDFromMatrix(inputMatrix), std::invalid_argument);
}

TEST(DDPackageTest, DDFromSingleElementMatrix) {
  const auto inputMatrix = CMat{{1}};

  constexpr auto nrQubits = 1U;
  const auto dd = std::make_unique<Package>(nrQubits);

  EXPECT_TRUE(dd->makeDDFromMatrix(inputMatrix).isOneTerminal());
}

constexpr TwoQubitGateMatrix CX_MAT{
    {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}};
constexpr TwoQubitGateMatrix CZ_MAT{
    {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}}};

TEST(DDPackageTest, TwoQubitControlledGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto gateMatrices =
      std::vector{std::pair{opToSingleQubitGateMatrix(qc::X), CX_MAT},
                  std::pair{opToSingleQubitGateMatrix(qc::Z), CZ_MAT}};

  // For every combination of control and target, test that the DD created by
  // makeTwoQubitGateDD is equal to the DD created by makeGateDD. This should
  // cover every scenario of the makeTwoQubitGateDD function.
  for (const auto& [gateMatrix, controlledGateMatrix] : gateMatrices) {
    for (Qubit control = 0; control < nrQubits; ++control) {
      for (Qubit target = 0; target < nrQubits; ++target) {
        if (control == target) {
          continue;
        }
        const auto controlledGateDD =
            dd->makeTwoQubitGateDD(controlledGateMatrix, control, target);
        const auto gateDD = dd->makeGateDD(
            gateMatrix, qc::Control{static_cast<qc::Qubit>(control)}, target);
        EXPECT_EQ(controlledGateDD, gateDD);
      }
    }
  }
}

TEST(DDPackageTest, TwoQubitControlledGateDDConstructionNegativeControls) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto gateMatrices =
      std::vector{std::pair{opToSingleQubitGateMatrix(qc::X), CX_MAT},
                  std::pair{opToSingleQubitGateMatrix(qc::Z), CZ_MAT}};

  // For every combination of controls, control type, and target, test that the
  // DD created by makeTwoQubitGateDD is equal to the DD created by makeGateDD.
  // This should cover every scenario of the makeTwoQubitGateDD function.
  for (const auto& [gateMatrix, controlledGateMatrix] : gateMatrices) {
    for (Qubit control0 = 0; control0 < nrQubits; ++control0) {
      for (Qubit control1 = 0; control1 < nrQubits; ++control1) {
        if (control0 == control1) {
          continue;
        }
        for (Qubit target = 0; target < nrQubits; ++target) {
          if (control0 == target || control1 == target) {
            continue;
          }
          for (const auto controlType :
               {qc::Control::Type::Pos, qc::Control::Type::Neg}) {
            const auto controlledGateDD = dd->makeTwoQubitGateDD(
                controlledGateMatrix, qc::Controls{{control0, controlType}},
                control1, target);
            const auto gateDD = dd->makeGateDD(
                gateMatrix, qc::Controls{{control0, controlType}, control1},
                target);
            EXPECT_EQ(controlledGateDD, gateDD);
          }
        }
      }
    }
  }
}

TEST(DDPackageTest, SWAPGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto swapGateDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::SWAP), *dd);

      auto gateDD = getDD(qc::StandardOperation(control, target, qc::X), *dd);
      gateDD = dd->multiply(
          gateDD, dd->multiply(
                      getDD(qc::StandardOperation(target, control, qc::X), *dd),
                      gateDD));

      EXPECT_EQ(swapGateDD, gateDD);
    }
  }
}

TEST(DDPackageTest, PeresGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto peresGateDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::Peres), *dd);

      auto gateDD = getDD(qc::StandardOperation(control, target, qc::X), *dd);
      gateDD = dd->multiply(getDD(qc::StandardOperation(control, qc::X), *dd),
                            gateDD);

      EXPECT_EQ(peresGateDD, gateDD);

      const auto peresInvDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::Peresdg),
          *dd);

      auto gateInvDD = getDD(qc::StandardOperation(control, qc::X), *dd);
      gateInvDD = dd->multiply(
          getDD(qc::StandardOperation(control, target, qc::X), *dd), gateInvDD);

      EXPECT_EQ(peresInvDD, gateInvDD);
    }
  }
}

TEST(DDPackageTest, iSWAPGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto iswapGateDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::iSWAP), *dd);

      auto gateDD = getDD(qc::StandardOperation(target, qc::S), *dd); // S q[1)
      gateDD = dd->multiply(getDD(qc::StandardOperation(control, qc::S), *dd),
                            gateDD); // S q[0)
      gateDD = dd->multiply(getDD(qc::StandardOperation(control, qc::H), *dd),
                            gateDD); // H q[0)
      gateDD = dd->multiply(
          getDD(qc::StandardOperation(control, target, qc::X), *dd),
          gateDD); // CX q[0], q[1]
      gateDD = dd->multiply(
          getDD(qc::StandardOperation(target, control, qc::X), *dd),
          gateDD); // CX q[1], q[0]
      gateDD = dd->multiply(getDD(qc::StandardOperation(target, qc::H), *dd),
                            gateDD); // H q[1)

      EXPECT_EQ(iswapGateDD, gateDD);

      const auto iswapInvGateDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::iSWAPdg),
          *dd);

      auto gateInvDD =
          getDD(qc::StandardOperation(target, qc::H), *dd); // H q[1)
      gateInvDD = dd->multiply(
          getDD(qc::StandardOperation(target, control, qc::X), *dd),
          gateInvDD); // CX q[1], q[0]
      gateInvDD = dd->multiply(
          getDD(qc::StandardOperation(control, target, qc::X), *dd),
          gateInvDD); // CX q[0], q[1]
      gateInvDD =
          dd->multiply(getDD(qc::StandardOperation(control, qc::H), *dd),
                       gateInvDD); // H q[0)
      gateInvDD =
          dd->multiply(getDD(qc::StandardOperation(control, qc::Sdg), *dd),
                       gateInvDD); // Sdag q[0]
      gateInvDD =
          dd->multiply(getDD(qc::StandardOperation(target, qc::Sdg), *dd),
                       gateInvDD); // Sdag q[1]

      EXPECT_EQ(iswapInvGateDD, gateInvDD);
    }
  }
}

TEST(DDPackageTest, DCXGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto dcxGateDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::DCX), *dd);

      const auto gateDD = dd->multiply(
          getDD(qc::StandardOperation(target, control, qc::X), *dd),
          getDD(qc::StandardOperation(control, target, qc::X), *dd));

      EXPECT_EQ(dcxGateDD, gateDD);
    }
  }
}

TEST(DDPackageTest, RZZGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto params = {0., PI_2, PI, 2 * PI};

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto rzzGateDD = getDD(
            qc::StandardOperation({control, target}, qc::RZZ, {param}), *dd);

        auto gateDD = getDD(qc::StandardOperation(control, target, qc::X), *dd);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation(target, qc::RZ, {param}), *dd), gateDD);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation(control, target, qc::X), *dd), gateDD);

        EXPECT_EQ(rzzGateDD, gateDD);
      }
    }
  }

  const auto identity = Package::makeIdent();
  const auto rzzZero = getDD(qc::StandardOperation({0, 1}, qc::RZZ, {0.}), *dd);
  EXPECT_EQ(rzzZero, identity);

  const auto rzzTwoPi =
      getDD(qc::StandardOperation({0, 1}, qc::RZZ, {2 * PI}), *dd);
  EXPECT_EQ(rzzTwoPi.p, identity.p);
  EXPECT_EQ(RealNumber::val(rzzTwoPi.w.r), -1.);

  const auto rzzPi = getDD(qc::StandardOperation({0, 1}, qc::RZZ, {PI}), *dd);
  auto zz = getDD(qc::StandardOperation(qc::Controls{}, 0, qc::Z), *dd);
  zz = dd->multiply(
      zz, getDD(qc::StandardOperation(qc::Controls{}, 1, qc::Z), *dd));
  EXPECT_EQ(rzzPi.p, zz.p);
}

TEST(DDPackageTest, RYYGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto params = {0., PI_2, PI};

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto ryyGateDD = getDD(
            qc::StandardOperation({control, target}, qc::RYY, {param}), *dd);

        // no controls are necessary on the RX gates since they cancel if the
        // controls are 0.
        auto gateDD =
            getDD(qc::StandardOperation(control, qc::RX, {PI_2}), *dd);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation(target, qc::RX, {PI_2}), *dd), gateDD);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation({control, target}, qc::RZZ, {param}),
                  *dd),
            gateDD);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation(target, qc::RX, {-PI_2}), *dd), gateDD);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation(control, qc::RX, {-PI_2}), *dd),
            gateDD);

        EXPECT_EQ(ryyGateDD, gateDD);
      }
    }
  }

  const auto identity = Package::makeIdent();
  const auto ryyZero = getDD(qc::StandardOperation({0, 1}, qc::RYY, {0.}), *dd);
  EXPECT_EQ(ryyZero, identity);

  const auto ryyPi = getDD(qc::StandardOperation({0, 1}, qc::RYY, {PI}), *dd);
  auto yy = getDD(qc::StandardOperation(qc::Controls{}, 0, qc::Y), *dd);
  yy = dd->multiply(
      yy, getDD(qc::StandardOperation(qc::Controls{}, 1, qc::Y), *dd));
  EXPECT_EQ(ryyPi.p, yy.p);
}

TEST(DDPackageTest, RXXGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto params = {0., PI_2, PI};

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto rxxGateDD = getDD(
            qc::StandardOperation({control, target}, qc::RXX, {param}), *dd);

        auto gateDD = getDD(qc::StandardOperation(control, qc::H), *dd);
        gateDD = dd->multiply(getDD(qc::StandardOperation(target, qc::H), *dd),
                              gateDD);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation({control, target}, qc::RZZ, {param}),
                  *dd),
            gateDD);
        gateDD = dd->multiply(getDD(qc::StandardOperation(target, qc::H), *dd),
                              gateDD);
        gateDD = dd->multiply(getDD(qc::StandardOperation(control, qc::H), *dd),
                              gateDD);

        EXPECT_EQ(rxxGateDD, gateDD);
      }
    }
  }

  const auto identity = Package::makeIdent();
  const auto rxxZero = getDD(qc::StandardOperation({0, 1}, qc::RXX, {0.}), *dd);
  EXPECT_EQ(rxxZero, identity);

  const auto rxxPi = getDD(qc::StandardOperation({0, 1}, qc::RXX, {PI}), *dd);
  auto xx = getDD(qc::StandardOperation(qc::Controls{}, 0, qc::X), *dd);
  xx = dd->multiply(
      xx, getDD(qc::StandardOperation(qc::Controls{}, 1, qc::X), *dd));
  EXPECT_EQ(rxxPi.p, xx.p);
}

TEST(DDPackageTest, RZXGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto params = {0., PI_2, PI};

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto rzxGateDD = getDD(
            qc::StandardOperation({control, target}, qc::RZX, {param}), *dd);

        // no controls are necessary on the H gates since they cancel if the
        // controls are 0.
        auto gateDD = getDD(qc::StandardOperation(target, qc::H), *dd);
        gateDD = dd->multiply(
            getDD(qc::StandardOperation({control, target}, qc::RZZ, {param}),
                  *dd),
            gateDD);
        gateDD = dd->multiply(getDD(qc::StandardOperation(target, qc::H), *dd),
                              gateDD);

        EXPECT_EQ(rzxGateDD, gateDD);
      }
    }
  }

  const auto identity = Package::makeIdent();
  const auto rzxZero = getDD(qc::StandardOperation({0, 1}, qc::RZX, {0.}), *dd);
  EXPECT_EQ(rzxZero, identity);

  const auto rzxPi = getDD(qc::StandardOperation({0, 1}, qc::RZX, {PI}), *dd);
  auto zx = getDD(qc::StandardOperation(qc::Controls{}, 0, qc::Z), *dd);
  zx = dd->multiply(
      zx, getDD(qc::StandardOperation(qc::Controls{}, 1, qc::X), *dd));
  EXPECT_EQ(rzxPi.p, zx.p);
}

TEST(DDPackageTest, ECRGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }

      const auto ecrGateDD = getDD(
          qc::StandardOperation(qc::Targets{control, target}, qc::ECR), *dd);

      auto gateDD =
          getDD(qc::StandardOperation({control, target}, qc::RZX, {PI_4}), *dd);
      gateDD = dd->multiply(getDD(qc::StandardOperation(control, qc::X), *dd),
                            gateDD);
      gateDD = dd->multiply(
          getDD(qc::StandardOperation({control, target}, qc::RZX, {-PI_4}),
                *dd),
          gateDD);

      EXPECT_EQ(ecrGateDD, gateDD);
    }
  }
}

TEST(DDPackageTest, XXMinusYYGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto thetaAngles = {0., PI_2, PI};
  const auto betaAngles = {0., PI_2, PI};

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }

      for (const auto& theta : thetaAngles) {
        for (const auto& beta : betaAngles) {
          const auto xxMinusYYGateDD =
              getDD(qc::StandardOperation({control, target}, qc::XXminusYY,
                                          {theta, beta}),
                    *dd);

          auto gateDD =
              getDD(qc::StandardOperation(target, qc::RZ, {-beta}), *dd);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RZ, {-PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::SX), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RZ, {PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::S), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, target, qc::X), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RY, {theta / 2.}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RY, {-theta / 2.}), *dd),
              gateDD);

          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, target, qc::X), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::Sdg), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RZ, {-PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::SXdg), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RZ, {PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RZ, {beta}), *dd),
              gateDD);

          EXPECT_EQ(xxMinusYYGateDD, gateDD);
        }
      }
    }
  }
}

TEST(DDPackageTest, XXPlusYYGateDDConstruction) {
  constexpr auto nrQubits = 5U;
  const auto dd = std::make_unique<Package>(nrQubits);

  const auto thetaAngles = {0., PI_2, PI};
  const auto betaAngles = {0., PI_2, PI};

  for (Qubit control = 0; control < nrQubits; ++control) {
    for (Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }

      for (const auto& theta : thetaAngles) {
        for (const auto& beta : betaAngles) {
          const auto xxPlusYYGateDD =
              getDD(qc::StandardOperation({control, target}, qc::XXplusYY,
                                          {theta, beta}),
                    *dd);
          auto gateDD =
              getDD(qc::StandardOperation(control, qc::RZ, {beta}), *dd);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RZ, {-PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::SX), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RZ, {PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::S), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, control, qc::X), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RY, {-theta / 2.}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RY, {-theta / 2.}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, control, qc::X), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::Sdg), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RZ, {-PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::SXdg), *dd), gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(target, qc::RZ, {PI_2}), *dd),
              gateDD);
          gateDD = dd->multiply(
              getDD(qc::StandardOperation(control, qc::RZ, {-beta}), *dd),
              gateDD);

          EXPECT_EQ(xxPlusYYGateDD, gateDD);
        }
      }
    }
  }
}

TEST(DDPackageTest, InnerProductTopNodeConjugation) {
  // Test comes from experimental results
  // 2 qubit state is rotated Rxx(-2) equivalent to
  // Ising model evolution up to a time T=1
  constexpr auto nrQubits = 2U;
  const auto dd = std::make_unique<Package>(nrQubits);
  const auto zeroState = dd->makeZeroState(nrQubits);
  const auto rxx = getDD(qc::StandardOperation({0, 1}, qc::RXX, {-2}), *dd);
  const auto op = getDD(qc::StandardOperation(0, qc::Z), *dd);

  const auto evolvedState = dd->multiply(rxx, zeroState);

  // Actual evolution results in approximately -0.416
  // If the top node in the inner product is not conjugated properly,
  // it will result in +0.416.
  EXPECT_NEAR(dd->expectationValue(op, evolvedState), -0.416, 0.001);
}

/**
 * @brief This is a regression test for a long lasting memory leak in the DD
 * package.
 * @details The memory leak was caused by a bug in the normalization routine
 * which was not properly returning a node to the memory manager. This occurred
 * whenever the multiplication of two DDs resulted in a zero terminal.
 */
TEST(DDPackageTest, DDNodeLeakRegressionTest) {
  constexpr auto nqubits = 1U;
  auto dd = std::make_unique<Package>(nqubits);

  const auto dd1 = dd->makeGateDD(MEAS_ZERO_MAT, 0U);
  const auto dd2 = dd->makeGateDD(MEAS_ONE_MAT, 0U);
  dd->multiply(dd1, dd2);
  dd->garbageCollect(true);
  EXPECT_EQ(dd->mMemoryManager.getStats().numUsed, 0U);
}

/**
 * @brief This is a regression test for a compute table bug with terminals.
 * @details The bug was caused by the assumption that `result.p == nullptr`
 * indicates that the lookup was unsuccessful. However, this is not the case
 * anymore since terminal DD nodes were replaced by a `nullptr` pointer.
 */
TEST(DDPackageTest, CTPerformanceRegressionTest) {
  constexpr auto nqubits = 1U;
  auto dd = std::make_unique<Package>(nqubits);

  const auto dd1 = dd->makeGateDD(MEAS_ZERO_MAT, 0U);
  const auto dd2 = dd->makeGateDD(MEAS_ONE_MAT, 0U);
  constexpr auto repetitions = 10U;
  for (auto i = 0U; i < repetitions; ++i) {
    dd->multiply(dd1, dd2);
  }
  const auto& ct = dd->matrixMatrixMultiplication;
  EXPECT_EQ(ct.getStats().lookups, repetitions);
  EXPECT_EQ(ct.getStats().hits, repetitions - 1U);

  // This additional check makes sure that no nodes are leaked.
  dd->garbageCollect(true);
  EXPECT_EQ(dd->mMemoryManager.getStats().numUsed, 0U);
}

TEST(DDPackageTest, DataStructureStatistics) {
  constexpr auto nqubits = 1U;
  const auto dd = std::make_unique<Package>(nqubits);
  const auto stats = getDataStructureStatistics();

  EXPECT_EQ(stats["vNode"]["size_B"], 64U);
  EXPECT_EQ(stats["vNode"]["alignment_B"], 8U);
  EXPECT_EQ(stats["mNode"]["size_B"], 112U);
  EXPECT_EQ(stats["mNode"]["alignment_B"], 8U);
  EXPECT_EQ(stats["dNode"]["size_B"], 112U);
  EXPECT_EQ(stats["dNode"]["alignment_B"], 8U);
  EXPECT_EQ(stats["vEdge"]["size_B"], 24U);
  EXPECT_EQ(stats["vEdge"]["alignment_B"], 8U);
  EXPECT_EQ(stats["mEdge"]["size_B"], 24U);
  EXPECT_EQ(stats["mEdge"]["alignment_B"], 8U);
  EXPECT_EQ(stats["dEdge"]["size_B"], 24U);
  EXPECT_EQ(stats["dEdge"]["alignment_B"], 8U);
  EXPECT_EQ(stats["RealNumber"]["size_B"], 16U);
  EXPECT_EQ(stats["RealNumber"]["alignment_B"], 8U);
}

TEST(DDPackageTest, DDStatistics) {
  constexpr auto nqubits = 2U;
  const auto dd = std::make_unique<Package>(nqubits);
  const auto dummyGate = getDD(qc::StandardOperation(0U, qc::X), *dd);
  EXPECT_NE(dummyGate.p, nullptr);
  const auto stats = getStatistics(*dd, true);

  std::cout << stats.dump(2) << "\n";
  EXPECT_TRUE(stats.contains("vector"));
  ASSERT_TRUE(stats.contains("matrix"));
  EXPECT_TRUE(stats.contains("density_matrix"));
  EXPECT_TRUE(stats.contains("real_numbers"));
  EXPECT_TRUE(stats.contains("compute_tables"));
  const auto& matrixStats = stats["matrix"];
  ASSERT_TRUE(matrixStats.contains("unique_table"));
  const auto& uniqueTableStats = matrixStats["unique_table"];
  EXPECT_TRUE(uniqueTableStats.contains("0"));
  EXPECT_TRUE(uniqueTableStats.contains("1"));
  EXPECT_TRUE(uniqueTableStats.contains("total"));
  ASSERT_TRUE(uniqueTableStats["0"].contains("num_buckets"));
  EXPECT_GT(uniqueTableStats["0"]["num_buckets"], 0);
  ASSERT_TRUE(uniqueTableStats["total"].contains("num_buckets"));
  EXPECT_GT(uniqueTableStats["total"]["num_buckets"], 0);
}

TEST(DDPackageTest, ReduceAncillaRegression) {
  const auto dd = std::make_unique<Package>(2);
  const auto inputMatrix =
      CMat{{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}};
  auto inputDD = dd->makeDDFromMatrix(inputMatrix);
  dd->incRef(inputDD);
  const auto outputDD = dd->reduceAncillae(inputDD, {true, false});

  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      CMat{{1, 0, 1, 0}, {1, 0, 1, 0}, {1, 0, -1, 0}, {1, 0, -1, 0}};

  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, VectorConjugate) {
  const auto dd = std::make_unique<Package>(2);

  EXPECT_EQ(dd->conjugate(vEdge::zero()), vEdge::zero());

  EXPECT_EQ(dd->conjugate(vEdge::one()), vEdge::one());
  EXPECT_EQ(dd->conjugate(vEdge::terminal(dd->cn.lookup(0., 1.))),
            vEdge::terminal(dd->cn.lookup(0., -1.)));

  CVec vec{{0., 0.5},
           {0.5 * SQRT2_2, 0.5 * SQRT2_2},
           {0., -0.5},
           {-0.5 * SQRT2_2, -0.5 * SQRT2_2}};

  const auto vecDD = dd->makeStateFromVector(vec);
  std::cout << "Vector:\n";
  vecDD.printVector();
  const auto conjVecDD = dd->conjugate(vecDD);
  std::cout << "Conjugated vector:\n";
  conjVecDD.printVector();

  const auto conjVec = conjVecDD.getVector();
  for (auto i = 0U; i < vec.size(); ++i) {
    constexpr auto tolerance = 1e-10;
    EXPECT_NEAR(conjVec[i].real(), vec[i].real(), tolerance);
    EXPECT_NEAR(conjVec[i].imag(), -vec[i].imag(), tolerance);
  }
}

TEST(DDPackageTest, ReduceAncillaIdentity) {
  const auto dd = std::make_unique<Package>(2);
  auto inputDD = Package::makeIdent();
  const auto outputDD = dd->reduceAncillae(inputDD, {true, true});

  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      CMat{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceAnicllaIdentityBeforeFirstNode) {
  const auto dd = std::make_unique<Package>(2);
  auto xGate = getDD(qc::StandardOperation(0, qc::X), *dd);
  const auto outputDD = dd->reduceAncillae(xGate, {false, true});

  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      CMat{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceAnicllaIdentityAfterLastNode) {
  const auto dd = std::make_unique<Package>(2);
  auto xGate = getDD(qc::StandardOperation(1, qc::X), *dd);
  dd->incRef(xGate);
  const auto outputDD = dd->reduceAncillae(xGate, {true, false});

  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      CMat{{0, 0, 1, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceAncillaIdentityBetweenTwoNodes) {
  const auto dd = std::make_unique<Package>(3);
  const auto xGate0 = getDD(qc::StandardOperation(0, qc::X), *dd);
  const auto xGate2 = getDD(qc::StandardOperation(2, qc::X), *dd);
  auto state = dd->multiply(xGate0, xGate2);

  dd->incRef(state);
  const auto outputDD = dd->reduceAncillae(state, {false, true, false});
  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      CMat{{0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
           {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
           {0, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0},
           {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentity) {
  const auto dd = std::make_unique<Package>(2);
  auto inputDD = Package::makeIdent();
  auto outputDD = dd->reduceGarbage(inputDD, {true, true});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected = CMat{{1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  outputDD = dd->reduceGarbage(inputDD, {true, true}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = CMat{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentityBeforeFirstNode) {
  const auto dd = std::make_unique<Package>(2);
  auto xGate = getDD(qc::StandardOperation(0, qc::X), *dd);
  auto outputDD = dd->reduceGarbage(xGate, {false, true});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected = CMat{{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  outputDD = dd->reduceGarbage(xGate, {false, true}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = CMat{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentityAfterLastNode) {
  const auto dd = std::make_unique<Package>(2);
  auto xGate = getDD(qc::StandardOperation(1, qc::X), *dd);
  dd->incRef(xGate);
  auto outputDD = dd->reduceGarbage(xGate, {true, false});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected = CMat{{0, 0, 1, 1}, {0, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  dd->incRef(xGate);
  outputDD = dd->reduceGarbage(xGate, {true, false}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = CMat{{0, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentityBetweenTwoNodes) {
  const auto dd = std::make_unique<Package>(3);
  const auto xGate0 = getDD(qc::StandardOperation(0, qc::X), *dd);
  const auto xGate2 = getDD(qc::StandardOperation(2, qc::X), *dd);
  auto state = dd->multiply(xGate0, xGate2);

  dd->incRef(state);
  auto outputDD = dd->reduceGarbage(state, {false, true, false});
  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected = CMat{{0, 0, 0, 0, 0, 1, 0, 1}, {0, 0, 0, 0, 1, 0, 1, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
                       {0, 1, 0, 1, 0, 0, 0, 0}, {1, 0, 1, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  dd->incRef(state);
  outputDD = dd->reduceGarbage(state, {false, true, false}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = CMat{{0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
                  {0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
                  {0, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0},
                  {0, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}
} // namespace dd
