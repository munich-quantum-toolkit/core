/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/GHZState.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/QPE.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"
#include "algorithms/WState.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace dd {

struct Experiment {
  std::unique_ptr<Package> dd;
  nlohmann::basic_json<> stats = nlohmann::basic_json<>::object();
  std::chrono::duration<double> runtime{};

  Experiment() = default;
  Experiment(const Experiment&) = delete;
  Experiment(Experiment&&) = default;
  Experiment& operator=(const Experiment&) = delete;
  Experiment& operator=(Experiment&&) = default;
  virtual ~Experiment() = default;

  [[nodiscard]] virtual bool success() const noexcept { return false; }
};

struct SimulationExperiment final : Experiment {
  SimulationExperiment() = default;
  VectorDD sim{};

  [[nodiscard]] bool success() const noexcept override {
    return sim.p != nullptr;
  }
};

struct FunctionalityConstructionExperiment final : Experiment {
  MatrixDD func{};

  [[nodiscard]] bool success() const noexcept override {
    return func.p != nullptr;
  }
};

namespace {
std::unique_ptr<SimulationExperiment>
benchmarkSimulate(const qc::QuantumComputation& qc) {
  auto exp = std::make_unique<SimulationExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  const auto in = makeZeroState(nq, *exp->dd);
  exp->sim = simulate(qc, in, *(exp->dd));
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = getStatistics(*exp->dd);
  return exp;
}

std::unique_ptr<FunctionalityConstructionExperiment>
benchmarkFunctionalityConstruction(const qc::QuantumComputation& qc) {
  auto exp = std::make_unique<FunctionalityConstructionExperiment>();
  const auto nq = qc.getNqubits();
  exp->dd = std::make_unique<Package>(nq);
  const auto start = std::chrono::high_resolution_clock::now();
  exp->func = buildFunctionality(qc, *(exp->dd));
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = getStatistics(*exp->dd);
  return exp;
}

std::unique_ptr<SimulationExperiment>
benchmarkSimulateGrover(const qc::Qubit nq,
                        const qc::GroverBitString& targetValue) {
  auto exp = std::make_unique<SimulationExperiment>();
  exp->dd = std::make_unique<Package>(nq + 1);
  auto& dd = *(exp->dd);
  const auto start = std::chrono::high_resolution_clock::now();

  // apply state preparation setup
  qc::QuantumComputation statePrep(nq + 1);
  appendGroverInitialization(statePrep);
  const auto s = buildFunctionality(statePrep, dd);
  auto e = dd.applyOperation(s, makeZeroState(nq + 1, dd));

  qc::QuantumComputation groverIteration(nq + 1);
  appendGroverOracle(groverIteration, targetValue);
  appendGroverDiffusion(groverIteration);

  auto iter = buildFunctionalityRecursive(groverIteration, dd);
  const auto iterations = qc::computeNumberOfIterations(nq);
  const std::bitset<128U> iterBits(iterations);
  const auto msb = static_cast<std::size_t>(std::floor(std::log2(iterations)));
  auto f = iter;
  dd.track(f);
  for (std::size_t j = 0U; j <= msb; ++j) {
    if (iterBits[j]) {
      e = dd.applyOperation(f, e);
    }
    if (j < msb) {
      f = dd.applyOperation(f, f);
    }
  }
  dd.untrack(f);
  exp->sim = e;
  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = getStatistics(*exp->dd);
  return exp;
}

std::unique_ptr<FunctionalityConstructionExperiment>
benchmarkFunctionalityConstructionGrover(
    const qc::Qubit nq, const qc::GroverBitString& targetValue) {
  auto exp = std::make_unique<FunctionalityConstructionExperiment>();
  exp->dd = std::make_unique<Package>(nq + 1);
  auto& dd = *(exp->dd);
  const auto start = std::chrono::high_resolution_clock::now();

  qc::QuantumComputation groverIteration(nq + 1);
  appendGroverOracle(groverIteration, targetValue);
  appendGroverDiffusion(groverIteration);

  const auto iter = buildFunctionalityRecursive(groverIteration, dd);
  auto e = iter;
  const auto iterations = qc::computeNumberOfIterations(nq);
  const std::bitset<128U> iterBits(iterations);
  const auto msb = static_cast<std::size_t>(std::floor(std::log2(iterations)));
  auto f = iter;
  dd.track(f);
  bool zero = !iterBits[0U];
  for (std::size_t j = 1U; j <= msb; ++j) {
    f = dd.applyOperation(f, f);
    if (iterBits[j]) {
      if (zero) {
        dd.track(f);
        dd.untrack(e);
        e = f;
        zero = false;
      } else {
        e = dd.applyOperation(f, e);
      }
    }
  }
  dd.untrack(f);

  // apply state preparation setup
  qc::QuantumComputation statePrep(nq + 1);
  appendGroverInitialization(statePrep);
  const auto s = buildFunctionality(statePrep, dd);
  exp->func = dd.applyOperation(e, s);

  const auto end = std::chrono::high_resolution_clock::now();
  exp->runtime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  exp->stats = getStatistics(*exp->dd);
  return exp;
}
} // namespace

static constexpr std::size_t SEED = 42U;

class BenchmarkDDPackage {
protected:
  void verifyAndSave(const std::string& name, const std::string& type,
                     qc::QuantumComputation& qc, const Experiment& exp) const {

    const std::string& filename = "results_" + inputFilename + ".json";

    nlohmann::basic_json<> j;
    std::fstream file(filename, std::ios::in | std::ios::out | std::ios::ate);
    if (!file.is_open()) {
      std::ofstream outputFile(filename);
      outputFile << nlohmann::json();
    } else if (file.tellg() == 0) {
      file << nlohmann::json();
    }
    file.close();

    std::ifstream ifs(filename);
    ifs >> j;
    ifs.close();

    auto& entry = j[name][type][std::to_string(qc.getNqubits())];

    entry["runtime"] = exp.runtime.count();

    // collect statistics from DD package
    entry["dd"] = exp.stats;

    std::ofstream ofs(filename);
    ofs << j.dump(2U);
    ofs.close();
  }

  std::string inputFilename;

  void runGHZ() const {
    constexpr std::array nqubits = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running GHZ Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createGHZState(nq);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("GHZ", "Simulation", qc, *exp);
    }
    std::cout << "Running GHZ Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createGHZState(nq);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("GHZ", "Functionality", qc, *exp);
    }
  }

  void runWState() const {
    constexpr std::array nqubits = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running WState Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createWState(nq);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("WState", "Simulation", qc, *exp);
    }
    std::cout << "Running WState Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createWState(nq);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("WState", "Functionality", qc, *exp);
    }
  }

  void runBV() const {
    constexpr std::array nqubits = {255U, 511U, 1023U, 2047U, 4095U};
    std::cout << "Running BV Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createBernsteinVazirani(nq, SEED);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("BV", "Simulation", qc, *exp);
    }
    std::cout << "Running BV Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createBernsteinVazirani(nq, SEED);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("BV", "Functionality", qc, *exp);
    }
  }

  void runQFT() const {
    constexpr std::array nqubitsSim = {64U, 128U, 256U, 512U, 1024U};
    std::cout << "Running QFT Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createQFT(nq, false);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QFT", "Simulation", qc, *exp);
    }
    constexpr std::array nqubitsFunc = {16U, 17U, 18U, 19U, 20U};
    std::cout << "Running QFT Functionality..." << '\n';
    for (const auto& nq : nqubitsFunc) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createQFT(nq, false);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QFT", "Functionality", qc, *exp);
    }
  }

  void runGrover() const {
    constexpr std::array nqubits = {27U, 31U, 35U, 39U};
    std::cout << "Running Grover Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      qc::GroverBitString targetValue;
      targetValue.set();
      auto qc = qc::createGrover(nq, targetValue);
      auto exp = benchmarkSimulateGrover(nq, targetValue);
      verifyAndSave("Grover", "Simulation", qc, *exp);
    }

    std::cout << "Running Grover Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      std::cout << "... with " << nq << " qubits" << '\n';
      qc::GroverBitString targetValue;
      targetValue.set();
      auto qc = qc::createGrover(nq, targetValue);
      auto exp = benchmarkFunctionalityConstructionGrover(nq, targetValue);
      verifyAndSave("Grover", "Functionality", qc, *exp);
    }
  }

  void runQPE() const {
    constexpr std::array nqubitsSim = {14U, 15U, 16U, 17U, 18U};
    std::cout << "Running QPE Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createQPE(nq, false, SEED);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QPE", "Simulation", qc, *exp);
    }
    std::cout << "Running QPE Functionality..." << '\n';
    constexpr std::array nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createQPE(nq, false, SEED);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QPE", "Functionality", qc, *exp);
    }
  }

  void runExactQPE() const {
    constexpr std::array nqubitsSim = {8U, 16U, 32U, 48U};
    std::cout << "Running QPE (exact) Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createQPE(nq, true, SEED);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QPE_Exact", "Simulation", qc, *exp);
    }
    std::cout << "Running QPE (exakt) Functionality..." << '\n';
    constexpr std::array nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createQPE(nq, true, SEED);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QPE_Exact", "Functionality", qc, *exp);
    }
  }

  void runRandomClifford() const {
    constexpr std::array nqubitsSim = {14U, 15U, 16U, 17U, 18U};
    std::cout << "Running RandomClifford Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createRandomCliffordCircuit(
          nq, static_cast<size_t>(nq) * nq, SEED);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("RandomClifford", "Simulation", qc, *exp);
    }
    std::cout << "Running RandomClifford Functionality..." << '\n';
    constexpr std::array nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      std::cout << "... with " << nq << " qubits" << '\n';
      auto qc = qc::createRandomCliffordCircuit(
          nq, static_cast<size_t>(nq) * nq, SEED);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("RandomClifford", "Functionality", qc, *exp);
    }
  }

public:
  explicit BenchmarkDDPackage(std::string filename)
      : inputFilename(std::move(filename)) {};

  void runAll() const {
    runGHZ();
    runWState();
    runBV();
    runQFT();
    runGrover();
    runQPE();
    runExactQPE();
    runRandomClifford();
  }
};

} // namespace dd

int main(const int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Exactly one argument is required to name the results file."
              << '\n';
    return 1;
  }
  try {
    const auto run = dd::BenchmarkDDPackage(
        argv[1]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    run.runAll();
  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Benchmarks done." << '\n';
  return 0;
}
