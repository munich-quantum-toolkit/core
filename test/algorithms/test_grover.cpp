#include "algorithms/Grover.hpp"
#include "dd/Benchmark.hpp"
#include "dd/FunctionalityConstruction.hpp"

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

class Grover
    : public testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {
protected:
  void TearDown() override {
    dd->decRef(sim);
    dd->decRef(func);
    dd->garbageCollect(true);

    // number of complex table entries after clean-up should equal initial
    // number of entries
    EXPECT_EQ(dd->cn.realCount(), initialComplexCount);

    // number of available cache entries after clean-up should equal initial
    // number of entries
    EXPECT_EQ(dd->cn.cacheCount(), initialCacheCount);
  }

  void SetUp() override {
    std::tie(nqubits, seed) = GetParam();
    dd = std::make_unique<dd::Package<>>(nqubits + 1);
    initialCacheCount = dd->cn.cacheCount();
    initialComplexCount = dd->cn.realCount();
  }

  std::size_t nqubits = 0;
  std::size_t seed = 0;
  std::unique_ptr<dd::Package<>> dd;
  std::unique_ptr<qc::Grover> qc;
  std::size_t initialCacheCount = 0;
  std::size_t initialComplexCount = 0;
  qc::VectorDD sim{};
  qc::MatrixDD func{};
};

constexpr std::size_t GROVER_MAX_QUBITS = 18;
constexpr std::size_t GROVER_NUM_SEEDS = 5;
constexpr dd::fp GROVER_ACCURACY = 1e-2;
constexpr dd::fp GROVER_GOAL_PROBABILITY = 0.9;

INSTANTIATE_TEST_SUITE_P(
    Grover, Grover,
    testing::Combine(
        testing::Range(static_cast<std::size_t>(2), GROVER_MAX_QUBITS + 1, 3),
        testing::Range(static_cast<std::size_t>(0), GROVER_NUM_SEEDS)),
    [](const testing::TestParamInfo<Grover::ParamType>& inf) {
      const auto nqubits = std::get<0>(inf.param);
      const auto seed = std::get<1>(inf.param);
      std::stringstream ss{};
      ss << nqubits + 1;
      if (nqubits == 0) {
        ss << "_qubit_";
      } else {
        ss << "_qubits_";
      }
      ss << seed;
      return ss.str();
    });

TEST_P(Grover, Functionality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

  qc->printStatistics(std::cout);
  auto x = '1' + qc->expected;
  std::reverse(x.begin(), x.end());
  std::replace(x.begin(), x.end(), '1', '2');

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionality(qc.get(), dd); });

  // amplitude of the searched-for entry should be 1
  auto c = func.getValueByPath(x);
  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, FunctionalityRecursive) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

  qc->printStatistics(std::cout);
  auto x = '1' + qc->expected;
  std::reverse(x.begin(), x.end());
  std::replace(x.begin(), x.end(), '1', '2');

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc.get(), dd); });

  // amplitude of the searched-for entry should be 1
  auto c = func.getValueByPath(x);
  EXPECT_NEAR(std::abs(c.real()), 1, GROVER_ACCURACY);
  EXPECT_NEAR(std::abs(c.imag()), 0, GROVER_ACCURACY);
  const auto prob = std::norm(c);
  EXPECT_GE(prob, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, Simulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = std::make_unique<qc::Grover>(nqubits, seed); });

  qc->printStatistics(std::cout);
  // there should be no error simulating the circuit
  const std::size_t shots = 1024;
  auto measurements = dd::benchmarkSimulateWithShots(*qc, shots);

  for (const auto& [state, count] : measurements) {
    std::cout << state << ": " << count << "\n";
  }

  auto correctShots = measurements[qc->expected];
  auto probability =
      static_cast<double>(correctShots) / static_cast<double>(shots);

  EXPECT_GE(probability, GROVER_GOAL_PROBABILITY);
}

TEST_P(Grover, BenchmarkBuildFunctionality) {
  qc = std::make_unique<qc::Grover>(nqubits, seed);
  const auto out = dd::benchmarkFunctionalityConstruction(*qc, true);
  EXPECT_NE(out->func.p, nullptr);
}
