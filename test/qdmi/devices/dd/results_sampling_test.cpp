/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * DDSIM QDMI Device - Results: sampling (histogram keys/values)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::vector<std::string> getShots(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t size = qdmi_test::querySize(job, QDMI_JOB_RESULT_SHOTS);
  std::string result(size, '\0');
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job, 0U, QDMI_JOB_RESULT_SHOTS, size, result.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_FALSE(result.empty());
  if (!result.empty()) {
    EXPECT_EQ(result.back(), '\0');
    result.pop_back();
  }
  return qdmi_test::splitCSV(result);
}

class HistogramTest : public ::testing::Test {
protected:
  using Histogram = std::pair<std::vector<std::string>, std::vector<size_t>>;
  static constexpr size_t NUM_SHOTS = 1024;
  static constexpr size_t NUM_QUBITS = 3;

  static Histogram runProgram(const QDMI_Program_Format format,
                              const std::string_view program,
                              const std::optional<int> seed = std::nullopt,
                              std::vector<std::string>* samples = nullptr) {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    EXPECT_EQ(qdmi_test::setProgram(j.job, format, program), QDMI_SUCCESS);
    EXPECT_EQ(qdmi_test::setShots(j.job, NUM_SHOTS), QDMI_SUCCESS);
    if (seed.has_value()) {
      EXPECT_EQ(qdmi_test::setSeed(j.job, *seed), QDMI_SUCCESS);
    }
    EXPECT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
    auto shots = getShots(j.job);
    EXPECT_EQ(shots.size(), NUM_SHOTS);
    EXPECT_EQ(shots, getShots(j.job));
    std::map<std::string, size_t> counts;
    for (const auto& shot : shots) {
      ++counts[shot];
    }
    const auto histogram = qdmi_test::getHistogram(j.job);
    const auto& [keys, values] = histogram;
    EXPECT_EQ(keys.size(), counts.size());
    EXPECT_EQ(keys.size(), values.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      EXPECT_EQ(counts.at(keys[i]), values.at(i));
    }
    if (samples != nullptr) {
      *samples = std::move(shots);
    }
    return histogram;
  }

  static void checkHistogram(const Histogram& hist) {
    const auto& [keys, vals] = hist;
    // Keys and values come from two independent device queries.
    // Check both vectors have the same size.
    ASSERT_EQ(keys.size(), vals.size());
    // Values should sum up to NUM_SHOTS.
    const auto sum = std::accumulate(vals.cbegin(), vals.cend(), size_t{0});
    EXPECT_EQ(sum, NUM_SHOTS);
    // Both keys '00' and '11' should be expected.
    ASSERT_EQ(keys.size(), 2U);
    // And no other keys should be expected.
    EXPECT_TRUE(std::ranges::all_of(
        keys, [](const auto& k) { return k == "00" || k == "11"; }));
  }

  /// Smoke check used for circuits whose distribution we do not know precisely.
  /// For example, multi-output adaptive programs.
  static void checkSmokeHistogram(const Histogram& hist) {
    const auto& [keys, vals] = hist;
    // Both vectors have the same size.
    ASSERT_EQ(keys.size(), vals.size());
    // Values sum up to NUM_SHOTS.
    const auto sum = std::accumulate(vals.cbegin(), vals.cend(), size_t{0});
    EXPECT_EQ(sum, NUM_SHOTS);
    // Every key is a NUM_QUBITS long bit string.
    EXPECT_TRUE(std::ranges::all_of(keys, [](const auto& k) {
      return k.size() == NUM_QUBITS && std::ranges::all_of(k, [](char c) {
               return c == '0' || c == '1';
             });
    }));
  }
};

class QIRHistogramTestModule : public HistogramTest {
protected:
  static std::string getProgram(const std::string_view file) {
    const std::string text = qdmi_test::getQIRProgram(file);
    llvm::LLVMContext context;
    llvm::SMDiagnostic err;
    auto const llvmModule = llvm::parseAssemblyString(text, err, context);
    EXPECT_NE(llvmModule, nullptr)
        << "parseAssemblyString failed: " << err.getMessage().str();
    if (llvmModule == nullptr) {
      return {};
    }
    std::string bitcodeBuffer;
    llvm::raw_string_ostream os(bitcodeBuffer);
    llvm::WriteBitcodeToFile(*llvmModule, os);
    os.flush();
    return bitcodeBuffer;
  }
};

class QIRHistogramTestString : public HistogramTest {};

} // namespace

TEST_F(HistogramTest, QASM3Program) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QASM3;
  constexpr std::string_view program = qdmi_test::QASM3_BELL_SAMPLING;
  checkHistogram(runProgram(format, program));
}

TEST_F(HistogramTest, QASM3ProgramWithoutMeasurements) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QASM3;
  constexpr std::string_view program = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
)qasm";
  checkHistogram(runProgram(format, program));
}

TEST_F(HistogramTest, QASM2Program) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QASM2;
  constexpr std::string_view program = qdmi_test::QASM2_BELL_SAMPLING;
  checkHistogram(runProgram(format, program));
}

TEST_F(HistogramTest, QASM3MultipleRegistersFollowQiskitOrder) {
  constexpr std::string_view program = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
bit[2] c0;
bit c1;
qubit[3] q;
x q[0];
x q[2];
c0[0] = measure q[0];
c0[1] = measure q[1];
c1 = measure q[2];
)qasm";
  const auto [keys, values] = runProgram(QDMI_PROGRAM_FORMAT_QASM3, program);
  EXPECT_EQ(keys, std::vector<std::string>{"101"});
  EXPECT_EQ(values, std::vector<size_t>{NUM_SHOTS});
}

TEST_F(QIRHistogramTestModule, BaseStatic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
  checkHistogram(runProgram(format, getProgram("BellPairStatic.ll")));
}

TEST_F(QIRHistogramTestString, BaseStatic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  checkHistogram(
      runProgram(format, qdmi_test::getQIRProgram("BellPairStatic.ll")));
}

TEST_F(QIRHistogramTestModule, BaseDynamic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
  checkHistogram(runProgram(format, getProgram("BellPairDynamic.ll")));
}

TEST_F(QIRHistogramTestString, BaseDynamic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  checkHistogram(
      runProgram(format, qdmi_test::getQIRProgram("BellPairDynamic.ll")));
}

TEST_F(QIRHistogramTestModule, Adaptive) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE;
  checkHistogram(runProgram(format, getProgram("BellPairAdaptive.ll")));
}

TEST_F(QIRHistogramTestString, Adaptive) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
  checkHistogram(
      runProgram(format, qdmi_test::getQIRProgram("BellPairAdaptive.ll")));
}

TEST_F(QIRHistogramTestModule, AdaptiveRecordOutputs) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE;
  checkSmokeHistogram(
      runProgram(format, getProgram("AdaptiveRecordOutputs.ll")));
}

TEST_F(QIRHistogramTestString, AdaptiveRecordOutputs) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
  checkSmokeHistogram(
      runProgram(format, qdmi_test::getQIRProgram("AdaptiveRecordOutputs.ll")));
}

TEST_F(HistogramTest, SeedReproducesQASMSampling) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QASM3;
  constexpr std::string_view program = qdmi_test::QASM3_BELL_SAMPLING;
  std::vector<std::string> first;
  std::vector<std::string> second;
  EXPECT_EQ(runProgram(format, program, 7, &first),
            runProgram(format, program, 7, &second));
  EXPECT_EQ(first, second);
  EXPECT_FALSE(std::ranges::is_sorted(first));
}

TEST_F(QIRHistogramTestString, SeedReproducesQIRSampling) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  const auto program = qdmi_test::getQIRProgram("BellPairStatic.ll");
  std::vector<std::string> first;
  std::vector<std::string> second;
  EXPECT_EQ(runProgram(format, program, 7, &first),
            runProgram(format, program, 7, &second));
  EXPECT_EQ(first, second);
  EXPECT_FALSE(std::ranges::is_sorted(first));
}

TEST_F(HistogramTest, QASM3DynamicShotsPreserveClassicalMapping) {
  constexpr std::string_view program = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] a;
bit[2] b;
a[0] = false;
a[1] = false;
b[0] = false;
b[1] = false;
h q[0];
a[1] = measure q[0];
if (a[1]) { x q[1]; }
b[0] = measure q[1];
)qasm";
  const auto [keys, values] = runProgram(QDMI_PROGRAM_FORMAT_QASM3, program, 7);
  EXPECT_EQ(keys, (std::vector<std::string>{"0000", "0110"}));
  EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t{0}),
            NUM_SHOTS);
}

TEST(ResultsSampling, EmptyQASM3YieldsEmptyHistogram) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3, "OPENQASM 3.0;"),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 4), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  constexpr std::array results{
      QDMI_JOB_RESULT_HIST_KEYS,
      QDMI_JOB_RESULT_HIST_VALUES,
  };
  char dummy{};
  for (const auto result : results) {
    size_t size = 1;
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, 0U, result, 0,
                                                    nullptr, &size),
              QDMI_SUCCESS);
    EXPECT_EQ(size, 0U);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, 0U, result, 0,
                                                    &dummy, nullptr),
              QDMI_SUCCESS);
  }
}

TEST(ResultsSampling, BufferTooSmallErrors) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 512), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const size_t shotsSize = qdmi_test::querySize(j.job, QDMI_JOB_RESULT_SHOTS);
  ASSERT_EQ(shotsSize, 512U * 3U);
  std::vector<char> shotsTooSmall(shotsSize - 1);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 0U, QDMI_JOB_RESULT_SHOTS, shotsTooSmall.size(),
                shotsTooSmall.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

  if (const size_t ks = qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_KEYS);
      ks > 0) {
    std::vector<char> tooSmall(ks - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, 0U, QDMI_JOB_RESULT_HIST_KEYS, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }

  if (const size_t vs =
          qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_VALUES);
      vs > 0) {
    std::vector<char> tooSmall(vs - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, 0U, QDMI_JOB_RESULT_HIST_VALUES, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(ResultsSampling, StateAndProbRequestsAreInvalidWhenShotsPositive) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 32), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          j.job, 0U, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 0U, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 0U, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0,
                nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          j.job, 0U, QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 0U, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0,
                nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 0U, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0,
                nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}
