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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR
#include "qir/runtime/Runtime.hpp"

#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#endif

namespace {

class HistogramTest : public ::testing::Test {
protected:
#ifdef BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR
  std::ostringstream sink;
  void SetUp() override { qir::Runtime::getInstance().setOstream(sink); }
  void TearDown() override { qir::Runtime::getInstance().resetOstream(); }
#endif

  using Histogram = std::pair<std::vector<std::string>, std::vector<size_t>>;
  static constexpr size_t SHOTS = 1024;

  static Histogram runProgram(const QDMI_Program_Format format,
                              const std::string_view program) {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    EXPECT_EQ(qdmi_test::setProgram(j.job, format, program), QDMI_SUCCESS);
    EXPECT_EQ(qdmi_test::setShots(j.job, SHOTS), QDMI_SUCCESS);
    EXPECT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
    return qdmi_test::getHistogram(j.job);
  }

  static void checkHistogram(const Histogram& hist) {
    const auto& [keys, vals] = hist;
    // Keys and values come from two independent device queries.
    // Check both vectors have the same size.
    ASSERT_EQ(keys.size(), vals.size());
    // Values should sum up to the number of SHOTS.
    const auto sum = std::accumulate(vals.cbegin(), vals.cend(), size_t{0});
    EXPECT_EQ(sum, SHOTS);
    // Both keys '00' and '11' should be expected.
    ASSERT_EQ(keys.size(), 2U);
    // And no other keys should be expected.
    EXPECT_TRUE(std::ranges::all_of(
        keys, [](const auto& k) { return k == "00" || k == "11"; }));
  }
};

#ifdef BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR
class QIRHistogramTestModule : public HistogramTest {
protected:
  static std::string getProgram(const std::string_view file) {
    const std::filesystem::path path =
        std::filesystem::path(QIR_FILES_DIR) / file;
    std::ifstream ifs(path);
    EXPECT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
    const std::string text(std::istreambuf_iterator<char>{ifs}, {});
    llvm::LLVMContext context;
    llvm::SMDiagnostic err;
    auto llvmModule = llvm::parseAssemblyString(text, err, context);
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

class QIRHistogramTestString : public HistogramTest {
protected:
  static std::string getProgram(const std::string_view file) {
    const std::filesystem::path path =
        std::filesystem::path(QIR_FILES_DIR) / file;
    std::ifstream ifs(path);
    EXPECT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
    return {std::istreambuf_iterator<char>{ifs}, {}};
  }
};
#endif

} // namespace

TEST_F(HistogramTest, QASM3Program) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QASM3;
  constexpr std::string_view program = qdmi_test::QASM3_BELL_SAMPLING;
  checkHistogram(runProgram(format, program));
}

#ifdef BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR
TEST_F(QIRHistogramTestModule, BaseStatic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
  checkHistogram(runProgram(format, getProgram("BellPairStatic.ll")));
}

TEST_F(QIRHistogramTestString, BaseStatic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  checkHistogram(runProgram(format, getProgram("BellPairStatic.ll")));
}

TEST_F(QIRHistogramTestModule, BaseDynamic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
  checkHistogram(runProgram(format, getProgram("BellPairDynamic.ll")));
}

TEST_F(QIRHistogramTestString, BaseDynamic) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  checkHistogram(runProgram(format, getProgram("BellPairDynamic.ll")));
}

TEST_F(QIRHistogramTestModule, Adaptive) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE;
  checkHistogram(runProgram(format, getProgram("BellPairAdaptive.ll")));
}

TEST_F(QIRHistogramTestString, Adaptive) {
  constexpr auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
  checkHistogram(runProgram(format, getProgram("BellPairAdaptive.ll")));
}
#endif

TEST(ResultsSampling, BufferTooSmallErrors) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 512), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  if (const size_t ks = qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_KEYS);
      ks > 0) {
    std::vector<char> tooSmall(ks - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_HIST_KEYS, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }

  if (const size_t vs =
          qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_VALUES);
      vs > 0) {
    std::vector<char> tooSmall(vs - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_HIST_VALUES, tooSmall.size(),
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

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          j.job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}
