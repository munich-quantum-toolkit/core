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

#include <cstddef>
#include <vector>

#ifdef BUILD_MQT_CORE_QDMI_WITH_QIR
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
#include <string>
#include <string_view>
#endif

namespace {

class HistogramKeysAndValuesSumToShots : public ::testing::Test {
protected:
#ifdef BUILD_MQT_CORE_QDMI_WITH_QIR
  std::ostringstream sink;
  void SetUp() override { qir::Runtime::getInstance().setOstream(sink); }
  void TearDown() override { qir::Runtime::getInstance().resetOstream(); }
#endif

  static void Run(const QDMI_Program_Format format,
                  const std::string_view program) {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    ASSERT_EQ(qdmi_test::setProgram(j.job, format, program), QDMI_SUCCESS);
    constexpr size_t shots = 1024;
    ASSERT_EQ(qdmi_test::setShots(j.job, shots), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

    auto [keys, vals] = qdmi_test::getHistogram(j.job);
    ASSERT_EQ(keys.size(), vals.size());
    size_t sum = 0U;
    for (const auto& v : vals) {
      sum += v;
    }
    EXPECT_EQ(sum, shots);
  }
};

} // namespace

TEST_F(HistogramKeysAndValuesSumToShots, QASM3Program) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QASM3;
  constexpr std::string_view program = qdmi_test::QASM3_BELL_SAMPLING;
  Run(format, program);
}

#ifdef BUILD_MQT_CORE_QDMI_WITH_QIR
TEST_F(HistogramKeysAndValuesSumToShots, QIRBaseModule) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  std::ifstream ifs(path);
  ASSERT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
  const std::string program(std::istreambuf_iterator<char>{ifs}, {});
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;
  auto module = llvm::parseAssemblyString(program, err, context);
  ASSERT_NE(module, nullptr)
      << "parseAssemblyString failed: " << err.getMessage().str();
  std::string bitcodeBuffer;
  llvm::raw_string_ostream os(bitcodeBuffer);
  llvm::WriteBitcodeToFile(*module, os);
  os.flush();
  Run(format, bitcodeBuffer);
}

TEST_F(HistogramKeysAndValuesSumToShots, QIRBaseString) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  std::ifstream ifs(path);
  ASSERT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
  const std::string program(std::istreambuf_iterator<char>{ifs}, {});
  Run(format, program);
}

TEST_F(HistogramKeysAndValuesSumToShots, QIRBaseModuleDynamic) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / "BellPairDynamic.ll";
  std::ifstream ifs(path);
  ASSERT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
  const std::string program(std::istreambuf_iterator<char>{ifs}, {});
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;
  auto module = llvm::parseAssemblyString(program, err, context);
  ASSERT_NE(module, nullptr)
      << "parseAssemblyString failed: " << err.getMessage().str();
  std::string bitcodeBuffer;
  llvm::raw_string_ostream os(bitcodeBuffer);
  llvm::WriteBitcodeToFile(*module, os);
  os.flush();
  Run(format, bitcodeBuffer);
}

TEST_F(HistogramKeysAndValuesSumToShots, QIRBaseStringDynamic) {
  constexpr QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / "BellPairDynamic.ll";
  std::ifstream ifs(path);
  ASSERT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
  const std::string program(std::istreambuf_iterator<char>{ifs}, {});
  Run(format, program);
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
