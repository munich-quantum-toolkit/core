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
 * DDSIM QDMI Device - Results: statevector (dense/sparse)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

namespace {

void expectBellState(const QDMI_Program_Format format,
                     const std::string_view program) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, format, program), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const auto vec = qdmi_test::getDenseState(j.job);
  ASSERT_EQ(vec.size(), 4U);
  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  EXPECT_NEAR(std::abs(vec[0]), invSqrt2, 1e-6);
  EXPECT_NEAR(std::abs(vec[1]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(vec[2]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(vec[3]), invSqrt2, 1e-6);
}

} // namespace

TEST(ResultsStatevector, QASM2YieldsBellState) {
  expectBellState(QDMI_PROGRAM_FORMAT_QASM2, qdmi_test::QASM2_BELL_STATE);
}

TEST(ResultsStatevector, QASM2IgnoresFinalMeasurements) {
  expectBellState(QDMI_PROGRAM_FORMAT_QASM2, qdmi_test::QASM2_BELL_SAMPLING);
}

TEST(ResultsStatevector, QASM3YieldsBellState) {
  expectBellState(QDMI_PROGRAM_FORMAT_QASM3, qdmi_test::QASM3_BELL_STATE);
}

TEST(ResultsStatevector, QASM3IgnoresFinalMeasurements) {
  expectBellState(QDMI_PROGRAM_FORMAT_QASM3, qdmi_test::QASM3_BELL_SAMPLING);
}

TEST(ResultsStatevector, EmptyQASM3YieldsEmptyResults) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3, "OPENQASM 3.0;"),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  constexpr std::array results{
      QDMI_JOB_RESULT_STATEVECTOR_DENSE,
      QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
      QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES,
      QDMI_JOB_RESULT_PROBABILITIES_DENSE,
      QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS,
      QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES,
  };
  char dummy{};
  for (const auto result : results) {
    size_t size = 1;
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(j.job, result, 0, nullptr, &size),
        QDMI_SUCCESS);
    EXPECT_EQ(size, 0U);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, result, 0, &dummy,
                                                    nullptr),
              QDMI_SUCCESS);
  }
}

TEST(ResultsStatevector, DenseNormalizedAndBufferTooSmall) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  auto const vec = qdmi_test::getDenseState(j.job);
  ASSERT_FALSE(vec.empty());
  auto norm = 0.0;
  for (const auto& v : vec) {
    norm += std::norm(v);
  }
  EXPECT_NEAR(norm, 1.0, 1e-6);

  const size_t sz =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE);
  if (sz > 0) {
    std::vector<char> tooSmall(sz - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(ResultsStatevector, SparseNormalizedAndBufferTooSmall) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  auto [keys, vals] = qdmi_test::getSparseState(j.job);
  ASSERT_EQ(keys.size(), vals.size());
  auto norm = 0.0;
  for (const auto& v : vals) {
    norm += std::norm(v);
  }
  EXPECT_NEAR(norm, 1.0, 1e-6);

  const size_t ksz =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS);
  if (ksz > 0) {
    std::vector<char> tooSmall(ksz - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
                  tooSmall.size(), tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  const size_t vsz =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES);
  if (vsz > 0) {
    std::vector<char> tooSmall(vsz - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES,
                  tooSmall.size(), tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(ResultsStatevector, SamplingRequestsInvalidWithShotsZero) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, QDMI_JOB_RESULT_SHOTS,
                                                  0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_HIST_VALUES, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(ResultsStatevector, QIRBaseStringYieldsBellState) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  const auto program = qdmi_test::getQIRProgram("BellPairStatic.ll");
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QIRBASESTRING, program),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const auto vec = qdmi_test::getDenseState(j.job);
  ASSERT_EQ(vec.size(), 4U);

  // Bell pair: amplitudes at |00> and |11> are 1/sqrt(2), |01> and |10> are 0.
  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  EXPECT_NEAR(std::abs(vec[0]), invSqrt2, 1e-6);
  EXPECT_NEAR(std::abs(vec[1]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(vec[2]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(vec[3]), invSqrt2, 1e-6);
}

TEST(ResultsStatevector, QIRPreservesPhaseWireOrderAndDeclaredWidth) {
  constexpr std::string_view program = R"(
define i64 @main() #0 {
  call void @__quantum__qis__gphase__body(double 0.3)
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__swap__body(ptr null, ptr inttoptr (i64 1 to ptr))
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  ret i64 0
}
declare void @__quantum__qis__gphase__body(double)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__swap__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr) #1
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="3" "required_num_results"="1" }
attributes #1 = { "irreversible" }
)";
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QIRBASESTRING, program),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  const auto values = qdmi_test::getDenseState(j.job);
  ASSERT_EQ(values.size(), 8);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(std::abs(values[i] - (i == 2 ? std::polar(1., 0.3)
                                             : std::complex<double>{})),
                0., 1e-12);
  }
}

TEST(ResultsStatevector, DenseSizesDoNotMaterializeUnaddressableVectors) {
  constexpr size_t bits = std::numeric_limits<size_t>::digits;
  const qdmi_test::SessionGuard session{};
  const std::array cases{
      std::array<size_t, 3>{
          bits - 5,
          size_t{1} << (bits - 1),
          size_t{1} << (bits - 2),
      },
      std::array<size_t, 3>{bits - 4, 0, size_t{1} << (bits - 1)},
      std::array<size_t, 3>{bits - 3, 0, 0},
      std::array<size_t, 3>{bits, 0, 0},
  };
  for (const auto& [qubits, stateSize, probabilitySize] : cases) {
    SCOPED_TRACE(qubits);
    const qdmi_test::JobGuard job{session.session};
    const auto program =
        "OPENQASM 3.0; qubit[" + std::to_string(qubits) + "] q; x q[0];";
    ASSERT_EQ(
        qdmi_test::setProgram(job.job, QDMI_PROGRAM_FORMAT_QASM3, program),
        QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job.job, 0), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
    for (const auto result : {
             QDMI_JOB_RESULT_STATEVECTOR_DENSE,
             QDMI_JOB_RESULT_PROBABILITIES_DENSE,
         }) {
      const auto expectedSize = result == QDMI_JOB_RESULT_STATEVECTOR_DENSE
                                    ? stateSize
                                    : probabilitySize;
      size_t size = 123;
      const auto status = MQT_DDSIM_QDMI_device_job_get_results(
          job.job, result, 0, nullptr, &size);
      if (expectedSize == 0) {
        EXPECT_EQ(status, QDMI_ERROR_OUTOFMEM);
        EXPECT_EQ(size, 123);
        continue;
      }
      ASSERT_EQ(status, QDMI_SUCCESS);
      EXPECT_EQ(size, expectedSize);
      double output = 42;
      EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    job.job, result, sizeof(output), &output, nullptr),
                QDMI_ERROR_INVALIDARGUMENT);
      EXPECT_EQ(output, 42);
    }
  }
}
