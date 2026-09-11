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
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"

#include "gtest/gtest.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <map>
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

TEST(ResultsStatevector, SparseResultsPreserveBasisOrderAndValuePairing) {
  constexpr std::string_view program = R"(OPENQASM 3;
include "stdgates.inc";
qubit[3] q;
ry(0.6) q[0];
x q[1];
h q[2];
s q[2];
)";
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  ASSERT_EQ(qdmi_test::setProgram(job.job, QDMI_PROGRAM_FORMAT_QASM3, program),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);

  const auto probabilities = qdmi_test::getSparseProbabilities(job.job);
  const auto state = qdmi_test::getSparseState(job.job);
  const std::vector<std::string> keys{"010", "011", "110", "111"};
  EXPECT_EQ(state.first, keys);
  EXPECT_EQ(probabilities.first, keys);
  ASSERT_EQ(state.second.size(), keys.size());
  ASSERT_EQ(probabilities.second.size(), keys.size());
  const auto cosine = std::cos(0.3) / std::numbers::sqrt2;
  const auto sine = std::sin(0.3) / std::numbers::sqrt2;
  const std::array<std::complex<double>, 4> expected{
      std::complex{cosine, 0.},
      std::complex{sine, 0.},
      std::complex{0., cosine},
      std::complex{0., sine},
  };
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_NEAR(std::abs(state.second[i] - expected[i]), 0., 1e-12);
    EXPECT_NEAR(probabilities.second[i], std::norm(expected[i]), 1e-12);
  }
  EXPECT_EQ(qdmi_test::getSparseState(job.job), state);
  EXPECT_EQ(qdmi_test::getSparseProbabilities(job.job), probabilities);
}

TEST(ResultsStatevector, SparseResultsRespectBasisIndexWidth) {
  constexpr size_t maxWidth = std::numeric_limits<size_t>::digits;
  for (const auto width : {maxWidth, maxWidth + 1}) {
    SCOPED_TRACE(width);
    const auto program =
        "define i64 @main() #0 {\n"
        "call void @__quantum__qis__x__body(ptr inttoptr (i64 " +
        std::to_string(width - 1) +
        " to ptr))\nret i64 0\n}\n"
        "declare void @__quantum__qis__x__body(ptr)\n"
        "attributes #0 = { \"entry_point\" \"qir_profiles\"=\"base_profile\" "
        "\"required_num_qubits\"=\"" +
        std::to_string(width) + "\" }\n";
    const qdmi_test::SessionGuard session{};
    const qdmi_test::JobGuard job{session.session};
    ASSERT_EQ(qdmi_test::setProgram(job.job, QDMI_PROGRAM_FORMAT_QIRBASESTRING,
                                    program),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job.job, 0), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
    QDMI_Job_Status status{};
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job.job, &status), QDMI_SUCCESS);
    ASSERT_EQ(status, QDMI_JOB_STATUS_DONE);
    if (width == maxWidth) {
      const auto [keys, values] = qdmi_test::getSparseState(job.job);
      EXPECT_EQ(keys,
                std::vector<std::string>{"1" + std::string(width - 1, '0')});
      EXPECT_EQ(values, std::vector<std::complex<double>>{1.});
    } else {
      for (const auto result : {
               QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
               QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES,
               QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS,
               QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES,
           }) {
        EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job.job, result, 0,
                                                        nullptr, nullptr),
                  QDMI_ERROR_NOTSUPPORTED);
      }
    }
  }
}

TEST(ResultsStatevector, SamplingRetainsStateWithoutChangingSamples) {
  const auto base = qdmi_test::getQIRProgram("BellPairStatic.ll");
  auto adaptive = base;
  adaptive.replace(adaptive.find("base_profile"),
                   std::string_view("base_profile").size(), "adaptive_profile");
  for (const auto format : {
           QDMI_PROGRAM_FORMAT_QASM2,
           QDMI_PROGRAM_FORMAT_QASM3,
           QDMI_PROGRAM_FORMAT_QIRBASESTRING,
           QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
       }) {
    SCOPED_TRACE(format);
    std::string program;
    if (format == QDMI_PROGRAM_FORMAT_QASM2) {
      program = qdmi_test::QASM2_BELL_SAMPLING;
    } else if (format == QDMI_PROGRAM_FORMAT_QASM3) {
      program = qdmi_test::QASM3_BELL_SAMPLING;
    } else {
      program = format == QDMI_PROGRAM_FORMAT_QIRBASESTRING ||
                        format == QDMI_PROGRAM_FORMAT_QIRBASEMODULE
                    ? base
                    : adaptive;
      if (format == QDMI_PROGRAM_FORMAT_QIRBASEMODULE ||
          format == QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE) {
        llvm::LLVMContext context;
        llvm::SMDiagnostic error;
        const auto llvmModule =
            llvm::parseAssemblyString(program, error, context);
        ASSERT_NE(llvmModule, nullptr);
        program.clear();
        llvm::raw_string_ostream stream(program);
        llvm::WriteBitcodeToFile(*llvmModule, stream);
      }
    }
    const qdmi_test::SessionGuard session{};
    const qdmi_test::JobGuard job{session.session};
    ASSERT_EQ(qdmi_test::setProgram(job.job, format, program), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job.job, 64), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setSeed(job.job, 7), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
    const auto counts = qdmi_test::getHistogram(job.job);
    const auto state = qdmi_test::getDenseState(job.job);
    ASSERT_EQ(state.size(), 4);
    EXPECT_NEAR(std::abs(state[0] - std::numbers::sqrt2 / 2), 0., 1e-12);
    EXPECT_NEAR(std::abs(state[3] - std::numbers::sqrt2 / 2), 0., 1e-12);
    EXPECT_NEAR(std::abs(state[1]) + std::abs(state[2]), 0., 1e-12);
    const auto sparse = qdmi_test::getSparseState(job.job);
    ASSERT_EQ(sparse.first.size(), 2);
    ASSERT_EQ(sparse.second.size(), 2);
    std::map<std::string, std::complex<double>> sparseMap;
    for (size_t i = 0; i < sparse.first.size(); ++i) {
      sparseMap.emplace(sparse.first[i], sparse.second[i]);
    }
    EXPECT_EQ(sparseMap, (std::map<std::string, std::complex<double>>{
                             {"00", state[0]}, {"11", state[3]}}));
    const auto probabilities = qdmi_test::getDenseProbabilities(job.job);
    ASSERT_EQ(probabilities.size(), 4);
    for (size_t i = 0; i < state.size(); ++i) {
      EXPECT_NEAR(probabilities[i], std::norm(state[i]), 1e-12);
    }
    const auto sparseProbabilities = qdmi_test::getSparseProbabilities(job.job);
    EXPECT_EQ(sparseProbabilities.first, sparse.first);
    EXPECT_EQ(sparseProbabilities.second,
              (std::vector<double>{probabilities[0], probabilities[3]}));
    EXPECT_EQ(qdmi_test::getDenseState(job.job), state);
    EXPECT_EQ(qdmi_test::getHistogram(job.job), counts);
    const qdmi_test::JobGuard repeated{session.session};
    ASSERT_EQ(qdmi_test::setProgram(repeated.job, format, program),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(repeated.job, 64), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setSeed(repeated.job, 7), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(repeated.job, 0), QDMI_SUCCESS);
    EXPECT_EQ(qdmi_test::getHistogram(repeated.job), counts);
  }
}

TEST(ResultsStatevector, QASM2YieldsBellState) {
  expectBellState(QDMI_PROGRAM_FORMAT_QASM2, qdmi_test::QASM2_BELL_STATE);
}

TEST(ResultsStatevector, QASMSamplingPreservesPhaseAndWireOrder) {
  constexpr std::string_view program = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
x q[0];
gphase(0.3);
swap q[0], q[1];
c = measure q;
)";
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  ASSERT_EQ(qdmi_test::setProgram(job.job, QDMI_PROGRAM_FORMAT_QASM3, program),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job.job, 16), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
  const auto state = qdmi_test::getDenseState(job.job);
  ASSERT_EQ(state.size(), 4);
  EXPECT_NEAR(std::abs(state[2] - std::polar(1., 0.3)), 0., 1e-12);
  const auto histogram = qdmi_test::getHistogram(job.job);
  EXPECT_EQ(histogram.first, (std::vector<std::string>{"10"}));
  EXPECT_EQ(histogram.second, (std::vector<size_t>{16}));
}

TEST(ResultsStatevector, QIRSamplingDoesNotExposeCollapsedTrajectories) {
  auto program = qdmi_test::getQIRProgram("BellPairStatic.ll");
  program.replace(program.find("base_profile"),
                  std::string_view("base_profile").size(), "adaptive_profile");
  const auto position = program.find("ret i64 0");
  ASSERT_NE(position, std::string::npos);
  program.insert(position, "call void @__quantum__qis__x__body(ptr null)\n");
  program += "\ndeclare void @__quantum__qis__x__body(ptr)\n";
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  ASSERT_EQ(qdmi_test::setProgram(
                job.job, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, program),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job.job, 16), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
  size_t size = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, &size),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_FALSE(qdmi_test::getHistogram(job.job).first.empty());
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

TEST(ResultsStatevector, EmptyProgramsRetainTheZeroQubitState) {
  for (const auto shots : {0U, 32U}) {
    for (const auto format :
         {QDMI_PROGRAM_FORMAT_QASM3, QDMI_PROGRAM_FORMAT_QIRBASESTRING}) {
      const qdmi_test::SessionGuard session{};
      const qdmi_test::JobGuard job{session.session};
      const auto* const program = format == QDMI_PROGRAM_FORMAT_QASM3
                                      ? "OPENQASM 3.0;"
                                      : R"(define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="0" "required_num_results"="0" })";
      ASSERT_EQ(qdmi_test::setProgram(job.job, format, program), QDMI_SUCCESS);
      ASSERT_EQ(qdmi_test::setShots(job.job, shots), QDMI_SUCCESS);
      ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
      EXPECT_EQ(qdmi_test::getDenseState(job.job),
                (std::vector<std::complex<double>>{{1., 0.}}));
      EXPECT_EQ(qdmi_test::getDenseProbabilities(job.job),
                (std::vector<double>{1.}));
      size_t size = 0;
      ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    job.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0,
                    nullptr, &size),
                QDMI_SUCCESS);
      EXPECT_EQ(size, 1);
      char key = 'x';
      ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    job.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 1, &key,
                    nullptr),
                QDMI_SUCCESS);
      EXPECT_EQ(key, '\0');
    }
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

  // Bell pair: amplitudes at |00⟩ and |11⟩ are 1/sqrt(2), |01⟩ and |10⟩ are 0.
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

TEST(ResultsStatevector, QIRAdaptiveStringAndBitcodePreserveDynamicState) {
  const auto text = qdmi_test::getQIRProgram("StatevectorAdaptive.ll");
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  const auto llvmModule = llvm::parseAssemblyString(text, error, context);
  ASSERT_NE(llvmModule, nullptr);
  std::string bitcode;
  llvm::raw_string_ostream stream(bitcode);
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  stream.flush();
  for (const auto format : {
           QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
       }) {
    SCOPED_TRACE(format);
    const qdmi_test::SessionGuard session{};
    const qdmi_test::JobGuard job{session.session};
    const auto& program =
        format == QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING ? text : bitcode;
    ASSERT_EQ(qdmi_test::setProgram(job.job, format, program), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job.job, 0), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
    QDMI_Job_Status status{};
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job.job, &status), QDMI_SUCCESS);
    ASSERT_EQ(status, QDMI_JOB_STATUS_DONE);
    const auto values = qdmi_test::getDenseState(job.job);
    ASSERT_EQ(values.size(), 16);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto expected = i == 4 || i == 7
                                ? std::polar(1. / std::numbers::sqrt2, 0.3)
                                : std::complex<double>{};
      EXPECT_NEAR(std::abs(values[i] - expected), 0., 1e-12);
    }
  }
}

TEST(ResultsStatevector, QIRAdaptiveRejectsNonTerminalMeasurementsAndFeedback) {
  for (const auto* body : {
           "call void @__quantum__qis__x__body(ptr null)\nret i64 0",
           "call void @__quantum__qis__reset__body(ptr null)\nret i64 0",
           "%r = call i1 @__quantum__rt__read_result(ptr null)\n"
           "br i1 %r, label %left, label %right\nleft: ret i64 0\nright: ret "
           "i64 0",
       }) {
    SCOPED_TRACE(body);
    const auto program = std::string(R"(
define i64 @main() #0 {
  call void @__quantum__qis__h__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr null)
)") + body + R"(
}
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__qis__reset__body(ptr)
declare i1 @__quantum__rt__read_result(ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
    const qdmi_test::SessionGuard session{};
    const qdmi_test::JobGuard job{session.session};
    ASSERT_EQ(qdmi_test::setProgram(
                  job.job, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, program),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job.job, 0), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
    QDMI_Job_Status status{};
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job.job, &status), QDMI_SUCCESS);
    EXPECT_EQ(status, QDMI_JOB_STATUS_FAILED);
    size_t size = 0;
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(
            job.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, &size),
        QDMI_ERROR_BADSTATE);
  }
}
