/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "helpers/test_utils.hpp"

#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include "gtest/gtest.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cassert>
#include <complex>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qdmi_test {

SessionGuard::SessionGuard(const char* qubits) {
  auto rc = MQT_DDSIM_QDMI_device_initialize();
  EXPECT_EQ(rc, QDMI_SUCCESS);
  rc = MQT_DDSIM_QDMI_device_session_alloc(&session);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  if (qubits != nullptr) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_session_set_parameter(
                  session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1,
                  std::strlen(qubits) + 1, qubits),
              QDMI_SUCCESS);
  }
  rc = MQT_DDSIM_QDMI_device_session_init(session);
  EXPECT_EQ(rc, QDMI_SUCCESS);
}

SessionGuard::~SessionGuard() {
  if (session != nullptr) {
    MQT_DDSIM_QDMI_device_session_free(session);
    session = nullptr;
  }
  MQT_DDSIM_QDMI_device_finalize();
}

JobGuard::JobGuard(MQT_DDSIM_QDMI_Device_Session s) {
  if (s != nullptr) {
    const auto rc = MQT_DDSIM_QDMI_device_session_create_device_job(s, &job);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
}

JobGuard::~JobGuard() {
  if (job != nullptr) {
    MQT_DDSIM_QDMI_device_job_free(job);
    job = nullptr;
  }
}

std::vector<MQT_DDSIM_QDMI_Site>
querySites(MQT_DDSIM_QDMI_Device_Session session) {
  size_t size = 0;
  auto rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  std::vector<MQT_DDSIM_QDMI_Site> sites(size / sizeof(MQT_DDSIM_QDMI_Site));
  rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_SITES, size,
      static_cast<void*>(sites.data()), nullptr);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  return sites;
}

std::vector<MQT_DDSIM_QDMI_Operation>
queryOperations(MQT_DDSIM_QDMI_Device_Session session) {
  size_t size = 0;
  auto rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_OPERATIONS, 0, nullptr, &size);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  std::vector<MQT_DDSIM_QDMI_Operation> ops(size /
                                            sizeof(MQT_DDSIM_QDMI_Operation));
  rc = MQT_DDSIM_QDMI_device_session_query_device_property(
      session, QDMI_DEVICE_PROPERTY_OPERATIONS, size,
      static_cast<void*>(ops.data()), nullptr);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  return ops;
}

int setProgram(MQT_DDSIM_QDMI_Device_Job job, const QDMI_Program_Format fmt,
               const std::string_view program) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // Text payloads include the trailing '\0' per the QDMI wire convention.
  // Binary payloads ship the exact byte count.
  // The `+1` is safe here because every existing call to `setProgram` with a
  // text format passes a `program` with a string literal or `std::string`, both
  // of which guarantee `'\0'` at `data()[size()]`.
  const bool isTextProgramFormat = fmt == QDMI_PROGRAM_FORMAT_QASM2 ||
                                   fmt == QDMI_PROGRAM_FORMAT_QASM3 ||
                                   fmt == QDMI_PROGRAM_FORMAT_QIRBASESTRING ||
                                   fmt == QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
  const auto bytesToSend =
      isTextProgramFormat ? program.size() + 1 : program.size();
  const void* const programData = program.data();
  return MQT_DDSIM_QDMI_device_job_set_programs(job, fmt, 1U, &bytesToSend,
                                                &programData);
}

int setShots(MQT_DDSIM_QDMI_Device_Job job, const size_t shots) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return MQT_DDSIM_QDMI_device_job_set_parameter(
      job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots);
}

int setSeed(MQT_DDSIM_QDMI_Device_Job job, const int seed) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return MQT_DDSIM_QDMI_device_job_set_parameter(
      job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM1, sizeof(int), &seed);
}

int submitAndWait(MQT_DDSIM_QDMI_Device_Job job, size_t timeoutSeconds) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (const int rc = MQT_DDSIM_QDMI_device_job_submit(job);
      rc != QDMI_SUCCESS) {
    return rc;
  }
  return MQT_DDSIM_QDMI_device_job_wait(job, timeoutSeconds);
}

size_t querySize(MQT_DDSIM_QDMI_Device_Job job, QDMI_Job_Result result) {
  size_t sz = 0;
  const auto rc =
      MQT_DDSIM_QDMI_device_job_get_results(job, 0U, result, 0, nullptr, &sz);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  return sz;
}

std::string getQIRProgram(const std::string_view file) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / file;
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    ADD_FAILURE() << "Failed to open " << path << ": "
                  << buffer.getError().message();
    return {};
  }
  return (*buffer)->getBuffer().str();
}

std::vector<std::string> splitCSV(const std::string& csv) {
  std::vector<std::string> out;
  for (const auto tok : llvm::split(csv, ',')) {
    if (!tok.empty()) {
      out.emplace_back(tok.str());
    }
  }
  return out;
}

namespace {
template <typename T>
std::vector<T> getResult(MQT_DDSIM_QDMI_Device_Job job,
                         const QDMI_Job_Result result) {
  std::vector<T> values(querySize(job, result) / sizeof(T));
  if (!values.empty()) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, 0U, result,
                                                    values.size() * sizeof(T),
                                                    values.data(), nullptr),
              QDMI_SUCCESS);
  }
  return values;
}

std::vector<std::string> getKeys(MQT_DDSIM_QDMI_Device_Job job,
                                 const QDMI_Job_Result result) {
  const auto keys = getResult<char>(job, result);
  return splitCSV(keys.empty() ? std::string{}
                               : std::string(keys.data(), keys.size() - 1));
}
} // namespace

std::pair<std::vector<std::string>, std::vector<size_t>>
getHistogram(MQT_DDSIM_QDMI_Device_Job job) {
  return {getKeys(job, QDMI_JOB_RESULT_HIST_KEYS),
          getResult<size_t>(job, QDMI_JOB_RESULT_HIST_VALUES)};
}

std::vector<std::complex<double>> getDenseState(MQT_DDSIM_QDMI_Device_Job job) {
  return getResult<std::complex<double>>(job,
                                         QDMI_JOB_RESULT_STATEVECTOR_DENSE);
}

std::pair<std::vector<std::string>, std::vector<std::complex<double>>>
getSparseState(MQT_DDSIM_QDMI_Device_Job job) {
  return {getKeys(job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS),
          getResult<std::complex<double>>(
              job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES)};
}

std::vector<double> getDenseProbabilities(MQT_DDSIM_QDMI_Device_Job job) {
  return getResult<double>(job, QDMI_JOB_RESULT_PROBABILITIES_DENSE);
}

std::pair<std::vector<std::string>, std::vector<double>>
getSparseProbabilities(MQT_DDSIM_QDMI_Device_Job job) {
  return {getKeys(job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS),
          getResult<double>(job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES)};
}

} // namespace qdmi_test
