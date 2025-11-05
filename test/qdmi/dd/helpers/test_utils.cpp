/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "helpers/test_utils.hpp"

#include <cassert>
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>

namespace qdmi_test {

SessionGuard::SessionGuard() {
  auto rc = MQT_DDSIM_QDMI_device_initialize();
  EXPECT_EQ(rc, QDMI_SUCCESS);
  rc = MQT_DDSIM_QDMI_device_session_alloc(&session);
  EXPECT_EQ(rc, QDMI_SUCCESS);
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
  int rc = MQT_DDSIM_QDMI_device_job_set_parameter(
      job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT, sizeof(QDMI_Program_Format),
      &fmt);
  if (rc != QDMI_SUCCESS && rc != QDMI_ERROR_NOTSUPPORTED) {
    return rc;
  }
  rc = MQT_DDSIM_QDMI_device_job_set_parameter(
      job, QDMI_DEVICE_JOB_PARAMETER_PROGRAM, program.size() + 1,
      program.data());
  return rc;
}

int setShots(MQT_DDSIM_QDMI_Device_Job job, const size_t shots) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return MQT_DDSIM_QDMI_device_job_set_parameter(
      job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots);
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
      MQT_DDSIM_QDMI_device_job_get_results(job, result, 0, nullptr, &sz);
  EXPECT_EQ(rc, QDMI_SUCCESS);
  return sz;
}

std::vector<std::string> splitCSV(const std::string& csv) {
  std::vector<std::string> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.emplace_back(tok);
    }
  }
  return out;
}

std::pair<std::vector<std::string>, std::vector<size_t>>
getHistogram(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t ks = querySize(job, QDMI_JOB_RESULT_HIST_KEYS);
  std::string keys(ks > 0 ? ks - 1 : 0, '\0');
  if (ks > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_HIST_KEYS, ks, keys.data(), nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  auto keyVec = splitCSV(keys);
  const size_t vs = querySize(job, QDMI_JOB_RESULT_HIST_VALUES);
  std::vector<size_t> vals(vs / sizeof(size_t));
  if (vs > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_HIST_VALUES, vs, vals.data(), nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  return {std::move(keyVec), std::move(vals)};
}

std::vector<std::complex<double>> getDenseState(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t sz = querySize(job, QDMI_JOB_RESULT_STATEVECTOR_DENSE);
  std::vector<double> buf(sz / sizeof(double));
  if (sz > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, sz, buf.data(), nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  std::vector<std::complex<double>> out;
  out.reserve(buf.size() / 2);
  for (size_t i = 0; i + 1 < buf.size(); i += 2) {
    out.emplace_back(buf[i], buf[i + 1]);
  }
  return out;
}

std::pair<std::vector<std::string>, std::vector<std::complex<double>>>
getSparseState(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t ks = querySize(job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS);
  std::string keys(ks > 0 ? ks - 1 : 0, '\0');
  if (ks > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, ks, keys.data(), nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  auto keyVec = splitCSV(keys);
  const size_t vs = querySize(job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES);
  std::vector<double> vals(vs / sizeof(double));
  if (vs > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, vs, vals.data(),
        nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  std::vector<std::complex<double>> out;
  out.reserve(vals.size() / 2);
  for (size_t i = 0; i + 1 < vals.size(); i += 2) {
    out.emplace_back(vals[i], vals[i + 1]);
  }
  return {std::move(keyVec), std::move(out)};
}

std::vector<double> getDenseProbabilities(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t sz = querySize(job, QDMI_JOB_RESULT_PROBABILITIES_DENSE);
  std::vector<double> out(sz / sizeof(double));
  if (sz > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, sz, out.data(), nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  return out;
}

std::pair<std::vector<std::string>, std::vector<double>>
getSparseProbabilities(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t ks = querySize(job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS);
  std::string keys(ks > 0 ? ks - 1 : 0, '\0');
  if (ks > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, ks, keys.data(),
        nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  auto keyVec = splitCSV(keys);
  const size_t vs = querySize(job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES);
  std::vector<double> vals(vs / sizeof(double));
  if (vs > 0) {
    const auto rc = MQT_DDSIM_QDMI_device_job_get_results(
        job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, vs, vals.data(),
        nullptr);
    EXPECT_EQ(rc, QDMI_SUCCESS);
  }
  return {std::move(keyVec), std::move(vals)};
}

} // namespace qdmi_test
