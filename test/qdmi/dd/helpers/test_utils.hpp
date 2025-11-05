/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * Test utilities for DDSIM QDMI device tests
 */
#pragma once

#include "mqt_ddsim_qdmi/device.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qdmi_test {

struct SessionGuard {
  MQT_DDSIM_QDMI_Device_Session session{nullptr};
  SessionGuard();
  ~SessionGuard();
  SessionGuard(const SessionGuard&) = delete;
  SessionGuard& operator=(const SessionGuard&) = delete;
};

struct JobGuard {
  MQT_DDSIM_QDMI_Device_Job job{nullptr};
  explicit JobGuard(MQT_DDSIM_QDMI_Device_Session s);
  ~JobGuard();
  JobGuard(const JobGuard&) = delete;
  JobGuard& operator=(const JobGuard&) = delete;
};

// Convenience queries
std::vector<MQT_DDSIM_QDMI_Site>
querySites(MQT_DDSIM_QDMI_Device_Session session);
std::vector<MQT_DDSIM_QDMI_Operation>
queryOperations(MQT_DDSIM_QDMI_Device_Session session);

// Convenience setters
int setProgram(MQT_DDSIM_QDMI_Device_Job job, QDMI_Program_Format fmt,
               std::string_view program);
int setShots(MQT_DDSIM_QDMI_Device_Job job, size_t shots);
int submitAndWait(MQT_DDSIM_QDMI_Device_Job job, size_t timeoutSeconds);

// Result helpers
size_t querySize(MQT_DDSIM_QDMI_Device_Job job, QDMI_Job_Result result);
std::pair<std::vector<std::string>, std::vector<size_t>>
getHistogram(MQT_DDSIM_QDMI_Device_Job job);
std::vector<std::complex<double>> getDenseState(MQT_DDSIM_QDMI_Device_Job job);
std::pair<std::vector<std::string>, std::vector<std::complex<double>>>
getSparseState(MQT_DDSIM_QDMI_Device_Job job);
std::vector<double> getDenseProbabilities(MQT_DDSIM_QDMI_Device_Job job);
std::pair<std::vector<std::string>, std::vector<double>>
getSparseProbabilities(MQT_DDSIM_QDMI_Device_Job job);

// Small helpers
std::vector<std::string> splitCSV(const std::string& csv);

} // namespace qdmi_test
