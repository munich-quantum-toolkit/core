/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Device.hpp
 * @brief The MQT QDMI device implementation for superconducting devices.
 */

#pragma once

#include "mqt_sc_qdmi/device.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct MQT_SC_QDMI_Device_Session_impl_d;

struct MQT_SC_QDMI_Site_impl_d {
  MQT_SC_QDMI_Device_Session_impl_d* owner = nullptr;
  uint64_t id = 0;
  std::optional<std::string> name;
  std::optional<uint64_t> t1;
  std::optional<uint64_t> t2;

  int queryProperty(QDMI_Site_Property property, size_t size, void* value,
                    size_t* sizeRet) const;
};

struct MQT_SC_QDMI_Operation_impl_d {
  struct Calibration {
    std::optional<uint64_t> duration;
    std::optional<double> fidelity;
  };

  MQT_SC_QDMI_Device_Session_impl_d* owner = nullptr;
  std::string name;
  size_t numParameters = 0;
  size_t numQubits = 0;
  std::vector<std::vector<MQT_SC_QDMI_Site>> supportedSites;
  std::vector<MQT_SC_QDMI_Site> flattenedSites;
  Calibration defaults;
  std::vector<std::pair<std::vector<MQT_SC_QDMI_Site>, Calibration>> overrides;

  int queryProperty(size_t numSites, const MQT_SC_QDMI_Site* sites,
                    size_t numParams, const double* params,
                    QDMI_Operation_Property property, size_t size, void* value,
                    size_t* sizeRet) const;
};

struct MQT_SC_QDMI_Device_Job_impl_d {
  explicit MQT_SC_QDMI_Device_Job_impl_d(
      MQT_SC_QDMI_Device_Session_impl_d* session)
      : session(session) {}

  MQT_SC_QDMI_Device_Session_impl_d* session;
  void free();
  int setParameter(QDMI_Device_Job_Parameter parameter, size_t size,
                   const void* value);
  int setPrograms(const QDMI_Program_Format* format, size_t count,
                  const size_t* sizes, const void* const* programs);
  int queryProperty(QDMI_Device_Job_Property property, size_t size, void* value,
                    size_t* sizeRet);
  int submit();
  int cancel();
  int check(QDMI_Job_Status* status);
  int wait(size_t timeout);
  int getResults(size_t programIndex, QDMI_Job_Result result, size_t size,
                 void* data, size_t* sizeRet);
};

struct MQT_SC_QDMI_Device_Session_impl_d {
  enum class Status : uint8_t { ALLOCATED, INITIALIZED };

  Status status = Status::ALLOCATED;
  std::optional<std::string> inlineConfiguration;
  std::optional<std::filesystem::path> fileConfiguration;
  std::string name;
  std::string durationUnit;
  double durationScaleFactor = 1.;
  size_t qubitsNum = 0;
  std::vector<std::unique_ptr<MQT_SC_QDMI_Site_impl_d>> siteStorage;
  std::vector<MQT_SC_QDMI_Site> sites;
  std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>> couplingMap;
  std::vector<std::unique_ptr<MQT_SC_QDMI_Operation_impl_d>> operationStorage;
  std::vector<MQT_SC_QDMI_Operation> operations;
  std::unordered_map<MQT_SC_QDMI_Device_Job,
                     std::unique_ptr<MQT_SC_QDMI_Device_Job_impl_d>>
      jobs;

  int init();
  int setParameter(QDMI_Device_Session_Parameter parameter, size_t size,
                   const void* value);
  int createDeviceJob(MQT_SC_QDMI_Device_Job* job);
  void freeDeviceJob(MQT_SC_QDMI_Device_Job job);
  int queryDeviceProperty(QDMI_Device_Property property, size_t size,
                          void* value, size_t* sizeRet) const;
  int querySiteProperty(MQT_SC_QDMI_Site site, QDMI_Site_Property property,
                        size_t size, void* value, size_t* sizeRet) const;
  int queryOperationProperty(MQT_SC_QDMI_Operation operation, size_t numSites,
                             const MQT_SC_QDMI_Site* sites, size_t numParams,
                             const double* params,
                             QDMI_Operation_Property property, size_t size,
                             void* value, size_t* sizeRet) const;
};
