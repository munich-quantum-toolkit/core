/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Device.cpp
 * @brief The MQT QDMI device implementation for superconducting devices.
 */

#include "qdmi/devices/sc/Device.hpp"

#include "mqt_sc_qdmi/constants.h"
#include "mqt_sc_qdmi/device.h"
#include "mqt_sc_qdmi/types.h"
#include "qdmi/common/Common.hpp"
#include "qdmi/common/DeviceConfiguration.hpp"
#include "qdmi/common/Diagnostics.hpp"
#include "qdmi/devices/sc/Configuration.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] bool
contains(const std::vector<std::vector<MQT_SC_QDMI_Site>>& haystack,
         const std::vector<MQT_SC_QDMI_Site>& needle) {
  return std::ranges::find(haystack, needle) != haystack.end();
}

[[nodiscard]] std::vector<MQT_SC_QDMI_Site>
materializeTuple(const std::vector<uint64_t>& indices,
                 const std::vector<MQT_SC_QDMI_Site>& sites) {
  std::vector<MQT_SC_QDMI_Site> tuple;
  tuple.reserve(indices.size());
  for (const auto index : indices) {
    if (index >= sites.size()) {
      throw std::invalid_argument("operation site index is out of range");
    }
    tuple.emplace_back(sites[index]);
  }
  return tuple;
}
} // namespace

int MQT_SC_QDMI_Device_Session_impl_d::init() {
  if (status != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  std::optional<qdmi::detail::LoadedDeviceConfiguration> loaded;
  try {
    int loadStatus = QDMI_SUCCESS;
    loaded = qdmi::detail::loadDeviceConfiguration(
        inlineConfiguration, fileConfiguration, "MQT_CORE_QDMI_SC_CONFIG_JSON",
        "MQT_CORE_QDMI_SC_CONFIG_FILE", "mqt-core-qdmi-sc-device.json",
        reinterpret_cast<const void*>(&MQT_SC_QDMI_device_initialize),
        loadStatus);
    if (!loaded) {
      return loadStatus;
    }
    const auto configuration = sc::readJSON(loaded->json, loaded->source);

    std::vector<std::unique_ptr<MQT_SC_QDMI_Site_impl_d>> newSiteStorage;
    std::vector<MQT_SC_QDMI_Site> newSites;
    newSiteStorage.reserve(configuration.numQubits);
    newSites.reserve(configuration.numQubits);
    for (uint64_t id = 0; id < configuration.numQubits; ++id) {
      auto site = std::make_unique<MQT_SC_QDMI_Site_impl_d>();
      site->owner = this;
      site->id = id;
      site->t1 = configuration.qubitProperties.defaults.t1;
      site->t2 = configuration.qubitProperties.defaults.t2;
      newSites.emplace_back(site.get());
      newSiteStorage.emplace_back(std::move(site));
    }
    for (const auto& override : configuration.qubitProperties.overrides) {
      auto& site = newSiteStorage.at(override.qubit);
      if (override.name) {
        site->name = override.name;
      }
      if (override.t1) {
        site->t1 = override.t1;
      }
      if (override.t2) {
        site->t2 = override.t2;
      }
    }

    std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>> newCouplingMap;
    newCouplingMap.reserve(configuration.couplings.size());
    for (const auto& [first, second] : configuration.couplings) {
      newCouplingMap.emplace_back(newSites.at(first), newSites.at(second));
    }

    std::vector<std::unique_ptr<MQT_SC_QDMI_Operation_impl_d>>
        newOperationStorage;
    std::vector<MQT_SC_QDMI_Operation> newOperations;
    newOperationStorage.reserve(configuration.operations.size());
    newOperations.reserve(configuration.operations.size());
    for (const auto& operationConfiguration : configuration.operations) {
      auto operation = std::make_unique<MQT_SC_QDMI_Operation_impl_d>();
      operation->owner = this;
      operation->name = operationConfiguration.name;
      operation->numParameters = operationConfiguration.numParameters;
      operation->numQubits = operationConfiguration.numQubits;
      operation->defaults = {
          .duration = operationConfiguration.duration,
          .fidelity = operationConfiguration.fidelity,
      };
      if (operationConfiguration.sites) {
        for (const auto& tuple : *operationConfiguration.sites) {
          operation->supportedSites.emplace_back(
              materializeTuple(tuple, newSites));
        }
      } else if (operation->numQubits == 1) {
        for (auto* const site : newSites) {
          operation->supportedSites.push_back({site});
        }
      } else if (operation->numQubits == 2) {
        for (const auto& [first, second] : newCouplingMap) {
          operation->supportedSites.push_back({first, second});
        }
      }
      for (const auto& tuple : operation->supportedSites) {
        operation->flattenedSites.insert(operation->flattenedSites.end(),
                                         tuple.begin(), tuple.end());
      }
      for (const auto& override : operationConfiguration.siteOverrides) {
        auto tuple = materializeTuple(override.sites, newSites);
        if (!contains(operation->supportedSites, tuple)) {
          throw std::invalid_argument(
              "operation site override is not a supported tuple");
        }
        operation->overrides.emplace_back(
            std::move(tuple), MQT_SC_QDMI_Operation_impl_d::Calibration{
                                  .duration = override.duration,
                                  .fidelity = override.fidelity,
                              });
      }
      newOperations.emplace_back(operation.get());
      newOperationStorage.emplace_back(std::move(operation));
    }

    name = configuration.name;
    durationUnit = configuration.durationUnit.unit;
    durationScaleFactor = configuration.durationUnit.scaleFactor;
    qubitsNum = static_cast<size_t>(configuration.numQubits);
    siteStorage = std::move(newSiteStorage);
    sites = std::move(newSites);
    couplingMap = std::move(newCouplingMap);
    operationStorage = std::move(newOperationStorage);
    operations = std::move(newOperations);
    status = Status::INITIALIZED;
    return QDMI_SUCCESS;
  } catch (const std::bad_alloc&) {
    const std::string_view source =
        loaded ? std::string_view(loaded->source)
               : std::string_view("selected configuration");
    qdmi::diagnostics::error(
        "Out of memory while initializing SC device from {}", source);
    return QDMI_ERROR_OUTOFMEM;
  } catch (const std::invalid_argument& error) {
    const std::string_view source =
        loaded ? std::string_view(loaded->source)
               : std::string_view("selected configuration");
    qdmi::diagnostics::error("Invalid SC device configuration from {}: {}",
                             source, error.what());
    return QDMI_ERROR_INVALIDARGUMENT;
  } catch (const std::exception& error) {
    const std::string_view source =
        loaded ? std::string_view(loaded->source)
               : std::string_view("selected configuration");
    qdmi::diagnostics::error("Failed to initialize SC device from {}: {}",
                             source, error.what());
    return QDMI_ERROR_FATAL;
  }
}

int MQT_SC_QDMI_Device_Session_impl_d::setParameter(
    const QDMI_Device_Session_Parameter parameter, const size_t size,
    const void* value) {
  if (parameter == QDMI_DEVICE_SESSION_PARAMETER_MAX ||
      IS_INVALID_ARGUMENT(parameter, QDMI_DEVICE_SESSION_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return qdmi::detail::setDeviceConfigurationParameter(
      parameter, size, value, inlineConfiguration, fileConfiguration);
}

int MQT_SC_QDMI_Device_Session_impl_d::createDeviceJob(
    MQT_SC_QDMI_Device_Job* job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  auto value = std::make_unique<MQT_SC_QDMI_Device_Job_impl_d>(this);
  *job = value.get();
  jobs.emplace(*job, std::move(value));
  return QDMI_SUCCESS;
}

void MQT_SC_QDMI_Device_Session_impl_d::freeDeviceJob(
    MQT_SC_QDMI_Device_Job job) {
  jobs.erase(job);
}

int MQT_SC_QDMI_Device_Session_impl_d::queryDeviceProperty(
    const QDMI_Device_Property property, const size_t size, void* value,
    size_t* sizeRet) const {
  if (status != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(property, QDMI_DEVICE_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_NAME, name.c_str(), property, size,
                      value, sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_VERSION, MQT_CORE_VERSION, property,
                      size, value, sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LIBRARYVERSION, QDMI_VERSION,
                      property, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_STATUS, QDMI_Device_Status,
                            QDMI_DEVICE_STATUS_IDLE, property, size, value,
                            sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_QUBITSNUM, size_t, qubitsNum,
                            property, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, size_t, 0,
                            property, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(
      QDMI_DEVICE_PROPERTY_PULSESUPPORT, QDMI_Device_Pulse_Support_Level,
      QDMI_DEVICE_PULSE_SUPPORT_LEVEL_NONE, property, size, value, sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_DURATIONUNIT, durationUnit.c_str(),
                      property, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR, double,
                            durationScaleFactor, property, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_SC_QDMI_Site, sites,
                    property, size, value, sizeRet)
#define SITE_PAIR std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_COUPLINGMAP, SITE_PAIR, couplingMap,
                    property, size, value, sizeRet)
#undef SITE_PAIR
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_SC_QDMI_Operation,
                    operations, property, size, value, sizeRet)
  if (property == QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS) {
    if (value != nullptr && size > 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (sizeRet != nullptr) {
      *sizeRet = 0;
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_SC_QDMI_Device_Session_impl_d::querySiteProperty(
    MQT_SC_QDMI_Site site, const QDMI_Site_Property property, const size_t size,
    void* value, size_t* sizeRet) const {
  if (status != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (site == nullptr || site->owner != this) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return site->queryProperty(property, size, value, sizeRet);
}

int MQT_SC_QDMI_Device_Session_impl_d::queryOperationProperty(
    MQT_SC_QDMI_Operation operation, const size_t numSites,
    const MQT_SC_QDMI_Site* suppliedSites, const size_t numParams,
    const double* params, const QDMI_Operation_Property property,
    const size_t size, void* value, size_t* sizeRet) const {
  if (status != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (operation == nullptr || operation->owner != this) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (suppliedSites != nullptr) {
    for (auto* const site : std::span{suppliedSites, numSites}) {
      if (site == nullptr || site->owner != this) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
    }
  }
  return operation->queryProperty(numSites, suppliedSites, numParams, params,
                                  property, size, value, sizeRet);
}

int MQT_SC_QDMI_Site_impl_d::queryProperty(const QDMI_Site_Property property,
                                           const size_t size, void* value,
                                           size_t* sizeRet) const {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(property, QDMI_SITE_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, uint64_t, id, property,
                            size, value, sizeRet)
  if (property == QDMI_SITE_PROPERTY_NAME) {
    if (!name) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    ADD_STRING_PROPERTY(QDMI_SITE_PROPERTY_NAME, name->c_str(), property, size,
                        value, sizeRet)
  }
  if (property == QDMI_SITE_PROPERTY_T1) {
    if (!t1) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T1, uint64_t, *t1, property,
                              size, value, sizeRet)
  }
  if (property == QDMI_SITE_PROPERTY_T2) {
    if (!t2) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T2, uint64_t, *t2, property,
                              size, value, sizeRet)
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_SC_QDMI_Operation_impl_d::queryProperty(
    const size_t numSites, const MQT_SC_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property property, const size_t size, void* value,
    size_t* sizeRet) const {
  if ((sites == nullptr) != (numSites == 0) ||
      (params == nullptr) != (numParams == 0) ||
      (value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(property, QDMI_OPERATION_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  std::vector<MQT_SC_QDMI_Site> tuple;
  if (sites != nullptr) {
    if (numSites != numQubits) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    const std::span suppliedSites{sites, numSites};
    tuple.assign(suppliedSites.begin(), suppliedSites.end());
    if (!contains(supportedSites, tuple)) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
  }
  if (params != nullptr && numParams != numParameters) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, name.c_str(), property,
                      size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            numParameters, property, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                            numQubits, property, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_OPERATION_PROPERTY_SITES, MQT_SC_QDMI_Site,
                    flattenedSites, property, size, value, sizeRet)
  const auto calibration = [&]() -> Calibration {
    if (!tuple.empty()) {
      if (const auto found = std::ranges::find(
              overrides, tuple,
              &std::pair<std::vector<MQT_SC_QDMI_Site>, Calibration>::first);
          found != overrides.end()) {
        return {
            .duration = found->second.duration ? found->second.duration
                                               : defaults.duration,
            .fidelity = found->second.fidelity ? found->second.fidelity
                                               : defaults.fidelity,
        };
      }
    }
    return defaults;
  }();
  if (property == QDMI_OPERATION_PROPERTY_DURATION) {
    if (!calibration.duration) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_DURATION, uint64_t,
                              *calibration.duration, property, size, value,
                              sizeRet)
  }
  if (property == QDMI_OPERATION_PROPERTY_FIDELITY) {
    if (!calibration.fidelity) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double,
                              *calibration.fidelity, property, size, value,
                              sizeRet)
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

void MQT_SC_QDMI_Device_Job_impl_d::free() { session->freeDeviceJob(this); }
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::setParameter(
    const QDMI_Device_Job_Parameter parameter, const size_t size,
    const void* value) {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(parameter, QDMI_DEVICE_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::queryProperty(
    const QDMI_Device_Job_Property property, const size_t size, void* value,
    size_t* /*sizeRet*/) {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(property, QDMI_DEVICE_JOB_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::submit() { return QDMI_ERROR_NOTSUPPORTED; }
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::cancel() { return QDMI_ERROR_NOTSUPPORTED; }
// NOLINTNEXTLINE(readability-non-const-parameter,readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::check(QDMI_Job_Status* status) {
  return status == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                           : QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::wait(size_t /*timeout*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int MQT_SC_QDMI_Device_Job_impl_d::getResults(const QDMI_Job_Result result,
                                              const size_t size, void* data,
                                              size_t* /*sizeRet*/) {
  if ((data != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(result, QDMI_JOB_RESULT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_SC_QDMI_device_initialize() { return QDMI_SUCCESS; }
int MQT_SC_QDMI_device_finalize() { return QDMI_SUCCESS; }
int MQT_SC_QDMI_device_session_alloc(MQT_SC_QDMI_Device_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // QDMI transfers ownership through its opaque C handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new (std::nothrow) MQT_SC_QDMI_Device_Session_impl_d;
  return *session == nullptr ? QDMI_ERROR_OUTOFMEM : QDMI_SUCCESS;
}
int MQT_SC_QDMI_device_session_init(MQT_SC_QDMI_Device_Session session) {
  return session == nullptr ? QDMI_ERROR_INVALIDARGUMENT : session->init();
}
void MQT_SC_QDMI_device_session_free(MQT_SC_QDMI_Device_Session session) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete session;
}
int MQT_SC_QDMI_device_session_set_parameter(
    MQT_SC_QDMI_Device_Session session,
    const QDMI_Device_Session_Parameter parameter, const size_t size,
    const void* value) {
  return session == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                            : session->setParameter(parameter, size, value);
}
int MQT_SC_QDMI_device_session_create_device_job(
    MQT_SC_QDMI_Device_Session session, MQT_SC_QDMI_Device_Job* job) {
  return session == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                            : session->createDeviceJob(job);
}
int MQT_SC_QDMI_device_session_retrieve_device_job_by_id(
    [[maybe_unused]] MQT_SC_QDMI_Device_Session session,
    [[maybe_unused]] const char* jobId,
    [[maybe_unused]] MQT_SC_QDMI_Device_Job* job) {
  return QDMI_ERROR_NOTSUPPORTED;
}
void MQT_SC_QDMI_device_job_free(MQT_SC_QDMI_Device_Job job) {
  if (job != nullptr) {
    job->free();
  }
}
int MQT_SC_QDMI_device_job_set_parameter(
    MQT_SC_QDMI_Device_Job job, const QDMI_Device_Job_Parameter parameter,
    const size_t size, const void* value) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                        : job->setParameter(parameter, size, value);
}
int MQT_SC_QDMI_device_job_query_property(
    MQT_SC_QDMI_Device_Job job, const QDMI_Device_Job_Property property,
    const size_t size, void* value, size_t* sizeRet) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                        : job->queryProperty(property, size, value, sizeRet);
}
int MQT_SC_QDMI_device_job_submit(MQT_SC_QDMI_Device_Job job) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT : job->submit();
}
int MQT_SC_QDMI_device_job_cancel(MQT_SC_QDMI_Device_Job job) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT : job->cancel();
}
int MQT_SC_QDMI_device_job_check(MQT_SC_QDMI_Device_Job job,
                                 QDMI_Job_Status* status) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT : job->check(status);
}
int MQT_SC_QDMI_device_job_wait(MQT_SC_QDMI_Device_Job job,
                                const size_t timeout) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT : job->wait(timeout);
}
int MQT_SC_QDMI_device_job_get_results(MQT_SC_QDMI_Device_Job job,
                                       const QDMI_Job_Result result,
                                       const size_t size, void* data,
                                       size_t* sizeRet) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                        : job->getResults(result, size, data, sizeRet);
}
int MQT_SC_QDMI_device_session_query_device_property(
    MQT_SC_QDMI_Device_Session session, const QDMI_Device_Property property,
    const size_t size, void* value, size_t* sizeRet) {
  return session == nullptr
             ? QDMI_ERROR_INVALIDARGUMENT
             : session->queryDeviceProperty(property, size, value, sizeRet);
}
int MQT_SC_QDMI_device_session_query_site_property(
    MQT_SC_QDMI_Device_Session session, MQT_SC_QDMI_Site site,
    const QDMI_Site_Property property, const size_t size, void* value,
    size_t* sizeRet) {
  return session == nullptr
             ? QDMI_ERROR_INVALIDARGUMENT
             : session->querySiteProperty(site, property, size, value, sizeRet);
}
int MQT_SC_QDMI_device_session_query_operation_property(
    MQT_SC_QDMI_Device_Session session, MQT_SC_QDMI_Operation operation,
    const size_t numSites, const MQT_SC_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property property, const size_t size, void* value,
    size_t* sizeRet) {
  return session == nullptr
             ? QDMI_ERROR_INVALIDARGUMENT
             : session->queryOperationProperty(operation, numSites, sites,
                                               numParams, params, property,
                                               size, value, sizeRet);
}
