/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief The MQT QDMI device implementation for its DD-based simulator.
 */

#include "qdmi/dd/Device.hpp"

#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/QuantumComputation.hpp"
#include "mqt_ddsim_qdmi/device.h"
#include "qasm3/Importer.hpp"
#include "qdmi/Common.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ranges>
#include <string>
#include <utility>

namespace {
constexpr uintptr_t OFFSET = 0x10000U;
template <typename T, std::size_t N> constexpr std::array<T, N> iotaArray() {
  std::array<T, N> result{};
  std::iota(result.begin(), result.end(), OFFSET);
  return result;
}

constexpr std::array<uintptr_t, std::numeric_limits<dd::Qubit>::max()> SITES =
    iotaArray<uintptr_t, std::numeric_limits<dd::Qubit>::max()>();

struct OperationInfo {
  const char* name{};
  std::size_t numSites{};
  std::size_t numParams{};
  bool isVariadic = false;
};

constexpr std::array OPERATIONS{
    OperationInfo{.name = "gphase", .numSites = 0, .numParams = 1},
    OperationInfo{.name = "i", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "x", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "cx", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "ccx", .numSites = 3, .numParams = 0},
    OperationInfo{
        .name = "mcx", .numSites = 0, .numParams = 0, .isVariadic = true},
    OperationInfo{.name = "y", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "cy", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "z", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "cz", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "ccz", .numSites = 3, .numParams = 0},
    OperationInfo{
        .name = "mcz", .numSites = 0, .numParams = 0, .isVariadic = true},
    OperationInfo{.name = "h", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "ch", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "s", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "cs", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "sdg", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "csdg", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "t", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "tdg", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "sx", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "csx", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "sxdg", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "r", .numSites = 1, .numParams = 2},
    OperationInfo{.name = "rx", .numSites = 1, .numParams = 1},
    OperationInfo{.name = "crx", .numSites = 2, .numParams = 1},
    OperationInfo{
        .name = "mcrx", .numSites = 0, .numParams = 1, .isVariadic = true},
    OperationInfo{.name = "ry", .numSites = 1, .numParams = 1},
    OperationInfo{.name = "cry", .numSites = 2, .numParams = 1},
    OperationInfo{
        .name = "mcry", .numSites = 0, .numParams = 1, .isVariadic = true},
    OperationInfo{.name = "rz", .numSites = 1, .numParams = 1},
    OperationInfo{.name = "crz", .numSites = 2, .numParams = 1},
    OperationInfo{
        .name = "mcrz", .numSites = 0, .numParams = 1, .isVariadic = true},
    OperationInfo{.name = "p", .numSites = 1, .numParams = 1},
    OperationInfo{.name = "cp", .numSites = 2, .numParams = 1},
    OperationInfo{
        .name = "mcp", .numSites = 0, .numParams = 1, .isVariadic = true},
    OperationInfo{.name = "u1", .numSites = 1, .numParams = 1},
    OperationInfo{.name = "cu1", .numSites = 2, .numParams = 1},
    OperationInfo{.name = "u2", .numSites = 1, .numParams = 2},
    OperationInfo{.name = "u", .numSites = 1, .numParams = 3},
    OperationInfo{.name = "u3", .numSites = 1, .numParams = 3},
    OperationInfo{.name = "cu3", .numSites = 2, .numParams = 3},
    OperationInfo{.name = "swap", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "cswap", .numSites = 3, .numParams = 0},
    OperationInfo{.name = "iswap", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "dcx", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "ecr", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "rxx", .numSites = 2, .numParams = 1},
    OperationInfo{.name = "ryy", .numSites = 2, .numParams = 1},
    OperationInfo{.name = "rzz", .numSites = 2, .numParams = 1},
    OperationInfo{.name = "rzx", .numSites = 2, .numParams = 1},
    OperationInfo{.name = "xx_minus_yy", .numSites = 2, .numParams = 2},
    OperationInfo{.name = "xx_plus_yy", .numSites = 2, .numParams = 2},
    OperationInfo{.name = "measure", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "reset", .numSites = 1, .numParams = 0},
    OperationInfo{
        .name = "barrier", .numSites = 0, .numParams = 0, .isVariadic = true},
    OperationInfo{
        .name = "if_else", .numSites = 0, .numParams = 0, .isVariadic = true}};

template <std::size_t N>
constexpr std::array<const OperationInfo*, N>
makeOperationAddresses(const std::array<OperationInfo, N>& ops) {
  std::array<const OperationInfo*, N> addresses{};
  for (std::size_t i = 0; i < N; ++i) {
    addresses[i] = &ops[i];
  }
  return addresses;
}
constexpr auto OPERATION_ADDRESSES = makeOperationAddresses(OPERATIONS);

constexpr std::array SUPPORTED_PROGRAM_FORMATS = {QDMI_PROGRAM_FORMAT_QASM2,
                                                  QDMI_PROGRAM_FORMAT_QASM3};

} // namespace

namespace qdmi::dd {
Device::Device()
    : name_("MQT Core DDSIM QDMI Device"),
      qubitsNum_(std::numeric_limits<::dd::Qubit>::max()) {}
auto Device::get() -> Device& {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* instance = new Device();
  // The instance is intentionally leaked to avoid static deinitialization
  // issues (cf. static (de)initialization order fiasco)
  return *instance;
}
auto Device::sessionAlloc(MQT_DDSIM_QDMI_Device_Session* session)
    -> QDMI_STATUS {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<MQT_DDSIM_QDMI_Device_Session_impl_d>();
  const std::scoped_lock<std::mutex> lock(sessionsMutex_);
  const auto& it =
      sessions_.emplace(uniqueSession.get(), std::move(uniqueSession)).first;
  // get the key, i.e., the raw pointer to the session from the map iterator
  *session = it->first;
  return QDMI_SUCCESS;
}
auto Device::sessionFree(MQT_DDSIM_QDMI_Device_Session session) -> void {
  if (session != nullptr) {
    const std::scoped_lock<std::mutex> lock(sessionsMutex_);
    if (const auto& it = sessions_.find(session); it != sessions_.end()) {
      sessions_.erase(it);
    }
  }
}
auto Device::queryProperty(const QDMI_Device_Property prop, const size_t size,
                           void* value, size_t* sizeRet) const -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_DEVICE_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_NAME, name_.c_str(), prop, size,
                      value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_VERSION, MQT_CORE_VERSION, prop,
                      size, value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LIBRARYVERSION, QDMI_VERSION, prop,
                      size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_STATUS, QDMI_Device_Status,
                            status_.load(), prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_QUBITSNUM, size_t, qubitsNum_,
                            prop, size, value, sizeRet)
  // This device never needs calibration
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, size_t, 0,
                            prop, size, value, sizeRet)
  // This device does not support pulse-level control
  ADD_SINGLE_VALUE_PROPERTY(
      QDMI_DEVICE_PROPERTY_PULSESUPPORT, QDMI_Device_Pulse_Support_Level,
      QDMI_DEVICE_PULSE_SUPPORT_LEVEL_NONE, prop, size, value, sizeRet)
  // Expose default length and time units
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LENGTHUNIT, "um", prop, size, value,
                      sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR, double, 1.0,
                            prop, size, value, sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_DURATIONUNIT, "ns", prop, size,
                      value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR, double,
                            1.0, prop, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_DDSIM_QDMI_Site, SITES,
                    prop, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_DDSIM_QDMI_Operation,
                    OPERATION_ADDRESSES, prop, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS,
                    QDMI_Program_Format, SUPPORTED_PROGRAM_FORMATS, prop, size,
                    value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}

auto Device::generateUniqueID() -> int {
  const std::scoped_lock<std::mutex> lock(rngMutex_);
  return dis_(rng_);
}
auto Device::setStatus(const QDMI_Device_Status status) -> void {
  status_.store(status);
}
auto Device::increaseRunningJobs() -> void {
  if (const auto prev = runningJobs_.fetch_add(1); prev == 0) {
    setStatus(QDMI_DEVICE_STATUS_BUSY);
  }
}
auto Device::decreaseRunningJobs() -> void {
  if (const auto prev = runningJobs_.fetch_sub(1); prev == 1) {
    setStatus(QDMI_DEVICE_STATUS_IDLE);
  }
}

} // namespace qdmi::dd

auto MQT_DDSIM_QDMI_Device_Session_impl_d::init() -> QDMI_STATUS {
  if (status_ != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status_ = Status::INITIALIZED;
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Session_impl_d::setParameter(
    const QDMI_Device_Session_Parameter param, const size_t size,
    const void* value) const -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_DEVICE_SESSION_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_DDSIM_QDMI_Device_Session_impl_d::createDeviceJob(
    // NOLINTNEXTLINE(readability-non-const-parameter)
    MQT_DDSIM_QDMI_Device_Job* job) -> QDMI_STATUS {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ == Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  auto uniqueJob = std::make_unique<MQT_DDSIM_QDMI_Device_Job_impl_d>(this);
  const std::scoped_lock<std::mutex> lock(jobsMutex_);
  *job = jobs_.emplace(uniqueJob.get(), std::move(uniqueJob)).first->first;
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Session_impl_d::freeDeviceJob(
    MQT_DDSIM_QDMI_Device_Job job) -> void {
  if (job != nullptr) {
    const std::scoped_lock<std::mutex> lock(jobsMutex_);
    jobs_.erase(job);
  }
}
auto MQT_DDSIM_QDMI_Device_Session_impl_d::queryDeviceProperty(
    const QDMI_Device_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> QDMI_STATUS {
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return qdmi::dd::Device::get().queryProperty(prop, size, value, sizeRet);
}
auto MQT_DDSIM_QDMI_Device_Session_impl_d::querySiteProperty(
    MQT_DDSIM_QDMI_Site site, const QDMI_Site_Property prop, const size_t size,
    void* value, size_t* sizeRet) const -> QDMI_STATUS {
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (site == nullptr || (value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_SITE_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const auto id = reinterpret_cast<uintptr_t>(site) - OFFSET;
  static_assert(sizeof(uintptr_t) == sizeof(size_t));
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, size_t, id, prop, size,
                            value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_DDSIM_QDMI_Device_Session_impl_d::queryOperationProperty(
    MQT_DDSIM_QDMI_Operation operation, const size_t numSites,
    const MQT_DDSIM_QDMI_Site* sites, const size_t numParams,
    const double* params, const QDMI_Operation_Property prop, const size_t size,
    void* value, size_t* sizeRet) const -> QDMI_STATUS {
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (operation == nullptr || (sites != nullptr && numSites == 0) ||
      (params != nullptr && numParams == 0) ||
      (value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_OPERATION_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const auto& [name_, numSites_, numParams_, isVariadic] =
      *reinterpret_cast<const OperationInfo*>(operation);
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, name_, prop, size, value,
                      sizeRet)
  if (!isVariadic) {
    if (sites != nullptr && numSites_ != numSites) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                              numSites_, prop, size, value, sizeRet)
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            numParams_, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double, 1.0, prop,
                            size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::free() -> void {
  // avoid freeing job while the asynchronous operation is still running
  if (jobHandle_.valid()) {
    jobHandle_.wait();
  }
  session_->freeDeviceJob(this);
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::setParameter(
    const QDMI_Device_Job_Parameter param, const size_t size, const void* value)
    -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_DEVICE_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_.load() != QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  switch (param) {
  case QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT:
    if (value != nullptr) {
      const auto format = *static_cast<const QDMI_Program_Format*>(value);
      if (IS_INVALID_ARGUMENT(format, QDMI_PROGRAM_FORMAT)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      if (format != QDMI_PROGRAM_FORMAT_QASM2 &&
          format != QDMI_PROGRAM_FORMAT_QASM3) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
      format_ = format;
    }
    return QDMI_SUCCESS;
  case QDMI_DEVICE_JOB_PARAMETER_PROGRAM:
    if (value != nullptr) {
      program_ = std::string(static_cast<const char*>(value), size - 1);
    }
    return QDMI_SUCCESS;
  case QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM:
    if (value != nullptr) {
      numShots_ = *static_cast<const size_t*>(value);
    }
    return QDMI_SUCCESS;
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::queryProperty(
    // NOLINTNEXTLINE(readability-non-const-parameter)
    const QDMI_Device_Job_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_DEVICE_JOB_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const auto str = std::to_string(id_);
  ADD_STRING_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_ID, str.c_str(), prop, size,
                      value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT,
                            QDMI_Program_Format, format_, prop, size, value,
                            sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAM, program_.c_str(), prop,
                      size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, size_t,
                            numShots_, prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::submit() -> QDMI_STATUS {
  if (status_.load() != QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status_.store(QDMI_JOB_STATUS_SUBMITTED);
  if (numShots_ > 0) {
    jobHandle_ = std::async(std::launch::async, [this]() {
      qdmi::dd::Device::get().increaseRunningJobs();
      status_.store(QDMI_JOB_STATUS_RUNNING);
      try {
        const auto qc = qasm3::Importer::imports(program_);
        counts_ = dd::sample(qc, numShots_);
        status_.store(QDMI_JOB_STATUS_DONE);
      } catch (const std::exception& e) {
        status_.store(QDMI_JOB_STATUS_FAILED);
        std::cerr << "Error: " << e.what() << '\n';
      }
      qdmi::dd::Device::get().decreaseRunningJobs();
    });
  } else {
    jobHandle_ = std::async(std::launch::async, [this]() {
      try {
        qdmi::dd::Device::get().increaseRunningJobs();
        status_.store(QDMI_JOB_STATUS_RUNNING);
        auto qc = qasm3::Importer::imports(program_);
        qc::CircuitOptimizer::removeFinalMeasurements(qc);
        const auto nqubits = qc.getNqubits();
        dd_ = std::make_unique<dd::Package>(nqubits);
        stateVecDD_ = dd::simulate(qc, dd::makeZeroState(nqubits, *dd_), *dd_);
        status_.store(QDMI_JOB_STATUS_DONE);
      } catch (const std::exception& e) {
        status_.store(QDMI_JOB_STATUS_FAILED);
        std::cerr << "Error: " << e.what() << '\n';
      }
      qdmi::dd::Device::get().decreaseRunningJobs();
    });
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::cancel() -> QDMI_STATUS {
  const auto s = status_.load();
  if (s == QDMI_JOB_STATUS_DONE || s == QDMI_JOB_STATUS_FAILED) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }

  if (s == QDMI_JOB_STATUS_CREATED) {
    status_.store(QDMI_JOB_STATUS_CANCELED);
    return QDMI_SUCCESS;
  }

  if (jobHandle_.valid()) {
    // Note: There is no direct way to cancel a running std::async task.
    // We can only wait for its completion here.
    jobHandle_.wait();
  }
  status_.store(QDMI_JOB_STATUS_CANCELED);
  return QDMI_SUCCESS;
}
// NOLINTNEXTLINE(readability-non-const-parameter)
auto MQT_DDSIM_QDMI_Device_Job_impl_d::check(QDMI_Job_Status* status) const
    -> QDMI_STATUS {
  if (status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *status = status_.load();
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::wait(const size_t timeout) const
    -> QDMI_STATUS {
  const auto s = status_.load();
  if (s == QDMI_JOB_STATUS_DONE || s == QDMI_JOB_STATUS_CANCELED ||
      s == QDMI_JOB_STATUS_FAILED) {
    return QDMI_SUCCESS;
  }
  if (!jobHandle_.valid() ||
      (s != QDMI_JOB_STATUS_SUBMITTED && s != QDMI_JOB_STATUS_QUEUED &&
       s != QDMI_JOB_STATUS_RUNNING)) {
    return QDMI_ERROR_BADSTATE;
  }

  if (timeout > 0) {
    if (const auto st = jobHandle_.wait_for(std::chrono::seconds(timeout));
        st == std::future_status::timeout) {
      return QDMI_ERROR_TIMEOUT;
    }
  } else {
    jobHandle_.wait();
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::getHistogram(
    const QDMI_Job_Result result, const size_t size, void* data,
    size_t* sizeRet) -> QDMI_STATUS {
  if (result == QDMI_JOB_RESULT_HIST_KEYS) {
    const size_t bitstringSize =
        counts_.empty() ? 0 : counts_.begin()->first.length();
    const size_t reqSize = counts_.size() * (bitstringSize + 1);
    if (sizeRet != nullptr) {
      *sizeRet = reqSize;
    }
    if (data != nullptr) {
      if (size < reqSize) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* dataPtr = static_cast<char*>(data);
      for (const auto& bitstring : counts_ | std::views::keys) {
        std::ranges::copy(bitstring, dataPtr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        dataPtr += bitstring.length();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = ',';
      }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      *(dataPtr - 1) = '\0'; // Replace last comma with null terminator
    }
  } else {
    // case QDMI_JOB_RESULT_HIST_VALUES:
    const size_t reqSize = counts_.size() * sizeof(size_t);
    if (sizeRet != nullptr) {
      *sizeRet = reqSize;
    }
    if (data != nullptr) {
      if (size < reqSize) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* dataPtr = static_cast<size_t*>(data);
      for (const auto& count : counts_ | std::views::values) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = count;
      }
    }
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::getStateVector(const size_t size,
                                                      void* data,
                                                      size_t* sizeRet)
    -> QDMI_STATUS {
  std::call_once(stateVecOnce_,
                 [this]() { stateVec_ = stateVecDD_.getVector(); });
  const size_t reqSize = stateVec_.size() * 2 * sizeof(double);
  if (data != nullptr) {
    if (size < reqSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::memcpy(data, stateVec_.data(), reqSize);
  }
  if (sizeRet != nullptr) {
    *sizeRet = reqSize;
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::getSparseResults(
    const QDMI_Job_Result result, const size_t size, void* data,
    size_t* sizeRet) -> QDMI_STATUS {
  std::call_once(stateVecSparseOnce_,
                 [this]() { stateVecSparse_ = stateVecDD_.getSparseVector(); });
  const size_t numQubits = stateVecDD_.p->v + 1;
  switch (result) {
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS: {
    const size_t reqSize = stateVecSparse_.size() * (numQubits + 1);
    if (data != nullptr) {
      if (size < reqSize) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* dataPtr = static_cast<char*>(data);
      for (const auto& i : stateVecSparse_ | std::views::keys) {
        for (size_t j = 0; j < numQubits; ++j) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          *dataPtr++ =
              ((i & (1ULL << (numQubits - j - 1ULL))) != 0U) ? '1' : '0';
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = ',';
      }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      *(dataPtr - 1) = '\0'; // Replace last comma with null terminator
    }
    if (sizeRet != nullptr) {
      *sizeRet = reqSize;
    }
    return QDMI_SUCCESS;
  }

  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES: {
    const size_t reqSize = stateVecSparse_.size() * 2 * sizeof(double);
    if (data != nullptr) {
      if (size < reqSize) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* dataPtr = static_cast<double*>(data);
      for (const auto& c : stateVecSparse_ | std::views::values) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = c.real();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = c.imag();
      }
    }
    if (sizeRet != nullptr) {
      *sizeRet = reqSize;
    }
    return QDMI_SUCCESS;
  }
  default: {
    // case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES:
    const size_t reqSize = stateVecSparse_.size() * sizeof(double);
    if (data != nullptr) {
      if (size < reqSize) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* dataPtr = static_cast<double*>(data);
      for (const auto& c : stateVecSparse_ | std::views::values) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = std::norm(c);
      }
    }
    if (sizeRet != nullptr) {
      *sizeRet = reqSize;
    }
  }
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::getProbabilities(const size_t size,
                                                        void* data,
                                                        size_t* sizeRet)
    -> QDMI_STATUS {
  if (stateVec_.empty()) {
    stateVec_ = stateVecDD_.getVector();
  }
  const size_t reqSize = stateVec_.size() * sizeof(double);
  if (data != nullptr) {
    if (size < reqSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto* dataPtr = static_cast<double*>(data);
    for (const auto& c : stateVec_) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      *dataPtr++ = std::norm(c);
    }
  }
  if (sizeRet != nullptr) {
    *sizeRet = reqSize;
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::getResults(const QDMI_Job_Result result,
                                                  const size_t size, void* data,
                                                  size_t* sizeRet)
    -> QDMI_STATUS {
  if ((data != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(result, QDMI_JOB_RESULT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_.load() != QDMI_JOB_STATUS_DONE) {
    return QDMI_ERROR_BADSTATE;
  }
  switch (result) {
  case QDMI_JOB_RESULT_HIST_KEYS:
  case QDMI_JOB_RESULT_HIST_VALUES:
    if (numShots_ == 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getHistogram(result, size, data, sizeRet);
  case QDMI_JOB_RESULT_STATEVECTOR_DENSE:
    if (numShots_ > 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getStateVector(size, data, sizeRet);
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS:
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES:
    if (numShots_ > 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getSparseResults(result, size, data, sizeRet);
  case QDMI_JOB_RESULT_PROBABILITIES_DENSE:
    if (numShots_ > 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getProbabilities(size, data, sizeRet);
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}

// QDMI uses a different naming convention for its C interface functions
// NOLINTBEGIN(readability-identifier-naming)
int MQT_DDSIM_QDMI_device_initialize() {
  // ensure the singleton is initialized
  std::ignore = qdmi::dd::Device::get();
  return QDMI_SUCCESS;
}

int MQT_DDSIM_QDMI_device_finalize() { return QDMI_SUCCESS; }

int MQT_DDSIM_QDMI_device_session_alloc(
    MQT_DDSIM_QDMI_Device_Session* session) {
  return qdmi::dd::Device::get().sessionAlloc(session);
}

int MQT_DDSIM_QDMI_device_session_init(MQT_DDSIM_QDMI_Device_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->init();
}

void MQT_DDSIM_QDMI_device_session_free(MQT_DDSIM_QDMI_Device_Session session) {
  qdmi::dd::Device::get().sessionFree(session);
}

int MQT_DDSIM_QDMI_device_session_set_parameter(
    MQT_DDSIM_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->setParameter(param, size, value);
}

int MQT_DDSIM_QDMI_device_session_create_device_job(
    MQT_DDSIM_QDMI_Device_Session session, MQT_DDSIM_QDMI_Device_Job* job) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->createDeviceJob(job);
}

void MQT_DDSIM_QDMI_device_job_free(MQT_DDSIM_QDMI_Device_Job job) {
  job->free();
}

int MQT_DDSIM_QDMI_device_job_set_parameter(
    MQT_DDSIM_QDMI_Device_Job job, const QDMI_Device_Job_Parameter param,
    const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->setParameter(param, size, value);
}

int MQT_DDSIM_QDMI_device_job_query_property(
    MQT_DDSIM_QDMI_Device_Job job, const QDMI_Device_Job_Property prop,
    const size_t size, void* value, size_t* size_ret) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->queryProperty(prop, size, value, size_ret);
}

int MQT_DDSIM_QDMI_device_job_submit(MQT_DDSIM_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->submit();
}

int MQT_DDSIM_QDMI_device_job_cancel(MQT_DDSIM_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->cancel();
}

int MQT_DDSIM_QDMI_device_job_check(MQT_DDSIM_QDMI_Device_Job job,
                                    QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->check(status);
}

int MQT_DDSIM_QDMI_device_job_wait(MQT_DDSIM_QDMI_Device_Job job,
                                   const size_t timeout) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->wait(timeout);
}

int MQT_DDSIM_QDMI_device_job_get_results(MQT_DDSIM_QDMI_Device_Job job,
                                          QDMI_Job_Result result,
                                          const size_t size, void* data,
                                          size_t* size_ret) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getResults(result, size, data, size_ret);
}

int MQT_DDSIM_QDMI_device_session_query_device_property(
    MQT_DDSIM_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* size_ret) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->queryDeviceProperty(prop, size, value, size_ret);
}

int MQT_DDSIM_QDMI_device_session_query_site_property(
    MQT_DDSIM_QDMI_Device_Session session, MQT_DDSIM_QDMI_Site site,
    const QDMI_Site_Property prop, const size_t size, void* value,
    size_t* size_ret) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->querySiteProperty(site, prop, size, value, size_ret);
}

int MQT_DDSIM_QDMI_device_session_query_operation_property(
    MQT_DDSIM_QDMI_Device_Session session, MQT_DDSIM_QDMI_Operation operation,
    const size_t num_sites, const MQT_DDSIM_QDMI_Site* sites,
    const size_t num_params, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* size_ret) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->queryOperationProperty(operation, num_sites, sites,
                                         num_params, params, prop, size, value,
                                         size_ret);
}
// NOLINTEND(readability-identifier-naming)
