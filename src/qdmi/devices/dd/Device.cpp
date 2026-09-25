/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Device.cpp
/// The MQT QDMI device implementation for its DD-based simulator.

#include "qdmi/devices/dd/Device.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/Package.hpp"
#include "mqt_ddsim_qdmi/device.h"
#include "qdmi/common/Common.hpp"

#include "Worker.hpp"
#include "WorkerProtocol.hpp"

#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {
/// Result reconstruction never performs DD arithmetic. Keep unused matrix
/// storage and compute caches minimal while retaining normal vector storage.
constexpr dd::DDPackageConfig RESULT_PACKAGE_CONFIG{
    .utMatNumBucket = 1,
    .utMatInitialAllocationSize = 1,
    .ctVecAddNumBucket = 1,
    .ctMatAddNumBucket = 1,
    .ctVecAddMagNumBucket = 1,
    .ctMatAddMagNumBucket = 1,
    .ctVecConjNumBucket = 1,
    .ctMatConjTransNumBucket = 1,
    .ctMatVecMultNumBucket = 1,
    .ctMatMatMultNumBucket = 1,
    .ctVecKronNumBucket = 1,
    .ctMatKronNumBucket = 1,
    .ctMatTraceNumBucket = 1,
    .ctVecInnerProdNumBucket = 1,
};

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
  bool supportsArbitraryPositiveControls = false;
};

constexpr auto ARBITRARY_POSITIVE_CONTROLS_METADATA =
    "mqt.compiler-target.v1:arbitrary-positive-controls";

constexpr auto controllableOperation(const char* name, const size_t numSites,
                                     const size_t numParams) -> OperationInfo {
  return OperationInfo{
      .name = name,
      .numSites = numSites,
      .numParams = numParams,
      .supportsArbitraryPositiveControls = true,
  };
}

constexpr std::array OPERATIONS{
    OperationInfo{.name = "gphase", .numSites = 0, .numParams = 1},
    controllableOperation("i", 1, 0),
    controllableOperation("x", 1, 0),
    OperationInfo{.name = "cx", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "ccx", .numSites = 3, .numParams = 0},
    OperationInfo{
        .name = "mcx",
        .numSites = 0,
        .numParams = 0,
        .isVariadic = true,
    },
    controllableOperation("y", 1, 0),
    OperationInfo{.name = "cy", .numSites = 2, .numParams = 0},
    controllableOperation("z", 1, 0),
    OperationInfo{.name = "cz", .numSites = 2, .numParams = 0},
    OperationInfo{.name = "ccz", .numSites = 3, .numParams = 0},
    controllableOperation("h", 1, 0),
    OperationInfo{.name = "ch", .numSites = 2, .numParams = 0},
    controllableOperation("s", 1, 0),
    OperationInfo{.name = "cs", .numSites = 2, .numParams = 0},
    controllableOperation("sdg", 1, 0),
    OperationInfo{.name = "csdg", .numSites = 2, .numParams = 0},
    controllableOperation("t", 1, 0),
    controllableOperation("tdg", 1, 0),
    controllableOperation("sx", 1, 0),
    OperationInfo{.name = "csx", .numSites = 2, .numParams = 0},
    controllableOperation("sxdg", 1, 0),
    controllableOperation("r", 1, 2),
    controllableOperation("rx", 1, 1),
    OperationInfo{.name = "crx", .numSites = 2, .numParams = 1},
    controllableOperation("ry", 1, 1),
    OperationInfo{.name = "cry", .numSites = 2, .numParams = 1},
    controllableOperation("rz", 1, 1),
    OperationInfo{.name = "crz", .numSites = 2, .numParams = 1},
    controllableOperation("p", 1, 1),
    OperationInfo{.name = "cp", .numSites = 2, .numParams = 1},
    OperationInfo{
        .name = "mcp",
        .numSites = 0,
        .numParams = 1,
        .isVariadic = true,
    },
    OperationInfo{.name = "u1", .numSites = 1, .numParams = 1},
    OperationInfo{.name = "cu1", .numSites = 2, .numParams = 1},
    controllableOperation("u2", 1, 2),
    controllableOperation("u", 1, 3),
    OperationInfo{.name = "u3", .numSites = 1, .numParams = 3},
    OperationInfo{.name = "cu3", .numSites = 2, .numParams = 3},
    controllableOperation("swap", 2, 0),
    OperationInfo{.name = "cswap", .numSites = 3, .numParams = 0},
    controllableOperation("iswap", 2, 0),
    controllableOperation("dcx", 2, 0),
    controllableOperation("ecr", 2, 0),
    controllableOperation("rxx", 2, 1),
    controllableOperation("ryy", 2, 1),
    controllableOperation("rzz", 2, 1),
    controllableOperation("rzx", 2, 1),
    controllableOperation("xx_minus_yy", 2, 2),
    controllableOperation("xx_plus_yy", 2, 2),
    controllableOperation("rccx", 3, 0),
    OperationInfo{.name = "measure", .numSites = 1, .numParams = 0},
    OperationInfo{.name = "reset", .numSites = 1, .numParams = 0},
    OperationInfo{
        .name = "barrier",
        .numSites = 0,
        .numParams = 0,
        .isVariadic = true,
    },
    OperationInfo{
        .name = "if_else",
        .numSites = 0,
        .numParams = 0,
        .isVariadic = true,
    },
};

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

constexpr std::array SUPPORTED_PROGRAM_FORMATS = {
    QDMI_PROGRAM_FORMAT_QASM2,
    QDMI_PROGRAM_FORMAT_QASM3,
    QDMI_PROGRAM_FORMAT_QIRBASESTRING,
    QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
    QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
    QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
};

[[nodiscard]] auto reportEmptyResult(size_t* sizeRet) -> QDMI_STATUS {
  if (sizeRet != nullptr) {
    *sizeRet = 0;
  }
  return QDMI_SUCCESS;
}

} // namespace

namespace qdmi::dd {
struct ProgramResult {
  size_t numShots_ = 0;
  std::optional<std::string> qirOutput_;
  /// The measurement counts for the job
  std::map<std::string, std::size_t> counts_;

  /// Measurement outcomes in sampling order.
  std::vector<std::string> shots_;

  /// Owns an extracted state or an uncollapsed terminal-sampling state.
  /// A null package means that no state result is available.
  std::unique_ptr<::dd::Package> dd_;

  /// The retained state, valid while dd_ owns its nodes.
  ::dd::VectorDD stateVecDD_{};

  /// The state vector for the job (only available if no mid-circuit
  /// measurements are used).
  ::dd::CVec stateVec_;

  /// Sparse amplitudes in ascending basis-index order (only available if no
  /// mid-circuit measurements are used).
  std::vector<std::pair<size_t, std::complex<::dd::fp>>> stateVecSparse_;

  /// One-time flags to lazily materialize vectors in a thread-safe way
  std::once_flag stateVecOnce_;
  std::once_flag stateVecSparseOnce_;

  /// Translate counts to QDMI histogram
  auto getHistogram(QDMI_Job_Result result, size_t size, void* data,
                    size_t* sizeRet) -> QDMI_STATUS;

  /// Copy ordered outcomes to the QDMI comma-separated shot representation.
  auto getShots(size_t size, void* data, size_t* sizeRet) const -> QDMI_STATUS;

  /// Translate the state vector DD to a dense state vector for QDMI
  auto getStateVector(size_t size, void* data, size_t* sizeRet) -> QDMI_STATUS;

  /// Translate the state vector DD to sparse representations for QDMI
  auto getSparseResults(QDMI_Job_Result result, size_t size, void* data,
                        size_t* sizeRet) -> QDMI_STATUS;

  /// Translate the state vector DD to a dense vector of probabilities for QDMI
  auto getProbabilities(size_t size, void* data, size_t* sizeRet)
      -> QDMI_STATUS;

  auto getResults(QDMI_Job_Result result, size_t size, void* data,
                  size_t* sizeRet) -> QDMI_STATUS;
};

struct Execution {
  struct Program {
    QDMI_Job_Status status = QDMI_JOB_STATUS_CREATED;
    bool cancelRequested = false;
    std::shared_ptr<Worker> worker;
    std::unique_ptr<ProgramResult> result;
  };
  std::mutex mutex;
  std::condition_variable completed;
  QDMI_Job_Status status = QDMI_JOB_STATUS_CREATED;
  size_t remaining = 0;
  std::vector<Program> programs;

  /// Called under mutex once a program has stopped or a queued program is
  /// cancelled.
  void finish() {
    if (--remaining != 0) {
      return;
    }
    status = QDMI_JOB_STATUS_DONE;
    for (const auto& program : programs) {
      if (program.status == QDMI_JOB_STATUS_FAILED) {
        status = QDMI_JOB_STATUS_FAILED;
        break;
      }
      if (program.status == QDMI_JOB_STATUS_CANCELED) {
        status = QDMI_JOB_STATUS_CANCELED;
      }
    }
    completed.notify_all();
  }
};

namespace {
struct Executor {
  WorkerPool workers;
  llvm::DefaultThreadPool threads{llvm::heavyweight_hardware_concurrency()};
};
Executor& executor() {
  static Executor instance;
  return instance;
}
} // namespace
} // namespace qdmi::dd

namespace qdmi::dd {
Device::Device()
    : name_("MQT Core DDSIM QDMI Device"),
      qubitsNum_(std::numeric_limits<::dd::Qubit>::max()) {}
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
  /// Target facts that QDMI v1.3 cannot encode compactly.
  /// TODO(#2093): Remove this compatibility marker when QDMI standardizes
  /// explicit unrestricted connectivity and operation applicability.
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_CUSTOM1,
                      "mqt.compiler-target.v1:all-to-all-homogeneous", prop,
                      size, value, sizeRet)
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
  const auto& [operationName, operationNumSites, operationNumParams, isVariadic,
               supportsArbitraryPositiveControls] =
      *reinterpret_cast<const OperationInfo*>(operation);
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, operationName, prop, size,
                      value, sizeRet)
  if (!isVariadic) {
    if (sites != nullptr && operationNumSites != numSites) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                              operationNumSites, prop, size, value, sizeRet)
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            operationNumParams, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double, 1.0, prop,
                            size, value, sizeRet)
  if (supportsArbitraryPositiveControls) {
    ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_CUSTOM1,
                        ARBITRARY_POSITIVE_CONTROLS_METADATA, prop, size, value,
                        sizeRet)
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
MQT_DDSIM_QDMI_Device_Job_impl_d::MQT_DDSIM_QDMI_Device_Job_impl_d(
    MQT_DDSIM_QDMI_Device_Session_impl_d* session)
    : session_(session), id_(qdmi::dd::Device::get().generateUniqueID()),
      execution_(std::make_shared<qdmi::dd::Execution>()) {}
MQT_DDSIM_QDMI_Device_Job_impl_d::~MQT_DDSIM_QDMI_Device_Job_impl_d() {
  std::ignore = wait(0);
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::free() -> void {
  session_->freeDeviceJob(this);
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::setParameter(
    const QDMI_Device_Job_Parameter param, const size_t size, const void* value)
    -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_DEVICE_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->status != QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  switch (param) {
  case QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT:
    if (value != nullptr) {
      if (size != sizeof(QDMI_Program_Format)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      const auto format = *static_cast<const QDMI_Program_Format*>(value);
      if (IS_INVALID_ARGUMENT(format, QDMI_PROGRAM_FORMAT)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      if (std::ranges::find(SUPPORTED_PROGRAM_FORMATS, format) ==
          SUPPORTED_PROGRAM_FORMATS.end()) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
      if (format_ != format) {
        programs_.clear();
        execution_->programs.clear();
      }
      format_ = format;
    }
    return QDMI_SUCCESS;
  case QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM:
    if (value != nullptr) {
      if (size != sizeof(size_t)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      numShots_ = *static_cast<const size_t*>(value);
    }
    return QDMI_SUCCESS;
  case QDMI_DEVICE_JOB_PARAMETER_CUSTOM1:
    if (value == nullptr) {
      return QDMI_SUCCESS;
    }
    if (size != sizeof(int) || *static_cast<const int*>(value) <= 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    seed_ = *static_cast<const int*>(value);
    return QDMI_SUCCESS;
  case QDMI_DEVICE_JOB_PARAMETER_CUSTOM2:
    if (value == nullptr) {
      return QDMI_SUCCESS;
    }
    if (size != sizeof(bool)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    captureQIROutput_ = *static_cast<const bool*>(value);
    return QDMI_SUCCESS;
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::setPrograms(
    const QDMI_Program_Format* format, size_t count, const size_t* sizes,
    const void* const* programs) -> QDMI_STATUS {
  if (format == nullptr || count == 0 ||
      IS_INVALID_ARGUMENT(*format, QDMI_PROGRAM_FORMAT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->status != QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (std::ranges::find(SUPPORTED_PROGRAM_FORMATS, *format) ==
      SUPPORTED_PROGRAM_FORMATS.end()) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (programs == nullptr) {
    return QDMI_SUCCESS;
  }
  if (sizes == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const bool text = !(*format == QDMI_PROGRAM_FORMAT_QIRBASEMODULE ||
                      *format == QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE);
  std::vector<std::string> copied;
  copied.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (programs[i] == nullptr || sizes[i] == 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::string_view bytes(static_cast<const char*>(programs[i]), sizes[i]);
    if (text) {
      if (bytes.back() != '\0') {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      bytes.remove_suffix(1);
      if (bytes.find('\0') != std::string_view::npos) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
    }
    copied.emplace_back(bytes);
  }
  std::vector<qdmi::dd::Execution::Program> states(count);
  programs_ = std::move(copied);
  execution_->programs = std::move(states);
  format_ = *format;
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::queryProperty(
    const QDMI_Device_Job_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_DEVICE_JOB_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (programs_.empty() && (prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAM ||
                            prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM ||
                            prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAMSTATUSES)) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT &&
      format_ == QDMI_PROGRAM_FORMAT_MAX) {
    return QDMI_ERROR_BADSTATE;
  }
  const auto id = std::to_string(id_);
  ADD_STRING_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_ID, id.c_str(), prop, size,
                      value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT,
                            QDMI_Program_Format, format_, prop, size, value,
                            sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, size_t,
                            numShots_, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM, size_t,
                            programs_.size(), prop, size, value, sizeRet)
  if (prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAMSTATUSES) {
    std::vector<QDMI_Job_Status> statuses;
    statuses.reserve(execution_->programs.size());
    for (const auto& program : execution_->programs) {
      statuses.push_back(program.status);
    }
    ADD_LIST_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAMSTATUSES, QDMI_Job_Status,
                      statuses, prop, size, value, sizeRet)
  }
  if (programs_.size() == 1) {
    const auto& program = programs_.front();
    if (!(format_ == QDMI_PROGRAM_FORMAT_QIRBASEMODULE ||
          format_ == QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE)) {
      ADD_STRING_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAM, program.c_str(),
                          prop, size, value, sizeRet)
    } else {
      ADD_LIST_PROPERTY(QDMI_DEVICE_JOB_PROPERTY_PROGRAM, char, program, prop,
                        size, value, sizeRet)
    }
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::submit() -> QDMI_STATUS {
  const auto execution = execution_;
  const std::scoped_lock lock(execution->mutex);
  if (execution->status != QDMI_JOB_STATUS_CREATED || programs_.empty()) {
    return QDMI_ERROR_BADSTATE;
  }
  if (captureQIROutput_ &&
      (numShots_ == 0 || format_ == QDMI_PROGRAM_FORMAT_QASM2 ||
       format_ == QDMI_PROGRAM_FORMAT_QASM3)) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  auto& executor = qdmi::dd::executor();
  execution->remaining = programs_.size();
  execution->status = QDMI_JOB_STATUS_QUEUED;
  for (auto& program : execution->programs) {
    program.status = QDMI_JOB_STATUS_QUEUED;
  }
  for (size_t index = 0; index < programs_.size(); ++index) {
    try {
      qdmi::dd::WorkerRequest request{.format = static_cast<int32_t>(format_),
                                      .program = programs_[index],
                                      .shots = numShots_,
                                      .seed = seed_,
                                      .captureOutput = captureQIROutput_};
      executor.threads.async([execution, index, request = std::move(request),
                              &executor] {
        std::shared_ptr<qdmi::dd::Worker> worker;
        {
          const std::scoped_lock guard(execution->mutex);
          auto& program = execution->programs[index];
          if (program.status != QDMI_JOB_STATUS_QUEUED) {
            return;
          }
          worker = executor.workers.acquire();
          program.worker = worker;
          program.status = QDMI_JOB_STATUS_RUNNING;
          execution->status = QDMI_JOB_STATUS_RUNNING;
        }
        auto& device = qdmi::dd::Device::get();
        device.increaseRunningJobs();
        bool reusable = false;
        std::unique_ptr<qdmi::dd::ProgramResult> result;
        try {
          qdmi::dd::WorkerResponse response;
          reusable = worker && worker->execute(request, response);
          if (reusable && response.succeeded &&
              response.shots.size() == request.shots) {
            result = std::make_unique<qdmi::dd::ProgramResult>();
            result->numShots_ = request.shots;
            result->shots_ = std::move(response.shots);
            result->qirOutput_ = std::move(response.output);
            for (const auto& shot : result->shots_) {
              ++result->counts_[shot];
            }
            if (response.state) {
              result->dd_ = std::make_unique<dd::Package>(
                  response.qubits, RESULT_PACKAGE_CONFIG);
              std::istringstream bytes(*response.state, std::ios::binary);
              result->stateVecDD_ =
                  result->dd_->deserialize<dd::vNode>(bytes, true);
              const auto root = result->stateVecDD_;
              const auto qubits =
                  root.isTerminal() ? 0U : static_cast<uint32_t>(root.p->v) + 1;
              if (qubits != response.qubits ||
                  bytes.peek() != std::char_traits<char>::eof()) {
                result.reset();
                reusable = false;
              } else {
                result->dd_->incRef(root);
              }
            }
          }
        } catch (const std::exception& error) {
          result.reset();
          reusable = false;
          std::cerr << "DDSIM worker failed: " << error.what() << '\n';
        }
        {
          const std::scoped_lock guard(execution->mutex);
          auto& program = execution->programs[index];
          program.worker.reset();
          if (!program.cancelRequested) {
            program.status =
                result ? QDMI_JOB_STATUS_DONE : QDMI_JOB_STATUS_FAILED;
            program.result = std::move(result);
          } else {
            program.status = QDMI_JOB_STATUS_CANCELED;
            reusable = false;
          }
        }
        if (worker) {
          executor.workers.release(worker, reusable);
        }
        device.decreaseRunningJobs();
        {
          const std::scoped_lock guard(execution->mutex);
          execution->finish();
        }
      });
    } catch (const std::exception& error) {
      execution->programs[index].status = QDMI_JOB_STATUS_FAILED;
      execution->finish();
      std::cerr << "Could not schedule DDSIM program: " << error.what() << '\n';
    }
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::cancel() -> QDMI_STATUS {
  std::vector<std::shared_ptr<qdmi::dd::Worker>> workers;
  {
    const std::scoped_lock lock(execution_->mutex);
    if (execution_->status == QDMI_JOB_STATUS_DONE ||
        execution_->status == QDMI_JOB_STATUS_FAILED) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (execution_->status == QDMI_JOB_STATUS_CREATED) {
      for (auto& program : execution_->programs) {
        program.status = QDMI_JOB_STATUS_CANCELED;
      }
      execution_->status = QDMI_JOB_STATUS_CANCELED;
      return QDMI_SUCCESS;
    }
    for (auto& program : execution_->programs) {
      if (program.status == QDMI_JOB_STATUS_QUEUED) {
        program.status = QDMI_JOB_STATUS_CANCELED;
        execution_->finish();
      } else if (program.status == QDMI_JOB_STATUS_RUNNING) {
        program.cancelRequested = true;
        if (program.worker) {
          workers.push_back(program.worker);
        }
      }
    }
  }
  for (const auto& worker : workers) {
    worker->terminate();
  }
  return wait(0);
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::check(QDMI_Job_Status* status) const
    -> QDMI_STATUS {
  if (status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  *status = execution_->status;
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::wait(size_t timeout) const
    -> QDMI_STATUS {
  std::unique_lock lock(execution_->mutex);
  if (execution_->status == QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  const auto complete = [&] { return execution_->remaining == 0; };
  if (timeout == 0) {
    execution_->completed.wait(lock, complete);
  } else if (!execution_->completed.wait_for(
                 lock, std::chrono::seconds(timeout), complete)) {
    return QDMI_ERROR_TIMEOUT;
  }
  return QDMI_SUCCESS;
}
auto qdmi::dd::ProgramResult::getShots(const size_t size, void* data,
                                       size_t* sizeRet) const -> QDMI_STATUS {
  const size_t required =
      std::accumulate(shots_.begin(), shots_.end(), size_t{0},
                      [](const size_t total, const auto& shot) {
                        return total + shot.size() + 1;
                      });
  if (sizeRet != nullptr) {
    *sizeRet = required;
  }
  if (data != nullptr) {
    if (size < required) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto output = std::span(static_cast<char*>(data), required);
    for (const auto& shot : shots_) {
      std::ranges::copy(shot, output.begin());
      output[shot.size()] = ',';
      output = output.subspan(shot.size() + 1);
    }
    if (required > 0) {
      std::span(static_cast<char*>(data), required).back() = '\0';
    }
  }
  return QDMI_SUCCESS;
}
auto qdmi::dd::ProgramResult::getHistogram(const QDMI_Job_Result result,
                                           const size_t size, void* data,
                                           size_t* sizeRet) -> QDMI_STATUS {
  if (counts_.size() == 1 && counts_.begin()->first.empty()) {
    return reportEmptyResult(sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_KEYS) {
    const size_t reqSize =
        std::accumulate(counts_.begin(), counts_.end(), size_t{0},
                        [](const size_t total, const auto& entry) {
                          return total + entry.first.size() + 1;
                        });
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
      // NOLINTNEXTLINE(misc-const-correctness): fills a mutable output buffer.
      auto* dataPtr = static_cast<size_t*>(data);
      for (const auto& count : counts_ | std::views::values) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *dataPtr++ = count;
      }
    }
  }
  return QDMI_SUCCESS;
}
auto qdmi::dd::ProgramResult::getStateVector(const size_t size, void* data,
                                             size_t* sizeRet) -> QDMI_STATUS {
  const auto numQubits = stateVecDD_.isTerminal()
                             ? 0U
                             : static_cast<size_t>(stateVecDD_.p->v) + 1U;
  constexpr size_t elementSize = 2 * sizeof(double);
  if (numQubits >= std::numeric_limits<size_t>::digits ||
      (std::numeric_limits<size_t>::max() >> numQubits) < elementSize) {
    return QDMI_ERROR_OUTOFMEM;
  }
  const size_t dimension = size_t{1} << numQubits;
  const size_t reqSize = dimension * elementSize;
  if (data != nullptr) {
    if (size < reqSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (dimension > stateVec_.max_size()) {
      return QDMI_ERROR_OUTOFMEM;
    }
    std::call_once(stateVecOnce_,
                   [this] { stateVec_ = stateVecDD_.getVector(); });
    std::memcpy(data, stateVec_.data(), reqSize);
  }
  if (sizeRet != nullptr) {
    *sizeRet = reqSize;
  }
  return QDMI_SUCCESS;
}
auto qdmi::dd::ProgramResult::getSparseResults(const QDMI_Job_Result result,
                                               const size_t size, void* data,
                                               size_t* sizeRet) -> QDMI_STATUS {
  const auto numQubits = stateVecDD_.isTerminal()
                             ? 0U
                             : static_cast<size_t>(stateVecDD_.p->v) + 1U;
  if (numQubits > std::numeric_limits<size_t>::digits) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  std::call_once(stateVecSparseOnce_, [this] {
    const auto sparse = stateVecDD_.getSparseVector();
    stateVecSparse_.assign(sparse.begin(), sparse.end());
    std::ranges::sort(stateVecSparse_, {},
                      &decltype(stateVecSparse_)::value_type::first);
  });
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
      // NOLINTNEXTLINE(misc-const-correctness): fills a mutable output buffer.
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
      // NOLINTNEXTLINE(misc-const-correctness): fills a mutable output buffer.
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
auto qdmi::dd::ProgramResult::getProbabilities(const size_t size, void* data,
                                               size_t* sizeRet) -> QDMI_STATUS {
  const auto numQubits = stateVecDD_.isTerminal()
                             ? 0U
                             : static_cast<size_t>(stateVecDD_.p->v) + 1U;
  constexpr size_t elementSize = sizeof(double);
  if (numQubits >= std::numeric_limits<size_t>::digits ||
      (std::numeric_limits<size_t>::max() >> numQubits) < elementSize) {
    return QDMI_ERROR_OUTOFMEM;
  }
  const size_t dimension = size_t{1} << numQubits;
  const size_t reqSize = dimension * elementSize;
  if (data != nullptr) {
    if (size < reqSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (dimension > stateVec_.max_size()) {
      return QDMI_ERROR_OUTOFMEM;
    }
    std::call_once(stateVecOnce_,
                   [this] { stateVec_ = stateVecDD_.getVector(); });
    // NOLINTNEXTLINE(misc-const-correctness): fills a mutable output buffer.
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
auto qdmi::dd::ProgramResult::getResults(const QDMI_Job_Result result,
                                         const size_t size, void* data,
                                         size_t* sizeRet) -> QDMI_STATUS {
  if (IS_INVALID_ARGUMENT(result, QDMI_JOB_RESULT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (qirOutput_) {
    ADD_STRING_PROPERTY(QDMI_JOB_RESULT_CUSTOM1, qirOutput_->c_str(), result,
                        size, data, sizeRet)
  }
  switch (result) {
  case QDMI_JOB_RESULT_SHOTS:
    if (numShots_ == 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getShots(size, data, sizeRet);
  case QDMI_JOB_RESULT_HIST_KEYS:
  case QDMI_JOB_RESULT_HIST_VALUES:
    if (numShots_ == 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getHistogram(result, size, data, sizeRet);
  case QDMI_JOB_RESULT_STATEVECTOR_DENSE:
    if (!dd_) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    return getStateVector(size, data, sizeRet);
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS:
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES:
    if (!dd_) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    return getSparseResults(result, size, data, sizeRet);
  case QDMI_JOB_RESULT_PROBABILITIES_DENSE:
    if (!dd_) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    return getProbabilities(size, data, sizeRet);
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}

auto MQT_DDSIM_QDMI_Device_Job_impl_d::getResults(size_t programIndex,
                                                  QDMI_Job_Result result,
                                                  size_t size, void* data,
                                                  size_t* sizeRet)
    -> QDMI_STATUS {
  if (IS_INVALID_ARGUMENT(result, QDMI_JOB_RESULT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  qdmi::dd::ProgramResult* programResult = nullptr;
  {
    const std::scoped_lock lock(execution_->mutex);
    if (execution_->programs.empty()) {
      return QDMI_ERROR_BADSTATE;
    }
    if (programIndex >= execution_->programs.size()) {
      return QDMI_ERROR_OUTOFRANGE;
    }
    const auto& program = execution_->programs[programIndex];
    if (program.status != QDMI_JOB_STATUS_DONE) {
      return QDMI_ERROR_BADSTATE;
    }
    programResult = program.result.get();
  }
  return programResult->getResults(result, size, data, sizeRet);
}

// QDMI uses a different naming convention for its C interface functions
// NOLINTBEGIN(readability-identifier-naming)
int MQT_DDSIM_QDMI_device_initialize() {
  // ensure the singleton is initialized
  std::ignore = qdmi::dd::Device::get();
  qdmi::dd::executor().workers.initialize();
  return QDMI_SUCCESS;
}

int MQT_DDSIM_QDMI_device_finalize() {
  auto& executor = qdmi::dd::executor();
  executor.workers.shutdown();
  executor.threads.wait();
  return QDMI_SUCCESS;
}

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

int MQT_DDSIM_QDMI_device_session_retrieve_device_job_by_id(
    [[maybe_unused]] MQT_DDSIM_QDMI_Device_Session session,
    [[maybe_unused]] const char* jobId, MQT_DDSIM_QDMI_Device_Job* /*job*/) {
  return QDMI_ERROR_NOTSUPPORTED;
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

int MQT_DDSIM_QDMI_device_job_set_programs(MQT_DDSIM_QDMI_Device_Job job,
                                           const QDMI_Program_Format* format,
                                           size_t count, const size_t* sizes,
                                           const void* const* programs) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  try {
    return job->setPrograms(format, count, sizes, programs);
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  }
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
                                          size_t programIndex,
                                          QDMI_Job_Result result,
                                          const size_t size, void* data,
                                          size_t* size_ret) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getResults(programIndex, result, size, data, size_ret);
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
