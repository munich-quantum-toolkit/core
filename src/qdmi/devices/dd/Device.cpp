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
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "mqt_ddsim_qdmi/device.h"
#include "qdmi/common/Common.hpp"

#include "Worker.hpp"
#include "WorkerProtocol.hpp"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <charconv>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <semaphore>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {
template <class Callable>
auto guardDeviceCall(Callable&& callable) noexcept -> int {
  try {
    return std::forward<Callable>(callable)();
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
}

/// Result reconstruction never performs DD arithmetic.
///
/// Keep unused matrix storage and compute caches minimal while retaining normal
/// vector storage.
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
namespace {
struct ProgramResult {
  std::optional<std::string> qirOutput_;
  std::optional<OutputTemplate> outputTemplate_;
  /// Comma-terminated QIR outcomes aligned with output records.
  std::string shots_;
  /// Histogram outcomes stay compact until serialized into a SHOTS buffer.
  std::map<std::string, size_t> compactCounts_;

  /// QIR histogram keys borrow from shots_.
  std::map<std::string_view, size_t> counts_;
  std::once_flag countsOnce_;

  /// Owns an extracted state or an uncollapsed terminal-sampling state.
  ///
  /// A null package means that no state result is available.
  std::unique_ptr<::dd::Package> dd_;

  /// The retained state, valid while dd_ owns its nodes.
  ::dd::VectorDD stateVecDD_{};

  /// Number of amplitudes visited by the sparse result traversal.
  size_t sparseAmplitudeCount_ = 0;
  std::once_flag sparseAmplitudeCountOnce_;

  /// Materialize the QIR stream directly into the caller's buffer.
  auto getQIROutput(size_t size, void* data, size_t* sizeRet) const
      -> QDMI_STATUS;

  /// Translate counts to QDMI histogram
  auto getHistogram(QDMI_Job_Result result, size_t size, void* data,
                    size_t* sizeRet) -> QDMI_STATUS;

  /// Copy outcomes to the QDMI comma-separated shot representation.
  auto getShots(size_t size, void* data, size_t* sizeRet) const -> QDMI_STATUS;

  /// Write dense amplitudes or probabilities into the caller's buffer.
  auto getDenseResults(QDMI_Job_Result result, size_t size, void* data,
                       size_t* sizeRet) const -> QDMI_STATUS;

  /// Translate the state vector DD to sparse representations for QDMI
  auto getSparseResults(QDMI_Job_Result result, size_t size, void* data,
                        size_t* sizeRet) -> QDMI_STATUS;

  auto getResults(QDMI_Job_Result result, size_t size, void* data,
                  size_t* sizeRet) -> QDMI_STATUS;
};

} // namespace

struct Execution {
  struct Program {
    std::string code;
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
  const size_t limit = [] {
    const size_t hardware = std::max(
        1U, llvm::heavyweight_hardware_concurrency().compute_thread_count());
    if (const char* raw = std::getenv("MQT_CORE_DD_WORKER_BUDGET")) {
      size_t parsed = 0;
      const auto* last =
          std::next(raw, static_cast<ptrdiff_t>(std::strlen(raw)));
      const auto [end, error] = std::from_chars(raw, last, parsed);
      if (error == std::errc{} && end == last && parsed > 0) {
        return std::min(parsed, hardware);
      }
    }
    return hardware;
  }();
  std::counting_semaphore<> slots{static_cast<ptrdiff_t>(limit)};
  WorkerPool workers;
  llvm::DefaultThreadPool threads{
      llvm::heavyweight_hardware_concurrency(static_cast<unsigned>(limit))};

  ~Executor() { workers.shutdown(); }
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
  *session = nullptr;
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
    std::unique_lock lock(sessionsMutex_);
    const auto removed = sessions_.extract(session);
    lock.unlock(); // Job destruction can wait for running programs.
  }
}
auto Device::queryProperty(const QDMI_Device_Property prop, const size_t size,
                           void* value, size_t* sizeRet) const -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      QDMI_IS_INVALID_ENUM_VALUE(prop, QDMI_DEVICE_PROPERTY)) {
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
                            runningJobs_.load() == 0 ? QDMI_DEVICE_STATUS_IDLE
                                                     : QDMI_DEVICE_STATUS_BUSY,
                            prop, size, value, sizeRet)
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
  // Target facts that QDMI v1.3 cannot encode compactly.
  // TODO(#2093): Remove this compatibility marker when QDMI standardizes
  // explicit unrestricted connectivity and operation applicability.
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
auto Device::increaseRunningJobs() -> void { ++runningJobs_; }
auto Device::decreaseRunningJobs() -> void { --runningJobs_; }

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
      QDMI_IS_INVALID_ENUM_VALUE(param, QDMI_DEVICE_SESSION_PARAMETER)) {
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
  *job = nullptr;
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
    std::unique_lock lock(jobsMutex_);
    const auto removed = jobs_.extract(job);
    lock.unlock(); // Do not block other jobs while this one finishes.
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
      QDMI_IS_INVALID_ENUM_VALUE(prop, QDMI_SITE_PROPERTY)) {
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
      QDMI_IS_INVALID_ENUM_VALUE(prop, QDMI_OPERATION_PROPERTY)) {
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
      QDMI_IS_INVALID_ENUM_VALUE(param, QDMI_DEVICE_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->status != QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  switch (param) {
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
  case QDMI_DEVICE_JOB_PARAMETER_CUSTOM3:
    if (value == nullptr) {
      return QDMI_SUCCESS;
    }
    if (size == sizeof(int)) {
      const auto requested = *static_cast<const int*>(value);
      if (requested < 0) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      maxWorkers_ = static_cast<size_t>(requested);
      return QDMI_SUCCESS;
    }
    if (size == sizeof(size_t)) {
      maxWorkers_ = *static_cast<const size_t*>(value);
      return QDMI_SUCCESS;
    }
    return QDMI_ERROR_INVALIDARGUMENT;
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::setPrograms(
    const QDMI_Program_Format format, size_t count, const size_t* sizes,
    const void* const* programs) -> QDMI_STATUS {
  if (count == 0 || QDMI_IS_INVALID_ENUM_VALUE(format, QDMI_PROGRAM_FORMAT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->status != QDMI_JOB_STATUS_CREATED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (std::ranges::find(SUPPORTED_PROGRAM_FORMATS, format) ==
      SUPPORTED_PROGRAM_FORMATS.end()) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (programs == nullptr) {
    return QDMI_SUCCESS;
  }
  if (sizes == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const bool text = format != QDMI_PROGRAM_FORMAT_QIRBASEMODULE &&
                    format != QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE;
  const std::span programSizes{sizes, count};
  const std::span programPointers{programs, count};
  std::vector<qdmi::dd::Execution::Program> copied;
  copied.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (programPointers[i] == nullptr || programSizes[i] == 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::string_view bytes(static_cast<const char*>(programPointers[i]),
                           programSizes[i]);
    if (text) {
      if (bytes.back() != '\0') {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      bytes.remove_suffix(1);
      if (bytes.find('\0') != std::string_view::npos) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
    }
    copied.push_back({.code = std::string(bytes)});
  }
  execution_->programs = std::move(copied);
  format_ = format;
  return QDMI_SUCCESS;
}

auto MQT_DDSIM_QDMI_Device_Job_impl_d::getProgram(const size_t programIndex,
                                                  const size_t size, void* data,
                                                  size_t* sizeRet) const
    -> QDMI_STATUS {
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->programs.empty()) {
    return QDMI_ERROR_BADSTATE;
  }
  if (programIndex >= execution_->programs.size()) {
    return QDMI_ERROR_OUTOFRANGE;
  }
  const auto& program = execution_->programs[programIndex].code;
  const bool text = format_ != QDMI_PROGRAM_FORMAT_QIRBASEMODULE &&
                    format_ != QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE;
  const size_t requiredSize = program.size() + (text ? 1U : 0U);
  if (data != nullptr) {
    if (size < requiredSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::memcpy(data, program.data(), program.size());
    if (text) {
      std::span(static_cast<char*>(data), requiredSize).back() = '\0';
    }
  }
  if (sizeRet != nullptr) {
    *sizeRet = requiredSize;
  }
  return QDMI_SUCCESS;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::getProgramStatus(
    const size_t programIndex, QDMI_Job_Status* status) const -> QDMI_STATUS {
  if (status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->programs.empty()) {
    return QDMI_ERROR_BADSTATE;
  }
  if (programIndex >= execution_->programs.size()) {
    return QDMI_ERROR_OUTOFRANGE;
  }
  *status = execution_->programs[programIndex].status;
  return QDMI_SUCCESS;
}

auto MQT_DDSIM_QDMI_Device_Job_impl_d::queryProperty(
    const QDMI_Device_Job_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> QDMI_STATUS {
  if ((value != nullptr && size == 0) ||
      QDMI_IS_INVALID_ENUM_VALUE(prop, QDMI_DEVICE_JOB_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::scoped_lock lock(execution_->mutex);
  if (execution_->programs.empty() &&
      prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM) {
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
                            execution_->programs.size(), prop, size, value,
                            sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_DDSIM_QDMI_Device_Job_impl_d::submit() -> QDMI_STATUS {
  const auto execution = execution_;
  const std::scoped_lock lock(execution->mutex);
  if (execution->status != QDMI_JOB_STATUS_CREATED ||
      execution->programs.empty()) {
    return QDMI_ERROR_BADSTATE;
  }
  if (captureQIROutput_ &&
      (numShots_ == 0 || format_ == QDMI_PROGRAM_FORMAT_QASM2 ||
       format_ == QDMI_PROGRAM_FORMAT_QASM3)) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  auto& executor = qdmi::dd::executor();
  const size_t perProgramLimit =
      std::max(size_t{1}, executor.limit / execution->programs.size());
  const size_t desired = std::min({
      maxWorkers_ == 0 ? perProgramLimit : maxWorkers_,
      perProgramLimit,
      std::max(size_t{1}, numShots_),
  });
  execution->remaining = execution->programs.size();
  execution->status = QDMI_JOB_STATUS_QUEUED;
  for (auto& program : execution->programs) {
    program.status = QDMI_JOB_STATUS_QUEUED;
  }
  for (size_t index = 0; index < execution->programs.size(); ++index) {
    try {
      qdmi::dd::WorkerRequest request{
          .format = static_cast<int32_t>(format_),
          .shots = numShots_,
          .seed = seed_,
          .captureOutput = captureQIROutput_,
          .maxWorkers = desired,
          .automaticWorkers = maxWorkers_ == 0,
      };
      executor.threads.async([pending = std::weak_ptr(execution), index,
                              request, &executor]() mutable {
        executor.slots.acquire();
        size_t grant = 1;
        const llvm::scope_exit releaseSlots(
            [&] { executor.slots.release(static_cast<ptrdiff_t>(grant)); });
        const auto execution = pending.lock();
        if (!execution) {
          return;
        }
        std::shared_ptr<qdmi::dd::Worker> worker;
        {
          const std::scoped_lock guard(execution->mutex);
          auto& program = execution->programs[index];
          if (program.status != QDMI_JOB_STATUS_QUEUED) {
            return;
          }
          request.program = program.code;
          program.status = QDMI_JOB_STATUS_RUNNING;
          execution->status = QDMI_JOB_STATUS_RUNNING;
        }
        auto& device = qdmi::dd::Device::get();
        device.increaseRunningJobs();
        bool reusable = false;
        std::unique_ptr<qdmi::dd::ProgramResult> result;
        const llvm::scope_exit completed([&] {
          {
            const std::scoped_lock guard(execution->mutex);
            auto& program = execution->programs[index];
            program.worker.reset();
            reusable = reusable && !program.cancelRequested;
          }
          if (worker) {
            try {
              executor.workers.release(worker, reusable);
            } catch (const std::exception& error) {
              result.reset();
              worker->terminate();
              std::cerr << "Could not release DDSIM worker: " << error.what()
                        << '\n';
            }
          }
          device.decreaseRunningJobs();
          const std::scoped_lock guard(execution->mutex);
          auto& program = execution->programs[index];
          if (program.cancelRequested) {
            program.status = QDMI_JOB_STATUS_CANCELED;
          } else {
            program.status =
                result ? QDMI_JOB_STATUS_DONE : QDMI_JOB_STATUS_FAILED;
            program.result = std::move(result);
          }
          execution->finish();
        });
        try {
          {
            const std::scoped_lock guard(execution->mutex);
            auto& program = execution->programs[index];
            if (program.cancelRequested) {
              return;
            }
            worker = executor.workers.acquire();
            program.worker = worker;
          }
          qdmi::dd::WorkerResponse response;
          reusable =
              worker && worker->execute(request, response, [&](size_t desired) {
                while (grant < desired && executor.slots.try_acquire()) {
                  ++grant;
                }
                return grant;
              });
          executor.slots.release(static_cast<ptrdiff_t>(grant - 1));
          grant = 1;
          uint64_t remaining = request.shots;
          bool validCounts = response.shotCount == 0;
          for (const auto& entry : response.counts) {
            const auto count = entry.second;
            if (count == 0 || count > remaining) {
              validCounts = false;
              break;
            }
            remaining -= count;
          }
          validCounts &= remaining == 0;
          const bool qasm = request.format == QDMI_PROGRAM_FORMAT_QASM2 ||
                            request.format == QDMI_PROGRAM_FORMAT_QASM3;
          const bool compact = !response.output && !response.outputTemplate &&
                               response.shotCount == 0;
          const bool validSamples =
              (qasm || compact) && request.shots != 0
                  ? validCounts
                  : response.counts.empty() &&
                        response.shotCount == request.shots;
          if (reusable && response.succeeded && validSamples) {
            result = std::make_unique<qdmi::dd::ProgramResult>();
            result->shots_ = std::move(response.shots);
            result->compactCounts_ = std::move(response.counts);
            result->qirOutput_ = std::move(response.output);
            result->outputTemplate_ = std::move(response.outputTemplate);
            if (response.state) {
              result->dd_ = std::make_unique<dd::Package>(
                  response.qubits, RESULT_PACKAGE_CONFIG);
              std::istringstream bytes(std::move(*response.state),
                                       std::ios::binary);
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
  size_t required = shots_.size();
  if (!compactCounts_.empty()) {
    for (const auto& [shot, count] : compactCounts_) {
      const auto width = shot.size() + 1;
      if (count > (std::numeric_limits<size_t>::max() - required) / width) {
        return QDMI_ERROR_OUTOFMEM;
      }
      required += count * width;
    }
  }
  if (sizeRet != nullptr) {
    *sizeRet = required;
  }
  if (data != nullptr) {
    if (size < required) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto output = std::span(static_cast<char*>(data), required);
    const auto writeShot = [&](std::string_view shot) {
      std::ranges::copy(shot, output.begin());
      output[shot.size()] = ',';
      output = output.subspan(shot.size() + 1);
    };
    if (compactCounts_.empty()) {
      std::ranges::copy(shots_, output.begin());
    } else {
      for (const auto& [shot, count] : compactCounts_) {
        for (size_t i = 0; i < count; ++i) {
          writeShot(shot);
        }
      }
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
  if (compactCounts_.empty()) {
    std::call_once(countsOnce_, [this] {
      decltype(counts_) counts;
      std::string_view remaining(shots_);
      while (!remaining.empty()) {
        const auto delimiter = remaining.find(',');
        ++counts[remaining.substr(0, delimiter)];
        remaining.remove_prefix(delimiter + 1);
      }
      counts_ = std::move(counts);
    });
  }
  const auto writeHistogram = [&](const auto& counts) -> QDMI_STATUS {
    if (counts.size() == 1 && counts.begin()->first.empty()) {
      return reportEmptyResult(sizeRet);
    }
    if (result == QDMI_JOB_RESULT_HIST_KEYS) {
      const size_t reqSize =
          std::accumulate(counts.begin(), counts.end(), size_t{0},
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
        for (const auto& bitstring : counts | std::views::keys) {
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
      const size_t reqSize = counts.size() * sizeof(size_t);
      if (sizeRet != nullptr) {
        *sizeRet = reqSize;
      }
      if (data != nullptr) {
        if (size < reqSize) {
          return QDMI_ERROR_INVALIDARGUMENT;
        }
        // NOLINTNEXTLINE(misc-const-correctness): writes into caller buffer.
        auto* dataPtr = static_cast<size_t*>(data);
        for (const auto& count : counts | std::views::values) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          *dataPtr++ = count;
        }
      }
    }
    return QDMI_SUCCESS;
  };
  return compactCounts_.empty() ? writeHistogram(counts_)
                                : writeHistogram(compactCounts_);
}
auto qdmi::dd::ProgramResult::getDenseResults(const QDMI_Job_Result result,
                                              const size_t size, void* data,
                                              size_t* sizeRet) const
    -> QDMI_STATUS {
  const auto numQubits = stateVecDD_.isTerminal()
                             ? 0U
                             : static_cast<size_t>(stateVecDD_.p->v) + 1U;
  const bool probabilities = result == QDMI_JOB_RESULT_PROBABILITIES_DENSE;
  const size_t components = probabilities ? 1 : 2;
  const size_t elementSize = components * sizeof(double);
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
    const std::span output(static_cast<double*>(data), dimension * components);
    std::ranges::fill(output, 0.);
    ::dd::traverseVector(
        stateVecDD_,
        [&output, probabilities](const size_t index,
                                 const std::complex<::dd::fp>& amplitude) {
          if (probabilities) {
            output[index] = std::norm(amplitude);
          } else {
            output[2 * index] = amplitude.real();
            output[(2 * index) + 1] = amplitude.imag();
          }
        });
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
  std::call_once(sparseAmplitudeCountOnce_, [this] {
    // shortcut: other shared paths repeat; memoize only with a memory bound.
    const auto count = [](const auto& self, const ::dd::vNode* node) -> size_t {
      if (::dd::vNode::isTerminal(node)) {
        return 1;
      }
      const auto& low = node->e[0];
      const auto& high = node->e[1];
      if (low.w.exactlyZero()) {
        return high.w.exactlyZero() ? 0 : self(self, high.p);
      }
      const auto left = self(self, low.p);
      if (high.w.exactlyZero()) {
        return left;
      }
      const auto right = low.p == high.p ? left : self(self, high.p);
      if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::bad_alloc{};
      }
      return left + right;
    };
    sparseAmplitudeCount_ = count(count, stateVecDD_.p);
  });
  if (sparseAmplitudeCount_ == 0) {
    return reportEmptyResult(sizeRet);
  }
  const bool keys = result == QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS ||
                    result == QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS;
  const bool complex = result == QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES;
  const size_t elementSize =
      keys ? numQubits + 1 : (complex ? 2U : 1U) * sizeof(double);
  if (sparseAmplitudeCount_ >
      std::numeric_limits<size_t>::max() / elementSize) {
    return QDMI_ERROR_OUTOFMEM;
  }
  const size_t reqSize = sparseAmplitudeCount_ * elementSize;
  if (data != nullptr) {
    if (size < reqSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (keys) {
      auto* dataPtr = static_cast<char*>(data);
      ::dd::traverseVector(
          stateVecDD_, [&dataPtr, numQubits](const size_t index,
                                             const std::complex<::dd::fp>&) {
            for (size_t j = numQubits; j > 0; --j) {
              // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
              *dataPtr++ = ((index >> (j - 1)) & 1U) != 0U ? '1' : '0';
            }
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            *dataPtr++ = ',';
          });
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      *(dataPtr - 1) = '\0'; // Replace last comma with null terminator
    } else if (complex) {
      // NOLINTNEXTLINE(misc-const-correctness): fills a mutable output buffer.
      auto* dataPtr = static_cast<double*>(data);
      ::dd::traverseVector(
          stateVecDD_,
          [&dataPtr](const size_t, const std::complex<::dd::fp>& amplitude) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            *dataPtr++ = amplitude.real();
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            *dataPtr++ = amplitude.imag();
          });
    } else {
      // NOLINTNEXTLINE(misc-const-correctness): fills a mutable output buffer.
      auto* dataPtr = static_cast<double*>(data);
      ::dd::traverseVector(
          stateVecDD_,
          [&dataPtr](const size_t, const std::complex<::dd::fp>& amplitude) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            *dataPtr++ = std::norm(amplitude);
          });
    }
  }
  if (sizeRet != nullptr) {
    *sizeRet = reqSize;
  }
  return QDMI_SUCCESS;
}
auto qdmi::dd::ProgramResult::getQIROutput(const size_t size, void* data,
                                           size_t* sizeRet) const
    -> QDMI_STATUS {
  if (!qirOutput_) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  size_t required = qirOutput_->size() + 1;
  if (outputTemplate_) {
    const size_t recordSize = outputTemplate_->text.size();
    const auto shots =
        shots_.size() / (outputTemplate_->resultOffsets.size() + 1);
    if (recordSize != 0 &&
        shots > (std::numeric_limits<size_t>::max() - required) / recordSize) {
      return QDMI_ERROR_OUTOFMEM;
    }
    required += shots * recordSize;
  }
  if (sizeRet != nullptr) {
    *sizeRet = required;
  }
  if (data == nullptr) {
    return QDMI_SUCCESS;
  }
  if (size < required) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto output = std::span(static_cast<char*>(data), required);
  std::ranges::copy(*qirOutput_, output.begin());
  output = output.subspan(qirOutput_->size());
  if (outputTemplate_) {
    const auto& record = *outputTemplate_;
    std::string_view remaining(shots_);
    const auto width = record.resultOffsets.size();
    while (!remaining.empty()) {
      std::ranges::copy(record.text, output.begin());
      for (size_t i = 0; i < record.resultOffsets.size(); ++i) {
        // QDMI bitstrings have the reverse of QIR recording order.
        output[record.resultOffsets[i]] = remaining[width - 1 - i];
      }
      output = output.subspan(record.text.size());
      remaining.remove_prefix(width + 1);
    }
  }
  output.front() = '\0';
  return QDMI_SUCCESS;
}

auto qdmi::dd::ProgramResult::getResults(const QDMI_Job_Result result,
                                         const size_t size, void* data,
                                         size_t* sizeRet) -> QDMI_STATUS {
  switch (result) {
  case QDMI_JOB_RESULT_CUSTOM1:
    return getQIROutput(size, data, sizeRet);
  case QDMI_JOB_RESULT_SHOTS:
    if (shots_.empty() && compactCounts_.empty()) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getShots(size, data, sizeRet);
  case QDMI_JOB_RESULT_HIST_KEYS:
  case QDMI_JOB_RESULT_HIST_VALUES:
    if (shots_.empty() && compactCounts_.empty()) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    return getHistogram(result, size, data, sizeRet);
  case QDMI_JOB_RESULT_STATEVECTOR_DENSE:
  case QDMI_JOB_RESULT_PROBABILITIES_DENSE:
    if (!dd_) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    return getDenseResults(result, size, data, sizeRet);
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS:
  case QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS:
  case QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES:
    if (!dd_) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    return getSparseResults(result, size, data, sizeRet);
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}

auto MQT_DDSIM_QDMI_Device_Job_impl_d::getResults(size_t programIndex,
                                                  QDMI_Job_Result result,
                                                  size_t size, void* data,
                                                  size_t* sizeRet)
    -> QDMI_STATUS {
  if (QDMI_IS_INVALID_ENUM_VALUE(result, QDMI_JOB_RESULT)) {
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
  return guardDeviceCall([] {
    std::ignore = qdmi::dd::Device::get();
    qdmi::dd::executor().workers.initialize();
    return QDMI_SUCCESS;
  });
}

int MQT_DDSIM_QDMI_device_finalize() {
  auto& executor = qdmi::dd::executor();
  executor.workers.shutdown();
  executor.threads.wait();
  return QDMI_SUCCESS;
}

int MQT_DDSIM_QDMI_device_session_alloc(
    MQT_DDSIM_QDMI_Device_Session* session) {
  return guardDeviceCall(
      [&] { return qdmi::dd::Device::get().sessionAlloc(session); });
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
  return guardDeviceCall([&] { return session->createDeviceJob(job); });
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
                                           const QDMI_Program_Format format,
                                           size_t count, const size_t* sizes,
                                           const void* const* programs) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return guardDeviceCall(
      [&] { return job->setPrograms(format, count, sizes, programs); });
}

int MQT_DDSIM_QDMI_device_job_get_program(MQT_DDSIM_QDMI_Device_Job job,
                                          const size_t programIndex,
                                          const size_t size, void* data,
                                          size_t* sizeRet) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                        : job->getProgram(programIndex, size, data, sizeRet);
}

int MQT_DDSIM_QDMI_device_job_get_program_status(MQT_DDSIM_QDMI_Device_Job job,
                                                 const size_t programIndex,
                                                 QDMI_Job_Status* status) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT
                        : job->getProgramStatus(programIndex, status);
}

int MQT_DDSIM_QDMI_device_job_query_property(
    MQT_DDSIM_QDMI_Device_Job job, const QDMI_Device_Job_Property prop,
    const size_t size, void* value, size_t* size_ret) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return guardDeviceCall(
      [&] { return job->queryProperty(prop, size, value, size_ret); });
}

int MQT_DDSIM_QDMI_device_job_submit(MQT_DDSIM_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return guardDeviceCall([&] { return job->submit(); });
}

int MQT_DDSIM_QDMI_device_job_cancel(MQT_DDSIM_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return guardDeviceCall([&] { return job->cancel(); });
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
  return guardDeviceCall([&] {
    return job->getResults(programIndex, result, size, data, size_ret);
  });
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
