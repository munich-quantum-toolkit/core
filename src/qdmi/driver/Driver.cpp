/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/driver/Driver.hpp"

#include "DeviceRegistry.hpp"
#include "qdmi/common/Common.hpp"

#include <qdmi/client.h>
#include <qdmi/device.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif // _WIN32

namespace qdmi {
#ifdef _WIN32
namespace {
/// Returns the directory of the currently loaded driver library.
[[nodiscard]] auto getDriverDirectory() -> std::filesystem::path {
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&getDriverDirectory),
                         &module) == 0) {
    return {};
  }

  std::wstring buffer(MAX_PATH, L'\0');
  DWORD size = 0;
  while (true) {
    size = GetModuleFileNameW(module, buffer.data(),
                              static_cast<DWORD>(buffer.size()));
    if (size == 0) {
      return {};
    }
    if (size < buffer.size()) {
      buffer.resize(size);
      break;
    }
    buffer.resize(buffer.size() * 2);
  }

  return std::filesystem::path(buffer).parent_path();
}

/// Loads the device library with the given name, searching in the driver
/// directory if no path is specified.
[[nodiscard]] auto loadDeviceLibrary(const std::string& libName) -> HMODULE {
  const auto requested = std::filesystem::path(libName);
  // Bare filenames are resolved relative to the Driver. Configured paths are
  // already absolute or relative to their declaring file.
  const auto path = requested.has_parent_path()
                        ? requested
                        : getDriverDirectory() / requested;
  // Search beside the device DLL for its dependencies. This is required for
  // device implementations such as DDSIM in an installed Python wheel.
  return LoadLibraryExW(path.wstring().c_str(), nullptr,
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
} // namespace

#define DL_OPEN(lib) loadDeviceLibrary((lib))
#define DL_SYM(lib, sym)                                                       \
  reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>((lib)), (sym)))
#define DL_CLOSE(lib) FreeLibrary(static_cast<HMODULE>((lib)))
#else
#define DL_OPEN(lib) dlopen((lib), RTLD_NOW | RTLD_LOCAL)
#define DL_SYM(lib, sym) dlsym((lib), (sym))
#define DL_CLOSE(lib) dlclose((lib))
#endif

DynamicDeviceLibrary::DynamicDeviceLibrary(const std::string& libName,
                                           const std::string& prefix)
    : libHandle_(DL_OPEN(libName.c_str())) {
  if (libHandle_ == nullptr) {
    throw std::runtime_error("Couldn't open the device library: " + libName);
  }

//===----------------------------------------------------------------------===//
// Macro for loading a symbol from the dynamic library.
// @param symbol is the name of the symbol to load.
#define LOAD_DYNAMIC_SYMBOL(symbol)                                            \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (symbol) = reinterpret_cast<decltype(symbol)>(                             \
        DL_SYM(libHandle_, symbolName.c_str()));                               \
    if ((symbol) == nullptr) {                                                 \
      throw std::runtime_error("Failed to load symbol: " + symbolName);        \
    }                                                                          \
  }

#define LOAD_OPTIONAL_DYNAMIC_SYMBOL(symbol)                                   \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (symbol) = reinterpret_cast<decltype(symbol)>(                             \
        DL_SYM(libHandle_, symbolName.c_str()));                               \
  }
  //===----------------------------------------------------------------------===//

  try {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    // load the function symbols from the dynamic library
    LOAD_DYNAMIC_SYMBOL(device_initialize)
    LOAD_DYNAMIC_SYMBOL(device_finalize)
    // device session interface
    LOAD_DYNAMIC_SYMBOL(device_session_alloc)
    LOAD_DYNAMIC_SYMBOL(device_session_init)
    LOAD_DYNAMIC_SYMBOL(device_session_free)
    LOAD_DYNAMIC_SYMBOL(device_session_set_parameter)
    // device job interface
    LOAD_DYNAMIC_SYMBOL(device_session_create_device_job)
    LOAD_OPTIONAL_DYNAMIC_SYMBOL(device_session_retrieve_device_job_by_id)
    LOAD_DYNAMIC_SYMBOL(device_job_free)
    LOAD_DYNAMIC_SYMBOL(device_job_set_parameter)
    LOAD_DYNAMIC_SYMBOL(device_job_query_property)
    LOAD_DYNAMIC_SYMBOL(device_job_submit)
    LOAD_DYNAMIC_SYMBOL(device_job_cancel)
    LOAD_DYNAMIC_SYMBOL(device_job_check)
    LOAD_DYNAMIC_SYMBOL(device_job_wait)
    LOAD_DYNAMIC_SYMBOL(device_job_get_results)
    // device query interface
    LOAD_DYNAMIC_SYMBOL(device_session_query_device_property)
    LOAD_DYNAMIC_SYMBOL(device_session_query_site_property)
    LOAD_DYNAMIC_SYMBOL(device_session_query_operation_property)
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    // Initialize the device library only after every required symbol is
    // available.
    throwIfError(device_initialize(), "Failed to initialize device library");
  } catch (...) {
    DL_CLOSE(libHandle_);
    libHandle_ = nullptr;
    throw;
  }
}

#undef LOAD_OPTIONAL_DYNAMIC_SYMBOL
#undef LOAD_DYNAMIC_SYMBOL

DynamicDeviceLibrary::~DynamicDeviceLibrary() {
  // Check if QDMI_device_finalize is not NULL before calling it.
  if (device_finalize != nullptr) {
    device_finalize();
  }
  // close the dynamic library
  if (libHandle_ != nullptr) {
    DL_CLOSE(libHandle_);
  }
}

namespace {
struct DynamicLibraryCache {
  std::mutex mutex;
  std::map<std::pair<std::string, std::string>,
           std::weak_ptr<DynamicDeviceLibrary>>
      libraries;
};

[[nodiscard]] auto dynamicLibraryCache() -> DynamicLibraryCache& {
  static DynamicLibraryCache cache;
  return cache;
}

[[nodiscard]] auto getDynamicDeviceLibrary(const std::string& libName,
                                           const std::string& prefix)
    -> std::shared_ptr<DynamicDeviceLibrary> {
  auto& cache = dynamicLibraryCache();
  const std::scoped_lock lock(cache.mutex);
  std::error_code error;
  auto canonicalPath = std::filesystem::weakly_canonical(
      std::filesystem::absolute(std::filesystem::path(libName), error), error);
  if (error) {
    canonicalPath = std::filesystem::path(libName).lexically_normal();
  }
  const auto key = std::pair{canonicalPath.string(), prefix};
  if (const auto library = cache.libraries[key].lock()) {
    return library;
  }
  auto library = std::make_shared<DynamicDeviceLibrary>(libName, prefix);
  cache.libraries[key] = library;
  return library;
}

template <class T>
void applyOverride(std::optional<T>& value,
                   const std::optional<T>& overrideValue) {
  if (overrideValue) {
    value = overrideValue;
  }
}

[[nodiscard]] auto mergeSessionConfig(const DeviceSessionConfig& defaults,
                                      const DeviceSessionConfig& overrides)
    -> DeviceSessionConfig {
  auto merged = defaults;
  applyOverride(merged.baseUrl, overrides.baseUrl);
  applyOverride(merged.token, overrides.token);
  applyOverride(merged.authFile, overrides.authFile);
  applyOverride(merged.authUrl, overrides.authUrl);
  applyOverride(merged.username, overrides.username);
  applyOverride(merged.password, overrides.password);
  applyOverride(merged.deviceConfiguration, overrides.deviceConfiguration);
  applyOverride(merged.custom1, overrides.custom1);
  applyOverride(merged.custom2, overrides.custom2);
  applyOverride(merged.custom3, overrides.custom3);
  applyOverride(merged.custom4, overrides.custom4);
  applyOverride(merged.custom5, overrides.custom5);
  return merged;
}
} // namespace

#undef DL_OPEN
#undef DL_SYM
#undef DL_CLOSE
} // namespace qdmi

QDMI_Device_impl_d::QDMI_Device_impl_d(
    std::shared_ptr<qdmi::DeviceLibrary> lib,
    const qdmi::DeviceSessionConfig& config,
    QDMI_Child_Device_impl_d* const childDevice)
    : library_(std::move(lib)) {
  if (library_->device_session_alloc(&deviceSession_) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to allocate device session");
  }

  // Set device session parameters from config
  auto setParameter = [this](const std::optional<std::string>& value,
                             QDMI_Device_Session_Parameter param) {
    if (value && library_->device_session_set_parameter) {
      const auto status =
          static_cast<QDMI_STATUS>(library_->device_session_set_parameter(
              deviceSession_, param, value->size() + 1, value->c_str()));
      if (status == QDMI_SUCCESS) {
        return;
      }

      if (status == QDMI_ERROR_NOTSUPPORTED) {
        SPDLOG_INFO(
            "Device session parameter {} not supported by device (skipped)",
            qdmi::toString(param));
        return;
      }
      library_->device_session_free(deviceSession_);
      std::ostringstream ss;
      ss << "Failed to set device session parameter " << qdmi::toString(param)
         << ": " << qdmi::toString(status);
      throw std::runtime_error(ss.str());
    }
  };

  setParameter(config.baseUrl, QDMI_DEVICE_SESSION_PARAMETER_BASEURL);
  setParameter(config.token, QDMI_DEVICE_SESSION_PARAMETER_TOKEN);
  if (config.authFile) {
    const std::optional authFile = config.authFile->string();
    setParameter(authFile, QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE);
  }
  setParameter(config.authUrl, QDMI_DEVICE_SESSION_PARAMETER_AUTHURL);
  setParameter(config.username, QDMI_DEVICE_SESSION_PARAMETER_USERNAME);
  setParameter(config.password, QDMI_DEVICE_SESSION_PARAMETER_PASSWORD);
  if (config.deviceConfiguration &&
      (config.custom1.has_value() || config.custom2.has_value())) {
    library_->device_session_free(deviceSession_);
    deviceSession_ = nullptr;
    throw std::invalid_argument(
        "Typed device configuration cannot be combined with raw custom1 or "
        "custom2 session parameters");
  }
  if (config.deviceConfiguration) {
    std::visit(
        [&](const auto& source) {
          using Source = std::decay_t<decltype(source)>;
          if constexpr (std::is_same_v<Source,
                                       qdmi::InlineDeviceConfiguration>) {
            setParameter(std::optional{source.json},
                         QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1);
          } else {
            setParameter(std::optional{source.path.string()},
                         QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2);
          }
        },
        *config.deviceConfiguration);
  }
  setParameter(config.custom1, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1);
  setParameter(config.custom2, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2);
  setParameter(config.custom3, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3);
  setParameter(config.custom4, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4);
  setParameter(config.custom5, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5);

  if (childDevice != nullptr) {
    const auto status =
        static_cast<QDMI_STATUS>(library_->device_session_set_parameter(
            deviceSession_, QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE,
            sizeof(QDMI_Child_Device), static_cast<const void*>(&childDevice)));
    if (status != QDMI_SUCCESS) {
      library_->device_session_free(deviceSession_);
      deviceSession_ = nullptr;
      std::ostringstream ss;
      ss << "Failed to select child device: " << qdmi::toString(status);
      throw std::runtime_error(ss.str());
    }
  }

  if (library_->device_session_init(deviceSession_) != QDMI_SUCCESS) {
    library_->device_session_free(deviceSession_);
    deviceSession_ = nullptr;
    throw std::runtime_error("Failed to initialize device session");
  }

  // Child sessions represent leaf devices in the QDMI multicore
  // workflow. Only top-level sessions discover and wrap child handles.
  if (childDevice != nullptr) {
    return;
  }

  size_t childrenSize = 0;
  auto status =
      static_cast<QDMI_STATUS>(library_->device_session_query_device_property(
          deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr,
          &childrenSize));
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    return;
  }
  if (status != QDMI_SUCCESS || childrenSize % sizeof(QDMI_Child_Device) != 0) {
    library_->device_session_free(deviceSession_);
    deviceSession_ = nullptr;
    if (status != QDMI_SUCCESS) {
      throw std::runtime_error("Failed to query child devices: " +
                               std::string(qdmi::toString(status)));
    }
    throw std::runtime_error("Device returned an invalid child device list");
  }

  std::vector<QDMI_Child_Device> children(childrenSize /
                                          sizeof(QDMI_Child_Device));
  if (childrenSize != 0) {
    status =
        static_cast<QDMI_STATUS>(library_->device_session_query_device_property(
            deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, childrenSize,
            static_cast<void*>(children.data()), nullptr));
    if (status != QDMI_SUCCESS) {
      library_->device_session_free(deviceSession_);
      deviceSession_ = nullptr;
      throw std::runtime_error("Failed to query child devices: " +
                               std::string(qdmi::toString(status)));
    }
  }

  try {
    childDevices_.reserve(children.size());
    for (auto* const child : children) {
      childDevices_.emplace_back(
          std::make_unique<QDMI_Device_impl_d>(library_, config, child));
    }
  } catch (...) {
    childDevices_.clear();
    library_->device_session_free(deviceSession_);
    deviceSession_ = nullptr;
    throw;
  }
}

auto QDMI_Device_impl_d::createJob(QDMI_Job* job) -> int {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  QDMI_Device_Job deviceJob = nullptr;
  auto result =
      library_->device_session_create_device_job(deviceSession_, &deviceJob);
  if (result != QDMI_SUCCESS) {
    return result;
  }
  auto uniqueJob = std::make_unique<QDMI_Job_impl_d>(deviceJob, this);
  auto* const jobHandle = uniqueJob.get();
  {
    const std::lock_guard lock(jobsMutex_);
    jobs_.emplace(jobHandle, std::move(uniqueJob));
  }
  *job = jobHandle;
  return QDMI_SUCCESS;
}

auto QDMI_Device_impl_d::retrieveJobById(const char* const jobId,
                                         QDMI_Job* const job) -> int {
  if (jobId == nullptr || *jobId == '\0' || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (library_->device_session_retrieve_device_job_by_id == nullptr) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  QDMI_Device_Job deviceJob = nullptr;
  const auto result = library_->device_session_retrieve_device_job_by_id(
      deviceSession_, jobId, &deviceJob);
  if (result != QDMI_SUCCESS) {
    return result;
  }
  auto uniqueJob = std::make_unique<QDMI_Job_impl_d>(deviceJob, this);
  auto* const jobHandle = uniqueJob.get();
  {
    const std::lock_guard lock(jobsMutex_);
    jobs_.emplace(jobHandle, std::move(uniqueJob));
  }
  *job = jobHandle;
  return QDMI_SUCCESS;
}

auto QDMI_Device_impl_d::freeJob(QDMI_Job job) -> void {
  std::unique_ptr<QDMI_Job_impl_d> ownedJob;
  {
    const std::lock_guard lock(jobsMutex_);
    if (const auto entry = jobs_.find(job); entry != jobs_.end()) {
      ownedJob = std::move(entry->second);
      jobs_.erase(entry);
    }
  }
}

auto QDMI_Device_impl_d::queryDeviceProperty(QDMI_Device_Property prop,
                                             const size_t size, void* value,
                                             size_t* sizeRet) const -> int {
  if (prop == QDMI_DEVICE_PROPERTY_CHILDDEVICES) {
    if (childDevices_.empty()) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    const auto requiredSize = childDevices_.size() * sizeof(QDMI_Device);
    if (value != nullptr) {
      if (size < requiredSize) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* devices = static_cast<QDMI_Device*>(value);
      std::ranges::transform(
          childDevices_, devices,
          [](const auto& child) -> QDMI_Device { return child.get(); });
    }
    if (sizeRet != nullptr) {
      *sizeRet = requiredSize;
    }
    return QDMI_SUCCESS;
  }
  return library_->device_session_query_device_property(deviceSession_, prop,
                                                        size, value, sizeRet);
}

auto QDMI_Device_impl_d::querySiteProperty(QDMI_Site site,
                                           QDMI_Site_Property prop,
                                           const size_t size, void* value,
                                           size_t* sizeRet) const -> int {
  return library_->device_session_query_site_property(
      deviceSession_, site, prop, size, value, sizeRet);
}

auto QDMI_Device_impl_d::queryOperationProperty(
    QDMI_Operation operation, const size_t numSites, const QDMI_Site* sites,
    const size_t numParams, const double* params, QDMI_Operation_Property prop,
    const size_t size, void* value, size_t* sizeRet) const -> int {
  return library_->device_session_query_operation_property(
      deviceSession_, operation, numSites, sites, numParams, params, prop, size,
      value, sizeRet);
}

namespace {
[[nodiscard]] auto toDeviceJobParameter(const QDMI_Job_Parameter& param)
    -> QDMI_Device_Job_Parameter {
  switch (param) {
  case QDMI_JOB_PARAMETER_PROGRAM:
    return QDMI_DEVICE_JOB_PARAMETER_PROGRAM;
  case QDMI_JOB_PARAMETER_PROGRAMFORMAT:
    return QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT;
  case QDMI_JOB_PARAMETER_SHOTSNUM:
    return QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM;
  case QDMI_JOB_PARAMETER_CUSTOM1:
    return QDMI_DEVICE_JOB_PARAMETER_CUSTOM1;
  case QDMI_JOB_PARAMETER_CUSTOM2:
    return QDMI_DEVICE_JOB_PARAMETER_CUSTOM2;
  case QDMI_JOB_PARAMETER_CUSTOM3:
    return QDMI_DEVICE_JOB_PARAMETER_CUSTOM3;
  case QDMI_JOB_PARAMETER_CUSTOM4:
    return QDMI_DEVICE_JOB_PARAMETER_CUSTOM4;
  case QDMI_JOB_PARAMETER_CUSTOM5:
    return QDMI_DEVICE_JOB_PARAMETER_CUSTOM5;
  default:
    return QDMI_DEVICE_JOB_PARAMETER_MAX;
  }
}
} // namespace

QDMI_Session_impl_d::QDMI_Session_impl_d(
    const std::vector<std::unique_ptr<QDMI_Device_impl_d>>& devices) {
  devices_.reserve(devices.size());
  std::ranges::transform(devices, std::back_inserter(devices_),
                         [](const auto& device) { return device.get(); });
}

QDMI_Session_impl_d::QDMI_Session_impl_d(
    const std::vector<QDMI_Device>& devices)
    : devices_(devices) {}

QDMI_Job_impl_d::~QDMI_Job_impl_d() {
  device_->getLibrary().device_job_free(deviceJob_);
}
auto QDMI_Job_impl_d::setParameter(QDMI_Job_Parameter param, const size_t size,
                                   const void* value) const -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device_->getLibrary().device_job_set_parameter(
      deviceJob_, toDeviceJobParameter(param), size, value);
}

namespace {
[[nodiscard]] auto toDeviceJobProperty(const QDMI_Job_Property& prop)
    -> QDMI_Device_Job_Property {
  switch (prop) {
  case QDMI_JOB_PROPERTY_ID:
    return QDMI_DEVICE_JOB_PROPERTY_ID;
  case QDMI_JOB_PROPERTY_PROGRAM:
    return QDMI_DEVICE_JOB_PROPERTY_PROGRAM;
  case QDMI_JOB_PROPERTY_PROGRAMFORMAT:
    return QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT;
  case QDMI_JOB_PROPERTY_SHOTSNUM:
    return QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM;
  case QDMI_JOB_PROPERTY_QUEUEPOSITION:
    return QDMI_DEVICE_JOB_PROPERTY_QUEUEPOSITION;
  case QDMI_JOB_PROPERTY_CUSTOM1:
    return QDMI_DEVICE_JOB_PROPERTY_CUSTOM1;
  case QDMI_JOB_PROPERTY_CUSTOM2:
    return QDMI_DEVICE_JOB_PROPERTY_CUSTOM2;
  case QDMI_JOB_PROPERTY_CUSTOM3:
    return QDMI_DEVICE_JOB_PROPERTY_CUSTOM3;
  case QDMI_JOB_PROPERTY_CUSTOM4:
    return QDMI_DEVICE_JOB_PROPERTY_CUSTOM4;
  case QDMI_JOB_PROPERTY_CUSTOM5:
    return QDMI_DEVICE_JOB_PROPERTY_CUSTOM5;
  default:
    return QDMI_DEVICE_JOB_PROPERTY_MAX;
  }
}
} // namespace

auto QDMI_Job_impl_d::queryProperty(QDMI_Job_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) const -> int {
  return device_->getLibrary().device_job_query_property(
      deviceJob_, toDeviceJobProperty(prop), size, value, sizeRet);
}

auto QDMI_Job_impl_d::submit() const -> int {
  return device_->getLibrary().device_job_submit(deviceJob_);
}

auto QDMI_Job_impl_d::cancel() const -> int {
  return device_->getLibrary().device_job_cancel(deviceJob_);
}

auto QDMI_Job_impl_d::check(QDMI_Job_Status* status) const -> int {
  return device_->getLibrary().device_job_check(deviceJob_, status);
}

auto QDMI_Job_impl_d::wait(size_t timeout) const -> int {
  return device_->getLibrary().device_job_wait(deviceJob_, timeout);
}

auto QDMI_Job_impl_d::getResults(QDMI_Job_Result result, const size_t size,
                                 void* data, size_t* sizeRet) const -> int {
  return device_->getLibrary().device_job_get_results(deviceJob_, result, size,
                                                      data, sizeRet);
}

auto QDMI_Job_impl_d::free() -> void { device_->freeJob(this); }

auto QDMI_Session_impl_d::init() -> int {
  if (status_ != qdmi::SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status_ = qdmi::SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

auto QDMI_Session_impl_d::setParameter(QDMI_Session_Parameter param,
                                       const size_t size,
                                       const void* value) const -> int {
  if ((value != nullptr && size == 0) || param >= QDMI_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != qdmi::SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

auto QDMI_Session_impl_d::querySessionProperty(QDMI_Session_Property prop,
                                               size_t size, void* value,
                                               size_t* sizeRet) const -> int {
  if ((value != nullptr && size == 0) || prop >= QDMI_SESSION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != qdmi::SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == QDMI_SESSION_PROPERTY_DEVICES) {
    if (value != nullptr) {
      if (size < devices_.size() * sizeof(QDMI_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      memcpy(value, static_cast<const void*>(devices_.data()),
             devices_.size() * sizeof(QDMI_Device));
    }
    if (sizeRet != nullptr) {
      *sizeRet = devices_.size() * sizeof(QDMI_Device);
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

namespace qdmi {
namespace {
void validateDefinition(const DeviceDefinition& definition) {
  if (definition.id.empty()) {
    throw std::invalid_argument("Device definition ID must not be empty");
  }
  if (definition.library.empty()) {
    throw std::invalid_argument("Device definition library must not be empty");
  }
  if (definition.prefix.empty()) {
    throw std::invalid_argument("Device definition prefix must not be empty");
  }
  if (definition.session.deviceConfiguration &&
      (definition.session.custom1 || definition.session.custom2)) {
    throw std::invalid_argument(
        "Typed device configuration cannot be combined with raw custom1 or "
        "custom2 session parameters");
  }
}
} // namespace

auto Driver::get() -> Driver& {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* instance = new Driver();
  return *instance;
}

Driver::Driver() {
  const detail::DeviceRegistry registry;
  disabledDeviceIds_.insert(registry.disabledIds().begin(),
                            registry.disabledIds().end());
  for (const auto& definition : registry.definitions()) {
    registerDevice(definition);
    clientDefinitionIds_.emplace_back(definition.id);
  }
}

void Driver::registerDevice(DeviceDefinition definition, const bool replace) {
  validateDefinition(definition);
  std::unique_lock lock(stateMutex_);
  if (disabledDeviceIds_.contains(definition.id)) {
    if (!replace) {
      throw std::invalid_argument("QDMI device ID '" + definition.id +
                                  "' is disabled by configuration");
    }
    disabledDeviceIds_.erase(definition.id);
  }
  auto existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (existing == definitions_.end()) {
    definitions_.emplace_back(std::move(definition));
    return;
  }
  if (!replace) {
    throw std::invalid_argument("QDMI device ID '" + definition.id +
                                "' is already registered");
  }
  stateChanged_.wait(lock, [this, &definition] {
    return !openingDeviceIds_.contains(definition.id);
  });
  existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (openedDevices_.contains(definition.id)) {
    throw std::runtime_error("Cannot replace opened QDMI device ID '" +
                             definition.id + "'");
  }
  *existing = std::move(definition);
}

auto Driver::registerDeviceIfAbsent(DeviceDefinition definition) -> bool {
  validateDefinition(definition);
  const std::lock_guard lock(stateMutex_);
  if (disabledDeviceIds_.contains(definition.id) ||
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id) !=
          definitions_.end()) {
    return false;
  }
  definitions_.emplace_back(std::move(definition));
  return true;
}

auto Driver::registeredDeviceIds() const -> std::vector<std::string> {
  const std::lock_guard lock(stateMutex_);
  std::vector<std::string> ids;
  ids.reserve(definitions_.size());
  std::ranges::transform(definitions_, std::back_inserter(ids),
                         &DeviceDefinition::id);
  return ids;
}

auto Driver::open(const std::string_view id) -> QDMI_Device {
  const std::string deviceId{id};
  DeviceDefinition definition;
  {
    std::unique_lock lock(stateMutex_);
    stateChanged_.wait(lock, [this, &deviceId] {
      return !openingDeviceIds_.contains(deviceId);
    });
    if (disabledDeviceIds_.contains(deviceId)) {
      throw std::runtime_error("QDMI device ID '" + deviceId +
                               "' is disabled by configuration");
    }
    if (const auto opened = openedDevices_.find(deviceId);
        opened != openedDevices_.end()) {
      return opened->second;
    }
    const auto registered =
        std::ranges::find(definitions_, id, &DeviceDefinition::id);
    if (registered == definitions_.end()) {
      throw std::out_of_range("Unknown QDMI device ID '" + deviceId + "'");
    }
    definition = *registered;
    openingDeviceIds_.emplace(deviceId);
  }

  std::unique_ptr<QDMI_Device_impl_d> candidate;
  try {
    candidate = std::make_unique<QDMI_Device_impl_d>(
        getDynamicDeviceLibrary(definition.library.string(), definition.prefix),
        definition.session);
  } catch (...) {
    {
      const std::lock_guard lock(stateMutex_);
      openingDeviceIds_.erase(deviceId);
    }
    stateChanged_.notify_all();
    throw;
  }

  auto* device = candidate.get();
  try {
    const std::lock_guard lock(stateMutex_);
    const auto [opened, inserted] =
        openedDevices_.emplace(deviceId, candidate.get());
    if (inserted) {
      try {
        devices_.emplace_back(std::move(candidate));
      } catch (...) {
        openedDevices_.erase(opened);
        throw;
      }
    } else {
      device = opened->second;
    }
    openingDeviceIds_.erase(deviceId);
  } catch (...) {
    {
      const std::lock_guard lock(stateMutex_);
      openingDeviceIds_.erase(deviceId);
    }
    stateChanged_.notify_all();
    throw;
  }
  stateChanged_.notify_all();
  return device;
}

auto Driver::openFresh(const std::string_view id,
                       const DeviceSessionConfig& overrides)
    -> std::shared_ptr<QDMI_Device_impl_d> {
  DeviceDefinition definition;
  {
    const std::lock_guard lock(stateMutex_);
    if (disabledDeviceIds_.contains(std::string(id))) {
      throw std::runtime_error("QDMI device ID '" + std::string(id) +
                               "' is disabled by configuration");
    }
    const auto registered =
        std::ranges::find(definitions_, id, &DeviceDefinition::id);
    if (registered == definitions_.end()) {
      throw std::out_of_range("Unknown QDMI device ID '" + std::string(id) +
                              "'");
    }
    definition = *registered;
  }
  return std::make_shared<QDMI_Device_impl_d>(
      getDynamicDeviceLibrary(definition.library.string(), definition.prefix),
      mergeSessionConfig(definition.session, overrides));
}

void Driver::materializeClientCatalog() {
  std::call_once(clientCatalogOnce_, [this] {
    std::vector<std::string> definitionIds;
    {
      const std::lock_guard lock(stateMutex_);
      definitionIds = clientDefinitionIds_;
    }
    std::vector<QDMI_Device> clientDevices;
    clientDevices.reserve(definitionIds.size());
    for (const auto& id : definitionIds) {
      try {
        clientDevices.emplace_back(open(id));
      } catch (const std::exception& ex) {
        std::string library = "<unknown>";
        {
          const std::lock_guard lock(stateMutex_);
          if (const auto definition =
                  std::ranges::find(definitions_, id, &DeviceDefinition::id);
              definition != definitions_.end()) {
            library = definition->library.string();
          }
        }
        SPDLOG_WARN("Skipping configured QDMI device '{}' from '{}': {}", id,
                    library, ex.what());
      }
    }
    const std::lock_guard lock(stateMutex_);
    clientDevices_ = std::move(clientDevices);
  });
}

auto Driver::sessionAlloc(QDMI_Session* session) -> int {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  materializeClientCatalog();
  const std::lock_guard lock(stateMutex_);
  auto uniqueSession = std::make_unique<QDMI_Session_impl_d>(clientDevices_);
  auto* const sessionHandle = uniqueSession.get();
  sessions_.emplace(sessionHandle, std::move(uniqueSession));
  *session = sessionHandle;
  return QDMI_SUCCESS;
}

auto Driver::sessionFree(QDMI_Session session) -> void {
  std::unique_ptr<QDMI_Session_impl_d> ownedSession;
  {
    const std::lock_guard lock(stateMutex_);
    if (const auto entry = sessions_.find(session); entry != sessions_.end()) {
      ownedSession = std::move(entry->second);
      sessions_.erase(entry);
    }
  }
}
} // namespace qdmi

int QDMI_session_alloc(QDMI_Session* session) {
  return qdmi::Driver::get().sessionAlloc(session);
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->init();
}

void QDMI_session_free(QDMI_Session session) {
  qdmi::Driver::get().sessionFree(session);
}

int QDMI_session_set_parameter(QDMI_Session session,
                               QDMI_Session_Parameter param, const size_t size,
                               const void* value) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->setParameter(param, size, value);
}

int QDMI_session_query_session_property(QDMI_Session session,
                                        QDMI_Session_Property prop, size_t size,
                                        void* value, size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->querySessionProperty(prop, size, value, sizeRet);
}

int QDMI_device_create_job(QDMI_Device dev, QDMI_Job* job) {
  if (dev == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return dev->createJob(job);
}

int QDMI_session_retrieve_job_by_id(QDMI_Device device, const char* jobId,
                                    QDMI_Job* job) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->retrieveJobById(jobId, job);
}

void QDMI_job_free(QDMI_Job job) {
  if (job != nullptr) {
    job->free();
  }
}

int QDMI_job_set_parameter(QDMI_Job job, QDMI_Job_Parameter param,
                           const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->setParameter(param, size, value);
}

int QDMI_job_query_property(QDMI_Job job, QDMI_Job_Property prop,
                            const size_t size, void* value, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->queryProperty(prop, size, value, sizeRet);
}

int QDMI_job_submit(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->submit();
}

int QDMI_job_cancel(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->cancel();
}

int QDMI_job_check(QDMI_Job job, QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->check(status);
}

int QDMI_job_wait(QDMI_Job job, size_t timeout) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->wait(timeout);
}

int QDMI_job_get_results(QDMI_Job job, QDMI_Job_Result result,
                         const size_t size, void* data, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getResults(result, size, data, sizeRet);
}

int QDMI_device_query_device_property(QDMI_Device device,
                                      QDMI_Device_Property prop,
                                      const size_t size, void* value,
                                      size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->queryDeviceProperty(prop, size, value, sizeRet);
}

int QDMI_device_query_site_property(QDMI_Device device, QDMI_Site site,
                                    QDMI_Site_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->querySiteProperty(site, prop, size, value, sizeRet);
}

int QDMI_device_query_operation_property(
    QDMI_Device device, QDMI_Operation operation, const size_t numSites,
    const QDMI_Site* sites, const size_t numParams, const double* params,
    QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->queryOperationProperty(operation, numSites, sites, numParams,
                                        params, prop, size, value, sizeRet);
}
