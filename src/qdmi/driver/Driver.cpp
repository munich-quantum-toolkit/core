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

#include "qdmi/common/Common.hpp"
#include "qdmi/common/Diagnostics.hpp"
#include "qdmi/driver/SessionConfig.hpp"

#include "DeviceRegistry.hpp"

#include "qdmi/client.h"
#include "qdmi/device.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include "qdmi/common/DeviceConfiguration.hpp"

#include <windows.h>
#else
#include <dlfcn.h>
#endif // _WIN32

namespace qdmi {
#ifdef _WIN32
namespace {
/// Loads the device library with the given name, searching in the driver
/// directory if no path is specified.
[[nodiscard]] auto loadDeviceLibrary(const std::string& libName) -> HMODULE {
  const auto requested = std::filesystem::path(libName);
  // Bare filenames are resolved relative to the Driver. Configured paths are
  // already absolute or relative to their declaring file.
  const auto path = requested.has_parent_path()
                        ? requested
                        : detail::moduleDirectory(reinterpret_cast<const void*>(
                              &loadDeviceLibrary)) /
                              requested;
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

Result<std::shared_ptr<DynamicDeviceLibrary>>
DynamicDeviceLibrary::create(const std::string& libName,
                             const std::string& prefix) {
  auto library = std::shared_ptr<DynamicDeviceLibrary>(
      new DynamicDeviceLibrary(DL_OPEN(libName.c_str())));
  if (library->libHandle_ == nullptr) {
    return Error{
        .status = QDMI_ERROR_LIBNOTFOUND,
        .message = "Couldn't open the device library: " + libName,
    };
  }
  if (auto error = library->initialize(prefix)) {
    return std::move(*error);
  }
  return library;
}

std::optional<Error>
DynamicDeviceLibrary::initialize(const std::string& prefix) {
//===----------------------------------------------------------------------===//
// Macro for loading a symbol from the dynamic library.
// @param symbol is the name of the symbol to load.
#define LOAD_DYNAMIC_SYMBOL(symbol)                                            \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (symbol) = reinterpret_cast<decltype(symbol)>(                             \
        DL_SYM(libHandle_, symbolName.c_str()));                               \
    if ((symbol) == nullptr) {                                                 \
      return Error{.status = QDMI_ERROR_NOTFOUND,                              \
                   .message = "Failed to load symbol: " + symbolName};         \
    }                                                                          \
  }

#define LOAD_OPTIONAL_DYNAMIC_SYMBOL(symbol)                                   \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (symbol) = reinterpret_cast<decltype(symbol)>(                             \
        DL_SYM(libHandle_, symbolName.c_str()));                               \
  }
  //===----------------------------------------------------------------------===//

  /// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  /// load the function symbols from the dynamic library
  LOAD_DYNAMIC_SYMBOL(device_initialize)
  LOAD_DYNAMIC_SYMBOL(device_finalize)
  /// device session interface
  LOAD_DYNAMIC_SYMBOL(device_session_alloc)
  LOAD_DYNAMIC_SYMBOL(device_session_init)
  LOAD_DYNAMIC_SYMBOL(device_session_free)
  LOAD_DYNAMIC_SYMBOL(device_session_set_parameter)
  /// device job interface
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
  /// device query interface
  LOAD_DYNAMIC_SYMBOL(device_session_query_device_property)
  LOAD_DYNAMIC_SYMBOL(device_session_query_site_property)
  LOAD_DYNAMIC_SYMBOL(device_session_query_operation_property)
  /// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  /// Initialize the device library only after every required symbol is
  /// available.
  if (auto error = checkError(device_initialize(),
                              "Failed to initialize device library")) {
    return error;
  }
  initialized_ = true;
  return std::nullopt;
}

#undef LOAD_OPTIONAL_DYNAMIC_SYMBOL
#undef LOAD_DYNAMIC_SYMBOL

DynamicDeviceLibrary::~DynamicDeviceLibrary() {
  if (initialized_) {
    device_finalize();
  }
  if (libHandle_ != nullptr) {
    DL_CLOSE(libHandle_);
  }
}

namespace {
struct DynamicLibraryCache {
  struct Module {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<DynamicDeviceLibrary>> providers;
  };
  std::mutex mutex;
  std::map<void*, Module> libraries;
};

[[nodiscard]] auto dynamicLibraryCache() -> DynamicLibraryCache& {
  /// Match Driver::get(): providers must outlive sessions in global
  /// destructors. NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* cache = new DynamicLibraryCache();
  return *cache;
}
} // namespace

[[nodiscard]] auto getDynamicDeviceLibrary(const std::string& libName,
                                           const std::string& prefix)
    -> Result<std::shared_ptr<DynamicDeviceLibrary>> {
  auto& cache = dynamicLibraryCache();
  const auto closeLibrary = [](void* handle) { DL_CLOSE(handle); };
  std::unique_ptr<void, decltype(closeLibrary)> handle(DL_OPEN(libName.c_str()),
                                                       closeLibrary);
  if (!handle) {
    return Error{
        .status = QDMI_ERROR_LIBNOTFOUND,
        .message = "Couldn't open the device library: " + libName,
    };
  }
  auto& module = [&]() -> auto& {
    const std::scoped_lock lock(cache.mutex);
    return cache.libraries[handle.get()];
  }();
  /// Modules may contain providers that share initialization state. Keep their
  /// initialization serialized without blocking unrelated modules.
  const std::scoped_lock lock(module.mutex);
  auto& providers = module.providers;
  if (const auto found = providers.find(prefix); found != providers.end()) {
    return found->second;
  }
  /// The private constructor takes ownership of the already loaded module.
  /// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto library = std::shared_ptr<DynamicDeviceLibrary>(
      new DynamicDeviceLibrary(handle.release()));
  if (auto error = library->initialize(prefix)) {
    return std::move(*error);
  }
  providers.emplace(prefix, library);
  return library;
}

#undef DL_OPEN
#undef DL_SYM
#undef DL_CLOSE
} // namespace qdmi

qdmi::Result<std::unique_ptr<QDMI_Device_impl_d>>
QDMI_Device_impl_d::create(std::shared_ptr<qdmi::DeviceLibrary> library,
                           const qdmi::DeviceSessionConfig& config,
                           QDMI_Child_Device childDevice) {
  if (!library) {
    return qdmi::Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Missing device library",
    };
  }
  auto device = std::unique_ptr<QDMI_Device_impl_d>(
      new QDMI_Device_impl_d(std::move(library)));
  if (auto error = device->initialize(config, childDevice)) {
    return std::move(*error);
  }
  return device;
}

std::optional<qdmi::Error>
QDMI_Device_impl_d::initialize(const qdmi::DeviceSessionConfig& config,
                               QDMI_Child_Device childDevice) {
  if (auto error =
          qdmi::checkError(library_->device_session_alloc(&deviceSession_),
                           "Failed to allocate device session")) {
    return error;
  }
  if (deviceSession_ == nullptr) {
    return qdmi::Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Device returned a null session handle",
    };
  }
  /// All views borrow NUL-terminated strings for this synchronous call.
  const auto setParameter = [&](const std::optional<std::string_view> value,
                                const QDMI_Device_Session_Parameter param)
      -> std::optional<qdmi::Error> {
    if (!value || library_->device_session_set_parameter == nullptr) {
      return std::nullopt;
    }
    const auto status = library_->device_session_set_parameter(
        deviceSession_, param, value->size() + 1, value->data());
    if (status == QDMI_ERROR_NOTSUPPORTED) {
      qdmi::diagnostics::info(
          "Device session parameter {} not supported by device (skipped)",
          qdmi::toString(param));
      return std::nullopt;
    }
    if (auto error = qdmi::checkError(
            status, std::string("Failed to set device session parameter ") +
                        qdmi::toString(param))) {
      return error;
    }
    return std::nullopt;
  };
  if (auto error =
          setParameter(config.baseUrl, QDMI_DEVICE_SESSION_PARAMETER_BASEURL)) {
    return error;
  }
  if (auto error =
          setParameter(config.token, QDMI_DEVICE_SESSION_PARAMETER_TOKEN)) {
    return error;
  }
  if (config.authFile) {
    if (auto error = setParameter(config.authFile->string(),
                                  QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE)) {
      return error;
    }
  }
  if (auto error =
          setParameter(config.authUrl, QDMI_DEVICE_SESSION_PARAMETER_AUTHURL)) {
    return error;
  }
  if (auto error = setParameter(config.username,
                                QDMI_DEVICE_SESSION_PARAMETER_USERNAME)) {
    return error;
  }
  if (auto error = setParameter(config.password,
                                QDMI_DEVICE_SESSION_PARAMETER_PASSWORD)) {
    return error;
  }
  if (config.deviceConfiguration && (config.custom1 || config.custom2)) {
    return qdmi::Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message =
            "Typed device configuration cannot be combined with raw custom1 or "
            "custom2 session parameters",
    };
  }
  if (config.deviceConfiguration) {
    auto error = std::visit(
        [&](const auto& source) {
          using Source = std::decay_t<decltype(source)>;
          if constexpr (std::is_same_v<Source,
                                       qdmi::InlineDeviceConfiguration>) {
            return setParameter(source.json,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1);
          } else {
            return setParameter(source.path.string(),
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2);
          }
        },
        *config.deviceConfiguration);
    if (error) {
      return error;
    }
  }
  if (auto error =
          setParameter(config.custom1, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1)) {
    return error;
  }
  if (auto error =
          setParameter(config.custom2, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2)) {
    return error;
  }
  if (auto error =
          setParameter(config.custom3, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3)) {
    return error;
  }
  if (auto error =
          setParameter(config.custom4, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4)) {
    return error;
  }
  if (auto error =
          setParameter(config.custom5, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5)) {
    return error;
  }
  if (childDevice != nullptr) {
    if (auto error = qdmi::checkError(
            library_->device_session_set_parameter(
                deviceSession_, QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE,
                sizeof(QDMI_Child_Device),
                static_cast<const void*>(&childDevice)),
            "Failed to select child device")) {
      return error;
    }
  }
  if (auto error =
          qdmi::checkError(library_->device_session_init(deviceSession_),
                           "Failed to initialize device session")) {
    return error;
  }
  /// Child sessions are leaves; only the parent discovers children.
  if (childDevice != nullptr) {
    return std::nullopt;
  }
  size_t childrenSize = 0;
  const auto status = library_->device_session_query_device_property(
      deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr,
      &childrenSize);
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  if (auto error = qdmi::checkError(status, "Failed to query child devices")) {
    return error;
  }
  if (childrenSize % sizeof(QDMI_Child_Device) != 0) {
    return qdmi::Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Device returned an invalid child device list",
    };
  }
  std::vector<QDMI_Child_Device> children(childrenSize /
                                          sizeof(QDMI_Child_Device));
  if (!children.empty()) {
    if (auto error = qdmi::checkError(
            library_->device_session_query_device_property(
                deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, childrenSize,
                static_cast<void*>(children.data()), nullptr),
            "Failed to query child devices")) {
      return error;
    }
  }
  childDevices_.reserve(children.size());
  for (auto* const child : children) {
    if (child == nullptr) {
      return qdmi::Error{
          .status = QDMI_ERROR_FATAL,
          .message = "Device returned a null child device handle",
      };
    }
    auto device = create(library_, config, child);
    if (auto* error = std::get_if<qdmi::Error>(&device)) {
      return std::move(*error);
    }
    childDevices_.emplace_back(std::get<0>(std::move(device)));
  }
  return std::nullopt;
}

auto QDMI_Device_impl_d::createJob(QDMI_Job* job) -> int {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *job = nullptr;
  QDMI_Device_Job deviceJob = nullptr;
  auto const result =
      library_->device_session_create_device_job(deviceSession_, &deviceJob);
  if (result != QDMI_SUCCESS && result != QDMI_WARN_GENERAL) {
    return result;
  }
  if (deviceJob == nullptr) {
    return QDMI_ERROR_FATAL;
  }
  auto uniqueJob = std::make_unique<QDMI_Job_impl_d>(deviceJob, this);
  auto* const jobHandle = uniqueJob.get();
  {
    const std::scoped_lock lock(jobsMutex_);
    jobs_.emplace(jobHandle, std::move(uniqueJob));
  }
  *job = jobHandle;
  return result;
}

auto QDMI_Device_impl_d::retrieveJobById(const char* const jobId,
                                         QDMI_Job* const job) -> int {
  if (jobId == nullptr || *jobId == '\0' || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (library_->device_session_retrieve_device_job_by_id == nullptr) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  *job = nullptr;
  QDMI_Device_Job deviceJob = nullptr;
  const auto result = library_->device_session_retrieve_device_job_by_id(
      deviceSession_, jobId, &deviceJob);
  if (result != QDMI_SUCCESS && result != QDMI_WARN_GENERAL) {
    return result;
  }
  if (deviceJob == nullptr) {
    return QDMI_ERROR_FATAL;
  }
  auto uniqueJob = std::make_unique<QDMI_Job_impl_d>(deviceJob, this);
  auto* const jobHandle = uniqueJob.get();
  {
    const std::scoped_lock lock(jobsMutex_);
    jobs_.emplace(jobHandle, std::move(uniqueJob));
  }
  *job = jobHandle;
  return result;
}

auto QDMI_Device_impl_d::freeJob(QDMI_Job job) -> void {
  std::unique_ptr<QDMI_Job_impl_d> ownedJob;
  {
    const std::scoped_lock lock(jobsMutex_);
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
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_SESSION_PARAMETER)) {
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
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_SESSION_PROPERTY)) {
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
std::optional<Error> validateDefinition(const DeviceDefinition& definition) {
  if (definition.id.empty()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Device definition ID must not be empty",
    };
  }
  if (definition.library.empty()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Device definition library must not be empty",
    };
  }
  if (definition.prefix.empty()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Device definition prefix must not be empty",
    };
  }
  if (definition.session.deviceConfiguration &&
      (definition.session.custom1 || definition.session.custom2)) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message =
            "Typed device configuration cannot be combined with raw custom1 or "
            "custom2 session parameters",
    };
  }
  return std::nullopt;
}
} // namespace

auto Driver::get() -> Driver& {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* instance = new Driver();
  return *instance;
}

std::optional<Error> Driver::initialize() {
  if (initialized_) {
    return std::nullopt;
  }
  auto result = detail::DeviceRegistry::discover();
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  auto const& registry = std::get<0>(result);
  for (const auto& definition : registry.definitions()) {
    if (auto error = validateDefinition(definition)) {
      return error;
    }
  }
  auto definitions = registry.definitions();
  std::unordered_set<std::string> disabled(registry.disabledIds().begin(),
                                           registry.disabledIds().end());
  std::vector<std::string> ids;
  ids.reserve(definitions.size());
  for (const auto& definition : definitions) {
    ids.emplace_back(definition.id);
  }
  definitions_ = std::move(definitions);
  disabledDeviceIds_ = std::move(disabled);
  clientDefinitionIds_ = std::move(ids);
  initialized_ = true;
  return std::nullopt;
}

std::optional<Error> Driver::registerDevice(DeviceDefinition definition,
                                            const bool replace) {
  if (auto error = validateDefinition(definition)) {
    return std::move(error);
  }
  std::unique_lock lock(stateMutex_);
  if (auto error = initialize()) {
    return std::move(error);
  }
  if (disabledDeviceIds_.contains(definition.id)) {
    if (!replace) {
      return Error{
          .status = QDMI_ERROR_INVALIDARGUMENT,
          .message = "QDMI device ID '" + definition.id +
                     "' is disabled by configuration",
      };
    }
    disabledDeviceIds_.erase(definition.id);
  }
  auto existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (existing == definitions_.end()) {
    definitions_.emplace_back(std::move(definition));
    return std::nullopt;
  }
  if (!replace) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message =
            "QDMI device ID '" + definition.id + "' is already registered",
    };
  }
  stateChanged_.wait(lock, [this, &definition] {
    return !openingDeviceIds_.contains(definition.id);
  });
  existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (openedDevices_.contains(definition.id)) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message =
            "Cannot replace opened QDMI device ID '" + definition.id + "'",
    };
  }
  *existing = std::move(definition);
  return std::nullopt;
}

auto Driver::registerDeviceIfAbsent(DeviceDefinition definition)
    -> Result<bool> {
  if (auto error = validateDefinition(definition)) {
    return std::move(*error);
  }
  const std::scoped_lock lock(stateMutex_);
  if (auto error = initialize()) {
    return std::move(*error);
  }
  if (disabledDeviceIds_.contains(definition.id) ||
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id) !=
          definitions_.end()) {
    return false;
  }
  definitions_.emplace_back(std::move(definition));
  return true;
}

auto Driver::registeredDeviceIds() -> Result<std::vector<std::string>> {
  const std::scoped_lock lock(stateMutex_);
  if (auto error = initialize()) {
    return std::move(*error);
  }
  std::vector<std::string> ids;
  ids.reserve(definitions_.size());
  std::ranges::transform(definitions_, std::back_inserter(ids),
                         &DeviceDefinition::id);
  return ids;
}

auto Driver::open(const std::string_view id) -> Result<QDMI_Device> {
  const std::string deviceId{id};
  DeviceDefinition definition;
  {
    std::unique_lock lock(stateMutex_);
    if (auto error = initialize()) {
      return std::move(*error);
    }
    stateChanged_.wait(lock, [this, &deviceId] {
      return !openingDeviceIds_.contains(deviceId);
    });
    if (disabledDeviceIds_.contains(deviceId)) {
      return Error{
          .status = QDMI_ERROR_BADSTATE,
          .message =
              "QDMI device ID '" + deviceId + "' is disabled by configuration",
      };
    }
    if (const auto opened = openedDevices_.find(deviceId);
        opened != openedDevices_.end()) {
      return opened->second.get();
    }
    const auto registered =
        std::ranges::find(definitions_, id, &DeviceDefinition::id);
    if (registered == definitions_.end()) {
      return Error{
          .status = QDMI_ERROR_OUTOFRANGE,
          .message = "Unknown QDMI device ID '" + deviceId + "'",
      };
    }
    definition = *registered;
    openingDeviceIds_.emplace(deviceId);
  }

  /// Unblock waiting open/replace calls on every return path.
  struct Opening {
    Driver* driver;
    const std::string* id;
    ~Opening() {
      {
        const std::scoped_lock lock(driver->stateMutex_);
        driver->openingDeviceIds_.erase(*id);
      }
      driver->stateChanged_.notify_all();
    }
  };
  const Opening opening{.driver = this, .id = &deviceId};
  auto library =
      getDynamicDeviceLibrary(definition.library.string(), definition.prefix);
  if (auto* error = std::get_if<Error>(&library)) {
    return std::move(*error);
  }
  auto candidate = QDMI_Device_impl_d::create(std::get<0>(std::move(library)),
                                              definition.session);
  if (auto* error = std::get_if<Error>(&candidate)) {
    return std::move(*error);
  }
  const std::scoped_lock lock(stateMutex_);
  const auto [opened, inserted] =
      openedDevices_.emplace(deviceId, std::get<0>(std::move(candidate)));
  return opened->second.get();
}

auto Driver::openFresh(const std::string_view id,
                       const DeviceSessionConfig& overrides)
    -> Result<std::shared_ptr<QDMI_Device_impl_d>> {
  DeviceDefinition definition;
  {
    const std::scoped_lock lock(stateMutex_);
    if (auto error = initialize()) {
      return std::move(*error);
    }
    if (disabledDeviceIds_.contains(std::string(id))) {
      return Error{
          .status = QDMI_ERROR_BADSTATE,
          .message = "QDMI device ID '" + std::string(id) +
                     "' is disabled by configuration",
      };
    }
    const auto registered =
        std::ranges::find(definitions_, id, &DeviceDefinition::id);
    if (registered == definitions_.end()) {
      return Error{
          .status = QDMI_ERROR_OUTOFRANGE,
          .message = "Unknown QDMI device ID '" + std::string(id) + "'",
      };
    }
    definition = *registered;
  }
  auto library =
      getDynamicDeviceLibrary(definition.library.string(), definition.prefix);
  if (auto* error = std::get_if<Error>(&library)) {
    return std::move(*error);
  }
  auto device = QDMI_Device_impl_d::create(
      std::get<0>(std::move(library)),
      detail::mergeSessionConfig(std::move(definition.session), overrides));
  if (auto* error = std::get_if<Error>(&device)) {
    return std::move(*error);
  }
  return std::shared_ptr<QDMI_Device_impl_d>(std::get<0>(std::move(device)));
}

void Driver::materializeClientCatalog() {
  std::call_once(clientCatalogOnce_, [this] {
    std::vector<std::string> definitionIds;
    {
      const std::scoped_lock lock(stateMutex_);
      definitionIds = clientDefinitionIds_;
    }
    std::vector<QDMI_Device> clientDevices;
    clientDevices.reserve(definitionIds.size());
    for (const auto& id : definitionIds) {
      auto result = open(id);
      if (const auto* error = std::get_if<Error>(&result)) {
        qdmi::diagnostics::warn("Skipping configured QDMI device '{}': {}", id,
                                error->message);
      } else {
        clientDevices.emplace_back(std::get<0>(result));
      }
    }
    const std::scoped_lock lock(stateMutex_);
    clientDevices_ = std::move(clientDevices);
  });
}

auto Driver::sessionAlloc(QDMI_Session* session) -> int {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  {
    const std::scoped_lock lock(stateMutex_);
    if (auto error = initialize()) {
      qdmi::diagnostics::warn("{}", error->message);
      return error->status;
    }
  }
  materializeClientCatalog();
  const std::scoped_lock lock(stateMutex_);
  auto uniqueSession = std::make_unique<QDMI_Session_impl_d>(clientDevices_);
  auto* const sessionHandle = uniqueSession.get();
  sessions_.emplace(sessionHandle, std::move(uniqueSession));
  *session = sessionHandle;
  return QDMI_SUCCESS;
}

auto Driver::sessionFree(QDMI_Session session) -> void {
  std::unique_ptr<QDMI_Session_impl_d> ownedSession;
  {
    const std::scoped_lock lock(stateMutex_);
    if (const auto entry = sessions_.find(session); entry != sessions_.end()) {
      ownedSession = std::move(entry->second);
      sessions_.erase(entry);
    }
  }
}
} // namespace qdmi

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
