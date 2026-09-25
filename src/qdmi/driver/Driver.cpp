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
#include "qdmi/driver/SessionConfig.hpp"

#include "DeviceRegistry.hpp"
#include "support/DiagnosticFormatting.hpp"
#include "support/Diagnostics.hpp"

#include "qdmi/client.h"
#include "qdmi/device.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ScopeExit.h"

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
  const auto requested = detail::pathFromString(libName);
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

mlir::FailureOr<std::shared_ptr<DynamicDeviceLibrary>>
DynamicDeviceLibrary::create(const std::string& libName,
                             const std::string& prefix) {
  auto library = std::shared_ptr<DynamicDeviceLibrary>(
      new DynamicDeviceLibrary(DL_OPEN(libName.c_str())));
  if (library->libHandle_ == nullptr) {
    return qdmi::emitError(QDMI_ERROR_LIBNOTFOUND,
                           "Couldn't open the device library: " + libName);
  }
  if (mlir::failed(library->initialize(prefix))) {
    return mlir::failure();
  }
  return library;
}

mlir::LogicalResult
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
      return qdmi::emitError(QDMI_ERROR_NOTFOUND,                              \
                             "Failed to load symbol: " + symbolName);          \
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
  LOAD_DYNAMIC_SYMBOL(device_job_set_programs)
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
  if (mlir::failed(checkError(device_initialize(),
                              "Failed to initialize device library"))) {
    return mlir::failure();
  }
  initialized_ = true;
  return mlir::success();
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
    -> mlir::FailureOr<std::shared_ptr<DynamicDeviceLibrary>> {
  auto& cache = dynamicLibraryCache();
  const auto closeLibrary = [](void* handle) { DL_CLOSE(handle); };
  std::unique_ptr<void, decltype(closeLibrary)> handle(DL_OPEN(libName.c_str()),
                                                       closeLibrary);
  if (!handle) {
    return qdmi::emitError(QDMI_ERROR_LIBNOTFOUND,
                           "Couldn't open the device library: " + libName);
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
  if (mlir::failed(library->initialize(prefix))) {
    return mlir::failure();
  }
  providers.emplace(prefix, library);
  return library;
}

#undef DL_OPEN
#undef DL_SYM
#undef DL_CLOSE
} // namespace qdmi

mlir::FailureOr<std::unique_ptr<QDMI_Device_impl_d>>
QDMI_Device_impl_d::create(std::shared_ptr<qdmi::DeviceLibrary> library,
                           const qdmi::DeviceSessionConfig& config,
                           QDMI_Child_Device childDevice, std::string id,
                           const bool strict) {
  if (!library) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "Missing device library");
  }
  auto device = std::unique_ptr<QDMI_Device_impl_d>(
      new QDMI_Device_impl_d(std::move(library)));
  device->id_ = std::move(id);
  if (mlir::failed(device->initialize(config, childDevice, strict))) {
    return mlir::failure();
  }
  return device;
}

mlir::LogicalResult
QDMI_Device_impl_d::initialize(const qdmi::DeviceSessionConfig& config,
                               QDMI_Child_Device childDevice,
                               const bool strict) {
  if (mlir::failed(
          qdmi::checkError(library_->device_session_alloc(&deviceSession_),
                           "Failed to allocate device session"))) {
    return mlir::failure();
  }
  if (deviceSession_ == nullptr) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "Device returned a null session handle");
  }
  /// All views borrow NUL-terminated strings for this synchronous call.
  const auto setParameter =
      [&](const std::optional<std::string_view> value,
          const QDMI_Device_Session_Parameter param) -> mlir::LogicalResult {
    if (!value) {
      return mlir::success();
    }
    if (library_->device_session_set_parameter == nullptr) {
      return strict ? qdmi::emitError(QDMI_ERROR_NOTSUPPORTED,
                                      "Device has no session parameter setter")
                    : mlir::success();
    }
    const auto status = library_->device_session_set_parameter(
        deviceSession_, param, value->size() + 1, value->data());
    if (status == QDMI_ERROR_NOTSUPPORTED && !strict) {
      ::mqt::diagnostics::info(
          "Device session parameter {} not supported by device (skipped)",
          qdmi::toString(param));
      return mlir::success();
    }
    if (mlir::failed(qdmi::checkError(
            status, std::string("Failed to set device session parameter ") +
                        qdmi::toString(param)))) {
      return mlir::failure();
    }
    return mlir::success();
  };
  if (mlir::failed(setParameter(config.baseUrl,
                                QDMI_DEVICE_SESSION_PARAMETER_BASEURL))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.token, QDMI_DEVICE_SESSION_PARAMETER_TOKEN))) {
    return mlir::failure();
  }
  if (config.authFile) {
    if (mlir::failed(setParameter(qdmi::detail::pathToString(*config.authFile),
                                  QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE))) {
      return mlir::failure();
    }
  }
  if (mlir::failed(setParameter(config.authUrl,
                                QDMI_DEVICE_SESSION_PARAMETER_AUTHURL))) {
    return mlir::failure();
  }
  if (mlir::failed(setParameter(config.username,
                                QDMI_DEVICE_SESSION_PARAMETER_USERNAME))) {
    return mlir::failure();
  }
  if (mlir::failed(setParameter(config.password,
                                QDMI_DEVICE_SESSION_PARAMETER_PASSWORD))) {
    return mlir::failure();
  }
  if (config.deviceConfiguration && (config.custom1 || config.custom2)) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Typed device configuration cannot be combined with raw custom1 or "
        "custom2 session parameters");
  }
  if (config.deviceConfiguration) {
    auto const error = std::visit(
        [&](const auto& source) {
          using Source = std::decay_t<decltype(source)>;
          if constexpr (std::is_same_v<Source,
                                       qdmi::InlineDeviceConfiguration>) {
            return setParameter(source.json,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1);
          } else {
            return setParameter(qdmi::detail::pathToString(source.path),
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2);
          }
        },
        *config.deviceConfiguration);
    if (mlir::failed(error)) {
      return mlir::failure();
    }
  }
  if (mlir::failed(setParameter(config.custom1,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1))) {
    return mlir::failure();
  }
  if (mlir::failed(setParameter(config.custom2,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2))) {
    return mlir::failure();
  }
  if (mlir::failed(setParameter(config.custom3,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3))) {
    return mlir::failure();
  }
  if (mlir::failed(setParameter(config.custom4,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4))) {
    return mlir::failure();
  }
  if (mlir::failed(setParameter(config.custom5,
                                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5))) {
    return mlir::failure();
  }
  if ((childDevice != nullptr) &&
      mlir::failed(qdmi::checkError(
          library_->device_session_set_parameter(
              deviceSession_, QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE,
              sizeof(QDMI_Child_Device),
              static_cast<const void*>(&childDevice)),
          "Failed to select child device"))) {
    return mlir::failure();
  }

  if (mlir::failed(
          qdmi::checkError(library_->device_session_init(deviceSession_),
                           "Failed to initialize device session"))) {
    return mlir::failure();
  }
  /// Child sessions are leaves; only the parent discovers children.
  if (childDevice != nullptr) {
    return mlir::success();
  }
  size_t childrenSize = 0;
  const auto status = library_->device_session_query_device_property(
      deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr,
      &childrenSize);
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    return mlir::success();
  }
  if (mlir::failed(qdmi::checkError(status, "Failed to query child devices"))) {
    return mlir::failure();
  }
  if (childrenSize % sizeof(QDMI_Child_Device) != 0) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "Device returned an invalid child device list");
  }
  std::vector<QDMI_Child_Device> children(childrenSize /
                                          sizeof(QDMI_Child_Device));
  if ((!children.empty()) &&
      mlir::failed(qdmi::checkError(
          library_->device_session_query_device_property(
              deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, childrenSize,
              static_cast<void*>(children.data()), nullptr),
          "Failed to query child devices"))) {
    return mlir::failure();
  }

  childDevices_.reserve(children.size());
  for (auto* const child : children) {
    if (child == nullptr) {
      return qdmi::emitError(QDMI_ERROR_FATAL,
                             "Device returned a null child device handle");
    }
    auto device = create(library_, config, child, {}, strict);
    if (mlir::failed(device)) {
      return mlir::failure();
    }
    childDevices_.emplace_back((*std::move(device)));
  }
  return mlir::success();
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
  if (!id_.empty()) {
    ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_ID, id_.c_str(), prop, size, value,
                        sizeRet)
  }
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

QDMI_Session_impl_d::QDMI_Session_impl_d(
    std::shared_ptr<QDMI_Device_impl_d> device)
    : devices_{device.get()}, ownedDevice_(std::move(device)) {}

QDMI_Job_impl_d::~QDMI_Job_impl_d() {
  device_->getLibrary().device_job_free(deviceJob_);
}
auto QDMI_Job_impl_d::setParameter(QDMI_Job_Parameter param, const size_t size,
                                   const void* value) const -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (param == 1) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  return device_->getLibrary().device_job_set_parameter(
      deviceJob_, toDeviceJobParameter(param), size, value);
}

auto QDMI_Job_impl_d::setPrograms(const QDMI_Program_Format* const format,
                                  const size_t count, const size_t* const sizes,
                                  const void* const* const programs) const
    -> int {
  return device_->getLibrary().device_job_set_programs(deviceJob_, format,
                                                       count, sizes, programs);
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
  case QDMI_JOB_PROPERTY_PROGRAMSNUM:
    return QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM;
  case QDMI_JOB_PROPERTY_PROGRAMSTATUSES:
    return QDMI_DEVICE_JOB_PROPERTY_PROGRAMSTATUSES;
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

auto QDMI_Job_impl_d::getResults(const size_t programIndex,
                                 QDMI_Job_Result result, const size_t size,
                                 void* data, size_t* sizeRet) const -> int {
  return device_->getLibrary().device_job_get_results(
      deviceJob_, programIndex, result, size, data, sizeRet);
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
mlir::LogicalResult validatePath(const std::filesystem::path& path,
                                 const std::string_view description) {
  if (path.empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           std::string(description) + " must not be empty");
  }
  if (path.native().find(std::filesystem::path::value_type{}) !=
      std::filesystem::path::string_type::npos) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           std::string(description) +
                               " must not contain null bytes");
  }
  return mlir::success();
}

mlir::LogicalResult validateSessionConfig(const DeviceSessionConfig& config) {
  if (config.authFile && mlir::failed(validatePath(
                             *config.authFile, "Device session auth file"))) {
    return mlir::failure();
  }
  if (const auto* source = config.deviceConfiguration
                               ? std::get_if<FileDeviceConfiguration>(
                                     &*config.deviceConfiguration)
                               : nullptr;
      source != nullptr &&
      mlir::failed(validatePath(source->path, "Device configuration file"))) {
    return mlir::failure();
  }
  if (config.deviceConfiguration && (config.custom1 || config.custom2)) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Typed device configuration cannot be combined with raw custom1 or "
        "custom2 session parameters");
  }
  return mlir::success();
}

mlir::LogicalResult validateDefinition(const DeviceDefinition& definition) {
  if (mlir::failed(detail::validateDeviceId(definition.id))) {
    return mlir::failure();
  }
  if (mlir::failed(
          validatePath(definition.library, "Device definition library"))) {
    return mlir::failure();
  }
  if (definition.prefix.empty() ||
      definition.prefix.find('\0') != std::string::npos) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Device definition prefix must not be empty or contain null bytes");
  }
  if (mlir::failed(validateSessionConfig(definition.session))) {
    return mlir::failure();
  }
  return mlir::success();
}
} // namespace

auto Driver::get() -> Driver& {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* instance = new Driver();
  return *instance;
}

mlir::LogicalResult Driver::initialize() {
  if (initialized_) {
    return mlir::success();
  }
  const llvm::scope_exit rollback([&] {
    if (!initialized_) {
      detail::rollbackDeviceManifestFreeze();
    }
  });
  auto result = detail::DeviceRegistry::discover();
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  auto const& registry = (*result);
  for (const auto& definition : registry.definitions()) {
    if (mlir::failed(validateDefinition(definition))) {
      return mlir::failure();
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
  return mlir::success();
}

mlir::LogicalResult Driver::registerDevice(DeviceDefinition definition,
                                           const bool replace) {
  if (mlir::failed(validateDefinition(definition))) {
    return mlir::failure();
  }
  std::unique_lock lock(stateMutex_);
  if (mlir::failed(initialize())) {
    return mlir::failure();
  }
  if (disabledDeviceIds_.contains(definition.id)) {
    if (!replace) {
      return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                             "QDMI device ID '" + definition.id +
                                 "' is disabled by configuration");
    }
    disabledDeviceIds_.erase(definition.id);
  }
  auto existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (existing == definitions_.end()) {
    definitions_.emplace_back(std::move(definition));
    return mlir::success();
  }
  if (!replace) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "QDMI device ID '" + definition.id +
                               "' is already registered");
  }
  stateChanged_.wait(lock, [this, &definition] {
    return !openingDeviceIds_.contains(definition.id);
  });
  existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (openedDevices_.contains(definition.id)) {
    return qdmi::emitError(QDMI_ERROR_BADSTATE,
                           "Cannot replace opened QDMI device ID '" +
                               definition.id + "'");
  }
  *existing = std::move(definition);
  return mlir::success();
}

auto Driver::registerDeviceIfAbsent(DeviceDefinition definition)
    -> mlir::FailureOr<bool> {
  if (mlir::failed(validateDefinition(definition))) {
    return mlir::failure();
  }
  const std::scoped_lock lock(stateMutex_);
  if (mlir::failed(initialize())) {
    return mlir::failure();
  }
  if (disabledDeviceIds_.contains(definition.id) ||
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id) !=
          definitions_.end()) {
    return false;
  }
  definitions_.emplace_back(std::move(definition));
  return true;
}

auto Driver::registeredDeviceIds()
    -> mlir::FailureOr<std::vector<std::string>> {
  const std::scoped_lock lock(stateMutex_);
  if (mlir::failed(initialize())) {
    return mlir::failure();
  }
  std::vector<std::string> ids;
  ids.reserve(definitions_.size());
  std::ranges::transform(definitions_, std::back_inserter(ids),
                         &DeviceDefinition::id);
  return ids;
}

auto Driver::open(const std::string_view id) -> mlir::FailureOr<QDMI_Device> {
  const std::string deviceId{id};
  DeviceDefinition definition;
  {
    std::unique_lock lock(stateMutex_);
    if (mlir::failed(initialize())) {
      return mlir::failure();
    }
    stateChanged_.wait(lock, [this, &deviceId] {
      return !openingDeviceIds_.contains(deviceId);
    });
    if (disabledDeviceIds_.contains(deviceId)) {
      return qdmi::emitError(QDMI_ERROR_PERMISSIONDENIED,
                             "QDMI device ID '" + deviceId +
                                 "' is disabled by configuration");
    }
    if (const auto opened = openedDevices_.find(deviceId);
        opened != openedDevices_.end()) {
      return opened->second.get();
    }
    const auto registered =
        std::ranges::find(definitions_, id, &DeviceDefinition::id);
    if (registered == definitions_.end()) {
      return qdmi::emitError(QDMI_ERROR_NOTFOUND,
                             "Unknown QDMI device ID '" + deviceId + "'");
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
  auto library = getDynamicDeviceLibrary(
      detail::pathToString(definition.library), definition.prefix);
  if (mlir::failed(library)) {
    return mlir::failure();
  }
  auto candidate = QDMI_Device_impl_d::create(
      (*std::move(library)), definition.session, nullptr, definition.id);
  if (mlir::failed(candidate)) {
    return mlir::failure();
  }
  const std::scoped_lock lock(stateMutex_);
  const auto [opened, inserted] =
      openedDevices_.emplace(deviceId, (*std::move(candidate)));
  return opened->second.get();
}

auto Driver::openFresh(const std::string_view id,
                       const DeviceSessionConfig& overrides, const bool strict)
    -> mlir::FailureOr<std::shared_ptr<QDMI_Device_impl_d>> {
  DeviceDefinition definition;
  {
    const std::scoped_lock lock(stateMutex_);
    if (mlir::failed(initialize())) {
      return mlir::failure();
    }
    if (disabledDeviceIds_.contains(std::string(id))) {
      return qdmi::emitError(QDMI_ERROR_PERMISSIONDENIED,
                             "QDMI device ID '" + std::string(id) +
                                 "' is disabled by configuration");
    }
    const auto registered =
        std::ranges::find(definitions_, id, &DeviceDefinition::id);
    if (registered == definitions_.end()) {
      return qdmi::emitError(QDMI_ERROR_NOTFOUND, "Unknown QDMI device ID '" +
                                                      std::string(id) + "'");
    }
    definition = *registered;
  }
  auto library = getDynamicDeviceLibrary(
      detail::pathToString(definition.library), definition.prefix);
  if (mlir::failed(library)) {
    return mlir::failure();
  }
  const auto config =
      detail::mergeSessionConfig(std::move(definition.session), overrides);
  if (mlir::failed(validateSessionConfig(config))) {
    return mlir::failure();
  }
  auto device = QDMI_Device_impl_d::create(*std::move(library), config, nullptr,
                                           definition.id, strict);
  if (mlir::failed(device)) {
    return mlir::failure();
  }
  return std::shared_ptr<QDMI_Device_impl_d>((*std::move(device)));
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
      ::mqt::ScopedDiagnosticHandler const context(
          [&](const ::mqt::Diagnostic& diagnostic) {
            auto warning = diagnostic;
            warning.severity = ::mqt::DiagnosticSeverity::Warning;
            warning.message = "Skipping configured QDMI device '" + id +
                              "': " + warning.message;
            ::mqt::emitDiagnostic(warning);
            return mlir::success();
          });
      auto result = open(id);
      if (mlir::succeeded(result)) {
        clientDevices.emplace_back(*result);
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
  *session = nullptr;
  {
    const std::scoped_lock lock(stateMutex_);
    const auto status = qdmi::invokeStatus([&] { return initialize(); });
    if (status != QDMI_SUCCESS) {
      return status;
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

auto Driver::sessionAllocForDevice(const std::string_view id,
                                   const DeviceSessionConfig& config,
                                   QDMI_Session* const session) -> int {
  if (id.empty() || session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  return qdmi::invokeStatus([&]() -> mlir::LogicalResult {
    auto device = openFresh(id, config, true);
    if (mlir::failed(device)) {
      return mlir::failure();
    }
    auto uniqueSession =
        std::make_unique<QDMI_Session_impl_d>(std::move(*device));
    auto* const sessionHandle = uniqueSession.get();
    const std::scoped_lock lock(stateMutex_);
    sessions_.emplace(sessionHandle, std::move(uniqueSession));
    *session = sessionHandle;
    return mlir::success();
  });
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

int QDMI_job_set_programs(QDMI_Job job, const QDMI_Program_Format* format,
                          const size_t count, const size_t* sizes,
                          const void* const* programs) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->setPrograms(format, count, sizes, programs);
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

int QDMI_job_get_results(QDMI_Job job, const size_t programIndex,
                         QDMI_Job_Result result, const size_t size, void* data,
                         size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getResults(programIndex, result, size, data, sizeRet);
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
