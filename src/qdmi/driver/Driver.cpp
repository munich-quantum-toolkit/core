/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Driver.hpp"

#include "qdmi/common/Common.hpp"
#include "qdmi/common/Diagnostics.hpp"
#include "qdmi/driver/DriverExtension.hpp"

#include "DeviceRegistry.hpp"
#include "SessionConfig.hpp"

#include "qdmi/client.h"
#include "qdmi/device.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
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
[[nodiscard]] auto loadDeviceLibrary(const std::string& libName)
    -> llvm::Expected<HMODULE> {
  auto requested = detail::pathFromString(libName);
  if (!requested) {
    return requested.takeError();
  }
  // Bare filenames are resolved relative to the Driver. Configured paths are
  // already absolute or relative to their declaring file.
  auto path = *requested;
  if (!requested->has_parent_path()) {
    auto directory = detail::moduleDirectory(
        reinterpret_cast<const void*>(&loadDeviceLibrary));
    if (!directory) {
      return directory.takeError();
    }
    path = *directory / *requested;
  }
  // Search beside the device DLL for its dependencies. This is required for
  // device implementations such as DDSIM in an installed Python wheel.
  return LoadLibraryExW(path.c_str(), nullptr,
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
} // namespace

#define DL_SYM(lib, sym)                                                       \
  reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>((lib)), (sym)))
#define DL_CLOSE(lib) FreeLibrary(static_cast<HMODULE>((lib)))
#else
#define DL_OPEN(lib) dlopen((lib), RTLD_NOW | RTLD_LOCAL)
#define DL_SYM(lib, sym) dlsym((lib), (sym))
#define DL_CLOSE(lib) dlclose((lib))
#endif

llvm::Error LoadedDeviceAPI::initialize(const std::string& prefix) {
//===----------------------------------------------------------------------===//
// Macro for loading a symbol from the dynamic library.
// @param symbol is the name of the symbol to load.
#define LOAD_DYNAMIC_SYMBOL(symbol)                                            \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (symbol) = reinterpret_cast<decltype(symbol)>(                             \
        DL_SYM(libHandle_, symbolName.c_str()));                               \
    if ((symbol) == nullptr) {                                                 \
      return qdmi::makeError(QDMI_ERROR_FATAL,                                 \
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
  LOAD_DYNAMIC_SYMBOL(device_job_set_programs)
  LOAD_DYNAMIC_SYMBOL(device_job_get_program)
  LOAD_DYNAMIC_SYMBOL(device_job_get_program_status)
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
  if (auto error = checkError(device_initialize(),
                              "Failed to initialize device library")) {
    return error;
  }
  initialized_ = true;
  return llvm::Error::success();
}

#undef LOAD_OPTIONAL_DYNAMIC_SYMBOL
#undef LOAD_DYNAMIC_SYMBOL

LoadedDeviceAPI::~LoadedDeviceAPI() {
  if (initialized_) {
    device_finalize();
  }
  if (libHandle_ != nullptr) {
    DL_CLOSE(libHandle_);
  }
}

namespace {
struct DeviceAPICache {
  struct Module {
    std::mutex mutex;
    llvm::StringMap<std::unique_ptr<LoadedDeviceAPI>> apis;
  };
  std::mutex mutex;
  llvm::DenseMap<void*, std::unique_ptr<Module>> libraries;
};

[[nodiscard]] auto deviceAPICache() -> DeviceAPICache& {
  // Match Driver::get(): APIs must outlive sessions in global destructors.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* cache = new DeviceAPICache();
  return *cache;
}
} // namespace

[[nodiscard]] auto loadDeviceAPI(const std::string& libName,
                                 const std::string& prefix)
    -> llvm::Expected<LoadedDeviceAPI*> {
  auto& cache = deviceAPICache();
  const auto closeLibrary = [](void* handle) { DL_CLOSE(handle); };
#ifdef _WIN32
  auto loaded = loadDeviceLibrary(libName);
  if (!loaded) {
    return loaded.takeError();
  }
  auto* rawHandle = static_cast<void*>(*loaded);
#else
  auto* rawHandle = DL_OPEN(libName.c_str());
#endif
  std::unique_ptr<void, decltype(closeLibrary)> handle(rawHandle, closeLibrary);
  if (!handle) {
    return qdmi::makeError(QDMI_ERROR_LIBNOTFOUND,
                           "Couldn't open the device library: " + libName);
  }
  auto& module = [&]() -> auto& {
    const std::scoped_lock lock(cache.mutex);
    auto& module = cache.libraries[handle.get()];
    if (!module) {
      module = std::make_unique<DeviceAPICache::Module>();
    }
    return *module;
  }();
  // Modules may contain devices that share initialization state. Keep their
  // initialization serialized without blocking unrelated modules.
  const std::scoped_lock lock(module.mutex);
  auto& apis = module.apis;
  if (const auto found = apis.find(prefix); found != apis.end()) {
    return found->second.get();
  }
  // The private constructor takes ownership of the already loaded module.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto library =
      std::unique_ptr<LoadedDeviceAPI>(new LoadedDeviceAPI(handle.release()));
  if (auto error = library->initialize(prefix)) {
    return error;
  }
  return apis.try_emplace(prefix, std::move(library)).first->second.get();
}

#undef DL_OPEN
#undef DL_SYM
#undef DL_CLOSE
} // namespace qdmi

llvm::Expected<std::unique_ptr<QDMI_Device_impl_d>> QDMI_Device_impl_d::create(
    const qdmi::DeviceAPI& library, const qdmi::DeviceSessionConfig& config,
    QDMI_Child_Device childDevice, std::string id, const bool strict) {
  auto device =
      std::unique_ptr<QDMI_Device_impl_d>(new QDMI_Device_impl_d(library));
  device->id_ = std::move(id);
  if (auto error = device->initialize(config, childDevice, strict)) {
    return error;
  }
  return device;
}

llvm::Error
QDMI_Device_impl_d::initialize(const qdmi::DeviceSessionConfig& config,
                               QDMI_Child_Device childDevice,
                               const bool strict) {
  if (auto error = qdmi::checkError(api_->device_session_alloc(&deviceSession_),
                                    "Failed to allocate device session")) {
    return error;
  }
  if (deviceSession_ == nullptr) {
    return qdmi::makeError(QDMI_ERROR_FATAL,
                           "Device returned a null session handle");
  }
  // All views borrow NUL-terminated strings for this synchronous call.
  const auto setParameter =
      [&](const std::optional<std::string_view> value,
          const QDMI_Device_Session_Parameter param) -> llvm::Error {
    if (!value) {
      return llvm::Error::success();
    }
    if (api_->device_session_set_parameter == nullptr) {
      return strict ? qdmi::makeError(QDMI_ERROR_NOTSUPPORTED,
                                      "Device has no session parameter setter")
                    : llvm::Error::success();
    }
    const auto status = api_->device_session_set_parameter(
        deviceSession_, param, value->size() + 1, value->data());
    if (status == QDMI_ERROR_NOTSUPPORTED && !strict) {
      qdmi::diagnostics::info(
          "Device session parameter {0} not supported by device (skipped)",
          qdmi::toString(param));
      return llvm::Error::success();
    }
    if (auto error = qdmi::checkError(
            status, llvm::Twine("Failed to set device session parameter ") +
                        qdmi::toString(param))) {
      return error;
    }
    return llvm::Error::success();
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
    auto filename = qdmi::detail::pathToString(*config.authFile);
    if (!filename) {
      return filename.takeError();
    }
    if (auto error =
            setParameter(*filename, QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE)) {
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
    return qdmi::makeError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Typed device configuration cannot be combined with raw custom1 or "
        "custom2 session parameters");
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
            auto filename = qdmi::detail::pathToString(source.path);
            if (!filename) {
              return filename.takeError();
            }
            return setParameter(*filename,
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
  if ((childDevice != nullptr)) {
    if (auto error = qdmi::checkError(
            api_->device_session_set_parameter(
                deviceSession_, QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE,
                sizeof(QDMI_Child_Device),
                static_cast<const void*>(&childDevice)),
            "Failed to select child device")) {
      return error;
    }
  }

  if (auto error = qdmi::checkError(api_->device_session_init(deviceSession_),
                                    "Failed to initialize device session")) {
    return error;
  }
  // Child sessions are leaves; only the parent discovers children.
  if (childDevice != nullptr) {
    return llvm::Error::success();
  }
  size_t childrenSize = 0;
  const auto status = api_->device_session_query_device_property(
      deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr,
      &childrenSize);
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    return llvm::Error::success();
  }
  if (auto error = qdmi::checkError(status, "Failed to query child devices")) {
    return error;
  }
  if (childrenSize % sizeof(QDMI_Child_Device) != 0) {
    return qdmi::makeError(QDMI_ERROR_FATAL,
                           "Device returned an invalid child device list");
  }
  std::vector<QDMI_Child_Device> children(childrenSize /
                                          sizeof(QDMI_Child_Device));
  if ((!children.empty())) {
    if (auto error = qdmi::checkError(
            api_->device_session_query_device_property(
                deviceSession_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, childrenSize,
                static_cast<void*>(children.data()), nullptr),
            "Failed to query child devices")) {
      return error;
    }
  }

  childDevices_.reserve(children.size());
  for (auto* const child : children) {
    if (child == nullptr) {
      return qdmi::makeError(QDMI_ERROR_FATAL,
                             "Device returned a null child device handle");
    }
    auto device = create(*api_, config, child, {}, strict);
    if (!device) {
      return device.takeError();
    }
    childDevices_.emplace_back(std::move(*device));
  }
  return llvm::Error::success();
}

QDMI_Device_impl_d::~QDMI_Device_impl_d() {
  jobs_.clear();
  childDevices_.clear();
  if (deviceSession_ != nullptr) {
    api_->device_session_free(deviceSession_);
  }
}

auto QDMI_Device_impl_d::adoptJob(const int status, QDMI_Device_Job deviceJob,
                                  std::unique_ptr<QDMI_Job_impl_d> wrapper,
                                  QDMI_Job* job) -> int {
  if (status != QDMI_SUCCESS && status != QDMI_WARN_GENERAL) {
    return status;
  }
  if (deviceJob == nullptr) {
    return QDMI_ERROR_FATAL;
  }
  wrapper->deviceJob_ = deviceJob;
  auto* const handle = wrapper.get();
  {
    const std::scoped_lock lock(jobsMutex_);
    jobs_.push_back(wrapper.release());
  }
  *job = handle;
  return status;
}

auto QDMI_Device_impl_d::createJob(QDMI_Job* job) -> int {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *job = nullptr;
  auto wrapper = std::unique_ptr<QDMI_Job_impl_d>(
      new (std::nothrow) QDMI_Job_impl_d(nullptr, this));
  if (!wrapper) {
    return QDMI_ERROR_OUTOFMEM;
  }
  QDMI_Device_Job deviceJob = nullptr;
  const auto status =
      api_->device_session_create_device_job(deviceSession_, &deviceJob);
  return adoptJob(status, deviceJob, std::move(wrapper), job);
}

auto QDMI_Device_impl_d::retrieveJobById(const char* const jobId,
                                         QDMI_Job* const job) -> int {
  if (jobId == nullptr || *jobId == '\0' || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *job = nullptr;
  if (api_->device_session_retrieve_device_job_by_id == nullptr) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  auto wrapper = std::unique_ptr<QDMI_Job_impl_d>(
      new (std::nothrow) QDMI_Job_impl_d(nullptr, this));
  if (!wrapper) {
    return QDMI_ERROR_OUTOFMEM;
  }
  QDMI_Device_Job deviceJob = nullptr;
  const auto status = api_->device_session_retrieve_device_job_by_id(
      deviceSession_, jobId, &deviceJob);
  return adoptJob(status, deviceJob, std::move(wrapper), job);
}

auto QDMI_Device_impl_d::freeJob(QDMI_Job job) -> void {
  std::unique_ptr<QDMI_Job_impl_d> ownedJob;
  {
    const auto lock = std::scoped_lock(jobsMutex_);
    ownedJob.reset(jobs_.remove(job));
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
  return api_->device_session_query_device_property(deviceSession_, prop, size,
                                                    value, sizeRet);
}

auto QDMI_Device_impl_d::querySiteProperty(QDMI_Site site,
                                           QDMI_Site_Property prop,
                                           const size_t size, void* value,
                                           size_t* sizeRet) const -> int {
  return api_->device_session_query_site_property(deviceSession_, site, prop,
                                                  size, value, sizeRet);
}

auto QDMI_Device_impl_d::queryOperationProperty(
    QDMI_Operation operation, const size_t numSites, const QDMI_Site* sites,
    const size_t numParams, const double* params, QDMI_Operation_Property prop,
    const size_t size, void* value, size_t* sizeRet) const -> int {
  return api_->device_session_query_operation_property(
      deviceSession_, operation, numSites, sites, numParams, params, prop, size,
      value, sizeRet);
}

namespace {
[[nodiscard]] auto toDeviceJobParameter(const QDMI_Job_Parameter& param)
    -> QDMI_Device_Job_Parameter {
  switch (param) {
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
    // Reserved slots 0 and 1 are representable but have no named enumerators.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    return static_cast<QDMI_Device_Job_Parameter>(param);
  }
}
} // namespace

QDMI_Session_impl_d::QDMI_Session_impl_d(
    llvm::ArrayRef<QDMI_Device> devices) noexcept
    : devices_(devices) {}

QDMI_Session_impl_d::QDMI_Session_impl_d(
    std::unique_ptr<QDMI_Device_impl_d> device) noexcept
    : ownedDevice_(std::move(device)), privateDevice_(ownedDevice_.get()) {
  devices_ = llvm::ArrayRef<QDMI_Device>(&privateDevice_, 1);
}

QDMI_Job_impl_d::~QDMI_Job_impl_d() {
  if (deviceJob_ != nullptr) {
    device_->api().device_job_free(deviceJob_);
  }
}
auto QDMI_Job_impl_d::setParameter(QDMI_Job_Parameter param, const size_t size,
                                   const void* value) const -> int {
  if ((value != nullptr && size == 0) ||
      QDMI_IS_INVALID_ENUM_VALUE(param, QDMI_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device_->api().device_job_set_parameter(
      deviceJob_, toDeviceJobParameter(param), size, value);
}

auto QDMI_Job_impl_d::setPrograms(const QDMI_Program_Format format,
                                  const size_t count, const size_t* const sizes,
                                  const void* const* const programs) const
    -> int {
  return device_->api().device_job_set_programs(deviceJob_, format, count,
                                                sizes, programs);
}

auto QDMI_Job_impl_d::getProgram(const size_t programIndex, const size_t size,
                                 void* data, size_t* sizeRet) const -> int {
  return device_->api().device_job_get_program(deviceJob_, programIndex, size,
                                               data, sizeRet);
}

auto QDMI_Job_impl_d::getProgramStatus(const size_t programIndex,
                                       QDMI_Job_Status* status) const -> int {
  return device_->api().device_job_get_program_status(deviceJob_, programIndex,
                                                      status);
}

namespace {
[[nodiscard]] auto toDeviceJobProperty(const QDMI_Job_Property& prop)
    -> QDMI_Device_Job_Property {
  switch (prop) {
  case QDMI_JOB_PROPERTY_ID:
    return QDMI_DEVICE_JOB_PROPERTY_ID;
  case QDMI_JOB_PROPERTY_PROGRAMFORMAT:
    return QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT;
  case QDMI_JOB_PROPERTY_SHOTSNUM:
    return QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM;
  case QDMI_JOB_PROPERTY_QUEUEPOSITION:
    return QDMI_DEVICE_JOB_PROPERTY_QUEUEPOSITION;
  case QDMI_JOB_PROPERTY_PROGRAMSNUM:
    return QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM;
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
  return device_->api().device_job_query_property(
      deviceJob_, toDeviceJobProperty(prop), size, value, sizeRet);
}

auto QDMI_Job_impl_d::submit() const -> int {
  return device_->api().device_job_submit(deviceJob_);
}

auto QDMI_Job_impl_d::cancel() const -> int {
  return device_->api().device_job_cancel(deviceJob_);
}

auto QDMI_Job_impl_d::check(QDMI_Job_Status* status) const -> int {
  return device_->api().device_job_check(deviceJob_, status);
}

auto QDMI_Job_impl_d::wait(size_t timeout) const -> int {
  return device_->api().device_job_wait(deviceJob_, timeout);
}

auto QDMI_Job_impl_d::getResults(const size_t programIndex,
                                 QDMI_Job_Result result, const size_t size,
                                 void* data, size_t* sizeRet) const -> int {
  return device_->api().device_job_get_results(deviceJob_, programIndex, result,
                                               size, data, sizeRet);
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
      QDMI_IS_INVALID_ENUM_VALUE(param, QDMI_SESSION_PARAMETER)) {
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
      QDMI_IS_INVALID_ENUM_VALUE(prop, QDMI_SESSION_PROPERTY)) {
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
llvm::Error validatePath(const std::filesystem::path& path,
                         const std::string_view description) {
  if (path.empty()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           std::string(description) + " must not be empty");
  }
  if (path.native().find(std::filesystem::path::value_type{}) !=
      std::filesystem::path::string_type::npos) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           std::string(description) +
                               " must not contain null bytes");
  }
  return llvm::Error::success();
}

llvm::Error validateSessionConfig(const DeviceSessionConfig& config) {
  if (config.authFile) {
    if (auto error =
            validatePath(*config.authFile, "Device session auth file")) {
      return error;
    }
  }
  if (const auto* source = config.deviceConfiguration
                               ? std::get_if<FileDeviceConfiguration>(
                                     &*config.deviceConfiguration)
                               : nullptr;
      source != nullptr) {
    if (auto error = validatePath(source->path, "Device configuration file")) {
      return error;
    }
  }
  if (config.deviceConfiguration && (config.custom1 || config.custom2)) {
    return qdmi::makeError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Typed device configuration cannot be combined with raw custom1 or "
        "custom2 session parameters");
  }
  return llvm::Error::success();
}

llvm::Error validateDefinition(const DeviceDefinition& definition) {
  if (auto error = detail::validateDeviceId(definition.id)) {
    return error;
  }
  if (auto error =
          validatePath(definition.library, "Device definition library")) {
    return error;
  }
  if (definition.prefix.empty() ||
      definition.prefix.find('\0') != std::string::npos) {
    return qdmi::makeError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Device definition prefix must not be empty or contain null bytes");
  }
  if (auto error = validateSessionConfig(definition.session)) {
    return error;
  }
  return llvm::Error::success();
}
} // namespace

auto Driver::get() -> Driver& {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* instance = new Driver();
  return *instance;
}

llvm::Error Driver::initialize() {
  if (initialized_) {
    return llvm::Error::success();
  }
  const llvm::scope_exit rollback([&] {
    if (!initialized_) {
      detail::rollbackDeviceManifestFreeze();
    }
  });
  auto result = detail::DeviceRegistry::discover();
  if (!result) {
    return result.takeError();
  }
  auto const& registry = (*result);
  for (const auto& definition : registry.definitions()) {
    if (auto error = validateDefinition(definition)) {
      return error;
    }
  }
  for (const auto& id : registry.disabledIds()) {
    disabledDeviceIds_.insert(id);
  }
  for (const auto& definition : registry.definitions()) {
    const auto entry = definitions_.try_emplace(definition.id).first;
    entry->second.definition = definition;
    registrationOrder_.push_back(&entry->second);
    clientDefinitionIds_.push_back(entry->getKey());
  }
  initialized_ = true;
  return llvm::Error::success();
}

llvm::Error Driver::registerDevice(DeviceDefinition definition,
                                   const bool replace) {
  if (auto error = validateDefinition(definition)) {
    return error;
  }
  std::unique_lock lock(stateMutex_);
  if (auto error = initialize()) {
    return error;
  }
  if (disabledDeviceIds_.contains(definition.id)) {
    if (!replace) {
      return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                             "QDMI device ID '" + definition.id +
                                 "' is disabled by configuration");
    }
    disabledDeviceIds_.erase(definition.id);
  }
  auto [entry, inserted] = definitions_.try_emplace(definition.id);
  auto& record = entry->second;
  if (inserted) {
    record.definition = std::move(definition);
    registrationOrder_.push_back(&record);
    return llvm::Error::success();
  }
  if (!replace) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           "QDMI device ID '" + definition.id +
                               "' is already registered");
  }
  stateChanged_.wait(lock, [&record] { return !record.opening; });
  if (record.device) {
    return qdmi::makeError(QDMI_ERROR_BADSTATE,
                           "Cannot replace opened QDMI device ID '" +
                               definition.id + "'");
  }
  record.definition = std::move(definition);
  return llvm::Error::success();
}

auto Driver::registerDeviceIfAbsent(DeviceDefinition definition)
    -> llvm::Expected<bool> {
  if (auto error = validateDefinition(definition)) {
    return error;
  }
  const auto lock = std::scoped_lock(stateMutex_);
  if (auto error = initialize()) {
    return error;
  }
  if (disabledDeviceIds_.contains(definition.id)) {
    return false;
  }
  auto [entry, inserted] = definitions_.try_emplace(definition.id);
  if (inserted) {
    entry->second.definition = std::move(definition);
    registrationOrder_.push_back(&entry->second);
  }
  return inserted;
}

auto Driver::registeredDeviceIds() -> llvm::Expected<std::vector<std::string>> {
  const auto lock = std::scoped_lock(stateMutex_);
  if (auto error = initialize()) {
    return error;
  }
  std::vector<std::string> ids;
  ids.reserve(registrationOrder_.size());
  for (const auto* entry : registrationOrder_) {
    ids.push_back(entry->definition.id);
  }
  return ids;
}

auto Driver::open(const std::string_view id) -> llvm::Expected<QDMI_Device> {
  RegistryEntry* record = nullptr;
  {
    std::unique_lock lock(stateMutex_);
    if (auto error = initialize()) {
      return error;
    }
    if (disabledDeviceIds_.contains(id)) {
      return qdmi::makeError(QDMI_ERROR_PERMISSIONDENIED,
                             "QDMI device ID '" + llvm::Twine(id) +
                                 "' is disabled by configuration");
    }
    const auto entry = definitions_.find(id);
    if (entry == definitions_.end()) {
      return qdmi::makeError(QDMI_ERROR_NOTFOUND, "Unknown QDMI device ID '" +
                                                      llvm::Twine(id) + "'");
    }
    record = &entry->second;
    stateChanged_.wait(lock, [record] { return !record->opening; });
    if (record->device) {
      return record->device.get();
    }
    record->opening = true;
  }
  const llvm::scope_exit finish([&] {
    {
      const auto lock = std::scoped_lock(stateMutex_);
      record->opening = false;
    }
    stateChanged_.notify_all();
  });
  // Replacement waits for opening; other insertions keep this entry stable.
  const auto& definition = record->definition;
  auto filename = detail::pathToString(definition.library);
  if (!filename) {
    return filename.takeError();
  }
  auto library = loadDeviceAPI(*filename, definition.prefix);
  if (!library) {
    return library.takeError();
  }
  auto candidate = QDMI_Device_impl_d::create(**library, definition.session,
                                              nullptr, definition.id);
  if (!candidate) {
    return candidate.takeError();
  }
  const auto lock = std::scoped_lock(stateMutex_);
  record->device = std::move(*candidate);
  return record->device.get();
}

auto Driver::openFresh(const std::string_view id,
                       const DeviceSessionConfig& overrides, const bool strict)
    -> llvm::Expected<std::unique_ptr<QDMI_Device_impl_d>> {
  DeviceDefinition definition;
  {
    const std::scoped_lock lock(stateMutex_);
    if (auto error = initialize()) {
      return error;
    }
    if (disabledDeviceIds_.contains(id)) {
      return qdmi::makeError(QDMI_ERROR_PERMISSIONDENIED,
                             "QDMI device ID '" + std::string(id) +
                                 "' is disabled by configuration");
    }
    const auto registered = definitions_.find(id);
    if (registered == definitions_.end()) {
      return qdmi::makeError(QDMI_ERROR_NOTFOUND, "Unknown QDMI device ID '" +
                                                      std::string(id) + "'");
    }
    definition = registered->second.definition;
  }
  const auto config =
      detail::mergeSessionConfig(std::move(definition.session), overrides);
  if (auto error = validateSessionConfig(config)) {
    return error;
  }
  auto filename = detail::pathToString(definition.library);
  if (!filename) {
    return filename.takeError();
  }
  auto library = loadDeviceAPI(*filename, definition.prefix);
  if (!library) {
    return library.takeError();
  }
  auto device = QDMI_Device_impl_d::create(**library, config, nullptr,
                                           definition.id, strict);
  if (!device) {
    return device.takeError();
  }
  return std::move(*device);
}

void Driver::materializeClientCatalog() {
  std::call_once(clientCatalogOnce_, [this] {
    llvm::SmallVector<QDMI_Device> clientDevices;
    clientDevices.reserve(clientDefinitionIds_.size());
    for (const auto& id : clientDefinitionIds_) {
      auto result = open(id);
      if (result) {
        clientDevices.emplace_back(*result);
      } else {
        qdmi::diagnostics::warn("Skipping configured QDMI device '{0}': {1}",
                                id, llvm::toString(result.takeError()));
      }
    }
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
    const auto status = qdmi::toStatus(initialize());
    if (status != QDMI_SUCCESS) {
      return status;
    }
  }
  materializeClientCatalog();
  // Ownership passes to QDMI_session_free through the C handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new (std::nothrow) QDMI_Session_impl_d(clientDevices_);
  return *session != nullptr ? QDMI_SUCCESS : QDMI_ERROR_OUTOFMEM;
}

auto Driver::sessionAllocForDevice(const std::string_view id,
                                   const DeviceSessionConfig& config,
                                   QDMI_Session* const session) -> int {
  if (id.empty() || session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  auto device = openFresh(id, config, true);
  if (!device) {
    return qdmi::toStatus(device.takeError());
  }
  // Ownership passes to QDMI_session_free through the C handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new (std::nothrow) QDMI_Session_impl_d(std::move(*device));
  return *session != nullptr ? QDMI_SUCCESS : QDMI_ERROR_OUTOFMEM;
}

} // namespace qdmi

uint32_t QDMI_driver_get_client_abi_version() {
  return QDMI_CLIENT_ABI_VERSION;
}

// The private C ABI fixes these exported symbol names.
// NOLINTBEGIN(readability-identifier-naming)
extern "C" QDMI_DRIVER_EXPORT int
MQT_CORE_QDMI_driver_add_manifest_v1(const char* const manifestPath) {
  if (manifestPath == nullptr || *manifestPath == '\0') {
    return QDMI_ERROR_INVALIDARGUMENT;
  }

  auto path = qdmi::detail::pathFromString(manifestPath);
  if (!path) {
    return qdmi::toStatus(path.takeError());
  }
  return qdmi::detail::stageDeviceManifest(*path);
}

extern "C" QDMI_DRIVER_EXPORT int MQT_CORE_QDMI_driver_registered_device_ids_v1(
    const size_t size, char* const ids, size_t* const sizeRet) {
  if ((ids == nullptr && sizeRet == nullptr) || (ids != nullptr && size == 0)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }

  auto idsValue = qdmi::Driver::get().registeredDeviceIds();
  if (!idsValue) {
    return qdmi::toStatus(idsValue.takeError());
  }
  size_t required = 0;
  for (const auto& id : *idsValue) {
    required += id.size() + 1;
  }
  if (sizeRet != nullptr) {
    *sizeRet = required;
  }
  if (ids != nullptr) {
    if (size < required) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto output = llvm::MutableArrayRef(ids, required);
    for (const auto& id : *idsValue) {
      std::memcpy(output.data(), id.c_str(), id.size() + 1);
      output = output.drop_front(id.size() + 1);
    }
  }
  return QDMI_SUCCESS;
}

extern "C" QDMI_DRIVER_EXPORT int
MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
    const char* const deviceId, const size_t deviceSessionJsonSize,
    const char* const deviceSessionJson, QDMI_Session* const session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  if (deviceId == nullptr || *deviceId == '\0' ||
      ((deviceSessionJson == nullptr) != (deviceSessionJsonSize == 0))) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  qdmi::DeviceSessionConfig config;
  if (const auto status = qdmi::detail::parseDeviceSessionJson(
          deviceSessionJson, deviceSessionJsonSize, config);
      status != QDMI_SUCCESS) {
    return status;
  }

  return qdmi::Driver::get().sessionAllocForDevice(deviceId, config, session);
}
// NOLINTEND(readability-identifier-naming)

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;

  return qdmi::Driver::get().sessionAlloc(session);
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->init();
}

void QDMI_session_free(QDMI_Session session) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete session;
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

int QDMI_job_set_programs(QDMI_Job job, const QDMI_Program_Format format,
                          const size_t count, const size_t* sizes,
                          const void* const* programs) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->setPrograms(format, count, sizes, programs);
}

int QDMI_job_get_program(QDMI_Job job, const size_t programIndex,
                         const size_t size, void* data, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getProgram(programIndex, size, data, sizeRet);
}

int QDMI_job_get_program_status(QDMI_Job job, const size_t programIndex,
                                QDMI_Job_Status* status) {
  if (job == nullptr || status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getProgramStatus(programIndex, status);
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
