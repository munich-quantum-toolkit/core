/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"

#include "qdmi/common/Common.hpp"
#include "qdmi/common/Diagnostics.hpp"

#include <qdmi/client.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

#ifndef MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME
#error                                                                         \
    "MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME must name the packaged Client driver"
#endif

namespace qdmi {
namespace {
/// Rejects the formats that `submitJob` cannot carry.
/// A batch job's program is a list of job handles rather than a byte blob, so
/// this API cannot express it at all. A calibration run has its own entry
/// point, because its payload is optional and it takes no shot count.
void rejectUnsupportedProgramFormat(const QDMI_Program_Format format) {
  if (format == QDMI_PROGRAM_FORMAT_BATCHJOB) {
    throw std::invalid_argument(
        "MQT Core does not support batch jobs. A batch job's program is a list "
        "of job handles, which this API cannot express");
  }
  if (format == QDMI_PROGRAM_FORMAT_CALIBRATION) {
    throw std::invalid_argument(
        "Use submitCalibrationJob (submit_calibration_job in Python) to "
        "trigger a calibration run");
  }
}

#ifdef _WIN32
using LibraryHandle = HMODULE;

[[nodiscard]] auto thisModuleDirectory() -> std::filesystem::path {
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&thisModuleDirectory),
                         &module) == 0) {
    throw std::runtime_error("Cannot locate the MQT Core QDMI module");
  }
  std::wstring buffer(MAX_PATH, L'\0');
  while (true) {
    const auto size = GetModuleFileNameW(module, buffer.data(),
                                         static_cast<DWORD>(buffer.size()));
    if (size == 0) {
      throw std::runtime_error("Cannot locate the MQT Core QDMI module");
    }
    if (size < buffer.size()) {
      buffer.resize(size);
      return std::filesystem::path(buffer).parent_path();
    }
    buffer.resize(buffer.size() * 2U);
  }
}

[[nodiscard]] auto openLibrary(const std::filesystem::path& path)
    -> LibraryHandle {
  return LoadLibraryExW(path.wstring().c_str(), nullptr,
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

[[nodiscard]] auto findSymbol(LibraryHandle library, const char* name)
    -> void* {
  return reinterpret_cast<void*>(GetProcAddress(library, name));
}

void closeLibrary(LibraryHandle library) { FreeLibrary(library); }
#else
using LibraryHandle = void*;

[[nodiscard]] auto thisModuleDirectory() -> std::filesystem::path {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&thisModuleDirectory), &info) == 0 ||
      info.dli_fname == nullptr) {
    throw std::runtime_error("Cannot locate the MQT Core QDMI module");
  }
  auto path = std::filesystem::path(info.dli_fname);
  if (!path.is_absolute()) {
#ifdef __linux__
    std::error_code error;
    path = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
      throw std::runtime_error("Cannot locate the MQT Core QDMI executable");
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    static_cast<void>(_NSGetExecutablePath(nullptr, &size));
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
      throw std::runtime_error("Cannot locate the MQT Core QDMI executable");
    }
    path = buffer.data();
#else
    throw std::runtime_error("Cannot resolve the MQT Core QDMI module path");
#endif
  }
  return std::filesystem::weakly_canonical(path).parent_path();
}

[[nodiscard]] auto openLibrary(const std::filesystem::path& path)
    -> LibraryHandle {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

[[nodiscard]] auto findSymbol(LibraryHandle library, const char* name)
    -> void* {
  return dlsym(library, name);
}

void closeLibrary(LibraryHandle library) { dlclose(library); }
#endif

[[nodiscard]] auto normalizePath(const std::filesystem::path& path)
    -> std::filesystem::path {
  if (path.empty()) {
    throw std::invalid_argument("QDMI Client driver path must not be empty");
  }
  std::error_code error;
  auto normalized = std::filesystem::weakly_canonical(
      std::filesystem::absolute(path, error), error);
  if (error) {
    normalized = std::filesystem::absolute(path).lexically_normal();
  }
  return normalized;
}

[[nodiscard]] auto packagedDriverPath() -> std::filesystem::path {
  const auto directory = thisModuleDirectory();
  auto filename = std::filesystem::path{MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME};
  for (const auto& candidate :
       {directory / filename, directory / "lib" / filename,
        directory / "bin" / filename,
        directory.parent_path() / "lib" / filename,
        directory.parent_path() / "bin" / filename}) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
#ifdef _WIN32
  return directory / filename;
#else
  return filename;
#endif
}

[[nodiscard]] auto requestedDriverPath(const SessionConfig& config)
    -> std::filesystem::path {
  if (config.driverPath) {
    return normalizePath(*config.driverPath);
  }
  if (const auto environment =
          detail::environmentUtf8("MQT_CORE_QDMI_DRIVER")) {
    return normalizePath(detail::pathFromUtf8(*environment));
  }
  const auto packaged = packagedDriverPath();
  return packaged.has_parent_path() ? normalizePath(packaged) : packaged;
}

struct LoadedClient {
  LibraryHandle library{};
  std::shared_ptr<detail::ClientApi> api;

  LoadedClient(LibraryHandle selectedLibrary,
               std::shared_ptr<detail::ClientApi> selectedApi)
      : library(selectedLibrary), api(std::move(selectedApi)) {}

  ~LoadedClient() {
    if (library != nullptr) {
      closeLibrary(library);
    }
  }

  LoadedClient(const LoadedClient&) = delete;
  LoadedClient& operator=(const LoadedClient&) = delete;
  LoadedClient(LoadedClient&& other) noexcept
      : library(std::exchange(other.library, nullptr)),
        api(std::move(other.api)) {}
};

template <class Function>
[[nodiscard]] auto loadSymbol(LibraryHandle library, const char* name)
    -> Function {
  /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto function = reinterpret_cast<Function>(findSymbol(library, name));
  if (function == nullptr) {
    throw std::runtime_error("QDMI Client driver is missing symbol " +
                             std::string(name));
  }
  return function;
}

[[nodiscard]] auto loadClient(const std::filesystem::path& path)
    -> LoadedClient {
  auto* const library = openLibrary(path);
  if (library == nullptr) {
    throw std::runtime_error("Cannot load QDMI Client driver '" +
                             detail::pathToUtf8(path) + "'");
  }

  try {
    auto api = std::make_shared<detail::ClientApi>();
    api->driverGetClientAbiVersion =
        loadSymbol<decltype(api->driverGetClientAbiVersion)>(
            library, "QDMI_driver_get_client_abi_version");
    const auto actualAbi = api->driverGetClientAbiVersion();
    if (QDMI_VERSION_MAJOR(actualAbi) !=
            QDMI_VERSION_MAJOR(QDMI_CLIENT_ABI_VERSION) ||
        QDMI_VERSION_MINOR(actualAbi) !=
            QDMI_VERSION_MINOR(QDMI_CLIENT_ABI_VERSION)) {
      throw std::runtime_error("QDMI Client driver has incompatible ABI " +
                               std::to_string(QDMI_VERSION_MAJOR(actualAbi)) +
                               "." +
                               std::to_string(QDMI_VERSION_MINOR(actualAbi)));
    }

#define LOAD_CLIENT_SYMBOL(field, symbol)                                      \
  api->field = loadSymbol<decltype(api->field)>(library, #symbol)
    LOAD_CLIENT_SYMBOL(sessionAlloc, QDMI_session_alloc);
    LOAD_CLIENT_SYMBOL(sessionInit, QDMI_session_init);
    LOAD_CLIENT_SYMBOL(sessionFree, QDMI_session_free);
    LOAD_CLIENT_SYMBOL(sessionSetParameter, QDMI_session_set_parameter);
    LOAD_CLIENT_SYMBOL(sessionQueryProperty,
                       QDMI_session_query_session_property);
    LOAD_CLIENT_SYMBOL(deviceCreateJob, QDMI_device_create_job);
    LOAD_CLIENT_SYMBOL(sessionRetrieveJobById, QDMI_session_retrieve_job_by_id);
    LOAD_CLIENT_SYMBOL(jobFree, QDMI_job_free);
    LOAD_CLIENT_SYMBOL(jobSetParameter, QDMI_job_set_parameter);
    LOAD_CLIENT_SYMBOL(jobQueryProperty, QDMI_job_query_property);
    LOAD_CLIENT_SYMBOL(jobSubmit, QDMI_job_submit);
    LOAD_CLIENT_SYMBOL(jobCancel, QDMI_job_cancel);
    LOAD_CLIENT_SYMBOL(jobCheck, QDMI_job_check);
    LOAD_CLIENT_SYMBOL(jobWait, QDMI_job_wait);
    LOAD_CLIENT_SYMBOL(jobGetResults, QDMI_job_get_results);
    LOAD_CLIENT_SYMBOL(deviceQueryProperty, QDMI_device_query_device_property);
    LOAD_CLIENT_SYMBOL(deviceQuerySiteProperty,
                       QDMI_device_query_site_property);
    LOAD_CLIENT_SYMBOL(deviceQueryOperationProperty,
                       QDMI_device_query_operation_property);
#undef LOAD_CLIENT_SYMBOL
    return {library, std::move(api)};
  } catch (...) {
    closeLibrary(library);
    throw;
  }
}

struct ClientSelection {
  std::mutex mutex;
  /// Keeps the selected driver loaded for the process.
  LibraryHandle library{};
  std::shared_ptr<const detail::ClientApi> api;
  std::filesystem::path path;
};

[[nodiscard]] auto clientSelection() -> ClientSelection& {
  /// The selected driver must remain loaded until process teardown.
  /// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* selection = new ClientSelection{};
  return *selection;
}

void selectClient(ClientSelection& selection, LoadedClient& loaded,
                  std::filesystem::path& path) noexcept {
  selection.library = std::exchange(loaded.library, nullptr);
  selection.api = loaded.api;
  selection.path.swap(path);
}

using SessionGuard =
    std::unique_ptr<QDMI_Session_impl_d, decltype(&QDMI_session_free)>;

void validateSessionAllocation(const int status, QDMI_Session session) {
  if (session == nullptr &&
      (status == QDMI_SUCCESS || status == QDMI_WARN_GENERAL)) {
    throw std::runtime_error("The QDMI Client driver returned a null session");
  }
  qdmi::throwIfError(status, "Allocating QDMI session");
}

[[nodiscard]] auto allocateSession(const SessionConfig& config)
    -> std::shared_ptr<detail::ClientSession> {
  auto& selection = clientSelection();
  const std::scoped_lock lock(selection.mutex);
  if (selection.api != nullptr) {
    if (config.driverPath &&
        normalizePath(*config.driverPath) != selection.path) {
      throw std::runtime_error(
          "The QDMI Client driver is already selected for this process");
    }
    const auto api = selection.api;
    QDMI_Session session = nullptr;
    const auto result = api->sessionAlloc(&session);
    SessionGuard guard{session, api->sessionFree};
    validateSessionAllocation(result, session);
    auto owner = std::make_shared<detail::ClientSession>(api, session);
    /// The ClientSession now owns the raw QDMI session.
    /// NOLINTNEXTLINE(bugprone-unused-return-value)
    guard.release();
    return owner;
  }

  auto path = requestedDriverPath(config);
  auto loaded = loadClient(path);
  const std::shared_ptr<const detail::ClientApi> api = loaded.api;
  QDMI_Session session = nullptr;
  const auto result = api->sessionAlloc(&session);
  SessionGuard guard{session, api->sessionFree};
  validateSessionAllocation(result, session);
  auto owner = std::make_shared<detail::ClientSession>(api, session);
  selectClient(selection, loaded, path);
  /// The ClientSession now owns the raw QDMI session.
  /// NOLINTNEXTLINE(bugprone-unused-return-value)
  guard.release();
  return owner;
}
} // namespace

detail::ClientSession::~ClientSession() {
  if (handle != nullptr) {
    api->sessionFree(handle);
  }
}

void detail::JobDeleter::operator()(QDMI_Job_impl_d* const job) const {
  if (job != nullptr) {
    session->api->jobFree(job);
  }
}

size_t Site::getIndex() const {
  return queryProperty<size_t>(QDMI_SITE_PROPERTY_INDEX);
}
std::optional<uint64_t> Site::getT1() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T1);
}
std::optional<uint64_t> Site::getT2() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T2);
}
std::optional<std::string> Site::getName() const {
  return queryProperty<std::optional<std::string>>(QDMI_SITE_PROPERTY_NAME);
}
std::optional<int64_t> Site::getXCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_XCOORDINATE);
}
std::optional<int64_t> Site::getYCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_YCOORDINATE);
}
std::optional<int64_t> Site::getZCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_ZCOORDINATE);
}
bool Site::isZone() const {
  return queryProperty<std::optional<bool>>(QDMI_SITE_PROPERTY_ISZONE)
      .value_or(false);
}
std::optional<uint64_t> Site::getXExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_XEXTENT);
}
std::optional<uint64_t> Site::getYExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_YEXTENT);
}
std::optional<uint64_t> Site::getZExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_ZEXTENT);
}
std::optional<uint64_t> Site::getModuleIndex() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_MODULEINDEX);
}
std::optional<uint64_t> Site::getSubmoduleIndex() const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_SITE_PROPERTY_SUBMODULEINDEX);
}
std::string Operation::getName(const std::vector<Site>& sites,
                               const std::vector<double>& params) const {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
std::optional<size_t>
Operation::getQubitsNum(const std::vector<Site>& sites,
                        const std::vector<double>& params) const {
  return queryProperty<std::optional<size_t>>(QDMI_OPERATION_PROPERTY_QUBITSNUM,
                                              sites, params);
}
size_t Operation::getParametersNum(const std::vector<Site>& sites,
                                   const std::vector<double>& params) const {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
std::optional<uint64_t>
Operation::getDuration(const std::vector<Site>& sites,
                       const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_DURATION, sites, params);
}
std::optional<double>
Operation::getFidelity(const std::vector<Site>& sites,
                       const std::vector<double>& params) const {
  return queryProperty<std::optional<double>>(QDMI_OPERATION_PROPERTY_FIDELITY,
                                              sites, params);
}
std::optional<uint64_t>
Operation::getInteractionRadius(const std::vector<Site>& sites,
                                const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS, sites, params);
}
std::optional<uint64_t>
Operation::getBlockingRadius(const std::vector<Site>& sites,
                             const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites, params);
}
std::optional<double>
Operation::getIdlingFidelity(const std::vector<Site>& sites,
                             const std::vector<double>& params) const {
  return queryProperty<std::optional<double>>(
      QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites, params);
}
bool Operation::isZoned() const {
  return queryProperty<std::optional<bool>>(QDMI_OPERATION_PROPERTY_ISZONED, {},
                                            {})
      .value_or(false);
}
std::optional<std::vector<Site>> Operation::getSites() const {
  const auto& qdmiSites = queryProperty<std::optional<std::vector<QDMI_Site>>>(
      QDMI_OPERATION_PROPERTY_SITES, {}, {});
  if (!qdmiSites.has_value()) {
    return std::nullopt;
  }
  std::vector<Site> returnedSites;
  returnedSites.reserve(qdmiSites->size());
  std::ranges::transform(*qdmiSites, std::back_inserter(returnedSites),
                         [this](const QDMI_Site& site) -> Site {
                           return {device_, session_, site};
                         });
  return returnedSites;
}
std::optional<std::vector<std::pair<Site, Site>>>
Operation::getSitePairs() const {
  if (const auto qubitsNum = getQubitsNum({}, {});
      !qubitsNum.has_value() || *qubitsNum != 2 || isZoned()) {
    return std::nullopt; // Not a 2-qubit operation or operation is zoned
  }

  const auto sitesOpt = getSites();
  if (!sitesOpt.has_value()) {
    return std::nullopt;
  }

  const auto& sitesVec = *sitesOpt;
  if (sitesVec.empty() || sitesVec.size() % 2 != 0) {
    return std::nullopt; // Invalid: no sites or odd number of sites
  }

  std::vector<std::pair<Site, Site>> pairs;
  pairs.reserve(sitesVec.size() / 2);

  for (size_t i = 0; i < sitesVec.size(); i += 2) {
    pairs.emplace_back(sitesVec[i], sitesVec[i + 1]);
  }

  return pairs;
}
std::optional<uint64_t>
Operation::getMeanShuttlingSpeed(const std::vector<Site>& sites,
                                 const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED, sites, params);
}
std::string Device::getId() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_ID);
}
std::string Device::getName() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_NAME);
}

std::string Device::getVersion() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_VERSION);
}

QDMI_Device_Status Device::getStatus() const {
  return queryProperty<QDMI_Device_Status>(QDMI_DEVICE_PROPERTY_STATUS);
}

std::string Device::getLibraryVersion() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LIBRARYVERSION);
}

size_t Device::getQubitsNum() const {
  return queryProperty<size_t>(QDMI_DEVICE_PROPERTY_QUBITSNUM);
}

std::vector<Site> Device::getSites() const {
  const auto& qdmiSites =
      queryProperty<std::vector<QDMI_Site>>(QDMI_DEVICE_PROPERTY_SITES);
  std::vector<Site> sites;
  sites.reserve(qdmiSites.size());
  std::ranges::transform(qdmiSites, std::back_inserter(sites),
                         [this](const QDMI_Site& site) -> Site {
                           return {device_, session_, site};
                         });
  return sites;
}

std::vector<Site> Device::getRegularSites() const {
  auto allSites = getSites();
  const auto newEnd = std::ranges::remove_if(
      allSites, [](const auto& s) { return s.isZone(); });
  allSites.erase(newEnd.begin(), newEnd.end());
  return allSites;
}

std::vector<Site> Device::getZones() const {
  const auto& allSites = getSites();
  std::vector<Site> zones;
  zones.reserve(3); // Reserve space for a typical max number of zones
  std::ranges::copy_if(allSites, std::back_inserter(zones),
                       [](const auto& s) { return s.isZone(); });
  return zones;
}

std::vector<Operation> Device::getOperations() const {
  const auto& qdmiOperations = queryProperty<std::vector<QDMI_Operation>>(
      QDMI_DEVICE_PROPERTY_OPERATIONS);
  return wrapOperations(qdmiOperations);
}

std::optional<std::vector<Operation>>
Device::queryCustomOperations(const CustomProperty property) const {
  const auto qdmiProperty = detail::toDeviceProperty(property);
  const auto handles = detail::queryHandleArray<QDMI_Operation>(
      [this, qdmiProperty](const size_t size, void* value, size_t* sizeRet) {
        return api().deviceQueryProperty(device_, qdmiProperty, size, value,
                                         sizeRet);
      },
      "custom operation list " +
          std::to_string(static_cast<unsigned>(property)));
  if (!handles.has_value()) {
    return std::nullopt;
  }
  return wrapOperations(*handles);
}

std::vector<Operation>
Device::wrapOperations(const std::span<const QDMI_Operation> operations) const {
  std::vector<Operation> wrappedOperations;
  wrappedOperations.reserve(operations.size());
  std::ranges::transform(operations, std::back_inserter(wrappedOperations),
                         [this](const QDMI_Operation& op) -> Operation {
                           return {device_, session_, op};
                         });
  return wrappedOperations;
}

std::optional<std::vector<std::pair<Site, Site>>>
Device::getCouplingMap() const {
  const auto& qdmiCouplingMap = queryProperty<
      std::optional<std::vector<std::pair<QDMI_Site, QDMI_Site>>>>(
      QDMI_DEVICE_PROPERTY_COUPLINGMAP);
  if (!qdmiCouplingMap.has_value()) {
    return std::nullopt;
  }

  std::vector<std::pair<Site, Site>> couplingMap;
  couplingMap.reserve(qdmiCouplingMap->size());
  std::ranges::transform(*qdmiCouplingMap, std::back_inserter(couplingMap),
                         [this](const std::pair<QDMI_Site, QDMI_Site>& pair)
                             -> std::pair<Site, Site> {
                           return {
                               Site{device_, session_, pair.first},
                               Site{device_, session_, pair.second},
                           };
                         });
  return couplingMap;
}

std::optional<size_t> Device::getNeedsCalibration() const {
  return queryProperty<std::optional<size_t>>(
      QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION);
}

std::optional<size_t> Device::getQueueLength() const {
  return queryProperty<std::optional<size_t>>(QDMI_DEVICE_PROPERTY_QUEUELENGTH);
}

std::optional<std::string> Device::getLengthUnit() const {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_LENGTHUNIT);
}

std::optional<double> Device::getLengthScaleFactor() const {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR);
}

std::optional<std::string> Device::getDurationUnit() const {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_DURATIONUNIT);
}

std::optional<double> Device::getDurationScaleFactor() const {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR);
}

std::optional<uint64_t> Device::getMinAtomDistance() const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_DEVICE_PROPERTY_MINATOMDISTANCE);
}

std::vector<QDMI_Program_Format> Device::getSupportedProgramFormats() const {
  return queryProperty<std::vector<QDMI_Program_Format>>(
      QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS);
}

std::vector<Device> Device::getChildDevices() const {
  size_t size = 0;
  auto result = api().deviceQueryProperty(
      device_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr, &size);
  if (result == QDMI_ERROR_NOTSUPPORTED) {
    return {};
  }
  qdmi::throwIfError(result, "Querying child devices size");
  if (size % sizeof(QDMI_Device) != 0) {
    throw std::runtime_error("Invalid child device list size");
  }

  std::vector<QDMI_Device> handles(size / sizeof(QDMI_Device));
  if (size != 0) {
    result = api().deviceQueryProperty(
        device_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
        static_cast<void*>(handles.data()), nullptr);
    qdmi::throwIfError(result, "Querying child devices");
  }

  std::vector<Device> devices;
  devices.reserve(handles.size());
  std::ranges::transform(handles, std::back_inserter(devices),
                         [this](QDMI_Device_impl_d* const handle) {
                           return Device(handle, session_);
                         });
  return devices;
}

Job Device::submitJob(const std::string& program,
                      const QDMI_Program_Format format, const size_t numShots,
                      const std::optional<CustomJobParameter>& custom1,
                      const std::optional<CustomJobParameter>& custom2,
                      const std::optional<CustomJobParameter>& custom3,
                      const std::optional<CustomJobParameter>& custom4,
                      const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    throw std::invalid_argument(
        "Binary program formats require exact-byte submission");
  }
  rejectUnsupportedProgramFormat(format);

  const auto bytes = std::as_bytes(
      std::span(program.c_str(), static_cast<size_t>(program.size() + 1)));
  return submitJob(bytes, format, numShots, custom1, custom2, custom3, custom4,
                   custom5);
}

Job Device::submitJob(const std::string& program,
                      const QDMI_Program_Format format,
                      const std::optional<CustomJobParameter>& custom1,
                      const std::optional<CustomJobParameter>& custom2,
                      const std::optional<CustomJobParameter>& custom3,
                      const std::optional<CustomJobParameter>& custom4,
                      const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    throw std::invalid_argument(
        "Binary program formats require exact-byte submission");
  }
  rejectUnsupportedProgramFormat(format);

  const auto bytes = std::as_bytes(
      std::span(program.c_str(), static_cast<size_t>(program.size() + 1)));
  return submitJob(bytes, format, custom1, custom2, custom3, custom4, custom5);
}

Job Device::submitJob(const std::span<const std::byte> program,
                      const QDMI_Program_Format format, const size_t numShots,
                      const std::optional<CustomJobParameter>& custom1,
                      const std::optional<CustomJobParameter>& custom2,
                      const std::optional<CustomJobParameter>& custom3,
                      const std::optional<CustomJobParameter>& custom4,
                      const std::optional<CustomJobParameter>& custom5) const {
  rejectUnsupportedProgramFormat(format);

  return submitJobImpl(format, program, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

Job Device::submitJob(const std::span<const std::byte> program,
                      const QDMI_Program_Format format,
                      const std::optional<CustomJobParameter>& custom1,
                      const std::optional<CustomJobParameter>& custom2,
                      const std::optional<CustomJobParameter>& custom3,
                      const std::optional<CustomJobParameter>& custom4,
                      const std::optional<CustomJobParameter>& custom5) const {
  rejectUnsupportedProgramFormat(format);

  return submitJobImpl(format, program, std::nullopt, custom1, custom2, custom3,
                       custom4, custom5);
}

Job Device::submitJobImpl(
    const QDMI_Program_Format format,
    const std::optional<std::span<const std::byte>> program,
    const std::optional<size_t> numShots,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  QDMI_Job job = nullptr;
  qdmi::throwIfError(api().deviceCreateJob(device_, &job),
                     "Creating job");
  Job jobWrapper{job, session_};

  qdmi::throwIfError(api().jobSetParameter(jobWrapper,
                                            QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                            sizeof(format), &format),
                     "Setting program format");
  if (program.has_value()) {
    qdmi::throwIfError(api().jobSetParameter(jobWrapper,
                                              QDMI_JOB_PARAMETER_PROGRAM,
                                              program->size(), program->data()),
                       "Setting program");
  }
  if (numShots.has_value()) {
    qdmi::throwIfError(api().jobSetParameter(jobWrapper,
                                              QDMI_JOB_PARAMETER_SHOTSNUM,
                                              sizeof(*numShots), &*numShots),
                       "Setting number of shots");
  }

  if (custom1.has_value()) {
    setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM1, *custom1);
  }
  if (custom2.has_value()) {
    setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM2, *custom2);
  }
  if (custom3.has_value()) {
    setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM3, *custom3);
  }
  if (custom4.has_value()) {
    setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM4, *custom4);
  }
  if (custom5.has_value()) {
    setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM5, *custom5);
  }

  qdmi::throwIfError(api().jobSubmit(jobWrapper), "Submitting job");
  return jobWrapper;
}

Job Device::submitCalibrationJob(
    const std::optional<std::span<const std::byte>> program,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  const auto payload =
      program.has_value() && !program->empty() ? program : std::nullopt;
  return submitJobImpl(QDMI_PROGRAM_FORMAT_CALIBRATION, payload, std::nullopt,
                       custom1, custom2, custom3, custom4, custom5);
}

Job Device::submitCalibrationJob(
    const std::string& program,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  const auto bytes = std::as_bytes(
      std::span(program.c_str(), static_cast<size_t>(program.size() + 1)));
  return submitCalibrationJob(bytes, custom1, custom2, custom3, custom4,
                              custom5);
}

Job Device::retrieveJobById(const std::string_view jobId) const {
  const std::string id{jobId};
  QDMI_Job job = nullptr;
  qdmi::throwIfError(api().sessionRetrieveJobById(device_, id.c_str(), &job),
                     "Retrieving job");
  return Job{job, session_};
}

void Device::setCustomJobParam(QDMI_Job job, const QDMI_Job_Parameter param,
                               const CustomJobParameter& value) const {
  std::visit(
      [&]<typename CustomValue>(const CustomValue& customValue) {
        using T = std::decay_t<CustomValue>;
        if constexpr (std::is_same_v<T, std::string>) {
          qdmi::throwIfError(api().jobSetParameter(job, param,
                                                   customValue.size() + 1,
                                                   customValue.c_str()),
                             "Setting custom parameter");
        } else {
          static_assert(std::is_trivially_copyable_v<T>,
                        "Custom job parameters must be trivially copyable");
          qdmi::throwIfError(
              api().jobSetParameter(job, param, sizeof(T), &customValue),
              "Setting custom parameter");
        }
      },
      value);
}

QDMI_Job_Status Job::check() const {
  QDMI_Job_Status status{};
  qdmi::throwIfError(api().jobCheck(job_.get(), &status),
                     "Checking job status");
  return status;
}

bool Job::wait(const size_t timeout) const {
  const auto ret = api().jobWait(job_.get(), timeout);
  if (ret == QDMI_SUCCESS) {
    return true;
  }
  if (ret == QDMI_ERROR_TIMEOUT) {
    return false;
  }
  qdmi::throwIfError(ret, "Waiting for job");
  qdmi::unreachable();
}

void Job::cancel() const {
  qdmi::throwIfError(api().jobCancel(job_.get()), "Cancelling job");
}

std::string Job::getId() const {
  size_t size = 0;
  qdmi::throwIfError(api().jobQueryProperty(job_.get(), QDMI_JOB_PROPERTY_ID, 0,
                                            nullptr, &size),
                     "Querying job ID size");
  std::string id(size, '\0');
  qdmi::throwIfError(api().jobQueryProperty(job_.get(), QDMI_JOB_PROPERTY_ID,
                                            size, id.data(), nullptr),
                     "Querying job ID");
  return detail::decodeText(std::move(id), "Job ID");
}

QDMI_Program_Format Job::getProgramFormat() const {
  QDMI_Program_Format format{};
  qdmi::throwIfError(api().jobQueryProperty(job_.get(),
                                            QDMI_JOB_PROPERTY_PROGRAMFORMAT,
                                            sizeof(format), &format, nullptr),
                     "Querying program format");
  return format;
}

std::vector<std::byte> Job::getProgramBytes() const {
  size_t size = 0;
  qdmi::throwIfError(api().jobQueryProperty(job_.get(),
                                            QDMI_JOB_PROPERTY_PROGRAM, 0,
                                            nullptr, &size),
                     "Querying program size");

  std::vector<std::byte> program(size);
  if (size != 0) {
    qdmi::throwIfError(api().jobQueryProperty(job_.get(),
                                              QDMI_JOB_PROPERTY_PROGRAM, size,
                                              program.data(), nullptr),
                       "Querying program");
  }
  return program;
}

std::string Job::getProgram() const {
  const auto format = getProgramFormat();
  if (isBinaryProgramFormat(format)) {
    throw std::invalid_argument(
        "Cannot decode a binary program as a string; use getProgramBytes()");
  }

  const auto program = getProgramBytes();
  if (program.empty() || program.back() != std::byte{0}) {
    throw std::invalid_argument(
        "Cannot decode program as a null-terminated string; use "
        "getProgramBytes() for binary payloads");
  }
  return {reinterpret_cast<const char*>(program.data()), program.size() - 1};
}

size_t Job::getNumShots() const {
  size_t numShots = 0;
  qdmi::throwIfError(
      api().jobQueryProperty(job_.get(), QDMI_JOB_PROPERTY_SHOTSNUM,
                             sizeof(numShots), &numShots, nullptr),
      "Querying number of shots");
  return numShots;
}

std::optional<size_t> Job::getQueuePosition() const {
  size_t queuePosition = 0;
  const auto result =
      api().jobQueryProperty(job_.get(), QDMI_JOB_PROPERTY_QUEUEPOSITION,
                             sizeof(queuePosition), &queuePosition, nullptr);
  return detail::queuePositionFromResult(result, queuePosition);
}

std::vector<std::string> Job::getShots() const {
  size_t shotsSize = 0;
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_SHOTS, 0, nullptr,
                                         &shotsSize),
                     "Querying shots size");

  if (shotsSize == 0) {
    return {};
  }

  std::string shots(shotsSize, '\0');
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_SHOTS, shotsSize,
                                         shots.data(), nullptr),
                     "Querying shots");
  shots.pop_back();

  return detail::parseShots(shots, getNumShots());
}

std::map<std::string, size_t> Job::getCounts() const {
  // Get the histogram keys
  size_t keysSize = 0;
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr,
                                         &keysSize),
                     "Querying histogram keys size");

  if (keysSize == 0) {
    return {}; // Empty histogram
  }

  std::string keys(keysSize, '\0');
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_HIST_KEYS, keysSize,
                                         keys.data(), nullptr),
                     "Querying histogram keys");
  keys.pop_back();

  // Get the histogram values
  size_t valuesSize = 0;
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_HIST_VALUES, 0,
                                         nullptr, &valuesSize),
                     "Querying histogram values size");

  if (valuesSize % sizeof(size_t) != 0) {
    throw std::runtime_error(
        "Invalid histogram values size: not a multiple of size_t");
  }

  std::vector<size_t> values(valuesSize / sizeof(size_t));
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_HIST_VALUES,
                                         valuesSize, values.data(), nullptr),
                     "Querying histogram values");

  // Parse the keys (comma-separated)
  std::map<std::string, size_t> counts;
  if (keys.empty() && values.size() == 1) {
    counts[""] = values.front();
    return counts;
  }
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx < values.size()) {
      counts[key] = values[idx];
      ++idx;
    }
  }

  if (idx != values.size()) {
    throw std::runtime_error("Histogram key/value count mismatch");
  }

  return counts;
}

std::vector<std::complex<double>> Job::getDenseStateVector() const {
  size_t size = 0;
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0,
                                         nullptr, &size),
                     "Querying dense state vector size");

  if (size % sizeof(std::complex<double>) != 0) {
    throw std::runtime_error(
        "Invalid state vector size: not a multiple of complex<double>");
  }

  std::vector<std::complex<double>> stateVector(size /
                                                sizeof(std::complex<double>));
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_STATEVECTOR_DENSE,
                                         size, stateVector.data(), nullptr),
                     "Querying dense state vector");
  return stateVector;
}

std::vector<double> Job::getDenseProbabilities() const {
  size_t size = 0;
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0,
                                         nullptr, &size),
                     "Querying dense probabilities size");

  if (size % sizeof(double) != 0) {
    throw std::runtime_error(
        "Invalid probabilities size: not a multiple of double");
  }

  std::vector<double> probabilities(size / sizeof(double));
  qdmi::throwIfError(api().jobGetResults(job_.get(),
                                         QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                                         size, probabilities.data(), nullptr),
                     "Querying dense probabilities");
  return probabilities;
}

std::map<std::string, std::complex<double>> Job::getSparseStateVector() const {
  size_t keysSize = 0;
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr,
                          &keysSize),
      "Querying sparse state vector keys size");

  if (keysSize == 0) {
    return {}; // Empty state vector
  }

  std::string keys(keysSize, '\0');
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, keysSize,
                          keys.data(), nullptr),
      "Querying sparse state vector keys");
  keys.pop_back();

  size_t valuesSize = 0;
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0, nullptr,
                          &valuesSize),
      "Querying sparse state vector values size");

  if (valuesSize % sizeof(std::complex<double>) != 0) {
    throw std::runtime_error(
        "Invalid sparse state vector values size: not a multiple of "
        "complex<double>");
  }

  std::vector<std::complex<double>> values(valuesSize /
                                           sizeof(std::complex<double>));
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, valuesSize,
                          values.data(), nullptr),
      "Querying sparse state vector values");

  // Parse the keys (comma-separated)
  std::map<std::string, std::complex<double>> stateVector;
  if (keys.empty() && values.size() == 1) {
    stateVector[""] = values.front();
    return stateVector;
  }
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx >= values.size()) {
      throw std::runtime_error("Sparse state vector key/value count mismatch");
    }
    stateVector[key] = values[idx];
    ++idx;
  }

  if (idx != values.size()) {
    throw std::runtime_error("Sparse state vector key/value count mismatch");
  }
  return stateVector;
}

std::map<std::string, double> Job::getSparseProbabilities() const {
  size_t keysSize = 0;
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0, nullptr,
                          &keysSize),
      "Querying sparse probabilities keys size");

  if (keysSize == 0) {
    return {}; // Empty probabilities
  }

  std::string keys(keysSize, '\0');
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, keysSize,
                          keys.data(), nullptr),
      "Querying sparse probabilities keys");
  keys =
      detail::decodeText(std::move(keys), "Sparse probabilities keys result");

  size_t valuesSize = 0;
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0,
                          nullptr, &valuesSize),
      "Querying sparse probabilities values size");

  if (valuesSize % sizeof(double) != 0) {
    throw std::runtime_error(
        "Invalid sparse probabilities values size: not a multiple of double");
  }

  std::vector<double> values(valuesSize / sizeof(double));
  qdmi::throwIfError(
      api().jobGetResults(job_.get(),
                          QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES,
                          valuesSize, values.data(), nullptr),
      "Querying sparse probabilities values");

  // Parse the keys (comma-separated)
  std::map<std::string, double> probabilities;
  if (keys.empty() && values.size() == 1) {
    probabilities[""] = values.front();
    return probabilities;
  }
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx >= values.size()) {
      throw std::runtime_error("Sparse probabilities key/value count mismatch");
    }
    probabilities[key] = values[idx];
    ++idx;
  }
  if (idx != values.size()) {
    throw std::runtime_error("Sparse probabilities key/value count mismatch");
  }
  return probabilities;
}

Device Session::openDevice(const std::string_view id,
                           const SessionConfig& config) {
  if (id.empty() || id.find('\0') != std::string_view::npos) {
    throw std::invalid_argument(
        "QDMI device ID must not be empty or contain null bytes");
  }
  Session session(config);
  auto devices = session.getDevices();
  std::string available;
  for (const auto& device : devices) {
    const auto candidateId = device.getId();
    if (candidateId == id) {
      return device;
    }
    if (!available.empty()) {
      available += ", ";
    }
    available += candidateId;
  }
  throw std::out_of_range("QDMI Client session has no device with ID '" +
                          std::string(id) + "'; available IDs: " + available);
}

Session::Session(const SessionConfig& config) {
  session_ = allocateSession(config);

  const auto setParameter = [this](const std::optional<std::string>& value,
                                   QDMI_Session_Parameter param) -> void {
    if (!value) {
      return;
    }
    const auto status = static_cast<QDMI_STATUS>(api().sessionSetParameter(
        session_->handle, param, value->size() + 1U, value->c_str()));
    if (status == QDMI_ERROR_NOTSUPPORTED) {
      qdmi::diagnostics::info("Session parameter {} not supported (skipped)",
                  qdmi::toString(param));
      return;
    }
    if (status != QDMI_SUCCESS) {
      std::ostringstream message;
      message << "Setting session parameter " << qdmi::toString(param) << ": "
              << qdmi::toString(status) << " (status = " << status << ")";
      qdmi::throwIfError(status, message.str());
    }
  };

  setParameter(config.token, QDMI_SESSION_PARAMETER_TOKEN);
  if (config.authFile) {
    const std::optional authFile = detail::pathToUtf8(*config.authFile);
    setParameter(authFile, QDMI_SESSION_PARAMETER_AUTHFILE);
  }
  setParameter(config.authUrl, QDMI_SESSION_PARAMETER_AUTHURL);
  setParameter(config.username, QDMI_SESSION_PARAMETER_USERNAME);
  setParameter(config.password, QDMI_SESSION_PARAMETER_PASSWORD);
  setParameter(config.projectId, QDMI_SESSION_PARAMETER_PROJECTID);
  setParameter(config.custom1, QDMI_SESSION_PARAMETER_CUSTOM1);
  setParameter(config.custom2, QDMI_SESSION_PARAMETER_CUSTOM2);
  setParameter(config.custom3, QDMI_SESSION_PARAMETER_CUSTOM3);
  setParameter(config.custom4, QDMI_SESSION_PARAMETER_CUSTOM4);
  setParameter(config.custom5, QDMI_SESSION_PARAMETER_CUSTOM5);

  qdmi::throwIfError(api().sessionInit(session_->handle),
                     "Initializing session");
}

std::vector<Device> Session::getDevices() {
  const auto& qdmiDevices =
      queryProperty<std::vector<QDMI_Device>>(QDMI_SESSION_PROPERTY_DEVICES);
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(qdmiDevices, std::back_inserter(devices),
                         [this](QDMI_Device_impl_d* const& dev) -> Device {
                           return {dev, session_};
                         });
  return devices;
}
} // namespace qdmi
